"""Nexus3D Math Engine - 3D mathematics for AI agents.

Provides vectors, matrices, quaternions, transforms, physics, and IK solvers.
All operations are NumPy-backed for high performance and batch processing.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import math


# ─── Vector Operations ───────────────────────────────────────────────────────

def vec3(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    """Create a 3D vector."""
    return np.array([x, y, z], dtype=np.float64)


def vec4(x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0) -> np.ndarray:
    """Create a 4D vector."""
    return np.array([x, y, z, w], dtype=np.float64)


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-10:
        return np.zeros_like(v)
    return v / n


def magnitude(v: np.ndarray) -> float:
    """Return the magnitude (length) of a vector."""
    return float(np.linalg.norm(v))


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the dot product of two vectors."""
    return float(np.dot(a, b))


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the cross product of two 3D vectors."""
    return np.cross(a, b)


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two vectors."""
    t = np.clip(t, 0.0, 1.0)
    return a + (b - a) * t


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(a - b))


def reflect(incident: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Reflect a vector around a normal."""
    return incident - 2.0 * dot(incident, normal) * normal


def refract(incident: np.ndarray, normal: np.ndarray, eta: float) -> np.ndarray:
    """Refract a vector through a surface. Returns None on total internal reflection."""
    cos_i = -dot(incident, normal)
    sin2_t = eta ** 2 * (1.0 - cos_i ** 2)
    if sin2_t > 1.0:
        return None  # Total internal reflection
    cos_t = math.sqrt(1.0 - sin2_t)
    return eta * incident + (eta * cos_i - cos_t) * normal


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in radians between two vectors."""
    cos_angle = np.clip(dot(normalize(a), normalize(b)), -1.0, 1.0)
    return float(np.arccos(cos_angle))


def project_onto(v: np.ndarray, onto: np.ndarray) -> np.ndarray:
    """Project vector v onto vector onto."""
    return dot(v, onto) / dot(onto, onto) * onto


def perpendicular(v: np.ndarray) -> np.ndarray:
    """Return a vector perpendicular to the given vector."""
    if abs(v[0]) < abs(v[1]):
        return normalize(cross(v, vec3(1, 0, 0)))
    else:
        return normalize(cross(v, vec3(0, 1, 0)))


# ─── Matrix Operations ───────────────────────────────────────────────────────

def mat4_identity() -> np.ndarray:
    """4x4 identity matrix."""
    return np.eye(4, dtype=np.float64)


def mat4_translate(tx: float, ty: float, tz: float) -> np.ndarray:
    """4x4 translation matrix."""
    m = np.eye(4, dtype=np.float64)
    m[0, 3] = tx
    m[1, 3] = ty
    m[2, 3] = tz
    return m


def mat4_scale(sx: float, sy: float, sz: float) -> np.ndarray:
    """4x4 scale matrix."""
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def mat4_rotate_x(angle_rad: float) -> np.ndarray:
    """4x4 rotation matrix around X axis."""
    m = np.eye(4, dtype=np.float64)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    m[1, 1] = c;  m[1, 2] = -s
    m[2, 1] = s;  m[2, 2] = c
    return m


def mat4_rotate_y(angle_rad: float) -> np.ndarray:
    """4x4 rotation matrix around Y axis."""
    m = np.eye(4, dtype=np.float64)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    m[0, 0] = c;  m[0, 2] = s
    m[2, 0] = -s; m[2, 2] = c
    return m


def mat4_rotate_z(angle_rad: float) -> np.ndarray:
    """4x4 rotation matrix around Z axis."""
    m = np.eye(4, dtype=np.float64)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    m[0, 0] = c;  m[0, 1] = -s
    m[1, 0] = s;  m[1, 1] = c
    return m


def mat4_rotate_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """4x4 rotation matrix around an arbitrary axis (Rodrigues' formula)."""
    axis = normalize(axis)
    K = np.array([
        [0, -axis[2], axis[1], 0],
        [axis[2], 0, -axis[0], 0],
        [-axis[1], axis[0], 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.float64)
    return np.eye(4, dtype=np.float64) + math.sin(angle_rad) * K + (1 - math.cos(angle_rad)) * (K @ K)


def mat4_look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """Construct a look-at view matrix."""
    if up is None:
        up = vec3(0, 1, 0)
    forward = normalize(target - eye)
    right = normalize(cross(forward, up))
    true_up = cross(right, forward)

    m = np.eye(4, dtype=np.float64)
    m[0, :3] = right
    m[1, :3] = true_up
    m[2, :3] = -forward
    m[0, 3] = -dot(right, eye)
    m[1, 3] = -dot(true_up, eye)
    m[2, 3] = dot(forward, eye)
    return m


def mat4_perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Perspective projection matrix."""
    fov_rad = math.radians(fov_deg)
    f = 1.0 / math.tan(fov_rad / 2.0)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def mat4_orthographic(left: float, right: float, bottom: float, top: float, near: float, far: float) -> np.ndarray:
    """Orthographic projection matrix."""
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m


def decompose_matrix(m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a 4x4 transform matrix into (translation, rotation_quat, scale)."""
    translation = m[:3, 3].copy()
    scale = np.array([np.linalg.norm(m[:3, 0]), np.linalg.norm(m[:3, 1]), np.linalg.norm(m[:3, 2])])

    rot_mat = m[:3, :3].copy()
    rot_mat[:, 0] /= scale[0]
    rot_mat[:, 1] /= scale[1]
    rot_mat[:, 2] /= scale[2]

    # Convert rotation matrix to quaternion
    rotation = rotation_matrix_to_quat(rot_mat)
    return translation, rotation, scale


def compose_matrix(translation: np.ndarray, rotation: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Compose a 4x4 transform matrix from translation, rotation (quaternion), and scale."""
    T = mat4_translate(translation[0], translation[1], translation[2])
    R3 = quat_to_rotation_matrix(rotation)
    R = np.eye(4, dtype=np.float64)
    R[:3, :3] = R3
    S = mat4_scale(scale[0], scale[1], scale[2])
    return T @ R @ S


# ─── Quaternion Operations ───────────────────────────────────────────────────

def quat(w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    """Create a quaternion [w, x, y, z]."""
    q = np.array([w, x, y, z], dtype=np.float64)
    return normalize_quat(q)


def quat_identity() -> np.ndarray:
    """Identity quaternion (no rotation)."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion."""
    n = np.linalg.norm(q)
    if n < 1e-10:
        return quat_identity()
    return q / n


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of a quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Inverse of a quaternion."""
    return quat_conjugate(q) / np.dot(q, q)


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix."""
    q = normalize_quat(q)
    w, x, y, z = q
    m = np.eye(3, dtype=np.float64)
    m[0, 0] = 1 - 2*(y*y + z*z);  m[0, 1] = 2*(x*y - w*z);      m[0, 2] = 2*(x*z + w*y)
    m[1, 0] = 2*(x*y + w*z);      m[1, 1] = 1 - 2*(x*x + z*z);  m[1, 2] = 2*(y*z - w*x)
    m[2, 0] = 2*(x*z - w*y);      m[2, 1] = 2*(y*z + w*x);      m[2, 2] = 1 - 2*(x*x + y*y)
    return m


def rotation_matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return np.array([0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s], dtype=np.float64)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s], dtype=np.float64)
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s], dtype=np.float64)
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s], dtype=np.float64)


def quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Create quaternion from axis-angle representation."""
    axis = normalize(axis)
    half = angle_rad / 2.0
    s = math.sin(half)
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def quat_to_axis_angle(q: np.ndarray) -> Tuple[np.ndarray, float]:
    """Convert quaternion to axis-angle representation."""
    q = normalize_quat(q)
    if q[0] < 0:
        q = -q
    angle = 2.0 * math.acos(np.clip(q[0], -1.0, 1.0))
    s = math.sqrt(1.0 - q[0] ** 2)
    if s < 1e-10:
        return vec3(0, 1, 0), 0.0
    axis = q[1:] / s
    return axis, angle


def quat_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions."""
    a = normalize_quat(a)
    b = normalize_quat(b)
    dot_val = np.dot(a, b)

    # Take shorter path
    if dot_val < 0.0:
        b = -b
        dot_val = -dot_val

    dot_val = np.clip(dot_val, -1.0, 1.0)

    if dot_val > 0.9995:
        # Linear interpolation for very close quaternions
        result = a + (b - a) * t
        return normalize_quat(result)

    theta_0 = math.acos(dot_val)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    wa = math.cos(theta) - dot_val * sin_theta / sin_theta_0
    wb = sin_theta / sin_theta_0
    return normalize_quat(wa * a + wb * b)


def quat_to_euler(q: np.ndarray, order: str = 'XYZ') -> np.ndarray:
    """Convert quaternion to Euler angles (radians)."""
    q = normalize_quat(q)
    w, x, y, z = q

    if order == 'XYZ':
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    elif order == 'ZYX':
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    else:
        raise ValueError(f"Unsupported Euler order: {order}")

    return np.array([roll, pitch, yaw], dtype=np.float64)


def euler_to_quat(angles: np.ndarray, order: str = 'XYZ') -> np.ndarray:
    """Convert Euler angles (radians) to quaternion."""
    result = quat_identity()
    axis_map = {'X': vec3(1, 0, 0), 'Y': vec3(0, 1, 0), 'Z': vec3(0, 0, 1)}
    for i, axis_name in enumerate(order):
        result = quat_multiply(quat_from_axis_angle(axis_map[axis_name], angles[i]), result)
    return result


# ─── Geometry Utilities ──────────────────────────────────────────────────────

def point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """Check if point p lies inside triangle abc using barycentric coordinates."""
    v0 = c - a
    v1 = b - a
    v2 = p - a
    dot00 = dot(v0, v0)
    dot01 = dot(v0, v1)
    dot02 = dot(v0, v2)
    dot11 = dot(v1, v1)
    dot12 = dot(v1, v2)
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    return (u >= 0) and (v >= 0) and (u + v <= 1)


def ray_plane_intersect(ray_origin: np.ndarray, ray_dir: np.ndarray,
                        plane_point: np.ndarray, plane_normal: np.ndarray) -> Optional[float]:
    """Intersect a ray with a plane. Returns distance t or None."""
    denom = dot(ray_dir, plane_normal)
    if abs(denom) < 1e-10:
        return None
    t = dot(plane_point - ray_origin, plane_normal) / denom
    return t if t >= 0 else None


def ray_triangle_intersect(ray_origin: np.ndarray, ray_dir: np.ndarray,
                           v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> Optional[float]:
    """Möller–Trumbore ray-triangle intersection. Returns distance t or None."""
    EPSILON = 1e-10
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = cross(ray_dir, edge2)
    a = dot(edge1, h)
    if abs(a) < EPSILON:
        return None
    f = 1.0 / a
    s = ray_origin - v0
    u = f * dot(s, h)
    if u < 0.0 or u > 1.0:
        return None
    q = cross(s, edge1)
    v = f * dot(ray_dir, q)
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * dot(edge2, q)
    return t if t > EPSILON else None


def ray_sphere_intersect(ray_origin: np.ndarray, ray_dir: np.ndarray,
                         center: np.ndarray, radius: float) -> Optional[Tuple[float, float]]:
    """Intersect a ray with a sphere. Returns (t_near, t_far) or None."""
    oc = ray_origin - center
    a = dot(ray_dir, ray_dir)
    b = 2.0 * dot(oc, ray_dir)
    c = dot(oc, oc) - radius * radius
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None
    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    return (t1, t2)


def ray_aabb_intersect(ray_origin: np.ndarray, ray_dir: np.ndarray,
                       aabb_min: np.ndarray, aabb_max: np.ndarray) -> Optional[Tuple[float, float]]:
    """Intersect a ray with an axis-aligned bounding box. Returns (t_near, t_far) or None."""
    t_min = -np.inf
    t_max = np.inf
    for i in range(3):
        if abs(ray_dir[i]) < 1e-10:
            if ray_origin[i] < aabb_min[i] or ray_origin[i] > aabb_max[i]:
                return None
        else:
            t1 = (aabb_min[i] - ray_origin[i]) / ray_dir[i]
            t2 = (aabb_max[i] - ray_origin[i]) / ray_dir[i]
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return None
    return (float(t_min), float(t_max))


def closest_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Find the closest point on line segment ab to point p."""
    ab = b - a
    t = np.clip(dot(p - a, ab) / max(dot(ab, ab), 1e-10), 0.0, 1.0)
    return a + t * ab


def triangle_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute the normal of a triangle."""
    return normalize(cross(v1 - v0, v2 - v0))


def triangle_area(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute the area of a triangle."""
    return float(0.5 * magnitude(cross(v1 - v0, v2 - v0)))


def triangle_centroid(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute the centroid of a triangle."""
    return (v0 + v1 + v2) / 3.0


def compute_bounding_box(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute AABB of a set of vertices."""
    return vertices.min(axis=0), vertices.max(axis=0)


def compute_centroid(vertices: np.ndarray) -> np.ndarray:
    """Compute the centroid of a mesh."""
    return vertices.mean(axis=0)


def compute_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Compute the volume of a closed mesh using signed tetrahedra method."""
    volume = 0.0
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        volume += dot(v0, cross(v1, v2)) / 6.0
    return abs(volume)


def compute_surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Compute total surface area of a mesh."""
    area = 0.0
    for face in faces:
        area += triangle_area(vertices[face[0]], vertices[face[1]], vertices[face[2]])
    return area


# ─── Physics Utilities ───────────────────────────────────────────────────────

@dataclass
class PhysicsBody:
    """A rigid body for physics simulation."""
    mass: float = 1.0
    position: np.ndarray = field(default_factory=lambda: vec3())
    velocity: np.ndarray = field(default_factory=lambda: vec3())
    acceleration: np.ndarray = field(default_factory=lambda: vec3())
    rotation: np.ndarray = field(default_factory=quat_identity)
    angular_velocity: np.ndarray = field(default_factory=lambda: vec3())
    damping: float = 0.99
    angular_damping: float = 0.98
    is_static: bool = False

    @property
    def inertia_tensor(self) -> np.ndarray:
        """Simplified inertia tensor (sphere approximation)."""
        I = 2.0 / 5.0 * self.mass
        return np.eye(3) * I

    def apply_force(self, force: np.ndarray):
        """Apply a force to the body."""
        if not self.is_static:
            self.acceleration += force / self.mass

    def apply_impulse(self, impulse: np.ndarray):
        """Apply an instantaneous impulse."""
        if not self.is_static:
            self.velocity += impulse / self.mass

    def apply_torque(self, torque: np.ndarray):
        """Apply torque for angular acceleration."""
        if not self.is_static:
            I = self.inertia_tensor
            self.angular_velocity += np.linalg.solve(I, torque)

    def step(self, dt: float):
        """Advance the simulation by dt seconds."""
        if self.is_static:
            return
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        self.velocity *= self.damping
        self.acceleration = vec3()

        # Angular integration
        ang_speed = magnitude(self.angular_velocity)
        if ang_speed > 1e-10:
            axis = self.angular_velocity / ang_speed
            dq = quat_from_axis_angle(axis, ang_speed * dt)
            self.rotation = normalize_quat(quat_multiply(dq, self.rotation))
            self.angular_velocity *= self.angular_damping


@dataclass
class Gravity:
    """Gravity constants for different environments."""
    EARTH: float = 9.80665        # m/s²
    MOON: float = 1.62           # m/s²
    MARS: float = 3.721          # m/s²
    JUPITER: float = 24.79       # m/s²
    ZERO_G: float = 0.0          # m/s²


def apply_gravity(body: PhysicsBody, g: float = 9.80665, dt: float = 1.0 / 60.0):
    """Apply gravity to a physics body."""
    body.apply_force(vec3(0, -g * body.mass, 0))
    body.step(dt)


def resolve_collision(a: PhysicsBody, b: PhysicsBody, normal: np.ndarray,
                      restitution: float = 0.5, friction: float = 0.3):
    """Resolve a collision between two physics bodies."""
    rel_vel = b.velocity - a.velocity
    vel_along_normal = dot(rel_vel, normal)

    if vel_along_normal > 0:
        return  # Already separating

    j = -(1.0 + restitution) * vel_along_normal
    j /= (1.0 / a.mass) + (1.0 / b.mass) if not (a.is_static or b.is_static) else 1.0 / (a.mass if not a.is_static else b.mass)

    impulse = j * normal
    if not a.is_static:
        a.velocity -= impulse / a.mass
    if not b.is_static:
        b.velocity += impulse / b.mass

    # Friction
    tangent = rel_vel - vel_along_normal * normal
    if magnitude(tangent) > 1e-10:
        tangent = normalize(tangent)
        jt = -dot(rel_vel, tangent)
        jt /= (1.0 / a.mass) + (1.0 / b.mass) if not (a.is_static or b.is_static) else 1.0
        jt = np.clip(jt, -abs(j) * friction, abs(j) * friction)
        friction_impulse = jt * tangent
        if not a.is_static:
            a.velocity -= friction_impulse / a.mass
        if not b.is_static:
            b.velocity += friction_impulse / b.mass


# ─── IK Solvers ──────────────────────────────────────────────────────────────

def solve_2bone_ik(shoulder: np.ndarray, target: np.ndarray,
                    length_a: float, length_b: float,
                    pole: np.ndarray = None, bend_factor: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Solve a 2-bone IK chain (e.g., upper arm + forearm).
    
    Returns (elbow_position, wrist_position).
    bend_factor: 0 = fully straight, 1 = maximum bend.
    """
    to_target = target - shoulder
    dist = magnitude(to_target)

    # Clamp distance to reachable range
    max_reach = length_a + length_b
    min_reach = abs(length_a - length_b)
    dist = np.clip(dist, min_reach + 0.001, max_reach - 0.001)

    # Recalculate target at clamped distance
    if magnitude(to_target) > 1e-10:
        target_dir = normalize(to_target)
    else:
        target_dir = vec3(0, 0, 1)

    clamped_target = shoulder + target_dir * dist

    # Law of cosines to find elbow angle
    cos_angle = np.clip((length_a ** 2 + dist ** 2 - length_b ** 2) / (2 * length_a * dist), -1.0, 1.0)
    angle_a = math.acos(cos_angle)
    angle_b = math.acos(np.clip((length_a ** 2 + length_b ** 2 - dist ** 2) / (2 * length_a * length_b), -1.0, 1.0))

    # Apply bend factor
    angle_a = angle_a * bend_factor

    # Determine the plane of the bend
    if pole is not None:
        pole_dir = normalize(pole - shoulder)
        bend_axis = normalize(cross(target_dir, pole_dir))
    else:
        # Default: bend in the plane perpendicular to the chain
        bend_axis = normalize(cross(target_dir, vec3(0, 1, 0)))
        if magnitude(bend_axis) < 0.1:
            bend_axis = normalize(cross(target_dir, vec3(1, 0, 0)))

    # Calculate elbow position
    elbow_offset = quat_from_axis_angle(bend_axis, angle_a)
    elbow_dir = quat_to_rotation_matrix(elbow_offset) @ target_dir
    elbow = shoulder + elbow_dir * length_a

    # Recalculate wrist to be exactly at target
    wrist = target
    return elbow, wrist


def solve_fabrik(joints: List[np.ndarray], target: np.ndarray,
                 tolerances: float = 0.01, max_iterations: int = 100) -> List[np.ndarray]:
    """FABRIK (Forward And Backward Reaching Inverse Kinematics) solver.
    
    joints: List of joint positions from root to end-effector.
    target: Desired position for the end-effector.
    Returns the solved joint positions.
    """
    joints = [j.copy() for j in joints]
    n = len(joints)
    if n < 2:
        return [target]

    # Calculate bone lengths
    lengths = []
    for i in range(n - 1):
        lengths.append(distance(joints[i], joints[i + 1]))
    total_length = sum(lengths)

    # Check if target is reachable
    if distance(joints[0], target) > total_length:
        # Target unreachable: stretch towards it
        for i in range(n - 1):
            direction = normalize(target - joints[i])
            joints[i + 1] = joints[i] + direction * lengths[i]
        return joints

    # Root stays fixed
    base = joints[0].copy()

    for iteration in range(max_iterations):
        # Check if close enough
        if distance(joints[-1], target) < tolerances:
            break

        # Forward reaching: from end-effector to base
        joints[-1] = target.copy()
        for i in range(n - 2, -1, -1):
            direction = normalize(joints[i] - joints[i + 1])
            joints[i] = joints[i + 1] + direction * lengths[i]

        # Backward reaching: from base to end-effector
        joints[0] = base.copy()
        for i in range(n - 1):
            direction = normalize(joints[i + 1] - joints[i])
            joints[i + 1] = joints[i] + direction * lengths[i]

    return joints


def solve_ccd(joints: List[np.ndarray], target: np.ndarray,
              tolerances: float = 0.01, max_iterations: int = 100,
              angles_limit: float = math.pi) -> List[np.ndarray]:
    """CCD (Cyclic Coordinate Descent) IK solver.
    
    Iteratively rotates each joint to minimize distance to target.
    """
    joints = [j.copy() for j in joints]
    n = len(joints)
    if n < 2:
        return [target]

    for iteration in range(max_iterations):
        if distance(joints[-1], target) < tolerances:
            break

        for i in range(n - 2, -1, -1):
            # Vector from joint to end-effector
            to_end = joints[-1] - joints[i]
            # Vector from joint to target
            to_target = target - joints[i]

            if magnitude(to_end) < 1e-10 or magnitude(to_target) < 1e-10:
                continue

            # Angle and axis of rotation
            cos_angle = np.clip(dot(normalize(to_end), normalize(to_target)), -1.0, 1.0)
            angle = math.acos(cos_angle)
            angle = np.clip(angle, -angles_limit, angles_limit)

            if angle < 1e-6:
                continue

            axis = normalize(cross(to_end, to_target))
            rot = quat_from_axis_angle(axis, angle)

            # Rotate all subsequent joints
            for j in range(i + 1, n):
                offset = joints[j] - joints[i]
                rot_mat = quat_to_rotation_matrix(rot)
                joints[j] = joints[i] + rot_mat @ offset

    return joints


# ─── Color & Lighting Math ───────────────────────────────────────────────────

def rgb_to_hsv(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """Convert RGB (0-1) to HSV (0-1)."""
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn
    h = 0.0
    if d != 0:
        if mx == r:
            h = ((g - b) / d) % 6
        elif mx == g:
            h = (b - r) / d + 2
        else:
            h = (r - g) / d + 4
        h /= 6.0
    s = 0.0 if mx == 0 else d / mx
    return h, s, mx


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    """Convert HSV (0-1) to RGB (0-1)."""
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """Convert hex color string to RGB (0-1)."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return r, g, b


def fresnel_schlick(dot_ni: float, f0: float) -> float:
    """Schlick's Fresnel approximation."""
    return f0 + (1.0 - f0) * (1.0 - dot_ni) ** 5


def ggx_distribution(n_dot_h: float, roughness: float) -> float:
    """GGX/Trowbridge-Reitz normal distribution function."""
    a = roughness ** 2
    a2 = a ** 2
    denom = n_dot_h ** 2 * (a2 - 1.0) + 1.0
    return a2 / (math.pi * denom ** 2)


def smith_geometry(n_dot_v: float, roughness: float) -> float:
    """Smith's geometry function (Schlick-GGX)."""
    r = roughness + 1.0
    k = (r ** 2) / 8.0
    return n_dot_v / (n_dot_v * (1.0 - k) + k)


# ─── Bezier / Spline Math ───────────────────────────────────────────────────

def bezier_cubic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
    """Evaluate a cubic Bezier curve at parameter t."""
    u = 1.0 - t
    return u**3 * p0 + 3 * u**2 * t * p1 + 3 * u * t**2 * p2 + t**3 * p3


def bezier_cubic_derivative(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
    """Evaluate the derivative of a cubic Bezier curve at parameter t."""
    u = 1.0 - t
    return 3 * u**2 * (p1 - p0) + 6 * u * t * (p2 - p1) + 3 * t**2 * (p3 - p2)


def catmull_rom_spline(points: List[np.ndarray], num_samples: int = 100) -> List[np.ndarray]:
    """Generate a Catmull-Rom spline through a list of points."""
    if len(points) < 2:
        return points

    result = []
    # Extend points with phantom endpoints
    extended = [2 * points[0] - points[1]] + points + [2 * points[-1] - points[-2]]

    for i in range(len(extended) - 3):
        for j in range(num_samples):
            t = j / num_samples
            t2 = t * t
            t3 = t2 * t
            p0, p1, p2, p3 = extended[i], extended[i + 1], extended[i + 2], extended[i + 3]
            point = 0.5 * (
                (2 * p1) +
                (-p0 + p2) * t +
                (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            result.append(point)
    result.append(points[-1])
    return result
