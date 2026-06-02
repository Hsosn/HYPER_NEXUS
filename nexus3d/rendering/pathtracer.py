"""Monte Carlo Path Tracer for Nexus3D.

A production-grade physically-based path tracer with:
  - Multi-bounce global illumination (Russian roulette termination)
  - Cook-Torrance specular BRDF (GGX NDF, Smith geometry, Schlick Fresnel)
  - Lambertian diffuse for non-metallic surfaces
  - Perfect specular reflections for mirrors
  - Dielectric refraction / transmission (Snell's law, total internal reflection)
  - Subsurface scattering (SSS) random-walk approximation
  - Soft shadows via area-light sampling
  - Ray-based ambient occlusion pass
  - BVH acceleration structure with SAH construction
  - HDR tonemapping (ACES Filmic, Reinhard, Linear)
  - sRGB gamma correction
  - Thin-lens depth-of-field camera
  - Progressive (accumulative) rendering

Pure Python / NumPy – no external dependencies beyond numpy and Pillow.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow is required: pip install Pillow")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPSILON = 1e-7
INF = float("inf")
PI = math.pi
TWO_PI = 2.0 * math.pi
INV_PI = 1.0 / PI
INV_TWO_PI = 1.0 / TWO_PI

# ---------------------------------------------------------------------------
# PTMaterial
# ---------------------------------------------------------------------------

@dataclass
class PTMaterial:
    """Physically-based material definition for the path tracer."""

    albedo: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.8], dtype=np.float64))
    metallic: float = 0.0
    roughness: float = 0.5
    emission: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64))
    transmission: float = 0.0
    ior: float = 1.5
    sss_radius: float = 0.0
    alpha: float = 1.0

    def __post_init__(self):
        self.albedo = np.asarray(self.albedo, dtype=np.float64).copy()
        self.emission = np.asarray(self.emission, dtype=np.float64).copy()

    # ---- helpers ----------------------------------------------------------
    @property
    def is_emissive(self) -> bool:
        return float(np.max(self.emission)) > EPSILON

    @property
    def is_specular(self) -> bool:
        return self.metallic >= 1.0 and self.roughness < 0.01

    @property
    def is_glass(self) -> bool:
        return self.transmission > 0.0

    @property
    def is_sss(self) -> bool:
        return self.sss_radius > 0.0

    @property
    def is_pure_mirror(self) -> bool:
        return self.is_specular and not self.is_glass

    def F0(self) -> np.ndarray:
        """Fresnel reflectance at normal incidence."""
        return self.albedo * self.metallic + (1.0 - self.metallic) * 0.04


# ---------------------------------------------------------------------------
# Vector / ray helpers (operating on numpy arrays of shape (3,))
# ---------------------------------------------------------------------------

def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > EPSILON else v

def _dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0]*b[0] + a[1]*b[1] + a[2]*b[2])

def _dot_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1)

def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ], dtype=np.float64)

def _reflect(incident: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return incident - 2.0 * _dot(incident, normal) * normal

def _refract(incident: np.ndarray, normal: np.ndarray, eta: float) -> Tuple[Optional[np.ndarray], float]:
    """Snell's law refraction. Returns (refracted_dir, cos_theta_t) or (None, cos_i)."""
    cos_i = -_dot(incident, normal)
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    if sin2_t > 1.0:
        return None, cos_i  # total internal reflection
    cos_t = math.sqrt(1.0 - sin2_t)
    return _norm(eta * incident + (eta * cos_i - cos_t) * normal), cos_t

def _schlick(cos_theta: float, f0_val: float) -> float:
    """Schlick Fresnel approximation for scalar reflectance."""
    return f0_val + (1.0 - f0_val) * max(0.0, 1.0 - cos_theta) ** 5

def _schlick_vec(cos_theta: float, f0: np.ndarray) -> np.ndarray:
    """Schlick Fresnel for vector F0."""
    return f0 + (1.0 - f0) * max(0.0, 1.0 - cos_theta) ** 5

def _fresnel_dielectric(cos_i: float, ior_i: float, ior_t: float) -> float:
    """Full Fresnel equations for dielectric (unpolarised)."""
    eta = ior_i / ior_t
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    if sin2_t > 1.0:
        return 1.0  # TIR
    cos_t = math.sqrt(max(0.0, 1.0 - sin2_t))
    rs = ((ior_t * cos_i - ior_i * cos_t) / (ior_t * cos_i + ior_i * cos_t)) ** 2
    rp = ((ior_i * cos_i - ior_t * cos_t) / (ior_i * cos_i + ior_t * cos_t)) ** 2
    return (rs + rp) * 0.5


# ---------------------------------------------------------------------------
# Random sampling helpers
# ---------------------------------------------------------------------------

_RNG: np.random.RandomState = np.random.RandomState()

def _set_seed(seed: Optional[int] = None):
    global _RNG
    _RNG = np.random.RandomState(seed)

def _rand() -> float:
    return float(_RNG.random())

def _rand2() -> Tuple[float, float]:
    return float(_RNG.random()), float(_RNG.random())

def _rand3() -> Tuple[float, float, float]:
    return float(_RNG.random()), float(_RNG.random()), float(_RNG.random())

def _cosine_hemisphere() -> np.ndarray:
    """Cosine-weighted random direction on the unit hemisphere (+Y)."""
    r1, r2 = _rand2()
    phi = TWO_PI * r1
    cos_theta = math.sqrt(r2)
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    return np.array([
        math.cos(phi) * sin_theta,
        cos_theta,
        math.sin(phi) * sin_theta,
    ], dtype=np.float64)

def _uniform_hemisphere() -> np.ndarray:
    """Uniform random direction on the unit hemisphere (+Y)."""
    r1, r2 = _rand2()
    phi = TWO_PI * r1
    cos_theta = r2
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    return np.array([
        math.cos(phi) * sin_theta,
        cos_theta,
        math.sin(phi) * sin_theta,
    ], dtype=np.float64)

def _uniform_sphere() -> np.ndarray:
    """Uniform random direction on the unit sphere."""
    r1, r2 = _rand2()
    phi = TWO_PI * r1
    cos_theta = 2.0 * r2 - 1.0
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    return np.array([
        math.cos(phi) * sin_theta,
        cos_theta,
        math.sin(phi) * sin_theta,
    ], dtype=np.float64)

def _uniform_disk() -> Tuple[float, float]:
    """Uniform random point on the unit disk."""
    r1, r2 = _rand2()
    r = math.sqrt(r1)
    theta = TWO_PI * r2
    return r * math.cos(theta), r * math.sin(theta)

def _ggx_hemisphere(roughness: float, normal: np.ndarray,
                    incident: np.ndarray) -> np.ndarray:
    """GGX VNDF importance sampling.

    Returns a world-space direction sampled from the GGX distribution
    conditioned on the incoming direction *incident*.
    """
    alpha = max(roughness * roughness, EPSILON)

    # Build local frame from normal
    w_up = normal
    if abs(w_up[0]) > 0.9:
        w_right = _norm(_cross(np.array([0.0, 1.0, 0.0]), w_up))
    else:
        w_right = _norm(_cross(np.array([1.0, 0.0, 0.0]), w_up))
    w_btw = _cross(w_up, w_right)

    # Transform incident to local space
    wi_local = np.array([_dot(incident, w_right), _dot(incident, w_up), _dot(incident, w_btw)], dtype=np.float64)

    # GGX VNDF sampling (Heitz 2018)
    wi_local = _norm(wi_local)
    v = np.array([
        alpha * wi_local[0],
        alpha * wi_local[1],
        wi_local[2],
    ], dtype=np.float64)
    length_v = np.linalg.norm(v)
    if length_v < EPSILON:
        v = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        v = v / length_v

    # Sample with stretch-on-ellipse trick
    r1, r2 = _rand2()
    a = 1.0 / (1.0 + v[2])
    b = v[2] - 1.0
    p = a * (r1 * a - b)
    z = (p * p - 1.0) * a + 1.0
    if z <= 0.0:
        z = EPSILON
    fac = math.sqrt(max(0.0, (z * z - 1.0) / (p * p - 1.0))) if abs(p * p - 1.0) > EPSILON else 1.0
    xi1 = p / z
    xi2 = fac * (r2 - 0.5)

    # Sample normal from the stretched VNDF
    n_stretched = np.array([xi1, xi2, 1.0], dtype=np.float64)
    n_stretched = _norm(n_stretched)

    # Un-stretch
    n_local = _norm(np.array([
        alpha * n_stretched[0],
        alpha * n_stretched[1],
        max(n_stretched[2], EPSILON),
    ], dtype=np.float64))

    # Transform back to world space
    wo = n_local[0] * w_right + n_local[1] * w_up + n_local[2] * w_btw
    return _norm(wo)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class Ray:
    """A ray with origin and direction."""
    __slots__ = ("origin", "direction", "t_min", "t_max")

    def __init__(self, origin: np.ndarray, direction: np.ndarray,
                 t_min: float = EPSILON, t_max: float = INF):
        self.origin = origin
        self.direction = _norm(direction)
        self.t_min = t_min
        self.t_max = t_max

    def at(self, t: float) -> np.ndarray:
        return self.origin + t * self.direction


class HitRecord:
    """Record of the closest intersection."""
    __slots__ = ("t", "point", "normal", "material", "u", "v", "front_face")

    def __init__(self):
        self.t: float = INF
        self.point: np.ndarray = np.zeros(3)
        self.normal: np.ndarray = np.zeros(3)
        self.material: Optional[PTMaterial] = None
        self.u: float = 0.0
        self.v: float = 0.0
        self.front_face: bool = True

    def set_face_normal(self, ray: Ray, outward_normal: np.ndarray):
        self.front_face = _dot(ray.direction, outward_normal) < 0.0
        self.normal = outward_normal if self.front_face else -outward_normal


class Sphere:
    """Axis-aligned sphere primitive."""

    def __init__(self, center: np.ndarray, radius: float, material: PTMaterial):
        self.center = np.asarray(center, dtype=np.float64).copy()
        self.radius = float(radius)
        self.material = material

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        r = np.array([self.radius, self.radius, self.radius])
        return self.center - r, self.center + r

    def intersect(self, ray: Ray, rec: HitRecord) -> bool:
        oc = ray.origin - self.center
        a = _dot(ray.direction, ray.direction)
        half_b = _dot(oc, ray.direction)
        c = _dot(oc, oc) - self.radius * self.radius
        disc = half_b * half_b - a * c
        if disc < 0.0:
            return False
        sq = math.sqrt(disc)
        root = (-half_b - sq) / a
        if root < ray.t_min or root > ray.t_max:
            root = (-half_b + sq) / a
            if root < ray.t_min or root > ray.t_max:
                return False
        rec.t = root
        rec.point = ray.at(root)
        outward_normal = (rec.point - self.center) / self.radius
        rec.set_face_normal(ray, outward_normal)
        rec.material = self.material
        # UV (spherical mapping)
        nx, ny, nz = outward_normal
        rec.u = 0.5 + math.atan2(nz, nx) * INV_TWO_PI
        rec.v = 0.5 + math.asin(np.clip(ny, -1.0, 1.0)) * INV_PI
        return True


class Triangle:
    """Single triangle primitive."""

    def __init__(self, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                 material: PTMaterial):
        self.v0 = np.asarray(v0, dtype=np.float64).copy()
        self.v1 = np.asarray(v1, dtype=np.float64).copy()
        self.v2 = np.asarray(v2, dtype=np.float64).copy()
        self.material = material
        self.edge1 = self.v1 - self.v0
        self.edge2 = self.v2 - self.v0
        self._normal = _norm(_cross(self.edge1, self.edge2))

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        mn = np.minimum(np.minimum(self.v0, self.v1), self.v2)
        mx = np.maximum(np.maximum(self.v0, self.v1), self.v2)
        return mn, mx

    def intersect(self, ray: Ray, rec: HitRecord) -> bool:
        """Möller–Trumbore intersection."""
        h = _cross(ray.direction, self.edge2)
        a = _dot(self.edge1, h)
        if abs(a) < EPSILON:
            return False
        f = 1.0 / a
        s = ray.origin - self.v0
        u = f * _dot(s, h)
        if u < 0.0 or u > 1.0:
            return False
        q = _cross(s, self.edge1)
        v = f * _dot(ray.direction, q)
        if v < 0.0 or u + v > 1.0:
            return False
        t = f * _dot(self.edge2, q)
        if t < ray.t_min or t > ray.t_max:
            return False
        rec.t = t
        rec.point = ray.at(t)
        rec.set_face_normal(ray, self._normal)
        rec.material = self.material
        rec.u = u
        rec.v = v
        return True


class Plane:
    """Infinite plane primitive with optional checkerboard pattern."""

    def __init__(self, point: np.ndarray, normal: np.ndarray,
                 material: PTMaterial, checkerboard: bool = False,
                 checker_scale: float = 1.0):
        self.point = np.asarray(point, dtype=np.float64).copy()
        self.normal = _norm(np.asarray(normal, dtype=np.float64))
        self.material = material
        self.checkerboard = checkerboard
        self.checker_scale = checker_scale

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        # Return a huge axis-aligned box so BVH can include it
        b = np.array([1e6, 1e6, 1e6])
        return self.point - b, self.point + b

    def intersect(self, ray: Ray, rec: HitRecord) -> bool:
        denom = _dot(self.normal, ray.direction)
        if abs(denom) < EPSILON:
            return False
        t = _dot(self.point - ray.origin, self.normal) / denom
        if t < ray.t_min or t > ray.t_max:
            return False
        rec.t = t
        rec.point = ray.at(t)
        rec.set_face_normal(ray, self.normal)
        mat = self.material
        # Checkerboard override of albedo
        if self.checkerboard:
            p = rec.point * self.checker_scale
            ix = int(math.floor(p[0])) % 2
            iz = int(math.floor(p[2])) % 2
            if (ix + iz) % 2 == 0:
                mat = PTMaterial(
                    albedo=mat.albedo * 0.3,
                    metallic=mat.metallic,
                    roughness=mat.roughness,
                    emission=mat.emission,
                    transmission=mat.transmission,
                    ior=mat.ior,
                    sss_radius=mat.sss_radius,
                    alpha=mat.alpha,
                )
        rec.material = mat
        return True


# ---------------------------------------------------------------------------
# Area Light
# ---------------------------------------------------------------------------

class AreaLight:
    """Rectangular area light defined by centre, two axis vectors and emission colour."""

    def __init__(self, center: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray,
                 emission: np.ndarray, two_sided: bool = True):
        self.center = np.asarray(center, dtype=np.float64).copy()
        self.u_axis = np.asarray(u_axis, dtype=np.float64).copy()
        self.v_axis = np.asarray(v_axis, dtype=np.float64).copy()
        self.emission = np.asarray(emission, dtype=np.float64).copy()
        self.two_sided = two_sided
        self.normal = _norm(_cross(self.u_axis, self.v_axis))
        self.area = float(np.linalg.norm(_cross(self.u_axis, self.v_axis)))

    def sample(self, u: float, v: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (point_on_light, normal, pdf) for given random (u,v) in [0,1]^2."""
        pt = self.center + u * self.u_axis + v * self.v_axis
        nrm = self.normal if self.two_sided else self.normal
        return pt, nrm, 1.0 / self.area

    def sample_direction(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Convenience: random point on the light surface."""
        return self.sample(_rand(), _rand())


# ---------------------------------------------------------------------------
# BVH (Bounding Volume Hierarchy) with SAH construction
# ---------------------------------------------------------------------------

class BVHNode:
    """A node in the BVH tree."""

    __slots__ = ("bbox_min", "bbox_max", "left", "right", "object")

    def __init__(self):
        self.bbox_min: np.ndarray = np.zeros(3)
        self.bbox_max: np.ndarray = np.zeros(3)
        self.left: Optional[BVHNode] = None
        self.right: Optional[BVHNode] = None
        self.object: Optional[object] = None  # leaf reference

    @property
    def is_leaf(self) -> bool:
        return self.object is not None


def _bbox_union(a_min, a_max, b_min, b_max):
    return np.minimum(a_min, b_min), np.maximum(a_max, b_max)


def _bbox_centroid(bmin, bmax):
    return (bmin + bmax) * 0.5


def _bbox_surface_area(bmin, bmax):
    e = bmax - bmin
    return 2.0 * (e[0]*e[1] + e[1]*e[2] + e[0]*e[2])


def _bbox_hit(node: BVHNode, ray: Ray) -> bool:
    """Slab test for axis-aligned bounding box."""
    t_min = ray.t_min
    t_max = ray.t_max
    for i in range(3):
        inv_d = 1.0 / ray.direction[i] if abs(ray.direction[i]) > EPSILON else (
            1e18 if ray.direction[i] >= 0 else -1e18
        )
        t0 = (node.bbox_min[i] - ray.origin[i]) * inv_d
        t1 = (node.bbox_max[i] - ray.origin[i]) * inv_d
        if inv_d < 0:
            t0, t1 = t1, t0
        t_min = max(t_min, t0)
        t_max = min(t_max, t1)
        if t_max < t_min:
            return False
    return True


class BVH:
    """Bounding Volume Hierarchy acceleration structure."""

    def __init__(self, objects: list):
        self.root = self._build(objects)

    # ---- SAH build --------------------------------------------------------

    def _build(self, objects: list) -> BVHNode:
        if len(objects) == 0:
            node = BVHNode()
            node.bbox_min = np.array([-INF, -INF, -INF])
            node.bbox_max = np.array([INF, INF, INF])
            return node
        if len(objects) == 1:
            node = BVHNode()
            bmin, bmax = objects[0].bounding_box()
            node.bbox_min = bmin
            node.bbox_max = bmax
            node.object = objects[0]
            return node
        # Compute world bounds
        all_min = np.full(3, INF)
        all_max = np.full(3, -INF)
        bboxes = []
        for o in objects:
            bmin, bmax = o.bounding_box()
            bboxes.append((bmin, bmax))
            all_min = np.minimum(all_min, bmin)
            all_max = np.maximum(all_max, bmax)

        node = BVHNode()
        node.bbox_min = all_min
        node.bbox_max = all_max

        SA_total = _bbox_surface_area(all_min, all_max)
        if SA_total < EPSILON:
            node.object = objects[0]
            return node

        # SAH: find best split axis and position
        best_axis = 0
        best_pos = 0.0
        best_cost = INF

        for axis in range(3):
            # Bucket-based SAH (8 buckets)
            n_buckets = 8
            bucket_min = [np.full(3, INF) for _ in range(n_buckets)]
            bucket_max = [np.full(3, -INF) for _ in range(n_buckets)]
            bucket_count = [0] * n_buckets
            extent = all_max[axis] - all_min[axis]
            if extent < EPSILON:
                continue

            for idx, o in enumerate(objects):
                bmin, bmax = bboxes[idx]
                centroid = (bmin[axis] + bmax[axis]) * 0.5
                b = int((centroid - all_min[axis]) / extent * n_buckets)
                b = min(b, n_buckets - 1)
                bucket_min[b] = np.minimum(bucket_min[b], bmin)
                bucket_max[b] = np.maximum(bucket_max[b], bmax)
                bucket_count[b] += 1

            # Sweep from left and right
            left_area = [0.0] * n_buckets
            right_area = [0.0] * n_buckets
            left_count = 0
            left_min = np.full(3, INF)
            left_max = np.full(3, -INF)
            for i in range(n_buckets):
                if bucket_count[i] > 0:
                    left_min = np.minimum(left_min, bucket_min[i])
                    left_max = np.maximum(left_max, bucket_max[i])
                left_area[i] = _bbox_surface_area(left_min, left_max) if left_count > 0 or bucket_count[i] > 0 else 0.0
                left_count += bucket_count[i]

            right_count = 0
            right_min = np.full(3, INF)
            right_max = np.full(3, -INF)
            for i in range(n_buckets - 1, -1, -1):
                if bucket_count[i] > 0:
                    right_min = np.minimum(right_min, bucket_min[i])
                    right_max = np.maximum(right_max, bucket_max[i])
                right_area[i] = _bbox_surface_area(right_min, right_max) if right_count > 0 or bucket_count[i] > 0 else 0.0
                right_count += bucket_count[i]

            for i in range(n_buckets - 1):
                cost = 0.125 + (left_area[i] * left_area[i] + right_area[i+1] * right_area[i+1]) / SA_total
                if cost < best_cost:
                    best_cost = cost
                    best_axis = axis
                    best_pos = all_min[axis] + (i + 1) / n_buckets * extent

        if best_cost >= INF:
            # Fallback: split in half
            mid = len(objects) // 2
            node.left = self._build(objects[:mid])
            node.right = self._build(objects[mid:])
            return node

        # Partition
        left_objs = []
        right_objs = []
        for idx, o in enumerate(objects):
            bmin, bmax = bboxes[idx]
            centroid = (bmin[best_axis] + bmax[best_axis]) * 0.5
            if centroid < best_pos:
                left_objs.append(o)
            else:
                right_objs.append(o)

        if len(left_objs) == 0:
            left_objs, right_objs = right_objs[:1], right_objs[1:]
        if len(right_objs) == 0:
            right_objs, left_objs = left_objs[:1], left_objs[1:]

        node.left = self._build(left_objs)
        node.right = self._build(right_objs)
        return node

    # ---- traversal --------------------------------------------------------

    def intersect(self, ray: Ray, rec: HitRecord) -> bool:
        return self._intersect(self.root, ray, rec)

    def _intersect(self, node: BVHNode, ray: Ray, rec: HitRecord) -> bool:
        if not _bbox_hit(node, ray):
            return False
        if node.is_leaf:
            return node.object.intersect(ray, rec)
        hit = self._intersect(node.left, ray, rec)
        closer_ray = Ray(ray.origin, ray.direction, ray.t_min, rec.t)
        hit2 = self._intersect(node.right, closer_ray, rec)
        return hit or hit2


# ---------------------------------------------------------------------------
# Camera (thin-lens model)
# ---------------------------------------------------------------------------

class Camera:
    def __init__(self, position: np.ndarray, target: np.ndarray, fov: float,
                 aspect: float, aperture: float = 0.0, focal_distance: float = 10.0):
        self.position = np.asarray(position, dtype=np.float64).copy()
        forward = _norm(np.asarray(target, dtype=np.float64) - self.position)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(_dot(forward, world_up)) > 0.999:
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.right = _norm(_cross(forward, world_up))
        self.up = _cross(self.right, forward)
        self.forward = forward

        theta = fov * PI / 180.0
        half_h = math.tan(theta * 0.5)
        half_w = aspect * half_h
        self.lower_left = (self.position
                           - half_w * self.right
                           - half_h * self.up
                           + forward)
        self.horizontal = 2.0 * half_w * self.right
        self.vertical = 2.0 * half_h * self.up

        self.aperture = aperture
        self.focal_distance = focal_distance
        self.lens_radius = aperture * 0.5 if aperture > EPSILON else 0.0

    def get_ray(self, s: float, t: float) -> Ray:
        origin = self.position
        if self.lens_radius > EPSILON:
            rd = self.lens_radius * _uniform_disk()
            origin = origin + rd[0] * self.right + rd[1] * self.up
        target = (self.lower_left
                  + s * self.horizontal
                  + t * self.vertical
                  - origin)
        direction = _norm(target * self.focal_distance + origin - origin)
        # Simpler: direction toward the focal plane
        if self.lens_radius > EPSILON:
            focal_pt = self.position + self.forward * self.focal_distance
            direction = _norm(focal_pt - origin + self.lower_left - self.position
                              + s * self.horizontal + t * self.vertical
                              - self.forward * self.focal_distance
                              + self.lower_left + s * self.horizontal + t * self.vertical
                              - self.lower_left - s * self.horizontal - t * self.vertical)
            # Correct thin-lens: compute the point on the focal plane
            direction = _norm(
                self.position + self.forward * self.focal_distance
                - origin
            )
        return Ray(origin, _norm(self.lower_left + s * self.horizontal + t * self.vertical - origin))

    def get_ray_dof(self, s: float, t: float) -> Ray:
        """Proper thin-lens ray with depth-of-field."""
        rd = self.lower_left + s * self.horizontal + t * self.vertical  # point on virtual image plane

        origin = self.position.copy()
        if self.lens_radius > EPSILON:
            ru, rv = _uniform_disk()
            offset = self.lens_radius * (ru * self.right + rv * self.up)
            origin = origin + offset

        direction = _norm(rd - origin)
        return Ray(origin, direction)


# ---------------------------------------------------------------------------
# Onb (Orthonormal Basis) for shading
# ---------------------------------------------------------------------------

class Onb:
    """Orthonormal basis from a normal vector."""
    __slots__ = ("u", "v", "w")

    def __init__(self, n: np.ndarray):
        self.w = _norm(n)
        if abs(self.w[0]) > 0.9:
            a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.v = _norm(_cross(self.w, a))
        self.u = _cross(self.w, self.v)

    def local(self, v: np.ndarray) -> np.ndarray:
        """Transform from local shading space to world space."""
        return v[0] * self.u + v[1] * self.w + v[2] * self.v

    def world(self, v: np.ndarray) -> np.ndarray:
        """Transform from world space to local shading space."""
        return np.array([_dot(v, self.u), _dot(v, self.w), _dot(v, self.v)], dtype=np.float64)


# ---------------------------------------------------------------------------
# Cook-Torrance BRDF evaluation & sampling
# ---------------------------------------------------------------------------

def _ggx_ndf(alpha: float, cos_nh: float) -> float:
    """GGX / Trowbridge-Reitz normal distribution function."""
    a2 = alpha * alpha
    denom = cos_nh * cos_nh * (a2 - 1.0) + 1.0
    return a2 / (PI * denom * denom + EPSILON)


def _smith_g1(alpha: float, cos_nv: float) -> float:
    """Smith height-correlated geometry term (single direction)."""
    k = alpha * alpha * 0.5  # k = alpha^2 / 2 for IBL; for direct use (alpha+1)^2/8
    return cos_nv / (cos_nv * (1.0 - k) + k + EPSILON)


def _smith_ggx(alpha: float, cos_nv: float, cos_nl: float) -> float:
    """Full Smith G term G(l,v) = G1(l)*G1(v)."""
    return _smith_g1(alpha, cos_nv) * _smith_g1(alpha, cos_nl)


def _cook_torrance_eval(wo: np.ndarray, wi: np.ndarray, normal: np.ndarray,
                        mat: PTMaterial) -> np.ndarray:
    """Evaluate Cook-Torrance BRDF * cos(theta_i) for given pair of directions.

    wo = outgoing (toward camera), wi = incoming (toward light).
    Both assumed to be in the hemisphere of *normal*.
    """
    alpha = max(mat.roughness * mat.roughness, EPSILON)
    f0 = mat.F0()

    wh = _norm(wi + wo)
    cos_nh = max(_dot(normal, wh), 0.0)
    cos_nv = max(_dot(normal, wo), 0.0)
    cos_nl = max(_dot(normal, wi), 0.0)
    cos_hv = max(_dot(wh, wo), 0.0)

    if cos_nl <= 0.0 or cos_nv <= 0.0:
        return np.zeros(3)

    D = _ggx_ndf(alpha, cos_nh)
    G = _smith_ggx(alpha, cos_nv, cos_nl)
    F = _schlick_vec(cos_hv, f0)

    denom = 4.0 * cos_nv * cos_nl + EPSILON
    specular = (D * G * F) / denom

    # Diffuse (Lambertian) – energy-conserving: kd = (1-F)(1-metallic)
    kd = (1.0 - F) * (1.0 - mat.metallic)
    diffuse = kd * mat.albedo * INV_PI

    return (diffuse + specular) * cos_nl


def _sample_brdf(wo: np.ndarray, normal: np.ndarray, mat: PTMaterial
                 ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Sample a direction *wi* from the BRDF.

    Returns (wi, throughput, pdf) where pdf is the probability of the sample
    and throughput is the pre-multiplied BRDF / pdf for estimator.
    """
    alpha = max(mat.roughness * mat.roughness, EPSILON)
    f0 = mat.F0()

    # Choose between diffuse and specular based on Fresnel
    cos_nv = max(_dot(wo, normal), 0.0)
    F_avg = float(np.mean(_schlick_vec(cos_nv, f0)))
    diffuse_weight = (1.0 - F_avg) * (1.0 - mat.metallic)
    specular_weight = F_avg + mat.metallic
    total_w = diffuse_weight + specular_weight + EPSILON

    r = _rand()
    onb = Onb(normal)

    if mat.transmission > 0.0:
        # Glass: handle refraction / reflection
        return _sample_glass(wo, normal, mat)

    if r < specular_weight / total_w:
        # ---- Specular: GGX importance sampling ---------------------------
        wi_local = _ggx_hemisphere(mat.roughness, normal, wo)
        wi = _norm(wi_local)
        cos_nl = max(_dot(normal, wi), 0.0)
        if cos_nl < EPSILON:
            return _sample_fallback_diffuse(wo, normal, mat)

        wh = _norm(wi + wo)
        cos_nh = max(_dot(normal, wh), 0.0)
        cos_hv = max(_dot(wh, wo), 0.0)

        D = _ggx_ndf(alpha, cos_nh)
        G = _smith_ggx(alpha, cos_nv, cos_nl)
        F = _schlick_vec(cos_hv, f0)

        pdf = D * cos_nh / (4.0 * cos_hv + EPSILON)
        if pdf < EPSILON:
            return _sample_fallback_diffuse(wo, normal, mat)

        specular = (D * G * F) / (4.0 * cos_nv * cos_nl + EPSILON)
        throughput = specular * cos_nl / pdf * (specular_weight / total_w)
        return wi, throughput, pdf

    else:
        # ---- Diffuse: cosine-weighted hemisphere --------------------------
        wi_local = _cosine_hemisphere()
        wi = onb.local(wi_local)
        cos_nl = max(_dot(normal, wi), 0.0)
        if cos_nl < EPSILON:
            return _sample_fallback_diffuse(wo, normal, mat)

        F = _schlick_vec(cos_nv, f0)
        kd = (1.0 - F) * (1.0 - mat.metallic)
        pdf = cos_nl * INV_PI
        throughput = kd * mat.albedo * INV_PI * cos_nl / (pdf + EPSILON) * (diffuse_weight / total_w)
        return wi, throughput, pdf


def _sample_fallback_diffuse(wo: np.ndarray, normal: np.ndarray,
                             mat: PTMaterial) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fallback diffuse sample when specular fails (e.g. grazing angle)."""
    onb = Onb(normal)
    wi_local = _cosine_hemisphere()
    wi = onb.local(wi_local)
    cos_nl = max(_dot(normal, wi), 0.0)
    pdf = cos_nl * INV_PI + EPSILON
    throughput = mat.albedo * (1.0 - mat.metallic) * INV_PI * cos_nl / pdf
    return wi, throughput, pdf


# ---------------------------------------------------------------------------
# Glass / dielectric sampling
# ---------------------------------------------------------------------------

def _sample_glass(wo: np.ndarray, normal: np.ndarray, mat: PTMaterial
                  ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Sample refraction or reflection for a dielectric material."""
    cos_i = max(-_dot(wo, normal), 0.0)
    entering = cos_i > 0.0
    if not entering:
        normal = -normal
        cos_i = max(-_dot(wo, normal), 0.0)

    ior_i = 1.0 if entering else mat.ior
    ior_t = mat.ior if entering else 1.0
    eta = ior_i / ior_t

    reflectance = _fresnel_dielectric(cos_i, ior_i, ior_t)

    if _rand() < reflectance:
        # Reflection
        wi = _reflect(wo, normal)
        return wi, np.array([1.0, 1.0, 1.0]), 1.0
    else:
        # Refraction
        refracted, cos_t = _refract(wo, normal, eta)
        if refracted is None:
            # TIR
            wi = _reflect(wo, normal)
            return wi, np.array([1.0, 1.0, 1.0]), 1.0
        # Attenuation through medium
        # Beer's law attenuation
        thickness = 1.0
        attenuation = np.exp(-thickness * (1.0 - mat.albedo) * 2.0)
        return refracted, attenuation, 1.0


# ---------------------------------------------------------------------------
# Mirror (perfect specular) sampling
# ---------------------------------------------------------------------------

def _sample_mirror(wo: np.ndarray, normal: np.ndarray,
                   mat: PTMaterial) -> Tuple[np.ndarray, np.ndarray, float]:
    wi = _reflect(wo, normal)
    return wi, mat.albedo.copy(), 1.0


# ---------------------------------------------------------------------------
# Subsurface scattering approximation
# ---------------------------------------------------------------------------

def _sample_sss(ray: Ray, hit: HitRecord, mat: PTMaterial,
                scene_objects: list, bvh: BVH, max_bounces: int,
                min_bounces: int) -> np.ndarray:
    """Random-walk SSS approximation.

    Casts multiple rays in random directions from the hit point,
    attenuated by exponential decay based on sss_radius.
    """
    onb = Onb(hit.normal)
    sss_samples = 8
    result = np.zeros(3)
    sigma_t = 1.0 / max(mat.sss_radius, EPSILON)

    for _ in range(sss_samples):
        # Random walk step
        r = _rand()
        dist = -math.log(max(r, EPSILON)) / sigma_t
        dist = min(dist, mat.sss_radius * 4.0)

        # Random direction biased into surface (slightly below)
        direction = onb.local(_cosine_hemisphere() * 0.3 - np.array([0.0, 0.7, 0.0]))
        direction = _norm(direction)

        origin = hit.point + hit.normal * EPSILON * 2.0
        end = origin + direction * dist

        # Check if we exit the surface
        sss_ray = Ray(origin, direction)
        rec = HitRecord()
        rec.t_max = dist + EPSILON
        if bvh.intersect(sss_ray, rec) and rec.material is mat:
            # Still inside – continue walk or scatter
            # Simplified: accumulate attenuated diffuse light
            contrib = _estimate_direct_sss(rec, scene_objects, bvh, mat)
            atten = math.exp(-sigma_t * rec.t)
            result += atten * contrib * mat.albedo
        else:
            # Exited – sample light
            light_contrib = _estimate_direct_sss_surface(end, hit.normal, scene_objects, bvh)
            atten = math.exp(-sigma_t * dist)
            result += atten * light_contrib * mat.albedo

    return result / sss_samples


def _estimate_direct_sss(hit: HitRecord, objects: list, bvh: BVH,
                         mat: PTMaterial) -> np.ndarray:
    """Estimate direct lighting at an SSS walk hit point."""
    return _estimate_direct_lighting(hit.point, hit.normal, hit, objects, bvh, 1.0)


def _estimate_direct_sss_surface(point: np.ndarray, normal: np.ndarray,
                                  objects: list, bvh: BVH) -> np.ndarray:
    """Estimate direct lighting at an SSS exit point."""
    rec = HitRecord()
    rec.point = point
    rec.normal = normal
    return _estimate_direct_lighting(point, normal, rec, objects, bvh, 1.0)


# ---------------------------------------------------------------------------
# Direct lighting estimation (for both SSS and NEE)
# ---------------------------------------------------------------------------

def _estimate_direct_lighting(point: np.ndarray, normal: np.ndarray,
                              hit: HitRecord, objects: list, bvh: BVH,
                              throughput: float) -> np.ndarray:
    """Sample all emissive objects in the scene as lights (MIS)."""
    result = np.zeros(3)
    for obj in objects:
        if not hasattr(obj, 'material') or obj.material is None:
            continue
        if not obj.material.is_emissive:
            continue

        # Treat emissive sphere as area light
        if isinstance(obj, Sphere):
            light_dir = obj.center - point
            dist_sq = float(np.dot(light_dir, light_dir))
            dist = math.sqrt(dist_sq)
            if dist < EPSILON:
                continue
            light_dir_n = light_dir / dist

            # Shadow test
            shadow_ray = Ray(point + normal * EPSILON, light_dir_n, EPSILON, dist - EPSILON)
            shadow_rec = HitRecord()
            if bvh.intersect(shadow_ray, shadow_rec):
                continue

            cos_nl = max(_dot(normal, light_dir_n), 0.0)
            # Area of sphere projected: approximate as 4πr²
            solid_angle = (PI * obj.radius * obj.radius) / dist_sq
            result += obj.material.emission * cos_nl * solid_angle / PI

    return result


# ---------------------------------------------------------------------------
# Tone-mapping
# ---------------------------------------------------------------------------

def _aces_filmic(x: float) -> float:
    """ACES filmic tone-mapping curve (single channel)."""
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    x = max(x, 0.0)
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def _tonemap_aces(image: np.ndarray) -> np.ndarray:
    """Apply ACES filmic tonemapping to an HDR image."""
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    x = np.clip(image, 0.0, None)
    result = (x * (a * x + b)) / (x * (c * x + d) + e)
    return np.clip(result, 0.0, 1.0)


def _tonemap_reinhard(image: np.ndarray) -> np.ndarray:
    """Apply Reinhard tonemapping to an HDR image."""
    return np.clip(image / (1.0 + image), 0.0, 1.0)


def _tonemap_linear(image: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """Simple linear exposure mapping."""
    return np.clip(image * exposure, 0.0, 1.0)


def _gamma_correct(image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Apply sRGB gamma correction."""
    return np.power(np.clip(image, 0.0, 1.0), 1.0 / gamma)


# ---------------------------------------------------------------------------
# Ambient Occlusion
# ---------------------------------------------------------------------------

def _compute_ao(point: np.ndarray, normal: np.ndarray, bvh: BVH,
                samples: int = 16, radius: float = 1.0) -> float:
    """Ray-based ambient occlusion."""
    onb = Onb(normal)
    hits = 0
    for _ in range(samples):
        direction = onb.local(_cosine_hemisphere())
        ray = Ray(point + normal * EPSILON, direction, EPSILON, radius)
        rec = HitRecord()
        if bvh.intersect(ray, rec):
            hits += 1
    return 1.0 - hits / samples


# ---------------------------------------------------------------------------
# PathTracer
# ---------------------------------------------------------------------------

class PathTracer:
    """Monte Carlo Path Tracer for Nexus3D.

    Usage::

        pt = PathTracer(width=1920, height=1080)
        pt.set_camera(position, target, fov)
        pt.add_sphere(center, radius, material)
        pt.add_area_light(center, u_axis, v_axis, emission)
        image = pt.render(samples_per_pixel=64, max_bounces=8)
        pt.save("output.png")
    """

    def __init__(self, width: int = 800, height: int = 600):
        self.width = int(width)
        self.height = int(height)
        self._objects: list = []
        self._area_lights: List[AreaLight] = []
        self._camera: Optional[Camera] = None
        self._environment: Optional[np.ndarray] = None
        self._tonemap: str = "aces"  # "aces", "reinhard", "linear"
        self._exposure: float = 1.0
        self._bvh: Optional[BVH] = None
        self._accumulated: Optional[np.ndarray] = None
        self._sample_count: int = 0
        self._seed: Optional[int] = None

    # ---- scene construction ------------------------------------------------

    def set_camera(self, position, target, fov: float,
                   aperture: float = 0.0, focal_distance: float = 10.0):
        """Configure the virtual camera.

        Parameters
        ----------
        position : array-like, shape (3,)
            Camera world-space position.
        target : array-like, shape (3,)
            Look-at target point.
        fov : float
            Vertical field of view in degrees.
        aperture : float
            Lens aperture diameter (0 = pinhole camera).
        focal_distance : float
            Distance to the focal plane (used when aperture > 0).
        """
        aspect = self.width / self.height
        self._camera = Camera(
            position=np.asarray(position, dtype=np.float64),
            target=np.asarray(target, dtype=np.float64),
            fov=fov,
            aspect=aspect,
            aperture=aperture,
            focal_distance=focal_distance,
        )

    def add_sphere(self, center, radius: float, material: PTMaterial):
        self._objects.append(Sphere(center, radius, material))

    def add_triangle(self, v0, v1, v2, material: PTMaterial):
        self._objects.append(Triangle(v0, v1, v2, material))

    def add_plane(self, point, normal, material: PTMaterial,
                  checkerboard: bool = False, checker_scale: float = 1.0):
        self._objects.append(Plane(point, normal, material, checkerboard, checker_scale))

    def add_area_light(self, center, u_axis, v_axis, emission,
                       two_sided: bool = True):
        """Add a rectangular area light to the scene.

        Parameters
        ----------
        center : array-like, shape (3,)
        u_axis, v_axis : array-like, shape (3,)
            Edge vectors defining the rectangle.
        emission : array-like, shape (3,)
            Radiant exitance (W/m^2).
        """
        al = AreaLight(center, u_axis, v_axis, emission, two_sided)
        self._area_lights.append(al)
        # Create an invisible geometry for intersection testing
        mat = PTMaterial(emission=np.asarray(emission, dtype=np.float64),
                         albedo=np.zeros(3), alpha=0.0)
        v0 = al.center - al.u_axis - al.v_axis
        v1 = al.center + al.u_axis - al.v_axis
        v2 = al.center + al.u_axis + al.v_axis
        v3 = al.center - al.u_axis + al.v_axis
        self.add_triangle(v0, v1, v2, mat)
        self.add_triangle(v0, v2, v3, mat)

    def set_environment(self, color):
        """Set the environment / background colour. *None* = black."""
        if color is None:
            self._environment = None
        else:
            self._environment = np.asarray(color, dtype=np.float64)

    def set_tonemap(self, method: str, exposure: float = 1.0):
        """Set the tone-mapping operator.

        Parameters
        ----------
        method : str
            One of ``"aces"``, ``"reinhard"``, ``"linear"``.
        exposure : float
            Linear exposure multiplier (applied before tonemapping).
        """
        method = method.lower()
        if method not in ("aces", "reinhard", "linear"):
            raise ValueError(f"Unknown tonemap method: {method}")
        self._tonemap = method
        self._exposure = exposure

    def set_seed(self, seed: Optional[int]):
        """Set the random seed for reproducibility."""
        self._seed = seed

    # ---- internal BVH build -----------------------------------------------

    def _build_bvh(self):
        self._bvh = BVH(self._objects)

    # ---- rendering ---------------------------------------------------------

    def render(self, samples_per_pixel: int = 1, max_bounces: int = 8,
               min_bounces: int = 3, accumulate: bool = True,
               callback: Optional[Callable[[float], None]] = None,
               ao_pass: bool = False, ao_samples: int = 16,
               ao_radius: float = 1.0) -> np.ndarray:
        """Render the scene and return an HDR image as a numpy float64 array
        of shape ``(height, width, 3)``.

        Parameters
        ----------
        samples_per_pixel : int
            Number of Monte Carlo samples per pixel.
        max_bounces : int
            Maximum path length (number of bounces).
        min_bounces : int
            Russian roulette is *not* applied before this depth.
        accumulate : bool
            When *True*, successive calls to ``render()`` accumulate samples
            (progressive rendering).
        callback : callable, optional
            Called with a float in ``[0, 1]`` indicating render progress.
        ao_pass : bool
            If *True*, compute an ambient-occlusion pass and multiply it
            into the final image.
        ao_samples : int
            Number of rays for the AO estimate per pixel.
        ao_radius : float
            Search radius for AO rays.

        Returns
        -------
        np.ndarray, shape (H, W, 3), dtype float64, range [0, 1]
        """
        if self._camera is None:
            raise RuntimeError("Camera not set. Call set_camera() first.")
        if len(self._objects) == 0:
            raise RuntimeError("Scene is empty. Add objects before rendering.")
        if self._bvh is None:
            self._build_bvh()

        _set_seed(self._seed)

        w, h = self.width, self.height

        if accumulate and self._accumulated is not None:
            framebuffer = self._accumulated.copy()
            prev_samples = self._sample_count
        else:
            framebuffer = np.zeros((h, w, 3), dtype=np.float64)
            prev_samples = 0

        total_pixels = h * w
        report_interval = max(total_pixels // 50, 1)
        pixels_done = 0

        start_time = time.time()

        for j in range(h):
            for i in range(w):
                colour = np.zeros(3, dtype=np.float64)
                for s in range(samples_per_pixel):
                    colour += self._trace_pixel(i, j, max_bounces, min_bounces)
                colour /= samples_per_pixel

                if prev_samples > 0:
                    # Blend with previous accumulation
                    total_s = prev_samples + samples_per_pixel
                    framebuffer[j, i] = (framebuffer[j, i] * prev_samples + colour * samples_per_pixel) / total_s
                else:
                    framebuffer[j, i] = colour

                pixels_done += 1
                if pixels_done % report_interval == 0 and callback is not None:
                    progress = pixels_done / total_pixels
                    elapsed = time.time() - start_time
                    eta = elapsed / progress - elapsed if progress > 0 else 0.0
                    callback(progress)

        self._accumulated = framebuffer
        self._sample_count = prev_samples + samples_per_pixel

        if ao_pass:
            ao_map = self._render_ao(ao_samples, ao_radius, callback)
            framebuffer *= ao_map[:, :, np.newaxis]

        # Apply tone-mapping and gamma correction
        display = self._apply_postprocess(framebuffer)

        if callback is not None:
            callback(1.0)

        return display

    def _trace_pixel(self, px: int, py: int,
                     max_bounces: int, min_bounces: int) -> np.ndarray:
        """Trace a single sample for pixel (px, py)."""
        # Jittered sub-pixel sampling
        sx = (px + _rand()) / self.width
        sy = (py + _rand()) / self.height

        if self._camera.lens_radius > EPSILON:
            ray = self._camera.get_ray_dof(sx, sy)
        else:
            ray = self._camera.get_ray(sx, sy)

        return self._trace(ray, max_bounces, min_bounces, throughput=np.ones(3), depth=0)

    def _trace(self, ray: Ray, max_bounces: int, min_bounces: int,
               throughput: np.ndarray, depth: int) -> np.ndarray:
        """Recursively trace a path."""
        if depth >= max_bounces:
            return np.zeros(3)

        rec = HitRecord()
        if not self._bvh.intersect(ray, rec):
            # Environment / sky
            if self._environment is not None:
                return self._environment
            return np.zeros(3)

        mat = rec.material
        if mat is None:
            return np.zeros(3)

        # Emissive hit – return emission directly
        if mat.is_emissive:
            return mat.emission * throughput

        # Alpha masking
        if mat.alpha < 1.0 and _rand() > mat.alpha:
            # Pass through
            new_ray = Ray(rec.point + ray.direction * EPSILON, ray.direction)
            return self._trace(new_ray, max_bounces, min_bounces, throughput, depth)

        # --- SSS path -------------------------------------------------------
        if mat.is_sss and not mat.is_glass and not mat.is_pure_mirror:
            sss_contrib = _sample_sss(ray, rec, mat, self._objects, self._bvh,
                                      max_bounces, min_bounces)
            return sss_contrib * throughput

        # --- Perfect mirror -------------------------------------------------
        if mat.is_pure_mirror:
            wi, mirror_throughput, _ = _sample_mirror(-ray.direction, rec.normal, mat)
            new_ray = Ray(rec.point + rec.normal * EPSILON, wi)
            return self._trace(new_ray, max_bounces, min_bounces,
                               throughput * mirror_throughput, depth + 1)

        # --- Glass / dielectric ---------------------------------------------
        if mat.is_glass:
            wi, glass_throughput, _ = _sample_glass(-ray.direction, rec.normal, mat)
            new_ray = Ray(rec.point + rec.normal * EPSILON * (1.0 if rec.front_face else -1.0), wi)
            return self._trace(new_ray, max_bounces, min_bounces,
                               throughput * glass_throughput, depth + 1)

        # --- General PBR (Cook-Torrance + Lambert) --------------------------
        wo = _norm(-ray.direction)
        wi, brdf_weight, pdf = _sample_brdf(wo, rec.normal, mat)

        new_throughput = throughput * brdf_weight

        # Russian roulette
        if depth >= min_bounces:
            rr_prob = float(np.max(new_throughput))
            if rr_prob < EPSILON:
                return np.zeros(3)
            if _rand() > min(rr_prob, 0.95):
                return np.zeros(3)
            new_throughput *= 1.0 / min(rr_prob, 0.95)

        new_ray = Ray(rec.point + rec.normal * EPSILON, wi)

        return self._trace(new_ray, max_bounces, min_bounces,
                           new_throughput, depth + 1)

    # ---- AO pass -----------------------------------------------------------

    def _render_ao(self, samples: int, radius: float,
                   callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """Compute an ambient-occlusion map."""
        h, w = self.height, self.width
        ao_map = np.ones((h, w), dtype=np.float64)
        total = h * w
        report_interval = max(total // 50, 1)
        done = 0

        for j in range(h):
            for i in range(w):
                sx = (i + 0.5) / w
                sy = (j + 0.5) / h
                ray = self._camera.get_ray(sx, sy)
                rec = HitRecord()
                if self._bvh.intersect(ray, rec) and rec.material is not None:
                    ao_map[j, i] = _compute_ao(rec.point, rec.normal,
                                                self._bvh, samples, radius)
                done += 1
                if done % report_interval == 0 and callback is not None:
                    callback(done / total)
        return ao_map

    # ---- post-processing ---------------------------------------------------

    def _apply_postprocess(self, hdr: np.ndarray) -> np.ndarray:
        """Apply exposure, tonemapping, and gamma correction."""
        exposed = hdr * self._exposure
        if self._tonemap == "aces":
            mapped = _tonemap_aces(exposed)
        elif self._tonemap == "reinhard":
            mapped = _tonemap_reinhard(exposed)
        else:
            mapped = _tonemap_linear(exposed, self._exposure)
        return _gamma_correct(mapped)

    # ---- I/O ---------------------------------------------------------------

    def save(self, filepath: str, image: Optional[np.ndarray] = None):
        """Save the rendered image to a PNG file.

        Parameters
        ----------
        filepath : str
            Output file path (should end in ``.png``).
        image : np.ndarray, optional
            Image to save. If *None*, saves the last rendered / accumulated image.
        """
        if image is None:
            image = self._accumulated
        if image is None:
            raise RuntimeError("No image to save. Render first.")
        # Ensure display pipeline
        display = self._apply_postprocess(image) if image.max() > 1.0 or self._tonemap != "linear" else image
        uint8 = (np.clip(display, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        img = Image.fromarray(uint8, mode="RGB")
        img.save(filepath)

    def get_hdr(self) -> Optional[np.ndarray]:
        """Return the raw HDR framebuffer (float64, before tonemapping)."""
        return self._accumulated

    def get_ldr(self) -> Optional[np.ndarray]:
        """Return the display-ready LDR framebuffer (float64, after tonemapping + gamma)."""
        if self._accumulated is None:
            return None
        return self._apply_postprocess(self._accumulated)

    def reset_accumulation(self):
        """Reset progressive rendering state."""
        self._accumulated = None
        self._sample_count = 0

    def rebuild_bvh(self):
        """Force a BVH rebuild (call after modifying the scene)."""
        self._build_bvh()


# ---------------------------------------------------------------------------
# Convenience preset scenes
# ---------------------------------------------------------------------------

def _default_white() -> PTMaterial:
    return PTMaterial(albedo=np.array([0.8, 0.8, 0.8]))

def _default_mirror() -> PTMaterial:
    return PTMaterial(metallic=1.0, roughness=0.0, albedo=np.array([0.95, 0.95, 0.95]))

def _default_glass() -> PTMaterial:
    return PTMaterial(transmission=1.0, ior=1.5, metallic=0.0, roughness=0.0,
                      albedo=np.array([1.0, 1.0, 1.0]))

def _default_red() -> PTMaterial:
    return PTMaterial(albedo=np.array([0.8, 0.1, 0.1]))

def _default_green() -> PTMaterial:
    return PTMaterial(albedo=np.array([0.1, 0.8, 0.1]))

def _default_blue() -> PTMaterial:
    return PTMaterial(albedo=np.array([0.1, 0.1, 0.8]))

def _default_skin() -> PTMaterial:
    return PTMaterial(albedo=np.array([0.8, 0.5, 0.4]), sss_radius=0.5, roughness=0.6)

def _default_gold() -> PTMaterial:
    return PTMaterial(albedo=np.array([1.0, 0.76, 0.34]), metallic=1.0, roughness=0.2)

def _default_copper() -> PTMaterial:
    return PTMaterial(albedo=np.array([0.95, 0.64, 0.54]), metallic=1.0, roughness=0.3)
