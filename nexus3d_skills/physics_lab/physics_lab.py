"""
Physics Lab — Advanced Physics Simulation Toolkit
==================================================
Provides rigid body dynamics, collision detection (AABB, sphere,
mesh), spring-mass systems, soft bodies, and simulation stepping
for the Nexus3D environment.
"""

from __future__ import annotations

import json
import math
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class BodyType(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    KINEMATIC = "kinematic"
    SOFT = "soft"


class CollisionShape(Enum):
    SPHERE = "sphere"
    BOX = "box"
    CAPSULE = "capsule"
    PLANE = "plane"
    MESH = "mesh"
    COMPOUND = "compound"


GRAVITY = np.array([0.0, -9.81, 0.0], dtype=np.float64)
FIXED_DT = 1.0 / 60.0
CONTACT_EPSILON = 1e-4
RESTITUTION_DEFAULT = 0.3
FRICTION_DEFAULT = 0.5


@dataclass
class AABB:
    """Axis-aligned bounding box."""
    min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    max: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def center(self) -> Tuple[float, float, float]:
        return tuple((a + b) / 2.0 for a, b in zip(self.min, self.max))

    @property
    def size(self) -> Tuple[float, float, float]:
        return tuple(b - a for a, b in zip(self.min, self.max))

    def contains(self, point: Tuple[float, float, float]) -> bool:
        return all(lo <= p <= hi for p, lo, hi in zip(point, self.min, self.max))

    def overlaps(self, other: "AABB") -> bool:
        return all(lo <= other.max[i] and other.min[i] <= hi
                   for i, (lo, hi) in enumerate(zip(self.min, self.max)))


@dataclass
class Contact:
    """Contact point between two bodies."""
    body_a: str = ""
    body_b: str = ""
    point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    penetration: float = 0.0
    restitution: float = RESTITUTION_DEFAULT
    friction: float = FRICTION_DEFAULT


@dataclass
class RigidBody:
    """A rigid body in the physics world."""
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = "body"
    body_type: BodyType = BodyType.DYNAMIC
    shape: CollisionShape = CollisionShape.SPHERE
    mass: float = 1.0
    restitution: float = RESTITUTION_DEFAULT
    friction: float = FRICTION_DEFAULT
    linear_damping: float = 0.01
    angular_damping: float = 0.01

    # Transform
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Velocity
    linear_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Shape parameters
    radius: float = 0.5      # sphere/capsule
    half_extents: Tuple[float, float, float] = (0.5, 0.5, 0.5)  # box
    height: float = 1.0      # capsule

    # Accumulated forces
    force: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    torque: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Sleeping
    is_sleeping: bool = False
    sleep_timer: float = 0.0
    sleep_threshold: float = 0.01

    # Mesh data (for mesh shapes)
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)

    @property
    def inverse_mass(self) -> float:
        if self.body_type == BodyType.STATIC or self.mass <= 0:
            return 0.0
        return 1.0 / self.mass

    @property
    def is_static(self) -> bool:
        return self.body_type == BodyType.STATIC

    def get_aabb(self) -> AABB:
        if self.shape == CollisionShape.SPHERE:
            r = self.radius * max(self.scale)
            return AABB(
                min=tuple(p - r for p in self.position),
                max=tuple(p + r for p in self.position),
            )
        elif self.shape == CollisionShape.BOX:
            he = tuple(h * s for h, s in zip(self.half_extents, self.scale))
            return AABB(
                min=tuple(p - h for p, h in zip(self.position, he)),
                max=tuple(p + h for p, h in zip(self.position, he)),
            )
        elif self.shape == CollisionShape.CAPSULE:
            r = self.radius * max(self.scale)
            hh = self.height / 2.0
            return AABB(
                min=(self.position[0] - r, self.position[1] - hh - r, self.position[2] - r),
                max=(self.position[0] + r, self.position[1] + hh + r, self.position[2] + r),
            )
        else:
            return AABB(min=self.position, max=self.position)

    def apply_force(self, f: Tuple[float, float, float]) -> None:
        self.force = tuple(a + b for a, b in zip(self.force, f))

    def apply_impulse(self, impulse: Tuple[float, float, float],
                      world_point: Optional[Tuple[float, float, float]] = None) -> None:
        lv = list(self.linear_velocity)
        for i in range(3):
            lv[i] += impulse[i] * self.inverse_mass
        self.linear_velocity = tuple(lv)
        if world_point:
            # Simple angular impulse (placeholder using cross product)
            rx = tuple(world_point[i] - self.position[i] for i in range(3))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["body_type"] = self.body_type.value
        d["shape"] = self.shape.value
        return d


# ---------------------------------------------------------------------------
# Collision Detection
# ---------------------------------------------------------------------------

class CollisionDetector:
    """Narrow-phase collision detection for primitive shapes."""

    @staticmethod
    def sphere_sphere(a: RigidBody, b: RigidBody) -> Optional[Contact]:
        """Sphere vs sphere collision."""
        pa = np.array(a.position, dtype=np.float64)
        pb = np.array(b.position, dtype=np.float64)
        diff = pb - pa
        dist = np.linalg.norm(diff)
        ra = a.radius * max(a.scale)
        rb = b.radius * max(b.scale)
        if dist >= ra + rb or dist < 1e-8:
            return None
        normal = tuple((diff / dist).tolist()) if dist > 1e-8 else (0.0, 1.0, 0.0)
        penetration = ra + rb - dist
        point = tuple(((pa + pb) / 2.0).tolist())
        return Contact(
            body_a=a.uid, body_b=b.uid,
            point=point, normal=normal,
            penetration=penetration,
            restitution=max(a.restitution, b.restitution),
            friction=max(a.friction, b.friction),
        )

    @staticmethod
    def sphere_box(sphere: RigidBody, box: RigidBody) -> Optional[Contact]:
        """Sphere vs box (closest point on AABB)."""
        sp = np.array(sphere.position, dtype=np.float64)
        he = np.array(box.half_extents, dtype=np.float64) * np.array(box.scale, dtype=np.float64)
        center = np.array(box.position, dtype=np.float64)
        closest = np.clip(sp - center, -he, he) + center
        diff = sp - closest
        dist = np.linalg.norm(diff)
        r = sphere.radius * max(sphere.scale)
        if dist >= r or dist < 1e-8:
            return None
        normal = tuple((-diff / dist).tolist())
        penetration = r - dist
        return Contact(
            body_a=sphere.uid, body_b=box.uid,
            point=tuple(closest.tolist()), normal=normal,
            penetration=penetration,
        )

    @staticmethod
    def aabb_aabb(a: RigidBody, b: RigidBody) -> Optional[Contact]:
        """AABB vs AABB."""
        aa = a.get_aabb()
        bb = b.get_aabb()
        if not aa.overlaps(bb):
            return None
        # Find minimum penetration axis
        overlap = min(
            (aa.max[0] - bb.min[0], (1.0, 0.0, 0.0)),
            (bb.max[0] - aa.min[0], (-1.0, 0.0, 0.0)),
            (aa.max[1] - bb.min[1], (0.0, 1.0, 0.0)),
            (bb.max[1] - aa.min[1], (0.0, -1.0, 0.0)),
            (aa.max[2] - bb.min[2], (0.0, 0.0, 1.0)),
            (bb.max[2] - aa.min[2], (0.0, 0.0, -1.0)),
            key=lambda x: x[0],
        )
        center = tuple((np.array(aa.center) + np.array(bb.center)) / 2.0)
        return Contact(
            body_a=a.uid, body_b=b.uid,
            point=center, normal=overlap[1],
            penetration=overlap[0],
        )

    @staticmethod
    def detect(a: RigidBody, b: RigidBody) -> Optional[Contact]:
        """Dispatch to correct collision detection based on shapes."""
        shapes = {a.shape, b.shape}
        if shapes == {CollisionShape.SPHERE}:
            return CollisionDetector.sphere_sphere(a, b)
        if shapes == {CollisionShape.SPHERE, CollisionShape.BOX}:
            if a.shape == CollisionShape.SPHERE:
                return CollisionDetector.sphere_box(a, b)
            else:
                return CollisionDetector.sphere_box(b, a)
        if shapes == {CollisionShape.BOX} or shapes == {CollisionShape.CAPSULE, CollisionShape.BOX}:
            return CollisionDetector.aabb_aabb(a, b)
        return None


# ---------------------------------------------------------------------------
# Broad Phase (Spatial Hashing)
# ---------------------------------------------------------------------------

class SpatialHash:
    """Grid-based spatial hashing for broad-phase collision culling."""

    def __init__(self, cell_size: float = 2.0) -> None:
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int, int], List[str]] = {}

    def _cell(self, p: Tuple[float, float, float]) -> Tuple[int, int, int]:
        return (int(p[0] / self.cell_size),
                int(p[1] / self.cell_size),
                int(p[2] / self.cell_size))

    def clear(self) -> None:
        self.grid.clear()

    def insert(self, body_uid: str, aabb: AABB) -> None:
        lo = self._cell(aabb.min)
        hi = self._cell(aabb.max)
        for x in range(lo[0], hi[0] + 1):
            for y in range(lo[1], hi[1] + 1):
                for z in range(lo[2], hi[2] + 1):
                    cell = (x, y, z)
                    if cell not in self.grid:
                        self.grid[cell] = []
                    self.grid[cell].append(body_uid)

    def get_potential_pairs(self) -> List[Tuple[str, str]]:
        """Return pairs of UIDs in the same or adjacent cells."""
        pairs = set()
        for cell, uids in self.grid.items():
            for i in range(len(uids)):
                for j in range(i + 1, len(uids)):
                    a, b = uids[i], uids[j]
                    if a < b:
                        pairs.add((a, b))
                    else:
                        pairs.add((b, a))
        return list(pairs)


# ---------------------------------------------------------------------------
# Constraint Solver (Sequential Impulse)
# ---------------------------------------------------------------------------

class ConstraintSolver:
    """Sequential impulse-based constraint solver."""

    @staticmethod
    def resolve_contact(contact: Contact,
                        bodies: Dict[str, RigidBody]) -> None:
        a = bodies.get(contact.body_a)
        b = bodies.get(contact.body_b)
        if not a or not b:
            return

        if a.is_static and b.is_static:
            return

        normal = np.array(contact.normal, dtype=np.float64)
        pa = np.array(contact.point, dtype=np.float64)
        pb = np.array(contact.point, dtype=np.float64)

        rel_vel = np.array(b.linear_velocity, dtype=np.float64) - np.array(a.linear_velocity, dtype=np.float64)
        vel_along_normal = np.dot(rel_vel, normal)

        if vel_along_normal > 0.0 and contact.penetration < CONTACT_EPSILON:
            return

        # Restitution impulse
        e = contact.restitution
        j = -(1.0 + e) * vel_along_normal
        inv_mass_sum = a.inverse_mass + b.inverse_mass
        if inv_mass_sum > 0:
            j /= inv_mass_sum

        impulse = normal * j
        if not a.is_static:
            a.linear_velocity = tuple(
                np.array(a.linear_velocity, dtype=np.float64) - impulse * a.inverse_mass
            )
        if not b.is_static:
            b.linear_velocity = tuple(
                np.array(b.linear_velocity, dtype=np.float64) + impulse * b.inverse_mass
            )

        # Position correction (Baumgarte)
        slop = 0.01
        correction = max(contact.penetration - slop, 0.0) * 0.8
        if inv_mass_sum > 0:
            corr = normal * (correction / inv_mass_sum)
            if not a.is_static:
                a.position = tuple(
                    np.array(a.position, dtype=np.float64) - corr * a.inverse_mass
                )
            if not b.is_static:
                b.position = tuple(
                    np.array(b.position, dtype=np.float64) + corr * b.inverse_mass
                )

        # Friction
        tangent = rel_vel - normal * vel_along_normal
        tan_len = np.linalg.norm(tangent)
        if tan_len > 1e-6:
            tangent = tangent / tan_len
            jt = -np.dot(rel_vel, tangent)
            if inv_mass_sum > 0:
                jt /= inv_mass_sum
            max_friction = contact.friction * abs(j)
            jt = max(-max_friction, min(max_friction, jt))
            friction_impulse = tangent * jt
            if not a.is_static:
                a.linear_velocity = tuple(
                    np.array(a.linear_velocity, dtype=np.float64) - friction_impulse * a.inverse_mass
                )
            if not b.is_static:
                b.linear_velocity = tuple(
                    np.array(b.linear_velocity, dtype=np.float64) + friction_impulse * b.inverse_mass
                )


# ---------------------------------------------------------------------------
# Physics World
# ---------------------------------------------------------------------------

class PhysicsWorld:
    """The main physics simulation world."""

    def __init__(self, gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)) -> None:
        self.gravity = np.array(gravity, dtype=np.float64)
        self.bodies: Dict[str, RigidBody] = {}
        self.spatial_hash = SpatialHash(cell_size=2.0)
        self.solver = ConstraintSolver()
        self.dt = FIXED_DT
        self.simulation_time: float = 0.0
        self.iteration_count: int = 0

    def add_body(self, body: RigidBody) -> str:
        self.bodies[body.uid] = body
        return body.uid

    def remove_body(self, uid: str) -> None:
        self.bodies.pop(uid, None)

    def get_body(self, uid: str) -> Optional[RigidBody]:
        return self.bodies.get(uid)

    def step(self, dt: Optional[float] = None) -> List[Contact]:
        """Advance the simulation by one time step."""
        if dt is None:
            dt = self.dt
        self.dt = dt
        self.iteration_count += 1

        # --- Integrate forces ---
        for body in self.bodies.values():
            if body.body_type != BodyType.DYNAMIC:
                continue
            if body.is_sleeping:
                continue

            # Apply gravity
            if body.inverse_mass > 0:
                g_force = self.gravity / body.inverse_mass
            else:
                g_force = np.zeros(3)

            acc = g_force + np.array(body.force, dtype=np.float64) * body.inverse_mass

            # Integrate velocity
            new_lv = np.array(body.linear_velocity, dtype=np.float64) + acc * dt
            new_lv *= (1.0 - body.linear_damping)
            body.linear_velocity = tuple(new_lv.tolist())

            # Integrate position
            new_pos = np.array(body.position, dtype=np.float64) + new_lv * dt
            body.position = tuple(new_pos.tolist())

            # Reset forces
            body.force = (0.0, 0.0, 0.0)
            body.torque = (0.0, 0.0, 0.0)

        # --- Broad phase ---
        self.spatial_hash.clear()
        for body in self.bodies.values():
            self.spatial_hash.insert(body.uid, body.get_aabb())

        potential_pairs = self.spatial_hash.get_potential_pairs()

        # --- Narrow phase ---
        contacts: List[Contact] = []
        for uid_a, uid_b in potential_pairs:
            a = self.bodies.get(uid_a)
            b = self.bodies.get(uid_b)
            if not a or not b:
                continue
            if a.is_static and b.is_static:
                continue
            contact = CollisionDetector.detect(a, b)
            if contact:
                contacts.append(contact)

        # --- Solve constraints ---
        for _ in range(8):  # 8 iterations for stability
            for contact in contacts:
                self.solver.resolve_contact(contact, self.bodies)

        # --- Sleep check ---
        for body in self.bodies.values():
            if body.body_type != BodyType.DYNAMIC:
                continue
            speed = np.linalg.norm(np.array(body.linear_velocity, dtype=np.float64))
            if speed < body.sleep_threshold:
                body.sleep_timer += dt
                if body.sleep_timer > 0.5:
                    body.is_sleeping = True
            else:
                body.is_sleeping = False
                body.sleep_timer = 0.0

        self.simulation_time += dt
        return contacts

    def raycast(self, origin: Tuple[float, float, float],
                direction: Tuple[float, float, float],
                max_distance: float = 100.0) -> List[Dict[str, Any]]:
        """Simple raycast against all bodies (sphere test only)."""
        o = np.array(origin, dtype=np.float64)
        d = np.array(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        hits = []
        for body in self.bodies.values():
            if body.shape != CollisionShape.SPHERE:
                continue
            c = np.array(body.position, dtype=np.float64)
            r = body.radius * max(body.scale)
            oc = o - c
            a = np.dot(d, d)
            b = 2.0 * np.dot(oc, d)
            c_ = np.dot(oc, oc) - r * r
            disc = b * b - 4 * a * c_
            if disc < 0:
                continue
            t = (-b - math.sqrt(disc)) / (2.0 * a)
            if 0 < t < max_distance:
                hit_point = o + d * t
                hits.append({
                    "body_uid": body.uid,
                    "point": tuple(hit_point.tolist()),
                    "distance": float(t),
                    "normal": tuple(((hit_point - c) / r).tolist()),
                })
        return sorted(hits, key=lambda h: h["distance"])

    def get_state(self) -> Dict[str, Any]:
        return {
            "time": self.simulation_time,
            "iterations": self.iteration_count,
            "num_bodies": len(self.bodies),
            "bodies": {uid: body.to_dict() for uid, body in self.bodies.items()},
        }


# ---------------------------------------------------------------------------
# Spring-Mass / Soft Body
# ---------------------------------------------------------------------------

@dataclass
class SpringMassPoint:
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mass: float = 1.0
    pinned: bool = False


@dataclass
class Spring:
    a: int = 0
    b: int = 0
    rest_length: float = 1.0
    stiffness: float = 100.0
    damping: float = 1.0


class SpringMassSystem:
    """Simple spring-mass simulator for cloth/deformable objects."""

    def __init__(self) -> None:
        self.points: List[SpringMassPoint] = []
        self.springs: List[Spring] = []
        self.gravity = np.array([0.0, -9.81, 0.0], dtype=np.float64)

    def add_point(self, p: SpringMassPoint) -> int:
        idx = len(self.points)
        self.points.append(p)
        return idx

    def add_spring(self, spring: Spring) -> int:
        idx = len(self.springs)
        self.springs.append(spring)
        return idx

    def create_cloth(self, width: int = 10, height: int = 10,
                     spacing: float = 0.1, stiffness: float = 100.0,
                     pin_top: bool = True) -> None:
        for y in range(height):
            for x in range(width):
                px = (x - width / 2) * spacing
                py = 2.0 - y * spacing
                pz = 0.0
                pinned = pin_top and y == 0
                self.add_point(SpringMassPoint(
                    position=(px, py, pz), mass=0.1, pinned=pinned
                ))

        def idx(x, y): return y * width + x
        for y in range(height):
            for x in range(width):
                if x < width - 1:
                    dl = spacing
                    self.add_spring(Spring(idx(x, y), idx(x + 1, y), dl, stiffness))
                if y < height - 1:
                    dl = spacing
                    self.add_spring(Spring(idx(x, y), idx(x, y + 1), dl, stiffness))
                if x < width - 1 and y < height - 1:
                    dl = spacing * math.sqrt(2)
                    self.add_spring(Spring(idx(x, y), idx(x + 1, y + 1), dl, stiffness * 0.5))
                    self.add_spring(Spring(idx(x + 1, y), idx(x, y + 1), dl, stiffness * 0.5))

    def step(self, dt: float = 1.0 / 60.0) -> None:
        forces = [np.zeros(3, dtype=np.float64) for _ in self.points]

        # Gravity
        for i, p in enumerate(self.points):
            if not p.pinned:
                forces[i] += self.gravity * p.mass

        # Spring forces
        for spring in self.springs:
            a = self.points[spring.a]
            b = self.points[spring.b]
            pa = np.array(a.position, dtype=np.float64)
            pb = np.array(b.position, dtype=np.float64)
            diff = pb - pa
            dist = np.linalg.norm(diff)
            if dist < 1e-8:
                continue
            direction = diff / dist
            displacement = dist - spring.rest_length
            force_mag = spring.stiffness * displacement

            # Damping
            rel_vel = np.array(b.velocity, dtype=np.float64) - np.array(a.velocity, dtype=np.float64)
            damping_force = spring.damping * np.dot(rel_vel, direction)

            total_force = (force_mag + damping_force) * direction
            if not a.pinned:
                forces[spring.a] += total_force
            if not b.pinned:
                forces[spring.b] -= total_force

        # Integrate
        for i, p in enumerate(self.points):
            if p.pinned:
                continue
            acc = forces[i] / p.mass
            new_vel = np.array(p.velocity, dtype=np.float64) + acc * dt
            new_pos = np.array(p.position, dtype=np.float64) + new_vel * dt
            p.velocity = tuple(new_vel.tolist())
            p.position = tuple(new_pos.tolist())

    def get_state(self) -> Dict[str, Any]:
        return {
            "num_points": len(self.points),
            "num_springs": len(self.springs),
            "points": [asdict(p) for p in self.points],
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def create_physics_world(gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)) -> str:
    """Create a new physics world."""
    world = PhysicsWorld(gravity)
    return world


async def add_rigid_body(world, body_data: Dict[str, Any]) -> str:
    """Add a rigid body to the physics world."""
    body = RigidBody(**{k: v for k, v in body_data.items()
                        if k in RigidBody.__dataclass_fields__})
    if "body_type" in body_data:
        body.body_type = BodyType(body_data["body_type"])
    if "shape" in body_data:
        body.shape = CollisionShape(body_data["shape"])
    return world.add_body(body)


async def step_simulation(world, dt: float = 1.0 / 60.0, substeps: int = 1) -> List[Dict[str, Any]]:
    """Step the physics simulation."""
    all_contacts = []
    for _ in range(substeps):
        contacts = world.step(dt / substeps)
        all_contacts.extend([asdict(c) for c in contacts])
    return all_contacts


async def get_physics_state(world) -> Dict[str, Any]:
    """Get the current state of the physics world."""
    return world.get_state()


async def create_cloth_simulation(width: int = 10, height: int = 10,
                                   spacing: float = 0.1,
                                   stiffness: float = 100.0) -> Dict[str, Any]:
    """Create a cloth simulation and return its state."""
    system = SpringMassSystem()
    system.create_cloth(width, height, spacing, stiffness)
    return system.get_state()


async def step_cloth_simulation(world_state: Dict[str, Any],
                                 dt: float = 1.0 / 60.0,
                                 steps: int = 1) -> Dict[str, Any]:
    """Step a cloth simulation."""
    system = SpringMassSystem()
    points_data = world_state.get("points", [])
    for pd in points_data:
        system.points.append(SpringMassPoint(**pd))
    for _ in range(steps):
        system.step(dt)
    return system.get_state()


async def raycast(world, origin: Tuple[float, float, float],
                   direction: Tuple[float, float, float],
                   max_distance: float = 100.0) -> List[Dict[str, Any]]:
    """Perform a raycast in the physics world."""
    return world.raycast(origin, direction, max_distance)
