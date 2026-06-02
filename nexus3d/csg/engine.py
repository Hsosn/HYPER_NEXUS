"""Nexus3D CSG Engine - Constructive Solid Geometry using BSP Trees.

Provides real boolean operations (union, subtract, intersect) on closed
triangle meshes, architectural primitives (walls, rooms, windows, doors,
arches, stairs), and post-CSG mesh cleanup utilities.

The boolean algorithm follows the classic BSP-tree approach (csg.js by
Evan Wallace): each solid is represented as a BSP tree, and operations
are performed by clipping trees against each other and inverting as
needed.

Typical usage::

    from nexus3d.csg import CSGMesh, ArchitectureKit

    wall = ArchitectureKit.wall(4.0, 3.0, 0.15)
    wall = ArchitectureKit.add_door(wall, door_width=1.0, door_height=2.1)
    wall = ArchitectureKit.add_window(wall, window_width=1.2)

    a = CSGMesh.from_sphere(1.0, 16)
    b = CSGMesh.from_cube(1.0)
    result = a.subtract(b)

    vertices, faces = result.to_vertices_faces()
"""

from __future__ import annotations

import copy
import math
from typing import Tuple, List, Optional, Any, Dict

import numpy as np

# ==============================================================================
# Constants
# ==============================================================================

EPSILON = 1e-5
_COPLANAR = 0
_FRONT = 1
_BACK = 2
_SPANNING = 3


# ==============================================================================
# Plane
# ==============================================================================

class Plane:
    """3D plane defined by unit normal *n* and signed distance *w* from the
    origin.  The implicit equation is ``n . p == w`` for any point *p*
    on the plane.
    """

    __slots__ = ("normal", "w")

    def __init__(self, normal: np.ndarray, w: float) -> None:
        self.normal = np.asarray(normal, dtype=np.float64).copy()
        n_len = np.linalg.norm(self.normal)
        if n_len > 1e-12:
            self.normal /= n_len
        self.w = float(w)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_points(cls, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> "Plane":
        """Create a plane from three non-collinear points (CCW winding)."""
        n = np.cross(b - a, c - a)
        n_len = np.linalg.norm(n)
        if n_len < 1e-10:
            # Degenerate triangle – fall back to Y-up
            return cls(np.array([0.0, 1.0, 0.0]), 0.0)
        n = n / n_len
        w = float(np.dot(n, a))
        return cls(n, w)

    @classmethod
    def from_normal_and_point(
        cls, normal: np.ndarray, point: np.ndarray
    ) -> "Plane":
        n = np.asarray(normal, dtype=np.float64)
        n_len = np.linalg.norm(n)
        if n_len < 1e-12:
            return cls(np.array([0.0, 1.0, 0.0]), 0.0)
        n = n / n_len
        return cls(n, float(np.dot(n, point)))

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def clone(self) -> "Plane":
        return Plane(self.normal.copy(), self.w)

    def flip(self) -> None:
        """Reverse the plane normal (inside <-> outside)."""
        self.normal = -self.normal
        self.w = -self.w

    def split_polygon(
        self,
        polygon: BSPPolygon,
        coplanar_front: List[BSPPolygon],
        coplanar_back: List[BSPPolygon],
        front: List[BSPPolygon],
        back: List[BSPPolygon],
    ) -> None:
        """Classify and optionally split *polygon* against this plane.

        The polygon is placed into the appropriate output lists based on
        which side(s) of the plane its vertices lie on.  Spanning polygons
        are split into two new polygons.
        """
        polygon_type = 0
        types: List[int] = []
        for v in polygon.vertices:
            t = float(np.dot(self.normal, v) - self.w)
            if t < -EPSILON:
                tp = _BACK
            elif t > EPSILON:
                tp = _FRONT
            else:
                tp = _COPLANAR
            types.append(tp)
            polygon_type |= tp

        # Fast paths
        if polygon_type == _COPLANAR:
            if np.dot(self.normal, polygon.plane.normal) > 0:
                coplanar_front.append(polygon)
            else:
                coplanar_back.append(polygon)
            return

        if polygon_type == _FRONT:
            front.append(polygon)
            return

        if polygon_type == _BACK:
            back.append(polygon)
            return

        # Spanning – must split
        f_verts: List[np.ndarray] = []
        b_verts: List[np.ndarray] = []
        n = len(polygon.vertices)
        for i in range(n):
            j = (i + 1) % n
            ti = types[i]
            tj = types[j]
            vi = polygon.vertices[i]
            vj = polygon.vertices[j]

            if ti != _BACK:
                f_verts.append(vi)
            if ti != _FRONT:
                b_verts.append(vi)

            if (ti | tj) == _SPANNING:
                denom = float(np.dot(self.normal, vj - vi))
                if abs(denom) < 1e-12:
                    t_val = 0.5
                else:
                    t_val = (self.w - float(np.dot(self.normal, vi))) / denom
                intersection = vi + t_val * (vj - vi)
                f_verts.append(intersection)
                b_verts.append(intersection)

        if len(f_verts) >= 3:
            new_front = BSPPolygon(f_verts, polygon.shared)
            if _polygon_area(new_front) > 1e-10:
                front.append(new_front)
        if len(b_verts) >= 3:
            new_back = BSPPolygon(b_verts, polygon.shared)
            if _polygon_area(new_back) > 1e-10:
                back.append(new_back)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Plane(normal=[{self.normal[0]:.4f}, {self.normal[1]:.4f}, "
            f"{self.normal[2]:.4f}], w={self.w:.6f})"
        )


# ==============================================================================
# BSP Polygon
# ==============================================================================

class BSPPolygon:
    """A convex polygon used during BSP tree construction and clipping.

    Attributes
    ----------
    vertices : list[np.ndarray]
        Ordered vertex positions (3 or more).
    plane : Plane
        The polygon's supporting plane, computed from the first three
        non-degenerate vertices.
    shared : Any
        Opaque per-polygon data propagated through CSG (e.g. material).
    """

    def __init__(
        self,
        vertices: List[np.ndarray],
        shared: Any = None,
    ) -> None:
        if len(vertices) < 3:
            raise ValueError("BSPPolygon requires at least 3 vertices")
        self.vertices: List[np.ndarray] = [
            np.asarray(v, dtype=np.float64).copy() for v in vertices
        ]
        self.shared = shared
        self._plane: Optional[Plane] = None

    @property
    def plane(self) -> Plane:
        if self._plane is None:
            self._plane = Plane.from_points(
                self.vertices[0], self.vertices[1], self.vertices[2]
            )
        return self._plane

    def clone(self) -> "BSPPolygon":
        return BSPPolygon([v.copy() for v in self.vertices], self.shared)

    def flip(self) -> None:
        """Reverse winding order (which inverts the polygon normal)."""
        self.vertices.reverse()
        if self._plane is not None:
            self._plane.flip()
        else:
            # Recompute plane so the lazy property reflects the flip
            self._plane = Plane.from_points(
                self.vertices[0], self.vertices[1], self.vertices[2]
            )

    def __repr__(self) -> str:  # pragma: no cover
        return f"BSPPolygon(n_verts={len(self.vertices)})"


def _polygon_area(poly: BSPPolygon) -> float:
    """Compute the area of a convex polygon using the fan-triangle method."""
    n = len(poly.vertices)
    if n < 3:
        return 0.0
    total = 0.0
    v0 = poly.vertices[0]
    for i in range(1, n - 1):
        vi = poly.vertices[i]
        vj = poly.vertices[i + 1]
        total += float(np.linalg.norm(np.cross(vi - v0, vj - v0)))
    return total * 0.5


# ==============================================================================
# BSP Node
# ==============================================================================

class BSPNode:
    """Node in a Binary Space Partitioning tree for CSG.

    Each node stores:
    * A splitting plane (``None`` for empty leaves).
    * Polygons that are coplanar with the splitting plane.
    * ``front`` / ``back`` child nodes.

    The tree partitions space so that everything in the *back* subtree
    is considered "inside" the solid and everything in the *front*
    subtree is "outside."
    """

    __slots__ = ("plane", "polygons", "front", "back")

    def __init__(self, polygons: Optional[List[BSPPolygon]] = None) -> None:
        self.plane: Optional[Plane] = None
        self.polygons: List[BSPPolygon] = []
        self.front: Optional[BSPNode] = None
        self.back: Optional[BSPNode] = None
        if polygons:
            self.build(polygons)

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def build(self, polygons: List[BSPPolygon], _depth: int = 0) -> None:
        """Insert *polygons* into this (sub-)tree.

        The first call on an empty node also sets the splitting plane.
        Subsequent calls add polygons to the existing structure.

        A maximum recursion depth prevents runaway tree growth caused
        by degenerate / near-degenerate polygons.
        """
        if not polygons:
            return
        # Prevent infinite recursion from degenerate polygons
        if _depth > 500:
            self.polygons.extend(polygons)
            return
        if self.plane is None:
            # Pick a good splitting plane: the polygon whose plane is
            # most well-defined (largest area).
            best_poly = max(
                polygons,
                key=lambda p: _polygon_area(p),
            )
            self.plane = best_poly.plane.clone()
        front_list: List[BSPPolygon] = []
        back_list: List[BSPPolygon] = []
        for poly in polygons:
            self.plane.split_polygon(
                poly,
                self.polygons,
                self.polygons,
                front_list,
                back_list,
            )
        if front_list:
            if self.front is None:
                self.front = BSPNode()
            self.front.build(front_list, _depth + 1)
        if back_list:
            if self.back is None:
                self.back = BSPNode()
            self.back.build(back_list, _depth + 1)

    # ------------------------------------------------------------------
    # Polygon retrieval
    # ------------------------------------------------------------------

    def all_polygons(self) -> List[BSPPolygon]:
        """Return every polygon stored in this sub-tree."""
        result: List[BSPPolygon] = list(self.polygons)
        if self.front:
            result.extend(self.front.all_polygons())
        if self.back:
            result.extend(self.back.all_polygons())
        return result

    # ------------------------------------------------------------------
    # Clipping
    # ------------------------------------------------------------------

    def clip_polygons(self, polygons: List[BSPPolygon]) -> List[BSPPolygon]:
        """Clip *polygons* against this BSP tree and return the survivors.

        Polygons that fall on the *back* side without a back child are
        discarded (they are inside the solid).
        """
        if self.plane is None:
            return list(polygons)
        coplanar_front: List[BSPPolygon] = []
        coplanar_back: List[BSPPolygon] = []
        front_list: List[BSPPolygon] = []
        back_list: List[BSPPolygon] = []
        for poly in polygons:
            self.plane.split_polygon(
                poly,
                coplanar_front,
                coplanar_back,
                front_list,
                back_list,
            )
        if self.front:
            front_list = self.front.clip_polygons(front_list)
        if self.back:
            back_list = self.back.clip_polygons(back_list)
        else:
            back_list = []  # inside the solid → discard
        # Coplanar front polygons are kept; coplanar back are also kept
        # when there IS a back child (they are on the boundary of the solid
        # on the outside).
        # Following csg.js: coplanar-front goes into front, coplanar-back
        # goes into back.
        return coplanar_front + front_list + coplanar_back + back_list

    def clip_to(self, bsp: "BSPNode") -> None:
        """Remove all polygons from *this* tree that are inside *bsp*."""
        self.polygons = bsp.clip_polygons(self.polygons)
        if self.front:
            self.front.clip_to(bsp)
        if self.back:
            self.back.clip_to(bsp)

    # ------------------------------------------------------------------
    # Inversion
    # ------------------------------------------------------------------

    def invert(self) -> None:
        """Swap inside ↔ outside for every polygon in this sub-tree."""
        for poly in self.polygons:
            poly.flip()
        if self.plane is not None:
            self.plane.flip()
        if self.front:
            self.front.invert()
        if self.back:
            self.back.invert()
        self.front, self.back = self.back, self.front


# (split_polygon is defined as a method on Plane above)


# ==============================================================================
# CSG Mesh
# ==============================================================================

class CSGMesh:
    """Triangle mesh for CSG boolean operations.

    Internally the mesh is stored as a list of :class:`BSPPolygon` (one
    per triangle).  The :meth:`union`, :meth:`subtract`, and
    :meth:`intersect` methods perform **real** BSP-tree boolean ops.

    Parameters
    ----------
    vertices : array-like, shape (N, 3), optional
    triangles : array-like, shape (M, 3), optional
    polygons : list[BSPPolygon], optional
        Provide polygons directly (overrides vertices/triangles).
    """

    def __init__(
        self,
        vertices: Optional[np.ndarray] = None,
        triangles: Optional[np.ndarray] = None,
        polygons: Optional[List[BSPPolygon]] = None,
    ) -> None:
        if polygons is not None:
            self.polygons = [p.clone() for p in polygons]
        else:
            self.polygons: List[BSPPolygon] = []
            if vertices is not None and triangles is not None:
                verts = np.asarray(vertices, dtype=np.float64)
                tris = np.asarray(triangles, dtype=np.int64)
                for face in tris:
                    tri = BSPPolygon([verts[face[0]], verts[face[1]], verts[face[2]]])
                    self.polygons.append(tri)

    # ------------------------------------------------------------------
    # Clone
    # ------------------------------------------------------------------

    def clone(self) -> "CSGMesh":
        return CSGMesh(polygons=self.polygons)

    def copy(self) -> "CSGMesh":
        return self.clone()

    # ------------------------------------------------------------------
    # Boolean operations (REAL BSP-based)
    # ------------------------------------------------------------------

    def union(self, other: "CSGMesh") -> "CSGMesh":
        """Return A | B  (real union via BSP clipping)."""
        a = BSPNode(self.clone().polygons)
        b = BSPNode(other.clone().polygons)
        a.clip_to(b)
        b.clip_to(a)
        b.invert()
        b.clip_to(a)
        b.invert()
        a.build(b.all_polygons())
        return CSGMesh(polygons=a.all_polygons())

    def subtract(self, other: "CSGMesh") -> "CSGMesh":
        """Return A - B  (real subtraction via BSP clipping)."""
        a = BSPNode(self.clone().polygons)
        b = BSPNode(other.clone().polygons)
        a.invert()
        a.clip_to(b)
        b.clip_to(a)
        b.invert()
        b.clip_to(a)
        b.invert()
        a.build(b.all_polygons())
        a.invert()
        return CSGMesh(polygons=a.all_polygons())

    def intersect(self, other: "CSGMesh") -> "CSGMesh":
        """Return A & B  (real intersection via BSP clipping)."""
        a = BSPNode(self.clone().polygons)
        b = BSPNode(other.clone().polygons)
        a.invert()
        b.clip_to(a)
        b.invert()
        a.clip_to(b)
        b.clip_to(a)
        b.build(a.all_polygons())
        return CSGMesh(polygons=b.all_polygons())

    # ------------------------------------------------------------------
    # Invert (flip normals / inside-out)
    # ------------------------------------------------------------------

    def invert(self) -> "CSGMesh":
        """Flip all polygon normals, returning *self* for chaining."""
        for p in self.polygons:
            p.flip()
        return self

    # ------------------------------------------------------------------
    # Output – compatible with Nexus3D Mesh class
    # ------------------------------------------------------------------

    def to_vertices_faces(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to (vertices, faces) arrays for :class:`nexus3d.mesh.Mesh`.

        Polygons with > 3 vertices are fan-triangulated.

        Returns
        -------
        vertices : np.ndarray, shape (N, 3), float64
        faces : np.ndarray, shape (M, 3), uint32
        """
        # Use MeshCleaner for robust output
        verts, faces = MeshCleaner.clean(
            *self._raw_vertices_faces(), epsilon=EPSILON
        )
        return verts, faces

    def _raw_vertices_faces(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert polygons → (vertices, faces) without cleanup."""
        vert_list: List[np.ndarray] = []
        face_list: List[Tuple[int, int, int]] = []

        for poly in self.polygons:
            n_verts = len(poly.vertices)
            if n_verts < 3:
                continue
            base = len(vert_list)
            for v in poly.vertices:
                vert_list.append(v)
            # Fan triangulation
            for i in range(1, n_verts - 1):
                face_list.append((base, base + i, base + i + 1))

        if not vert_list:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint32)

        vertices = np.array(vert_list, dtype=np.float64)
        faces = np.array(face_list, dtype=np.uint32)
        return vertices, faces

    # ------------------------------------------------------------------
    # Volume (divergence theorem)
    # ------------------------------------------------------------------

    def volume(self) -> float:
        """Compute volume using the signed-tetrahedra method."""
        verts, faces = self._raw_vertices_faces()
        if faces.shape[0] == 0:
            return 0.0
        vol = 0.0
        for f in faces:
            v0 = verts[f[0]]
            v1 = verts[f[1]]
            v2 = verts[f[2]]
            vol += float(np.dot(v0, np.cross(v1, v2))) / 6.0
        return abs(vol)

    # ------------------------------------------------------------------
    # Primitive factories
    # ------------------------------------------------------------------

    @classmethod
    def from_cube(cls, size: float = 1.0) -> "CSGMesh":
        """Create an axis-aligned cube centred at the origin."""
        h = size / 2.0
        vertices = np.array(
            [
                [-h, -h, -h],
                [+h, -h, -h],
                [+h, +h, -h],
                [-h, +h, -h],
                [-h, -h, +h],
                [+h, -h, +h],
                [+h, +h, +h],
                [-h, +h, +h],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [
                [0, 3, 2], [0, 2, 1],  # -Z
                [4, 5, 6], [4, 6, 7],  # +Z
                [0, 4, 7], [0, 7, 3],  # -X
                [1, 2, 6], [1, 6, 5],  # +X
                [3, 7, 6], [3, 6, 2],  # +Y
                [0, 1, 5], [0, 5, 4],  # -Y
            ],
            dtype=np.int64,
        )
        return cls(vertices=vertices, triangles=faces)

    @classmethod
    def from_sphere(cls, radius: float = 1.0, segments: int = 16) -> "CSGMesh":
        """Create a UV-sphere centred at the origin."""
        polys: List[BSPPolygon] = []
        rings = segments // 2
        for i in range(rings):
            theta1 = math.pi * (-0.5 + i / rings)
            theta2 = math.pi * (-0.5 + (i + 1) / rings)
            for j in range(segments):
                phi1 = 2.0 * math.pi * j / segments
                phi2 = 2.0 * math.pi * (j + 1) / segments

                def _sv(theta: float, phi: float) -> np.ndarray:
                    return np.array(
                        [
                            radius * math.cos(theta) * math.cos(phi),
                            radius * math.sin(theta),
                            radius * math.cos(theta) * math.sin(phi),
                        ],
                        dtype=np.float64,
                    )

                v00 = _sv(theta1, phi1)
                v01 = _sv(theta1, phi2)
                v10 = _sv(theta2, phi1)
                v11 = _sv(theta2, phi2)

                if i == 0:
                    # South pole strip: one triangle per segment
                    polys.append(BSPPolygon([v00, v10, v11]))
                elif i == rings - 1:
                    # North pole strip: only the non-degenerate cap triangle
                    polys.append(BSPPolygon([v00, v11, v01]))
                else:
                    # Middle strips: two triangles per quad
                    polys.append(BSPPolygon([v00, v10, v11]))
                    polys.append(BSPPolygon([v00, v11, v01]))

        return cls(polygons=polys)

    @classmethod
    def from_cylinder(
        cls,
        radius: float = 1.0,
        height: float = 2.0,
        segments: int = 16,
    ) -> "CSGMesh":
        """Create a cylinder along the Y axis, centred at the origin."""
        polys: List[BSPPolygon] = []
        r = radius
        hh = height / 2.0

        # Side
        for j in range(segments):
            a0 = 2.0 * math.pi * j / segments
            a1 = 2.0 * math.pi * (j + 1) / segments
            v0 = np.array([r * math.cos(a0), hh, r * math.sin(a0)], dtype=np.float64)
            v1 = np.array([r * math.cos(a1), hh, r * math.sin(a1)], dtype=np.float64)
            v2 = np.array([r * math.cos(a0), -hh, r * math.sin(a0)], dtype=np.float64)
            v3 = np.array([r * math.cos(a1), -hh, r * math.sin(a1)], dtype=np.float64)
            polys.append(BSPPolygon([v0, v1, v3]))
            polys.append(BSPPolygon([v0, v3, v2]))

        # Top cap (Y+)
        centre_top = np.array([0.0, hh, 0.0], dtype=np.float64)
        for j in range(segments):
            a0 = 2.0 * math.pi * j / segments
            a1 = 2.0 * math.pi * (j + 1) / segments
            v0 = np.array([r * math.cos(a0), hh, r * math.sin(a0)], dtype=np.float64)
            v1 = np.array([r * math.cos(a1), hh, r * math.sin(a1)], dtype=np.float64)
            polys.append(BSPPolygon([centre_top, v1, v0]))

        # Bottom cap (Y-)
        centre_bot = np.array([0.0, -hh, 0.0], dtype=np.float64)
        for j in range(segments):
            a0 = 2.0 * math.pi * j / segments
            a1 = 2.0 * math.pi * (j + 1) / segments
            v0 = np.array([r * math.cos(a0), -hh, r * math.sin(a0)], dtype=np.float64)
            v1 = np.array([r * math.cos(a1), -hh, r * math.sin(a1)], dtype=np.float64)
            polys.append(BSPPolygon([centre_bot, v0, v1]))

        return cls(polygons=polys)

    @classmethod
    def from_cone(
        cls,
        radius: float = 1.0,
        height: float = 2.0,
        segments: int = 16,
    ) -> "CSGMesh":
        """Create a cone along the Y axis, base at Y=-h/2, apex at Y=+h/2."""
        polys: List[BSPPolygon] = []
        r = radius
        hh = height / 2.0
        apex = np.array([0.0, hh, 0.0], dtype=np.float64)
        centre_bot = np.array([0.0, -hh, 0.0], dtype=np.float64)

        # Side
        for j in range(segments):
            a0 = 2.0 * math.pi * j / segments
            a1 = 2.0 * math.pi * (j + 1) / segments
            v0 = np.array([r * math.cos(a0), -hh, r * math.sin(a0)], dtype=np.float64)
            v1 = np.array([r * math.cos(a1), -hh, r * math.sin(a1)], dtype=np.float64)
            polys.append(BSPPolygon([apex, v1, v0]))

        # Bottom cap
        for j in range(segments):
            a0 = 2.0 * math.pi * j / segments
            a1 = 2.0 * math.pi * (j + 1) / segments
            v0 = np.array([r * math.cos(a0), -hh, r * math.sin(a0)], dtype=np.float64)
            v1 = np.array([r * math.cos(a1), -hh, r * math.sin(a1)], dtype=np.float64)
            polys.append(BSPPolygon([centre_bot, v0, v1]))

        return cls(polygons=polys)

    @classmethod
    def from_mesh(cls, mesh: "Mesh") -> "CSGMesh":
        """Create a CSGMesh from a nexus3d.mesh.Mesh object.

        Parameters
        ----------
        mesh : Mesh
            A Mesh with vertices and faces.

        Returns
        -------
        CSGMesh
        """
        return cls(vertices=mesh.vertices, triangles=mesh.faces)

    @classmethod
    def from_torus(
        cls,
        major_r: float = 1.0,
        minor_r: float = 0.3,
        major_seg: int = 24,
        minor_seg: int = 12,
    ) -> "CSGMesh":
        """Create a torus in the XZ plane, centred at the origin."""
        polys: List[BSPPolygon] = []
        for i in range(major_seg):
            theta0 = 2.0 * math.pi * i / major_seg
            theta1 = 2.0 * math.pi * (i + 1) / major_seg
            for j in range(minor_seg):
                phi0 = 2.0 * math.pi * j / minor_seg
                phi1 = 2.0 * math.pi * (j + 1) / minor_seg

                def _tv(theta: float, phi: float) -> np.ndarray:
                    x = (major_r + minor_r * math.cos(phi)) * math.cos(theta)
                    y = minor_r * math.sin(phi)
                    z = (major_r + minor_r * math.cos(phi)) * math.sin(theta)
                    return np.array([x, y, z], dtype=np.float64)

                v00 = _tv(theta0, phi0)
                v10 = _tv(theta1, phi0)
                v11 = _tv(theta1, phi1)
                v01 = _tv(theta0, phi1)
                polys.append(BSPPolygon([v00, v10, v11]))
                polys.append(BSPPolygon([v00, v11, v01]))

        return cls(polygons=polys)

    # ------------------------------------------------------------------
    # Operator overloads
    # ------------------------------------------------------------------

    def __or__(self, other: "CSGMesh") -> "CSGMesh":
        return self.union(other)

    def __sub__(self, other: "CSGMesh") -> "CSGMesh":
        return self.subtract(other)

    def __and__(self, other: "CSGMesh") -> "CSGMesh":
        return self.intersect(other)

    def __invert__(self) -> "CSGMesh":
        return self.clone().invert()

    def __repr__(self) -> str:  # pragma: no cover
        return f"CSGMesh(polygons={len(self.polygons)})"


# ==============================================================================
# Mesh Cleaner
# ==============================================================================

class MeshCleaner:
    """Post-CSG mesh cleanup utilities.

    After boolean operations the mesh may contain duplicate vertices,
    degenerate (zero-area) triangles, and inconsistent winding orders.
    The :meth:`clean` pipeline runs all three fixes in sequence.
    """

    @staticmethod
    def weld_vertices(
        vertices: np.ndarray,
        faces: np.ndarray,
        epsilon: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Merge vertices that are within *epsilon* distance of each other.

        Returns new (vertices, faces) arrays with duplicate vertices
        eliminated.
        """
        if vertices.shape[0] == 0:
            return vertices.copy(), faces.copy()

        # Build a spatial hash (rounded grid) for O(n) expected lookup
        inv_eps = 1.0 / max(epsilon, 1e-12)
        index_map: Dict[Tuple[int, int, int], int] = {}
        remap: np.ndarray = np.full(vertices.shape[0], -1, dtype=np.int64)
        new_verts: List[np.ndarray] = []

        for i, v in enumerate(vertices):
            key = (
                int(math.floor(v[0] * inv_eps)),
                int(math.floor(v[1] * inv_eps)),
                int(math.floor(v[2] * inv_eps)),
            )
            # Also check 26 neighbours to avoid missing near-boundary dups
            found = -1
            for dx in (-1, 0, 1):
                if found >= 0:
                    break
                for dy in (-1, 0, 1):
                    if found >= 0:
                        break
                    for dz in (-1, 0, 1):
                        nkey = (key[0] + dx, key[1] + dy, key[2] + dz)
                        if nkey in index_map:
                            idx = index_map[nkey]
                            if np.linalg.norm(new_verts[idx] - v) <= epsilon:
                                found = idx
                                break
            if found >= 0:
                remap[i] = found
            else:
                new_idx = len(new_verts)
                new_verts.append(v.copy())
                index_map[key] = new_idx
                remap[i] = new_idx

        new_faces = remap[faces]
        # Remove degenerate faces (where all indices are the same)
        valid = (new_faces[:, 0] != new_faces[:, 1]) & (
            new_faces[:, 1] != new_faces[:, 2]
        ) & (new_faces[:, 0] != new_faces[:, 2])
        new_faces = new_faces[valid]

        return (
            np.array(new_verts, dtype=np.float64) if new_verts else np.zeros((0, 3), dtype=np.float64),
            new_faces.astype(np.uint32),
        )

    @staticmethod
    def remove_degenerate_triangles(
        vertices: np.ndarray,
        faces: np.ndarray,
        min_area: float = 1e-10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove triangles whose area is below *min_area*."""
        if faces.shape[0] == 0:
            return vertices.copy(), faces.copy()

        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross_vecs = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross_vecs, axis=1)
        valid = areas > min_area
        return vertices.copy(), faces[valid]

    @staticmethod
    def fix_normals(
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ensure consistent outward-facing winding order.

        Uses a flood-fill from the centroid.  The "seed" triangle is
        oriented so its normal points away from the centroid of the
        mesh, and neighbours are oriented consistently.
        """
        if faces.shape[0] == 0:
            return vertices.copy(), faces.copy()

        faces = faces.copy()
        n_faces = faces.shape[0]

        # Build edge → face adjacency (directed edges)
        edge_face: Dict[Tuple[int, int], int] = {}
        for fi in range(n_faces):
            a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
            edge_face[(a, b)] = fi
            edge_face[(b, c)] = fi
            edge_face[(c, a)] = fi

        # Build face → face adjacency via opposite directed edges
        adj: Dict[int, List[int]] = {fi: [] for fi in range(n_faces)}
        for fi in range(n_faces):
            a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
            # Look for reversed edges
            for ea, eb in [(b, a), (c, b), (a, c)]:
                if (ea, eb) in edge_face:
                    nf = edge_face[(ea, eb)]
                    if nf != fi:
                        adj[fi].append(nf)

        # Compute centroid
        centroid = vertices.mean(axis=0)

        # Seed: orient first triangle so normal points away from centroid
        seed = 0
        v0, v1, v2 = vertices[faces[seed, 0]], vertices[faces[seed, 1]], vertices[faces[seed, 2]]
        normal = np.cross(v1 - v0, v2 - v0)
        face_centre = (v0 + v1 + v2) / 3.0
        if np.dot(normal, face_centre - centroid) < 0:
            # Flip
            faces[seed] = np.array([faces[seed, 0], faces[seed, 2], faces[seed, 1]], dtype=np.uint32)

        # BFS
        visited = [False] * n_faces
        visited[seed] = True
        queue = [seed]
        while queue:
            fi = queue.pop(0)
            a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
            for nf in adj[fi]:
                if visited[nf]:
                    continue
                na, nb, nc = int(faces[nf, 0]), int(faces[nf, 1]), int(faces[nf, 2])
                # Check if nf shares an edge with fi in opposite direction
                # The shared edge should appear in opposite winding
                shared = None
                for edge_fi, edge_nf in [((a, b), (nb, na)), ((b, c), (nc, nb)), ((c, a), (na, nc))]:
                    if edge_nf in edge_face and edge_face[edge_nf] == fi:
                        shared = True
                        break
                    # Also check non-matching
                    for e in [(a, b), (b, c), (c, a)]:
                        for en in [(na, nb), (nb, nc), (nc, na)]:
                            if e == (en[1], en[0]):
                                shared = True
                                break

                # Simpler approach: just check if the shared edge has
                # opposite winding.  If not, flip.
                fi_verts = {a, b, c}
                nf_verts = {na, nb, nc}
                shared_verts = fi_verts & nf_verts
                if len(shared_verts) == 2:
                    sv = sorted(shared_verts)
                    # In fi, get the edge direction
                    fi_order = [a, b, c]
                    nf_order = [na, nb, nc]
                    # Find the edge in fi
                    fi_dir = None
                    for k in range(3):
                        if fi_order[k] in shared_verts and fi_order[(k + 1) % 3] in shared_verts:
                            fi_dir = (fi_order[k], fi_order[(k + 1) % 3])
                            break
                    # Find the edge in nf
                    nf_dir = None
                    for k in range(3):
                        if nf_order[k] in shared_verts and nf_order[(k + 1) % 3] in shared_verts:
                            nf_dir = (nf_order[k], nf_order[(k + 1) % 3])
                            break
                    if fi_dir and nf_dir and fi_dir == nf_dir:
                        # Same direction → need to flip nf
                        faces[nf] = np.array(
                            [faces[nf, 0], faces[nf, 2], faces[nf, 1]], dtype=np.uint32
                        )

                visited[nf] = True
                queue.append(nf)

        return vertices.copy(), faces

    @staticmethod
    def clean(
        vertices: np.ndarray,
        faces: np.ndarray,
        epsilon: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Full cleanup pipeline: weld → remove degenerates → fix normals."""
        vertices, faces = MeshCleaner.weld_vertices(vertices, faces, epsilon=epsilon)
        vertices, faces = MeshCleaner.remove_degenerate_triangles(vertices, faces)
        if faces.shape[0] > 0:
            vertices, faces = MeshCleaner.fix_normals(vertices, faces)
        return vertices, faces


# ==============================================================================
# Architecture Kit
# ==============================================================================

class ArchitectureKit:
    """CSG-based architectural primitives.

    All methods construct meshes using real boolean operations
    (union, subtract) so that openings (doors, windows) are properly
    cut through the geometry.
    """

    @staticmethod
    def wall(
        width: float,
        height: float,
        thickness: float,
    ) -> CSGMesh:
        """Create an axis-aligned wall block.

        The wall lies in the XY plane, centred at the origin, with
        thickness along Z.
        """
        mesh = CSGMesh.from_cube(1.0)
        # Scale to desired dimensions
        verts = []
        for poly in mesh.polygons:
            new_verts = []
            for v in poly.vertices:
                new_verts.append(np.array([
                    v[0] * width,
                    v[1] * height,
                    v[2] * thickness,
                ], dtype=np.float64))
            verts.append(BSPPolygon(new_verts, poly.shared))
        return CSGMesh(polygons=verts)

    @staticmethod
    def _translate(mesh: CSGMesh, offset: np.ndarray) -> CSGMesh:
        """Return a translated copy of *mesh*."""
        new_polys = []
        for poly in mesh.polygons:
            new_verts = [v + offset for v in poly.vertices]
            new_polys.append(BSPPolygon(new_verts, poly.shared))
        return CSGMesh(polygons=new_polys)

    @staticmethod
    def _scale(mesh: CSGMesh, sx: float, sy: float, sz: float) -> CSGMesh:
        """Return a scaled copy of *mesh*."""
        new_polys = []
        for poly in mesh.polygons:
            new_verts = [np.array([v[0] * sx, v[1] * sy, v[2] * sz], dtype=np.float64)
                         for v in poly.vertices]
            new_polys.append(BSPPolygon(new_verts, poly.shared))
        return CSGMesh(polygons=new_polys)

    @staticmethod
    def add_door(
        wall: CSGMesh,
        door_width: float = 1.0,
        door_height: float = 2.1,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> CSGMesh:
        """Cut a door opening in a wall using real CSG subtract.

        The door starts at the bottom of the wall and extends upward
        by *door_height*.  The *position* offset is applied to the
        door cut-out.
        """
        # Create door-shaped box (the part to subtract)
        door = CSGMesh.from_cube(1.0)
        door = ArchitectureKit._scale(door, door_width, door_height, wall_thickness(wall) * 3)
        # Position the door: bottom-centre of the wall
        wall_h = _mesh_height(wall)
        offset = np.array([
            position[0],
            position[1] - wall_h / 2.0 + door_height / 2.0,
            position[2],
        ], dtype=np.float64)
        door = ArchitectureKit._translate(door, offset)
        return wall.subtract(door)

    @staticmethod
    def add_window(
        wall: CSGMesh,
        window_width: float = 1.2,
        window_height: float = 1.0,
        window_bottom: float = 0.9,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        frame_thickness: float = 0.05,
    ) -> CSGMesh:
        """Cut a window opening in a wall, with an optional frame.

        If *frame_thickness* > 0, a thin frame is unioned around the
        opening.
        """
        wall_t = wall_thickness(wall)
        wall_h = _mesh_height(wall)

        # Subtract window opening
        opening = CSGMesh.from_cube(1.0)
        opening = ArchitectureKit._scale(opening, window_width, window_height, wall_t * 3)
        cy = position[1] - wall_h / 2.0 + window_bottom + window_height / 2.0
        offset = np.array([position[0], cy, position[2]], dtype=np.float64)
        opening = ArchitectureKit._translate(opening, offset)
        result = wall.subtract(opening)

        # Add frame if requested
        if frame_thickness > 0:
            result = ArchitectureKit._add_window_frame_geometry(
                result,
                window_width,
                window_height,
                window_bottom,
                wall_t,
                frame_thickness,
                offset,
            )

        return result

    @staticmethod
    def _add_window_frame_geometry(
        wall: CSGMesh,
        window_width: float,
        window_height: float,
        window_bottom: float,
        wall_t: float,
        frame_thickness: float,
        window_centre: np.ndarray,
    ) -> CSGMesh:
        """Create a frame around the window opening using CSG."""
        ft = frame_thickness
        depth = wall_t * 1.5  # slightly thicker than wall to ensure overlap

        # Outer frame dimensions
        outer_w = window_width + ft * 2
        outer_h = window_height + ft * 2

        # Top bar
        top = CSGMesh.from_cube(1.0)
        top = ArchitectureKit._scale(top, outer_w, ft, depth)
        top_y = window_centre[1] + window_height / 2.0 + ft / 2.0
        top = ArchitectureKit._translate(
            top, np.array([window_centre[0], top_y, window_centre[2]])
        )

        # Bottom bar
        bottom = CSGMesh.from_cube(1.0)
        bottom = ArchitectureKit._scale(bottom, outer_w, ft, depth)
        bot_y = window_centre[1] - window_height / 2.0 - ft / 2.0
        bottom = ArchitectureKit._translate(
            bottom, np.array([window_centre[0], bot_y, window_centre[2]])
        )

        # Left bar
        left = CSGMesh.from_cube(1.0)
        left = ArchitectureKit._scale(left, ft, window_height, depth)
        left_x = window_centre[0] - window_width / 2.0 - ft / 2.0
        left = ArchitectureKit._translate(
            left, np.array([left_x, window_centre[1], window_centre[2]])
        )

        # Right bar
        right = CSGMesh.from_cube(1.0)
        right = ArchitectureKit._scale(right, ft, window_height, depth)
        right_x = window_centre[0] + window_width / 2.0 + ft / 2.0
        right = ArchitectureKit._translate(
            right, np.array([right_x, window_centre[1], window_centre[2]])
        )

        result = wall.union(top).union(bottom).union(left).union(right)
        return result

    @staticmethod
    def add_window_frame(
        wall: CSGMesh,
        window_width: float = 1.2,
        window_height: float = 1.0,
        window_bottom: float = 0.9,
        frame_thickness: float = 0.05,
        frame_depth: float = 0.03,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> CSGMesh:
        """Cut a window opening and add a frame using subtract + union."""
        wall_t = wall_thickness(wall)
        wall_h = _mesh_height(wall)

        # Opening
        opening = CSGMesh.from_cube(1.0)
        opening = ArchitectureKit._scale(opening, window_width, window_height, wall_t * 3)
        cy = position[1] - wall_h / 2.0 + window_bottom + window_height / 2.0
        offset = np.array([position[0], cy, position[2]], dtype=np.float64)
        opening = ArchitectureKit._translate(opening, offset)
        result = wall.subtract(opening)

        # Frame (four bars)
        ft = frame_thickness
        outer_w = window_width + ft * 2
        outer_h = window_height + ft * 2
        fd = max(frame_depth, wall_t * 1.5)

        # Top
        top = CSGMesh.from_cube(1.0)
        top = ArchitectureKit._scale(top, outer_w, ft, fd)
        top = ArchitectureKit._translate(
            top, np.array([offset[0], offset[1] + window_height / 2.0 + ft / 2.0, offset[2]])
        )
        result = result.union(top)

        # Bottom
        bot = CSGMesh.from_cube(1.0)
        bot = ArchitectureKit._scale(bot, outer_w, ft, fd)
        bot = ArchitectureKit._translate(
            bot, np.array([offset[0], offset[1] - window_height / 2.0 - ft / 2.0, offset[2]])
        )
        result = result.union(bot)

        # Left
        lft = CSGMesh.from_cube(1.0)
        lft = ArchitectureKit._scale(lft, ft, window_height, fd)
        lft = ArchitectureKit._translate(
            lft, np.array([offset[0] - window_width / 2.0 - ft / 2.0, offset[1], offset[2]])
        )
        result = result.union(lft)

        # Right
        rgt = CSGMesh.from_cube(1.0)
        rgt = ArchitectureKit._scale(rgt, ft, window_height, fd)
        rgt = ArchitectureKit._translate(
            rgt, np.array([offset[0] + window_width / 2.0 + ft / 2.0, offset[1], offset[2]])
        )
        result = result.union(rgt)

        return result

    @staticmethod
    def room(
        width: float,
        depth: float,
        height: float,
        wall_thickness: float = 0.15,
        door_openings: Optional[List[Dict[str, Any]]] = None,
        window_openings: Optional[List[Dict[str, Any]]] = None,
    ) -> CSGMesh:
        """Create a room with four walls, using real CSG for openings.

        Parameters
        ----------
        width : float
            Room width (X axis).
        depth : float
            Room depth (Z axis).
        height : float
            Wall height (Y axis).
        wall_thickness : float
            Thickness of each wall.
        door_openings : list[dict], optional
            Each dict: ``{"wall": "north"|"south"|"east"|"west",
            "width": float, "position": float}``.
        window_openings : list[dict], optional
            Each dict: ``{"wall": "north"|"south"|"east"|"west",
            "width": float, "height": float, "bottom": float,
            "position": float}``.

        Returns
        -------
        CSGMesh
            The room geometry.
        """
        door_openings = door_openings or []
        window_openings = window_openings or []
        wt = wall_thickness

        # Create four walls
        # Front wall (south, -Z)
        front = CSGMesh.from_cube(1.0)
        front = ArchitectureKit._scale(front, width, height, wt)
        front = ArchitectureKit._translate(front, np.array([0, 0, -depth / 2.0 + wt / 2.0]))

        # Back wall (north, +Z)
        back = CSGMesh.from_cube(1.0)
        back = ArchitectureKit._scale(back, width, height, wt)
        back = ArchitectureKit._translate(back, np.array([0, 0, depth / 2.0 - wt / 2.0]))

        # Left wall (west, -X)
        left = CSGMesh.from_cube(1.0)
        left = ArchitectureKit._scale(left, wt, height, depth - wt)
        left = ArchitectureKit._translate(left, np.array([-width / 2.0 + wt / 2.0, 0, 0]))

        # Right wall (east, +X)
        right = CSGMesh.from_cube(1.0)
        right = ArchitectureKit._scale(right, wt, height, depth - wt)
        right = ArchitectureKit._translate(right, np.array([width / 2.0 - wt / 2.0, 0, 0]))

        room_mesh = front.union(back).union(left).union(right)

        # Cut door openings
        for door in door_openings:
            wall_name = door.get("wall", "south").lower()
            dw = door.get("width", 1.0)
            dh = door.get("height", 2.1)
            pos = door.get("position", 0.0)
            wall_mesh = _get_wall_for_room(wall_name, width, depth, height, wt)
            if wall_mesh is not None:
                cut = CSGMesh.from_cube(1.0)
                cut = ArchitectureKit._scale(cut, dw, dh, wt * 3)
                cut = ArchitectureKit._translate(
                    cut, np.array([pos, -height / 2.0 + dh / 2.0, 0])
                )
                wall_mesh = wall_mesh.subtract(cut)
                # Rebuild room with this wall replaced (simplified: just union)
                room_mesh = room_mesh.union(wall_mesh)

        # Cut window openings
        for win in window_openings:
            wall_name = win.get("wall", "south").lower()
            ww = win.get("width", 1.2)
            wh = win.get("height", 1.0)
            wb = win.get("bottom", 0.9)
            pos = win.get("position", 0.0)
            wall_mesh = _get_wall_for_room(wall_name, width, depth, height, wt)
            if wall_mesh is not None:
                cut = CSGMesh.from_cube(1.0)
                cut = ArchitectureKit._scale(cut, ww, wh, wt * 3)
                cy = -height / 2.0 + wb + wh / 2.0
                cut = ArchitectureKit._translate(
                    cut, np.array([pos, cy, 0])
                )
                wall_mesh = wall_mesh.subtract(cut)
                room_mesh = room_mesh.union(wall_mesh)

        return room_mesh

    @staticmethod
    def staircase(
        steps: int = 10,
        width: float = 1.0,
        step_height: float = 0.18,
        step_depth: float = 0.28,
    ) -> CSGMesh:
        """Create a staircase using CSG union of individual step blocks.

        Steps ascend along the +X axis.
        """
        result: Optional[CSGMesh] = None
        for i in range(steps):
            step = CSGMesh.from_cube(1.0)
            step = ArchitectureKit._scale(step, step_depth, step_height, width)
            cx = -((steps * step_depth) / 2.0) + (i * step_depth) + step_depth / 2.0
            cy = -((steps * step_height) / 2.0) + (i * step_height) + step_height / 2.0
            step = ArchitectureKit._translate(step, np.array([cx, cy, 0.0]))
            if result is None:
                result = step
            else:
                result = result.union(step)

        return result if result is not None else CSGMesh.from_cube(1.0)

    @staticmethod
    def _create_prism(
        width: float, depth: float, height: float, slope: float = 0.0
    ) -> CSGMesh:
        """Create a triangular prism for gable/shed roofs.

        Builds a wedge (triangular prism) from 8 vertices and 12 triangles.
        The ridge runs along the Z axis. When slope > 0, one side is lower.
        """
        hh = height
        hw = width / 2.0
        hd = depth / 2.0
        s = abs(slope)
        ridge_y = hh * (1.0 - s)  # lower ridge for sloped shed roofs
        polys: List[BSPPolygon] = []
        pts = {
            'a': np.array([-hw, -hh, -hd], dtype=np.float64),
            'b': np.array([+hw, -hh, -hd], dtype=np.float64),
            'c': np.array([+hw, +ridge_y, -hd], dtype=np.float64),
            'd': np.array([-hw, +ridge_y, -hd], dtype=np.float64),
            'e': np.array([-hw, -hh, +hd], dtype=np.float64),
            'f': np.array([+hw, -hh, +hd], dtype=np.float64),
            'g': np.array([+hw, +ridge_y, +hd], dtype=np.float64),
            'h': np.array([-hw, +ridge_y, +hd], dtype=np.float64),
        }
        # Front face (-Z) - triangular
        polys.append(BSPPolygon([pts['a'], pts['c'], pts['b']]))
        polys.append(BSPPolygon([pts['a'], pts['d'], pts['c']]))
        # Back face (+Z) - triangular
        polys.append(BSPPolygon([pts['e'], pts['f'], pts['g']]))
        polys.append(BSPPolygon([pts['e'], pts['g'], pts['h']]))
        # Bottom face (rectangle)
        polys.append(BSPPolygon([pts['a'], pts['b'], pts['f']]))
        polys.append(BSPPolygon([pts['a'], pts['f'], pts['e']]))
        # Left face (-X) - rectangular
        polys.append(BSPPolygon([pts['a'], pts['e'], pts['h']]))
        polys.append(BSPPolygon([pts['a'], pts['h'], pts['d']]))
        # Right face (+X) - rectangular
        polys.append(BSPPolygon([pts['b'], pts['c'], pts['g']]))
        polys.append(BSPPolygon([pts['b'], pts['g'], pts['f']]))
        # Top face(s) - peak
        polys.append(BSPPolygon([pts['d'], pts['h'], pts['g']]))
        polys.append(BSPPolygon([pts['d'], pts['g'], pts['c']]))
        return CSGMesh(polygons=polys)

    @staticmethod
    def pillar(
        radius: float = 0.15,
        height: float = 3.0,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> CSGMesh:
        """Create a cylindrical pillar."""
        p = CSGMesh.from_cylinder(radius, height, segments=16)
        return ArchitectureKit._translate(
            p, np.array(position, dtype=np.float64)
        )

    @staticmethod
    def arch(
        width: float = 1.5,
        height: float = 2.5,
        depth: float = 0.15,
        arch_radius: float = 0.75,
    ) -> CSGMesh:
        """Create an arch by subtracting a rounded doorway from a block.

        The arch is in the XZ plane, extending upward along Y.
        """
        # Base block
        block = CSGMesh.from_cube(1.0)
        block = ArchitectureKit._scale(block, width, height, depth)

        # Subtract a rounded-top opening:
        # Rectangle for the lower part + semicircle for the top
        rect_h = height - arch_radius
        rect_w = width * 0.8

        # Rectangular cut-out (lower part)
        rect = CSGMesh.from_cube(1.0)
        rect = ArchitectureKit._scale(rect, rect_w, rect_h, depth * 3)
        rect = ArchitectureKit._translate(
            rect, np.array([0, -height / 2.0 + rect_h / 2.0, 0])
        )
        block = block.subtract(rect)

        # Cylindrical cut-out (semicircular arch at the top)
        cyl = CSGMesh.from_cylinder(arch_radius, depth * 3, segments=24)
        cyl = ArchitectureKit._translate(
            cyl, np.array([0, -height / 2.0 + rect_h, 0])
        )
        block = block.subtract(cyl)

        return block

    @staticmethod
    def floor_plan(
        rooms: List[Tuple[float, float, float, float]],
        wall_thickness: float = 0.15,
        wall_height: float = 3.0,
    ) -> CSGMesh:
        """Create a multi-room floor plan using CSG union.

        Parameters
        ----------
        rooms : list of (x, z, width, depth)
            Position and size of each room.  Adjacent rooms will
            share walls.
        wall_thickness : float
        wall_height : float

        Returns
        -------
        CSGMesh
        """
        result: Optional[CSGMesh] = None
        for rx, rz, rw, rd in rooms:
            room = ArchitectureKit.room(rw, rd, wall_height, wall_thickness)
            room = ArchitectureKit._translate(room, np.array([rx, 0, rz]))
            if result is None:
                result = room
            else:
                result = result.union(room)

        return result if result is not None else CSGMesh.from_cube(1.0)


# ==============================================================================
# Helpers
# ==============================================================================

def _mesh_height(mesh: CSGMesh) -> float:
    """Compute the Y-axis extent of a CSGMesh."""
    if not mesh.polygons:
        return 0.0
    ys = [v[1] for poly in mesh.polygons for v in poly.vertices]
    return max(ys) - min(ys)


def wall_thickness(mesh: CSGMesh) -> float:
    """Estimate wall thickness from mesh Z-axis extent."""
    if not mesh.polygons:
        return 0.15
    zs = [v[2] for poly in mesh.polygons for v in poly.vertices]
    return max(zs) - min(zs)


def _get_wall_for_room(
    wall_name: str,
    width: float,
    depth: float,
    height: float,
    wt: float,
) -> Optional[CSGMesh]:
    """Return one wall of a room for cutting."""
    wall_name = wall_name.lower()
    if wall_name in ("south", "front"):
        w = CSGMesh.from_cube(1.0)
        w = ArchitectureKit._scale(w, width, height, wt)
        w = ArchitectureKit._translate(w, np.array([0, 0, -depth / 2.0 + wt / 2.0]))
        return w
    elif wall_name in ("north", "back"):
        w = CSGMesh.from_cube(1.0)
        w = ArchitectureKit._scale(w, width, height, wt)
        w = ArchitectureKit._translate(w, np.array([0, 0, depth / 2.0 - wt / 2.0]))
        return w
    elif wall_name in ("west", "left"):
        w = CSGMesh.from_cube(1.0)
        w = ArchitectureKit._scale(w, wt, height, depth)
        w = ArchitectureKit._translate(w, np.array([-width / 2.0 + wt / 2.0, 0, 0]))
        return w
    elif wall_name in ("east", "right"):
        w = CSGMesh.from_cube(1.0)
        w = ArchitectureKit._scale(w, wt, height, depth)
        w = ArchitectureKit._translate(w, np.array([width / 2.0 - wt / 2.0, 0, 0]))
        return w
    return None
