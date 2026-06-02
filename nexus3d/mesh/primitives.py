"""Nexus3D Mesh Primitives — procedural geometry generators.

Each generator returns a dict with keys:
    vertices : np.ndarray, shape (N, 3)  – vertex positions
    faces    : np.ndarray, shape (M, 3)  – triangle indices (uint32)
    normals  : np.ndarray, shape (N, 3)  – per-vertex normals
    uvs      : np.ndarray, shape (N, 2)  – per-vertex texture coordinates

All geometry is built around the origin, centred when possible,
and uses counter-clockwise winding order for front faces.
"""

from __future__ import annotations

import math
from typing import Dict, Any

import numpy as np

from nexus3d.math3d.core import vec3, normalize, cross


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _push_quad(
    vertices: list[np.ndarray],
    normals: list[np.ndarray],
    uvs: list[np.ndarray],
    faces: list[tuple[int, int, int]],
    v00: np.ndarray, v10: np.ndarray, v11: np.ndarray, v01: np.ndarray,
    n: np.ndarray,
    uv00: tuple[float, float],
    uv10: tuple[float, float],
    uv11: tuple[float, float],
    uv01: tuple[float, float],
) -> None:
    """Append four vertices and two CCW triangles forming a quad."""
    idx = len(vertices)
    vertices.extend([v00, v10, v11, v01])
    normals.extend([n, n, n, n])
    uvs.extend([uv00, uv10, uv11, uv01])
    faces.append((idx, idx + 1, idx + 2))
    faces.append((idx, idx + 2, idx + 3))


# ---------------------------------------------------------------------------
# 1. Cube
# ---------------------------------------------------------------------------

def create_cube(size: float = 1.0) -> Dict[str, Any]:
    """Create a unit cube centred at the origin.

    Generates 24 vertices (4 per face) so that each face has independent
    normals for correct flat shading.

    Parameters
    ----------
    size : float
        Full side-length of the cube (default 1.0).

    Returns
    -------
    dict with keys ``vertices``, ``faces``, ``normals``, ``uvs``.
    """
    h = size / 2.0
    vertices: list[np.ndarray] = []
    normals:  list[np.ndarray] = []
    uvs:      list[np.ndarray] = []
    faces:    list[tuple[int, int, int]] = []

    # Six faces – each defined as four corners.
    # Convention: v00 = bottom-left, v10 = bottom-right,
    #             v11 = top-right,    v01 = top-left  (in face-local space).
    face_defs = [
        # (+X face)
        (vec3(h, -h, h),  vec3(h, -h, -h), vec3(h, h, -h),  vec3(h, h, h),
         vec3(1, 0, 0)),
        # (-X face)
        (vec3(-h, -h, -h), vec3(-h, -h, h),  vec3(-h, h, h),  vec3(-h, h, -h),
         vec3(-1, 0, 0)),
        # (+Y face)
        (vec3(-h, h, h),  vec3(h, h, h),   vec3(h, h, -h),  vec3(-h, h, -h),
         vec3(0, 1, 0)),
        # (-Y face)
        (vec3(-h, -h, -h), vec3(h, -h, -h),  vec3(h, -h, h),  vec3(-h, -h, h),
         vec3(0, -1, 0)),
        # (+Z face)
        (vec3(-h, -h, h),  vec3(h, -h, h),   vec3(h, h, h),   vec3(-h, h, h),
         vec3(0, 0, 1)),
        # (-Z face)
        (vec3(h, -h, -h),  vec3(-h, -h, -h), vec3(-h, h, -h), vec3(h, h, -h),
         vec3(0, 0, -1)),
    ]

    for v00, v10, v11, v01, n in face_defs:
        _push_quad(
            vertices, normals, uvs, faces,
            v00, v10, v11, v01, n,
            (0, 0), (1, 0), (1, 1), (0, 1),
        )

    return {
        "vertices": np.array(vertices, dtype=np.float64),
        "faces":    np.array(faces, dtype=np.uint32),
        "normals":  np.array(normals, dtype=np.float64),
        "uvs":      np.array(uvs, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# 2. UV Sphere
# ---------------------------------------------------------------------------

def create_sphere(radius: float = 1.0, rings: int = 32, segments: int = 32) -> Dict[str, Any]:
    """Create a UV-sphere centred at the origin.

    Parameters
    ----------
    radius : float
        Sphere radius (default 1.0).
    rings : int
        Number of horizontal rings (latitude bands).  Must be >= 2.
    segments : int
        Number of longitudinal segments.  Must be >= 3.

    Returns
    -------
    dict with keys ``vertices``, ``faces``, ``normals``, ``uvs``.
    """
    rings = max(rings, 2)
    segments = max(segments, 3)

    vertices: list[np.ndarray] = []
    normals:  list[np.ndarray] = []
    uvs:      list[np.ndarray] = []
    faces:    list[tuple[int, int, int]] = []

    # Top pole
    vertices.append(vec3(0, radius, 0))
    normals.append(vec3(0, 1, 0))
    uvs.append(np.array([0.5, 1.0]))

    # Rings (excluding poles)
    for i in range(1, rings):
        phi = math.pi * i / rings  # polar angle from top
        for j in range(segments):
            theta = 2.0 * math.pi * j / segments
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.cos(phi)
            z = radius * math.sin(phi) * math.sin(theta)
            n = normalize(vec3(x, y, z))
            vertices.append(vec3(x, y, z))
            normals.append(n)
            u = j / segments
            v = 1.0 - i / rings
            uvs.append(np.array([u, v]))

    # Bottom pole
    vertices.append(vec3(0, -radius, 0))
    normals.append(vec3(0, -1, 0))
    uvs.append(np.array([0.5, 0.0]))

    # Top-cap triangles
    for j in range(segments):
        next_j = (j + 1) % segments
        faces.append((0, 1 + j, 1 + next_j))

    # Middle quads (two triangles each)
    for i in range(rings - 2):
        for j in range(segments):
            curr = 1 + i * segments + j
            next_v = 1 + i * segments + (j + 1) % segments
            above = 1 + (i + 1) * segments + j
            above_next = 1 + (i + 1) * segments + (j + 1) % segments
            faces.append((curr, above, above_next))
            faces.append((curr, above_next, next_v))

    # Bottom-cap triangles
    bottom_idx = len(vertices) - 1
    base = 1 + (rings - 2) * segments
    for j in range(segments):
        next_j = (j + 1) % segments
        faces.append((bottom_idx, base + next_j, base + j))

    return {
        "vertices": np.array(vertices, dtype=np.float64),
        "faces":    np.array(faces, dtype=np.uint32),
        "normals":  np.array(normals, dtype=np.float64),
        "uvs":      np.array(uvs, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# 3. Cylinder
# ---------------------------------------------------------------------------

def create_cylinder(radius: float = 1.0, height: float = 2.0,
                    segments: int = 32) -> Dict[str, Any]:
    """Create a cylinder aligned along the Y-axis, centred at the origin.

    Parameters
    ----------
    radius : float
        Radius of the cylinder (default 1.0).
    height : float
        Full height of the cylinder (default 2.0).
    segments : int
        Number of radial segments.  Must be >= 3.

    Returns
    -------
    dict with keys ``vertices``, ``faces``, ``normals``, ``uvs``.
    """
    segments = max(segments, 3)
    half_h = height / 2.0

    vertices: list[np.ndarray] = []
    normals:  list[np.ndarray] = []
    uvs:      list[np.ndarray] = []
    faces:    list[tuple[int, int, int]] = []

    # Generate side vertices (top and bottom rings)
    top_indices: list[int] = []
    bottom_indices: list[int] = []
    for j in range(segments):
        theta = 2.0 * math.pi * j / segments
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        side_normal = vec3(cos_t, 0, sin_t)

        # Top ring
        top_indices.append(len(vertices))
        vertices.append(vec3(radius * cos_t, half_h, radius * sin_t))
        normals.append(side_normal)
        uvs.append(np.array([j / segments, 1.0]))

        # Bottom ring
        bottom_indices.append(len(vertices))
        vertices.append(vec3(radius * cos_t, -half_h, radius * sin_t))
        normals.append(side_normal)
        uvs.append(np.array([j / segments, 0.0]))

    # Side faces
    for j in range(segments):
        next_j = (j + 1) % segments
        # CCW winding
        faces.append((top_indices[j], top_indices[next_j], bottom_indices[next_j]))
        faces.append((top_indices[j], bottom_indices[next_j], bottom_indices[j]))

    # Top cap
    top_centre = len(vertices)
    vertices.append(vec3(0, half_h, 0))
    normals.append(vec3(0, 1, 0))
    uvs.append(np.array([0.5, 0.5]))
    for j in range(segments):
        next_j = (j + 1) % segments
        faces.append((top_centre, top_indices[next_j], top_indices[j]))

    # Bottom cap
    bottom_centre = len(vertices)
    vertices.append(vec3(0, -half_h, 0))
    normals.append(vec3(0, -1, 0))
    uvs.append(np.array([0.5, 0.5]))
    for j in range(segments):
        next_j = (j + 1) % segments
        faces.append((bottom_centre, bottom_indices[j], bottom_indices[next_j]))

    return {
        "vertices": np.array(vertices, dtype=np.float64),
        "faces":    np.array(faces, dtype=np.uint32),
        "normals":  np.array(normals, dtype=np.float64),
        "uvs":      np.array(uvs, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# 4. Cone
# ---------------------------------------------------------------------------

def create_cone(radius: float = 1.0, height: float = 2.0,
                segments: int = 32) -> Dict[str, Any]:
    """Create a cone aligned along the Y-axis with the apex at +Y.

    Parameters
    ----------
    radius : float
        Base radius (default 1.0).
    height : float
        Full height of the cone (default 2.0).
    segments : int
        Number of radial segments.  Must be >= 3.

    Returns
    -------
    dict with keys ``vertices``, ``faces``, ``normals``, ``uvs``.
    """
    segments = max(segments, 3)
    half_h = height / 2.0

    # Compute the slant height for correct normals
    slant = math.sqrt(radius * radius + height * height)
    ny = radius / slant
    nr = height / slant

    vertices: list[np.ndarray] = []
    normals:  list[np.ndarray] = []
    uvs:      list[np.ndarray] = []
    faces:    list[tuple[int, int, int]] = []

    # Apex vertex
    apex_idx = len(vertices)
    vertices.append(vec3(0, half_h, 0))
    normals.append(vec3(0, ny, 0))  # Approximate; real normal varies per face
    uvs.append(np.array([0.5, 1.0]))

    # Base ring vertices
    base_indices: list[int] = []
    for j in range(segments):
        theta = 2.0 * math.pi * j / segments
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        base_indices.append(len(vertices))
        vertices.append(vec3(radius * cos_t, -half_h, radius * sin_t))
        normals.append(normalize(vec3(nr * cos_t, ny, nr * sin_t)))
        uvs.append(np.array([j / segments, 0.0]))

    # Side faces – use per-face normals stored as apex duplicates for flat shading
    # We generate dedicated apex + base-edge vertices per triangle so that
    # normals are exact.
    side_start = len(vertices)
    for j in range(segments):
        next_j = (j + 1) % segments
        theta_mid = 2.0 * math.pi * (j + 0.5) / segments
        face_normal = normalize(vec3(
            math.cos(theta_mid),
            radius / height,
            math.sin(theta_mid),
        ))

        idx0 = len(vertices)
        vertices.append(vertices[apex_idx].copy())
        normals.append(face_normal)
        uvs.append(np.array([(j + 0.5) / segments, 1.0]))

        idx1 = len(vertices)
        vertices.append(vertices[base_indices[j]].copy())
        normals.append(face_normal)
        uvs.append(np.array([j / segments, 0.0]))

        idx2 = len(vertices)
        vertices.append(vertices[base_indices[next_j]].copy())
        normals.append(face_normal)
        uvs.append(np.array([(j + 1) / segments, 0.0]))

        faces.append((idx0, idx1, idx2))

    # Discard the generic apex and base ring used only as position templates
    vertices = vertices[side_start:]
    normals  = normals[side_start:]
    uvs      = uvs[side_start:]

    # Bottom cap – centre + fan
    cap_centre = len(vertices)
    vertices.append(vec3(0, -half_h, 0))
    normals.append(vec3(0, -1, 0))
    uvs.append(np.array([0.5, 0.5]))
    for j in range(segments):
        next_j = (j + 1) % segments
        idx_a = len(vertices)
        theta = 2.0 * math.pi * j / segments
        vertices.append(vec3(radius * math.cos(theta), -half_h, radius * math.sin(theta)))
        normals.append(vec3(0, -1, 0))
        uvs.append(np.array([0.5 + 0.5 * math.cos(theta), 0.5 + 0.5 * math.sin(theta)]))

        idx_b = len(vertices)
        theta2 = 2.0 * math.pi * next_j / segments
        vertices.append(vec3(radius * math.cos(theta2), -half_h, radius * math.sin(theta2)))
        normals.append(vec3(0, -1, 0))
        uvs.append(np.array([0.5 + 0.5 * math.cos(theta2), 0.5 + 0.5 * math.sin(theta2)]))

        faces.append((cap_centre, idx_a, idx_b))

    return {
        "vertices": np.array(vertices, dtype=np.float64),
        "faces":    np.array(faces, dtype=np.uint32),
        "normals":  np.array(normals, dtype=np.float64),
        "uvs":      np.array(uvs, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# 5. Torus
# ---------------------------------------------------------------------------

def create_torus(major_radius: float = 1.0, minor_radius: float = 0.3,
                 major_segments: int = 48,
                 minor_segments: int = 24) -> Dict[str, Any]:
    """Create a torus (donut) in the XZ plane, centred at the origin.

    Parameters
    ----------
    major_radius : float
        Distance from the centre of the tube to the centre of the torus.
    minor_radius : float
        Radius of the tube.
    major_segments : int
        Number of segments around the main ring.
    minor_segments : int
        Number of segments around the tube cross-section.

    Returns
    -------
    dict with keys ``vertices``, ``faces``, ``normals``, ``uvs``.
    """
    major_segments = max(major_segments, 3)
    minor_segments = max(minor_segments, 3)

    vertices: list[np.ndarray] = []
    normals:  list[np.ndarray] = []
    uvs:      list[np.ndarray] = []
    faces:    list[tuple[int, int, int]] = []

    # Build a grid of (major_segment, minor_segment) vertices
    for i in range(major_segments):
        theta = 2.0 * math.pi * i / major_segments
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        for j in range(minor_segments):
            phi = 2.0 * math.pi * j / minor_segments
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)

            # Position
            x = (major_radius + minor_radius * cos_phi) * cos_theta
            y = minor_radius * sin_phi
            z = (major_radius + minor_radius * cos_phi) * sin_theta

            # Normal (points outward from tube centre)
            nx = cos_phi * cos_theta
            ny = sin_phi
            nz = cos_phi * sin_theta

            vertices.append(vec3(x, y, z))
            normals.append(normalize(vec3(nx, ny, nz)))
            uvs.append(np.array([i / major_segments, j / minor_segments]))

    # Quads -> two triangles each
    for i in range(major_segments):
        ni = (i + 1) % major_segments
        for j in range(minor_segments):
            nj = (j + 1) % minor_segments

            curr  = i * minor_segments + j
            next_maj = ni * minor_segments + j
            next_both = ni * minor_segments + nj
            next_min = i * minor_segments + nj

            faces.append((curr, next_min, next_both))
            faces.append((curr, next_both, next_maj))

    return {
        "vertices": np.array(vertices, dtype=np.float64),
        "faces":    np.array(faces, dtype=np.uint32),
        "normals":  np.array(normals, dtype=np.float64),
        "uvs":      np.array(uvs, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# 6. Plane
# ---------------------------------------------------------------------------

def create_plane(width: float = 1.0, height: float = 1.0,
                 width_segments: int = 1,
                 height_segments: int = 1) -> Dict[str, Any]:
    """Create a rectangular plane in the XY plane, centred at the origin.

    Parameters
    ----------
    width : float
        Full width along the X-axis.
    height : float
        Full height along the Y-axis.
    width_segments : int
        Number of subdivisions along the width.
    height_segments : int
        Number of subdivisions along the height.

    Returns
    -------
    dict with keys ``vertices``, ``faces``, ``normals``, ``uvs``.
    """
    width_segments  = max(width_segments, 1)
    height_segments = max(height_segments, 1)

    vertices: list[np.ndarray] = []
    normals:  list[np.ndarray] = []
    uvs:      list[np.ndarray] = []
    faces:    list[tuple[int, int, int]] = []

    half_w = width / 2.0
    half_h = height / 2.0
    face_normal = vec3(0, 0, 1)

    for iy in range(height_segments + 1):
        for ix in range(width_segments + 1):
            u = ix / width_segments
            v = iy / height_segments
            x = -half_w + width * u
            y = -half_h + height * v
            vertices.append(vec3(x, y, 0))
            normals.append(face_normal)
            uvs.append(np.array([u, v]))

    # Triangulate grid
    for iy in range(height_segments):
        for ix in range(width_segments):
            a = iy * (width_segments + 1) + ix
            b = a + 1
            c = a + (width_segments + 1)
            d = c + 1
            faces.append((a, c, d))
            faces.append((a, d, b))

    return {
        "vertices": np.array(vertices, dtype=np.float64),
        "faces":    np.array(faces, dtype=np.uint32),
        "normals":  np.array(normals, dtype=np.float64),
        "uvs":      np.array(uvs, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# 7. Grid (XZ plane, wireframe)
# ---------------------------------------------------------------------------

def create_grid(size: float = 10.0, divisions: int = 10) -> Dict[str, Any]:
    """Create a flat grid in the XZ plane.

    Unlike other primitives, the ``faces`` key contains line-pairs (edges)
    of shape ``(E, 2)`` instead of triangles.

    Parameters
    ----------
    size : float
        Full extent of the grid (same along X and Z).
    divisions : int
        Number of cells per side.  Must be >= 1.

    Returns
    -------
    dict with keys ``vertices``, ``edges``, ``normals``, ``uvs``.
    ``edges`` is np.ndarray shape (E, 2) with uint32 indices.
    """
    divisions = max(divisions, 1)
    half = size / 2.0
    step = size / divisions

    vertices: list[np.ndarray] = []
    edges:    list[tuple[int, int]] = []

    # Lines along X (constant Z)
    for i in range(divisions + 1):
        z = -half + i * step
        idx = len(vertices)
        vertices.append(vec3(-half, 0, z))
        vertices.append(vec3(half, 0, z))
        edges.append((idx, idx + 1))

    # Lines along Z (constant X)
    for i in range(divisions + 1):
        x = -half + i * step
        idx = len(vertices)
        vertices.append(vec3(x, 0, -half))
        vertices.append(vec3(x, 0, half))
        edges.append((idx, idx + 1))

    n = len(vertices)
    return {
        "vertices": np.array(vertices, dtype=np.float64),
        "edges":    np.array(edges, dtype=np.uint32),
        "faces":    np.zeros((0, 3), dtype=np.uint32),  # wireframe grid, no triangles
        "normals":  np.tile(np.array([[0, 1, 0]], dtype=np.float64), (n, 1)),
        "uvs":      np.zeros((n, 2), dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# 8. Circle (XY plane, flat filled)
# ---------------------------------------------------------------------------

def create_circle(radius: float = 1.0, segments: int = 64) -> Dict[str, Any]:
    """Create a filled circle in the XY plane, centred at the origin.

    Parameters
    ----------
    radius : float
        Radius of the circle.
    segments : int
        Number of segments around the circumference.  Must be >= 3.

    Returns
    -------
    dict with keys ``vertices``, ``faces``, ``normals``, ``uvs``.
    """
    segments = max(segments, 3)

    vertices: list[np.ndarray] = []
    normals:  list[np.ndarray] = []
    uvs:      list[np.ndarray] = []
    faces:    list[tuple[int, int, int]] = []

    # Centre vertex
    centre = len(vertices)
    vertices.append(vec3(0, 0, 0))
    normals.append(vec3(0, 0, 1))
    uvs.append(np.array([0.5, 0.5]))

    for j in range(segments):
        theta = 2.0 * math.pi * j / segments
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        vertices.append(vec3(radius * cos_t, radius * sin_t, 0))
        normals.append(vec3(0, 0, 1))
        uvs.append(np.array([0.5 + 0.5 * cos_t, 0.5 + 0.5 * sin_t]))

    for j in range(segments):
        next_j = (j + 1) % segments
        faces.append((centre, 1 + j, 1 + next_j))

    return {
        "vertices": np.array(vertices, dtype=np.float64),
        "faces":    np.array(faces, dtype=np.uint32),
        "normals":  np.array(normals, dtype=np.float64),
        "uvs":      np.array(uvs, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# 9. Arrow
# ---------------------------------------------------------------------------

def create_arrow(length: float = 1.0, shaft_radius: float = 0.05,
                 head_radius: float = 0.15,
                 head_length: float = 0.2) -> Dict[str, Any]:
    """Create a 3D arrow (shaft + cone head) aligned along the +Y axis.

    Parameters
    ----------
    length : float
        Total length of the arrow (shaft + head).
    shaft_radius : float
        Radius of the cylindrical shaft.
    head_radius : float
        Radius of the conical head at its widest point.
    head_length : float
        Length of the conical head portion.

    Returns
    -------
    dict with keys ``vertices``, ``faces``, ``normals``, ``uvs``.
    """
    segments = 32
    shaft_length = length - head_length
    shaft_length = max(shaft_length, 0.0)

    vertices: list[np.ndarray] = []
    normals:  list[np.ndarray] = []
    uvs:      list[np.ndarray] = []
    faces:    list[tuple[int, int, int]] = []

    # ── Shaft (cylinder from y=0 to y=shaft_length) ─────────────────────
    shaft_bottom: list[int] = []
    shaft_top: list[int] = []
    for j in range(segments):
        theta = 2.0 * math.pi * j / segments
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        side_n = vec3(cos_t, 0, sin_t)

        shaft_bottom.append(len(vertices))
        vertices.append(vec3(shaft_radius * cos_t, 0, shaft_radius * sin_t))
        normals.append(side_n)
        uvs.append(np.array([j / segments, 0.0]))

        shaft_top.append(len(vertices))
        vertices.append(vec3(shaft_radius * cos_t, shaft_length, shaft_radius * sin_t))
        normals.append(side_n)
        uvs.append(np.array([j / segments, 1.0]))

    # Shaft side faces
    for j in range(segments):
        nj = (j + 1) % segments
        faces.append((shaft_bottom[j], shaft_top[j], shaft_top[nj]))
        faces.append((shaft_bottom[j], shaft_top[nj], shaft_bottom[nj]))

    # Shaft bottom cap
    cap_centre = len(vertices)
    vertices.append(vec3(0, 0, 0))
    normals.append(vec3(0, -1, 0))
    uvs.append(np.array([0.5, 0.5]))
    for j in range(segments):
        nj = (j + 1) % segments
        faces.append((cap_centre, shaft_bottom[nj], shaft_bottom[j]))

    # ── Head (cone from y=shaft_length to y=length) ─────────────────────
    # Flat-shaded cone: each side triangle gets its own copy of the apex
    # and base-edge vertices so normals are per-face.
    slant = math.sqrt(head_radius ** 2 + head_length ** 2)
    nr = head_length / slant
    ny = head_radius / slant

    for j in range(segments):
        nj = (j + 1) % segments
        theta_mid = 2.0 * math.pi * (j + 0.5) / segments
        face_n = normalize(vec3(
            nr * math.cos(theta_mid), ny, nr * math.sin(theta_mid),
        ))

        # Apex (per-face copy)
        apex = len(vertices)
        vertices.append(vec3(0, length, 0))
        normals.append(face_n)
        uvs.append(np.array([(j + 0.5) / segments, 1.0]))

        # Base edge A
        theta_a = 2.0 * math.pi * j / segments
        b0 = len(vertices)
        vertices.append(vec3(head_radius * math.cos(theta_a),
                             shaft_length,
                             head_radius * math.sin(theta_a)))
        normals.append(face_n)
        uvs.append(np.array([j / segments, 0.0]))

        # Base edge B
        theta_b = 2.0 * math.pi * nj / segments
        b1 = len(vertices)
        vertices.append(vec3(head_radius * math.cos(theta_b),
                             shaft_length,
                             head_radius * math.sin(theta_b)))
        normals.append(face_n)
        uvs.append(np.array([(j + 1) / segments, 0.0]))

        faces.append((apex, b0, b1))

    # Base cap of the cone head (covers the bottom of the conical head
    # so there is no gap between shaft-top and head-base).
    head_cap_centre = len(vertices)
    vertices.append(vec3(0, shaft_length, 0))
    normals.append(vec3(0, -1, 0))
    uvs.append(np.array([0.5, 0.5]))
    for j in range(segments):
        nj = (j + 1) % segments
        theta = 2.0 * math.pi * j / segments
        theta2 = 2.0 * math.pi * nj / segments

        ca = len(vertices)
        vertices.append(vec3(head_radius * math.cos(theta),
                             shaft_length,
                             head_radius * math.sin(theta)))
        normals.append(vec3(0, -1, 0))
        uvs.append(np.array([0.5 + 0.5 * math.cos(theta),
                             0.5 + 0.5 * math.sin(theta)]))

        cb = len(vertices)
        vertices.append(vec3(head_radius * math.cos(theta2),
                             shaft_length,
                             head_radius * math.sin(theta2)))
        normals.append(vec3(0, -1, 0))
        uvs.append(np.array([0.5 + 0.5 * math.cos(theta2),
                             0.5 + 0.5 * math.sin(theta2)]))

        faces.append((head_cap_centre, cb, ca))

    return {
        "vertices": np.array(vertices, dtype=np.float64),
        "faces":    np.array(faces, dtype=np.uint32),
        "normals":  np.array(normals, dtype=np.float64),
        "uvs":      np.array(uvs, dtype=np.float64),
    }
