"""Nexus3D Mesh — core geometry container for the headless 3D engine.

The :class:`Mesh` class wraps raw vertex / face / normal / UV data and
provides I/O (OBJ), geometric queries, transforms, and topology editing
operations (merge, subdivide, decimate, extrude, inset).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from nexus3d.math3d.core import (
    vec3,
    normalize,
    magnitude,
    triangle_normal,
    compute_bounding_box,
    compute_centroid,
    cross,
    dot,
    compute_volume,
    compute_surface_area,
    triangle_area,
)


class Mesh:
    """Core 3D mesh with vertices, faces, normals, UVs, and materials.

    Attributes
    ----------
    name : str
        Human-readable identifier.
    vertices : np.ndarray, shape (N, 3)
    faces : np.ndarray, shape (M, 3), dtype uint32
    normals : np.ndarray, shape (N, 3)
    uvs : np.ndarray, shape (N, 2)
    tangents : np.ndarray | None, shape (N, 3)
    material_name : str | None
    """

    def __init__(
        self,
        name: str = "Mesh",
        vertices: Optional[np.ndarray] = None,
        faces: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        uvs: Optional[np.ndarray] = None,
    ) -> None:
        self.name = name

        self.vertices = (
            np.asarray(vertices, dtype=np.float64) if vertices is not None
            else np.zeros((0, 3), dtype=np.float64)
        )
        self.faces = (
            np.asarray(faces, dtype=np.uint32) if faces is not None
            else np.zeros((0, 3), dtype=np.uint32)
        )
        self.normals = (
            np.asarray(normals, dtype=np.float64) if normals is not None
            else np.zeros_like(self.vertices)
        )
        self.uvs = (
            np.asarray(uvs, dtype=np.float64) if uvs is not None
            else np.zeros((self.vertices.shape[0], 2), dtype=np.float64)
        )

        self.tangents: Optional[np.ndarray] = None
        self.material_name: Optional[str] = None

        # Auto-compute normals when absent
        if self.vertices.shape[0] > 0 and self.normals.shape[0] == 0:
            self.compute_normals()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_primitive(cls, name: str, primitive_data: Dict[str, Any]) -> "Mesh":
        """Create a :class:`Mesh` from a primitive dict.

        ``primitive_data`` must contain at least ``vertices`` and ``faces``.
        ``normals`` and ``uvs`` are optional and default to zero-filled arrays.

        Parameters
        ----------
        name : str
            Mesh name.
        primitive_data : dict
            Output of any ``create_*`` function in :mod:`nexus3d.mesh.primitives`.

        Returns
        -------
        Mesh
        """
        vertices = primitive_data.get("vertices")
        faces = primitive_data.get("faces")
        if vertices is None or faces is None:
            raise ValueError("primitive_data must contain 'vertices' and 'faces'")

        normals = primitive_data.get("normals")
        uvs = primitive_data.get("uvs")
        return cls(
            name=name,
            vertices=vertices,
            faces=faces,
            normals=normals,
            uvs=uvs,
        )

    @classmethod
    def from_file(cls, filepath: str) -> "Mesh":
        """Load a mesh from a Wavefront OBJ file.

        Supports ``v``, ``vn``, ``vt``, and ``f`` directives.  Face formats
        accepted: ``f v v v``, ``f v/vt v/vt v/vt``, ``f v/vt/vn v/vt/vn v/vt/vn``,
        ``f v//vn v//vn v//vn``, and polygon faces with more than 3 vertices
        (automatically triangulated as fans).

        Parameters
        ----------
        filepath : str
            Path to the ``.obj`` file.

        Returns
        -------
        Mesh
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"OBJ file not found: {filepath}")

        positions: list[np.ndarray] = []
        vert_normals: list[np.ndarray] = []
        vert_uvs: list[np.ndarray] = []
        face_v: list[list[Tuple[int, int, int]]] = []  # (pos_idx, uv_idx, norm_idx)

        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                kind = parts[0]

                if kind == "v" and len(parts) >= 4:
                    positions.append(np.array(
                        [float(parts[1]), float(parts[2]), float(parts[3])],
                        dtype=np.float64,
                    ))
                elif kind == "vn" and len(parts) >= 4:
                    vert_normals.append(np.array(
                        [float(parts[1]), float(parts[2]), float(parts[3])],
                        dtype=np.float64,
                    ))
                elif kind == "vt" and len(parts) >= 3:
                    vert_uvs.append(np.array(
                        [float(parts[1]), float(parts[2])],
                        dtype=np.float64,
                    ))
                elif kind == "f":
                    indices: list[Tuple[int, int, int]] = []
                    for token in parts[1:]:
                        # OBJ indices are 1-based; negative = relative
                        components = token.split("/")
                        # Position index
                        pi = int(components[0])
                        # UV index (may be omitted)
                        ti = int(components[1]) if len(components) > 1 and components[1] else -1
                        # Normal index (may be omitted)
                        ni = int(components[2]) if len(components) > 2 and components[2] else -1
                        indices.append((pi, ti, ni))
                    face_v.append(indices)

        # ---- Build expanded arrays ----
        # OBJ uses per-face-vertex indices, so we must expand to unique vertices.
        v_list: list[np.ndarray] = []
        n_list: list[np.ndarray] = []
        uv_list: list[np.ndarray] = []
        tri_list: list[Tuple[int, int, int]] = []

        # Maps (pos_idx, uv_idx, norm_idx) -> expanded vertex index
        index_map: Dict[Tuple[int, int, int], int] = {}

        def _resolve(ref: Tuple[int, int, int]) -> int:
            """Return expanded vertex index, creating if necessary."""
            if ref in index_map:
                return index_map[ref]
            pi, ti, ni = ref
            # Convert 1-based (or negative) to 0-based
            pos_idx = pi - 1 if pi > 0 else len(positions) + pi
            v = positions[pos_idx].copy()

            n = np.zeros(3, dtype=np.float64)
            if ni != -1:
                norm_idx = ni - 1 if ni > 0 else len(vert_normals) + ni
                n = vert_normals[norm_idx].copy()

            uv = np.zeros(2, dtype=np.float64)
            if ti != -1:
                tex_idx = ti - 1 if ti > 0 else len(vert_uvs) + ti
                uv = vert_uvs[tex_idx].copy()

            idx = len(v_list)
            v_list.append(v)
            n_list.append(n)
            uv_list.append(uv)
            index_map[ref] = idx
            return idx

        for face in face_v:
            # Fan-triangulate polygons
            for i in range(1, len(face) - 1):
                i0 = _resolve(face[0])
                i1 = _resolve(face[i])
                i2 = _resolve(face[i + 1])
                tri_list.append((i0, i1, i2))

        name = path.stem
        mesh = cls(
            name=name,
            vertices=np.array(v_list, dtype=np.float64) if v_list else None,
            faces=np.array(tri_list, dtype=np.uint32) if tri_list else None,
            normals=np.array(n_list, dtype=np.float64) if n_list else None,
            uvs=np.array(uv_list, dtype=np.float64) if uv_list else None,
        )

        # If normals were missing from the OBJ, compute them
        if not vert_normals and mesh.vertex_count() > 0:
            mesh.compute_normals()

        return mesh

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_obj(self, filepath: str) -> None:
        """Export the mesh to a Wavefront OBJ file.

        Parameters
        ----------
        filepath : str
            Destination path (``.obj`` extension recommended).
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# Nexus3D OBJ export: {self.name}\n")
            if self.material_name:
                fh.write(f"mtllib {self.material_name}.mtl\n")
                fh.write(f"usemtl {self.material_name}\n")

            # Vertices (1-based)
            for v in self.vertices:
                fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Normals
            for n in self.normals:
                fh.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

            # UVs
            for uv in self.uvs:
                fh.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

            # Faces  (1-based indices)
            for f in self.faces:
                i0, i1, i2 = int(f[0]) + 1, int(f[1]) + 1, int(f[2]) + 1
                fh.write(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}\n")

    # ------------------------------------------------------------------
    # Normal & tangent computation
    # ------------------------------------------------------------------

    def compute_normals(self) -> None:
        """Compute smooth per-vertex normals by averaging face normals.

        For each face the (un-normalised) face normal is accumulated into
        each of its three vertex slots.  After all faces are processed
        every vertex normal is normalised.
        """
        n_verts = self.vertices.shape[0]
        if n_verts == 0:
            return

        accum = np.zeros((n_verts, 3), dtype=np.float64)

        for face in self.faces:
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]]
            v2 = self.vertices[face[2]]
            fn = cross(v1 - v0, v2 - v0)
            accum[face[0]] += fn
            accum[face[1]] += fn
            accum[face[2]] += fn

        # Normalise rows with non-zero length; leave zero-vectors as-is.
        lengths = np.linalg.norm(accum, axis=1, keepdims=True)
        mask = (lengths > 1e-12).flatten()
        accum[mask] /= lengths[mask]
        self.normals = accum

    def compute_tangents(self) -> np.ndarray:
        """Compute tangent vectors for normal mapping.

        Uses the Lengyel–Szonyi method to derive per-vertex tangent (and
        bitangent) vectors from triangles and their UV coordinates.

        Returns
        -------
        np.ndarray, shape (N, 3)
            The computed tangent vectors (also stored in ``self.tangents``).
        """
        n_verts = self.vertices.shape[0]
        if n_verts == 0 or self.faces.shape[0] == 0:
            self.tangents = np.zeros((n_verts, 3), dtype=np.float64)
            return self.tangents

        tan_accum = np.zeros((n_verts, 3), dtype=np.float64)
        bitan_accum = np.zeros((n_verts, 3), dtype=np.float64)

        for face in self.faces:
            i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])

            v0, v1, v2 = self.vertices[i0], self.vertices[i1], self.vertices[i2]
            uv0, uv1, uv2 = self.uvs[i0], self.uvs[i1], self.uvs[i2]

            e1 = v1 - v0
            e2 = v2 - v0

            duv1 = uv1 - uv0
            duv2 = uv2 - uv0

            denom = duv1[0] * duv2[1] - duv2[0] * duv1[1]
            if abs(denom) < 1e-12:
                continue
            r = 1.0 / denom

            s_dir = (duv2[1] * e1 - duv1[1] * e2) * r
            t_dir = (-duv2[0] * e1 + duv1[0] * e2) * r

            for idx in (i0, i1, i2):
                tan_accum[idx] += s_dir
                bitan_accum[idx] += t_dir

        # Orthogonalise tangents w.r.t. normals (Gram-Schmidt)
        self.tangents = np.zeros((n_verts, 3), dtype=np.float64)
        for i in range(n_verts):
            n = self.normals[i]
            t = tan_accum[i]
            # t = normalize(t - n * dot(n, t))
            t = t - n * dot(n, t)
            length = np.linalg.norm(t)
            if length > 1e-12:
                t /= length
            else:
                t = np.zeros(3, dtype=np.float64)
            self.tangents[i] = t

        return self.tangents

    # ------------------------------------------------------------------
    # Geometric queries
    # ------------------------------------------------------------------

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the axis-aligned bounding box as ``(min_corner, max_corner)``."""
        if self.vertices.shape[0] == 0:
            return vec3(), vec3()
        return compute_bounding_box(self.vertices)

    def get_centroid(self) -> np.ndarray:
        """Return the centroid (mean position) of all vertices."""
        if self.vertices.shape[0] == 0:
            return vec3()
        return compute_centroid(self.vertices)

    def get_volume(self) -> float:
        """Compute the signed-volume of a closed mesh using the divergence theorem.

        Returns 0.0 if the mesh has no faces.
        """
        if self.faces.shape[0] == 0:
            return 0.0
        return compute_volume(self.vertices, self.faces)

    def get_surface_area(self) -> float:
        """Compute the total surface area of all triangles."""
        if self.faces.shape[0] == 0:
            return 0.0
        return compute_surface_area(self.vertices, self.faces)

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def apply_transform(self, matrix_4x4: np.ndarray) -> None:
        """Transform all vertices by a 4×4 homogeneous matrix.

        Parameters
        ----------
        matrix_4x4 : np.ndarray, shape (4, 4)
            The transformation matrix.
        """
        if self.vertices.shape[0] == 0:
            return
        mat = np.asarray(matrix_4x4, dtype=np.float64).reshape(4, 4)

        # Convert (N, 3) -> (N, 4) with w=1
        ones = np.ones((self.vertices.shape[0], 1), dtype=np.float64)
        homogeneous = np.hstack([self.vertices, ones])  # (N, 4)
        transformed = (mat @ homogeneous.T).T  # (N, 4)
        self.vertices = transformed[:, :3]

        # Transform normals by the inverse-transpose of the upper-left 3×3
        normal_mat = np.linalg.inv(mat[:3, :3]).T
        for i in range(self.normals.shape[0]):
            n = normal_mat @ self.normals[i]
            length = np.linalg.norm(n)
            self.normals[i] = n / length if length > 1e-12 else np.zeros(3)

    # ------------------------------------------------------------------
    # Topology operations
    # ------------------------------------------------------------------

    def merge(self, other: "Mesh") -> None:
        """Merge another mesh into this one.

        Vertex, normal, UV and face arrays are concatenated.  The *other*
        mesh is not modified.

        Parameters
        ----------
        other : Mesh
            Mesh to merge in.
        """
        if other.vertices.shape[0] == 0:
            return

        offset = self.vertices.shape[0]

        self.vertices = np.vstack([self.vertices, other.vertices])
        self.normals  = np.vstack([self.normals, other.normals])
        self.uvs      = np.vstack([self.uvs, other.uvs])

        if other.faces.shape[0] > 0:
            shifted = other.faces + np.uint32(offset)
            self.faces = np.vstack([self.faces, shifted])

        if self.tangents is not None and other.tangents is not None:
            self.tangents = np.vstack([self.tangents, other.tangents])

    def subdivide(self, level: int = 1) -> "Mesh":
        """Return a new mesh with *level* rounds of simplified Loop subdivision.

        Each triangle is split into 4 sub-triangles by inserting a vertex
        at each edge midpoint.  This is a simplified (non-true-Loop) scheme
        suitable for uniform refinement.

        Parameters
        ----------
        level : int
            Number of subdivision iterations (default 1).

        Returns
        -------
        Mesh
            A new subdivided mesh.  The original is not modified.
        """
        current_verts = self.vertices.copy()
        current_faces = self.faces.copy()
        current_uvs = self.uvs.copy()

        for _ in range(level):
            edge_midpoints: Dict[Tuple[int, int], int] = {}
            new_verts: list[np.ndarray] = list(current_verts)
            new_uvs: list[np.ndarray] = list(current_uvs)
            new_faces: list[Tuple[int, int, int]] = []

            def _midpoint(i: int, j: int) -> int:
                key = (min(i, j), max(i, j))
                if key in edge_midpoints:
                    return edge_midpoints[key]
                mid_v = (current_verts[i] + current_verts[j]) / 2.0
                mid_uv = (current_uvs[i] + current_uvs[j]) / 2.0
                idx = len(new_verts)
                new_verts.append(mid_v)
                new_uvs.append(mid_uv)
                edge_midpoints[key] = idx
                return idx

            for face in current_faces:
                i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
                a = _midpoint(i0, i1)
                b = _midpoint(i1, i2)
                c = _midpoint(i2, i0)
                new_faces.append((i0, a, c))
                new_faces.append((i1, b, a))
                new_faces.append((i2, c, b))
                new_faces.append((a, b, c))

            current_verts = np.array(new_verts, dtype=np.float64)
            current_faces = np.array(new_faces, dtype=np.uint32)
            current_uvs   = np.array(new_uvs, dtype=np.float64)

        result = Mesh(
            name=f"{self.name}_subdivided",
            vertices=current_verts,
            faces=current_faces,
            uvs=current_uvs,
        )
        result.compute_normals()
        return result

    def decimate(self, target_faces: int) -> "Mesh":
        """Simplified mesh decimation by collapsing shortest edges.

        Iteratively collapses the shortest edge in the mesh until the
        face count reaches *target_faces*.  This is a naive, non-optimal
        approach suitable for quick reductions.  Topologically degenerate
        collapses are skipped.

        Parameters
        ----------
        target_faces : int
            Desired number of faces (minimum 4).

        Returns
        -------
        Mesh
            A new decimated mesh.  The original is not modified.
        """
        if self.face_count() <= target_faces:
            return self._copy()

        verts = self.vertices.tolist()
        norms = self.normals.tolist()
        uv_list = self.uvs.tolist()
        faces = [tuple(int(f) for f in face) for face in self.faces]

        # Build adjacency
        remaining: set[int] = set(range(len(verts)))
        edge_length_cache: Dict[Tuple[int, int], float] = {}

        def _edge_len(a: int, b: int) -> float:
            key = (min(a, b), max(a, b))
            if key not in edge_length_cache:
                edge_length_cache[key] = float(np.linalg.norm(
                    np.array(verts[a]) - np.array(verts[b])
                ))
            return edge_length_cache[key]

        max_iters = self.face_count() * 4  # Safety bound
        iters = 0

        while len(faces) > target_faces and iters < max_iters:
            iters += 1
            # Collect candidate edges
            edge_set: set[Tuple[int, int]] = set()
            for tri in faces:
                a, b, c = tri
                edge_set.add((min(a, b), max(a, b)))
                edge_set.add((min(b, c), max(b, c)))
                edge_set.add((min(a, c), max(a, c)))

            # Filter: both vertices must still exist
            valid_edges = [e for e in edge_set if e[0] in remaining and e[1] in remaining]
            if not valid_edges:
                break

            # Find shortest
            shortest = min(valid_edges, key=lambda e: _edge_len(e[0], e[1]))
            va, vb = shortest

            # Replace all occurrences of vb with va
            new_faces = []
            changed = False
            for tri in faces:
                a, b, c = tri
                na = va if a == vb else a
                nb = va if b == vb else b
                nc = va if c == vb else c
                # Skip degenerate triangles
                if len({na, nb, nc}) == 3:
                    new_faces.append((na, nb, nc))
                    if (na, nb, nc) != tri:
                        changed = True
            faces = new_faces
            if changed:
                remaining.discard(vb)

        # Compact vertex array
        idx_map: Dict[int, int] = {}
        compact_verts: list[np.ndarray] = []
        compact_norms: list[np.ndarray] = []
        compact_uvs: list[np.ndarray] = []
        for idx in sorted(remaining):
            idx_map[idx] = len(compact_verts)
            compact_verts.append(verts[idx])
            compact_norms.append(norms[idx])
            compact_uvs.append(uv_list[idx])

        compact_faces = [(idx_map[a], idx_map[b], idx_map[c]) for a, b, c in faces]

        result = Mesh(
            name=f"{self.name}_decimated",
            vertices=np.array(compact_verts, dtype=np.float64) if compact_verts else None,
            faces=np.array(compact_faces, dtype=np.uint32) if compact_faces else None,
            normals=np.array(compact_norms, dtype=np.float64) if compact_norms else None,
            uvs=np.array(compact_uvs, dtype=np.float64) if compact_uvs else None,
        )
        result.compute_normals()
        return result

    def boolean_union(self, other: "Mesh") -> "Mesh":
        """Placeholder for boolean CSG union.

        A full implementation requires a mesh boolean library (e.g.
        Manifold, Carve, or Cork).  This method currently just merges
        the two meshes.

        Parameters
        ----------
        other : Mesh
            Mesh to union with.

        Returns
        -------
        Mesh
            The merged result.
        """
        result = self._copy()
        result.merge(other)
        result.name = f"{self.name}_union_{other.name}"
        return result

    def extrude_face(self, face_index: int, distance: float) -> "Mesh":
        """Extrude a face outward along its normal by *distance*.

        Creates new vertices for the extruded face and connecting side
        triangles.

        Parameters
        ----------
        face_index : int
            Index into ``self.faces``.
        distance : float
            Extrusion distance (positive = along normal).

        Returns
        -------
        Mesh
            A new mesh with the extrusion applied.
        """
        result = self._copy()

        if face_index < 0 or face_index >= result.face_count():
            raise IndexError(f"face_index {face_index} out of range [0, {result.face_count()})")

        face = result.faces[face_index]
        v0 = result.vertices[face[0]]
        v1 = result.vertices[face[1]]
        v2 = result.vertices[face[2]]
        fn = triangle_normal(v0, v1, v2)
        offset = fn * distance

        # Create new vertices
        n_verts = result.vertices.shape[0]
        new_indices: list[int] = []
        for vi in face:
            result.vertices = np.vstack([result.vertices, result.vertices[vi] + offset])
            result.normals  = np.vstack([result.normals, fn])
            result.uvs      = np.vstack([result.uvs, result.uvs[vi]])
            new_indices.append(n_verts)
            n_verts += 1

        # Side faces
        num_face_verts = len(face)
        for i in range(num_face_verts):
            j = (i + 1) % num_face_verts
            a = int(face[i])
            b = int(face[j])
            c = new_indices[j]
            d = new_indices[i]
            # Two triangles per side quad (CCW)
            tri1 = (a, b, c)
            tri2 = (a, c, d)
            result.faces = np.vstack([result.faces, np.array([tri1, tri2], dtype=np.uint32)])

        # Replace original face with extruded face
        result.faces[face_index] = np.array(new_indices, dtype=np.uint32)
        result.compute_normals()
        return result

    def inset_face(self, face_index: int, amount: float) -> "Mesh":
        """Inset (bevel) a face inward by *amount*.

        Creates new inner vertices shrunk towards the face centroid and
        connects them to the original edges.

        Parameters
        ----------
        face_index : int
            Index into ``self.faces``.
        amount : float
            Inset amount as a fraction (0 = no inset, 1 = collapse to centroid).

        Returns
        -------
        Mesh
            A new mesh with the inset applied.
        """
        result = self._copy()

        if face_index < 0 or face_index >= result.face_count():
            raise IndexError(f"face_index {face_index} out of range [0, {result.face_count()})")

        face = result.faces[face_index]
        v0 = result.vertices[face[0]]
        v1 = result.vertices[face[1]]
        v2 = result.vertices[face[2]]
        fn = triangle_normal(v0, v1, v2)

        # Face centroid
        centroid = (v0 + v1 + v2) / 3.0

        # Create inner vertices
        n_verts = result.vertices.shape[0]
        inner_indices: list[int] = []
        for vi in face:
            inner_pos = result.vertices[vi] * (1.0 - amount) + centroid * amount
            result.vertices = np.vstack([result.vertices, inner_pos])
            result.normals  = np.vstack([result.normals, fn])
            result.uvs      = np.vstack([result.uvs, result.uvs[vi]])
            inner_indices.append(n_verts)
            n_verts += 1

        # Connect original edges to inner vertices
        num_face_verts = len(face)
        # Original face becomes the inner face
        result.faces[face_index] = np.array(inner_indices, dtype=np.uint32)

        for i in range(num_face_verts):
            j = (i + 1) % num_face_verts
            a = int(face[i])
            b = int(face[j])
            c = inner_indices[j]
            d = inner_indices[i]
            tri1 = (a, c, b)
            tri2 = (a, d, c)
            result.faces = np.vstack([result.faces, np.array([tri1, tri2], dtype=np.uint32)])

        result.compute_normals()
        return result

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the mesh to a JSON-friendly dictionary."""
        return {
            "name": self.name,
            "vertices": self.vertices.tolist(),
            "faces": self.faces.tolist(),
            "normals": self.normals.tolist(),
            "uvs": self.uvs.tolist(),
            "tangents": self.tangents.tolist() if self.tangents is not None else None,
            "material_name": self.material_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Mesh":
        """Deserialise a mesh from a dictionary (e.g. JSON payload).

        Parameters
        ----------
        data : dict
            Dictionary produced by :meth:`to_dict`.

        Returns
        -------
        Mesh
        """
        mesh = cls(
            name=data.get("name", "Mesh"),
            vertices=np.array(data["vertices"], dtype=np.float64),
            faces=np.array(data["faces"], dtype=np.uint32),
            normals=np.array(data.get("normals", []), dtype=np.float64),
            uvs=np.array(data.get("uvs", []), dtype=np.float64),
        )
        if data.get("tangents") is not None:
            mesh.tangents = np.array(data["tangents"], dtype=np.float64)
        mesh.material_name = data.get("material_name")
        return mesh

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def vertex_count(self) -> int:
        """Return the number of vertices."""
        return int(self.vertices.shape[0])

    def face_count(self) -> int:
        """Return the number of triangles."""
        return int(self.faces.shape[0])

    def _copy(self) -> "Mesh":
        """Return a deep copy of this mesh."""
        m = Mesh(
            name=self.name,
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            normals=self.normals.copy(),
            uvs=self.uvs.copy(),
        )
        m.tangents = self.tangents.copy() if self.tangents is not None else None
        m.material_name = self.material_name
        return m

    def __repr__(self) -> str:
        return (
            f"Mesh(name={self.name!r}, "
            f"vertices={self.vertex_count()}, "
            f"faces={self.face_count()})"
        )
