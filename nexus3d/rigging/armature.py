"""Nexus3D Rigging System — armature, bones, skin weights, and IK integration.

Provides the core rigging data structures for skeletal animation in the
headless 3D engine.  Bones are organised into an :class:`Armature` (skeleton)
tree, IK solvers from :mod:`nexus3d.math3d.core` are integrated for
real-time posing, and :class:`SkinWeights` handles linear blend skinning
so meshes deform with the skeleton.

Quaternion convention throughout is ``[w, x, y, z]``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from nexus3d.math3d.core import (
    vec3,
    quat_identity,
    normalize_quat,
    quat_multiply,
    quat_from_axis_angle,
    quat_to_rotation_matrix,
    mat4_identity,
    mat4_translate,
    compose_matrix,
    decompose_matrix,
    solve_2bone_ik,
    solve_fabrik,
    solve_ccd,
    normalize,
    magnitude,
    distance,
    closest_point_on_segment,
    dot,
    cross,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _rotation_align_y(direction: np.ndarray) -> np.ndarray:
    """Return a quaternion that rotates the Y-axis ``[0,1,0]`` onto *direction*.

    Handles the degenerate parallel / anti-parallel cases explicitly so
    there is no discontinuity.

    Parameters
    ----------
    direction : np.ndarray, shape (3,)
        Target direction (will be normalised internally).

    Returns
    -------
    np.ndarray, shape (4,)
        Unit quaternion ``[w, x, y, z]``.
    """
    d = normalize(direction)
    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    dot_val = float(np.clip(np.dot(y_axis, d), -1.0, 1.0))

    if dot_val > 1.0 - 1e-6:
        return quat_identity()
    if dot_val < -1.0 + 1e-6:
        # 180-degree rotation around any perpendicular axis
        if abs(d[0]) < 0.9:
            perp = normalize(cross(d, vec3(1.0, 0.0, 0.0)))
        else:
            perp = normalize(cross(d, vec3(0.0, 0.0, 1.0)))
        return quat_from_axis_angle(perp, math.pi)

    axis = normalize(cross(y_axis, d))
    angle = math.acos(dot_val)
    return quat_from_axis_angle(axis, angle)


# ─── Bone ─────────────────────────────────────────────────────────────────────


class Bone:
    """A single bone in an armature.

    A bone is defined by its *head* and *tail* positions expressed in the
    **parent bone's local coordinate system** (world space for root bones).
    The bone's primary axis is Y, pointing from head toward tail.

    The *local_transform* is a 4x4 matrix that maps bone-local coordinates
    (origin at head, Y toward tail) into parent space.  It is automatically
    rebuilt whenever :meth:`set_head_tail` is called or the roll angle
    changes.

    The *world_transform* cascades from the root and is recomputed by
    :meth:`Armature.update_transforms`.

    Attributes
    ----------
    name : str
        Unique identifier within the armature.
    head_pos : np.ndarray, shape (3,)
        Head position in parent space.
    tail_pos : np.ndarray, shape (3,)
        Tail position in parent space.
    parent : Bone | None
        Parent bone reference (``None`` for root bones).
    children : list[Bone]
        Direct child bones.
    local_transform : np.ndarray, shape (4, 4)
        Maps bone-local coords to parent space.
    world_transform : np.ndarray, shape (4, 4)
        Maps bone-local coords to world space.
    roll : float
        Twist angle in radians around the bone's primary axis.
    use_connect : bool
        If ``True`` the bone's head is rigidly attached to its parent's
        tail (cosmetic hint; the actual constraint depends on the IK solver).
    """

    def __init__(
        self,
        name: str,
        head_pos: Optional[np.ndarray] = None,
        tail_pos: Optional[np.ndarray] = None,
        parent: Optional["Bone"] = None,
    ) -> None:
        self.name: str = name
        self.head_pos: np.ndarray = (
            np.asarray(head_pos, dtype=np.float64).copy()
            if head_pos is not None
            else vec3()
        )
        self.tail_pos: np.ndarray = (
            np.asarray(tail_pos, dtype=np.float64).copy()
            if tail_pos is not None
            else vec3(0.0, 1.0, 0.0)
        )
        self.parent: Optional[Bone] = parent
        self.children: List[Bone] = []
        self.roll: float = 0.0
        self.use_connect: bool = parent is not None
        self.local_transform: np.ndarray = mat4_identity()
        self.world_transform: np.ndarray = mat4_identity()

        if parent is not None:
            parent.children.append(self)

        # Build the initial local transform from positions.
        self._rebuild_local_transform()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def direction(self) -> np.ndarray:
        """Normalised direction from head to tail."""
        d = self.tail_pos - self.head_pos
        m = magnitude(d)
        if m < 1e-10:
            return vec3(0.0, 1.0, 0.0)
        return d / m

    @property
    def length(self) -> float:
        """Euclidean distance from head to tail."""
        return float(distance(self.head_pos, self.tail_pos))

    @property
    def world_head(self) -> np.ndarray:
        """Head position transformed to world space."""
        return self.world_transform[:3, 3].copy()

    @property
    def world_tail(self) -> np.ndarray:
        """Tail position transformed to world space.

        Equivalent to transforming the bone-local point ``(0, length, 0)``
        through the world transform.
        """
        bone_local_tail = np.array([0.0, self.length, 0.0, 1.0], dtype=np.float64)
        return (self.world_transform @ bone_local_tail)[:3].copy()

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def set_head_tail(
        self,
        head: np.ndarray,
        tail: np.ndarray,
    ) -> None:
        """Set new head and tail positions (in parent space) and rebuild
        the local transform."""
        self.head_pos = np.asarray(head, dtype=np.float64).copy()
        self.tail_pos = np.asarray(tail, dtype=np.float64).copy()
        self._rebuild_local_transform()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild_local_transform(self) -> None:
        """Recompute :attr:`local_transform` from :attr:`head_pos`,
        :attr:`tail_pos`, and :attr:`roll`.

        The resulting matrix:

        * rotates the bone-local Y-axis onto the head→tail direction,
        * applies the roll twist around the (already-rotated) Y-axis,
        * then translates to ``head_pos`` in parent space.
        """
        d = self.tail_pos - self.head_pos
        length = magnitude(d)

        if length < 1e-10:
            # Degenerate zero-length bone: identity rotation at head.
            self.local_transform = mat4_translate(
                self.head_pos[0], self.head_pos[1], self.head_pos[2]
            )
            return

        direction = d / length

        # Orientation quaternion: Y-axis → bone direction.
        rot_quat = _rotation_align_y(direction)

        # Roll twist around the bone's local Y-axis.
        if abs(self.roll) > 1e-10:
            roll_quat = quat_from_axis_angle(
                np.array([0.0, 1.0, 0.0], dtype=np.float64), self.roll
            )
            # Compose: apply roll first (in bone-local), then orient.
            rot_quat = quat_multiply(rot_quat, roll_quat)
            rot_quat = normalize_quat(rot_quat)

        rot_mat = quat_to_rotation_matrix(rot_quat)
        self.local_transform = mat4_identity()
        self.local_transform[:3, :3] = rot_mat
        self.local_transform[:3, 3] = self.head_pos

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the bone to a JSON-friendly dictionary."""
        return {
            "name": self.name,
            "head_pos": self.head_pos.tolist(),
            "tail_pos": self.tail_pos.tolist(),
            "parent_name": self.parent.name if self.parent else None,
            "roll": self.roll,
            "use_connect": self.use_connect,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], parent: Optional["Bone"] = None) -> "Bone":
        """Deserialise a bone from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary produced by :meth:`to_dict`.
        parent : Bone | None
            Parent bone reference (the caller is responsible for linking).
        """
        bone = cls(
            name=data["name"],
            head_pos=np.array(data["head_pos"], dtype=np.float64),
            tail_pos=np.array(data["tail_pos"], dtype=np.float64),
            parent=parent,
        )
        bone.roll = float(data.get("roll", 0.0))
        bone.use_connect = bool(data.get("use_connect", parent is not None))
        return bone

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Bone(name={self.name!r}, "
            f"length={self.length:.4f}, "
            f"parent={self.parent.name if self.parent else None})"
        )


# ─── Armature ─────────────────────────────────────────────────────────────────


class Armature:
    """A complete armature (skeleton) for rigging.

    Bones are stored in a flat :attr:`bones` dictionary keyed by name and
    linked via :attr:`Bone.parent` / :attr:`Bone.children` references.
    :attr:`root_bones` tracks bones without a parent.

    Parameters
    ----------
    name : str
        Human-readable identifier for the armature.
    """

    def __init__(self, name: str = "Armature") -> None:
        self.name: str = name
        self.bones: Dict[str, Bone] = {}
        self.root_bones: List[Bone] = []

    # ------------------------------------------------------------------
    # Bone management
    # ------------------------------------------------------------------

    def add_bone(
        self,
        name: str,
        head: Optional[np.ndarray] = None,
        tail: Optional[np.ndarray] = None,
        parent_name: Optional[str] = None,
    ) -> Bone:
        """Add a bone, optionally connecting it to a parent.

        Parameters
        ----------
        name : str
            Unique bone name.
        head : np.ndarray | None
            Head position in parent space (or world space for roots).
        tail : np.ndarray | None
            Tail position in parent space (or world space for roots).
        parent_name : str | None
            Name of the parent bone.  ``None`` creates a root bone.

        Returns
        -------
        Bone
            The newly created bone.
        """
        if name in self.bones:
            raise ValueError(f"Bone '{name}' already exists in armature '{self.name}'.")

        parent: Optional[Bone] = None
        if parent_name is not None:
            parent = self.bones.get(parent_name)
            if parent is None:
                raise KeyError(
                    f"Parent bone '{parent_name}' not found in armature '{self.name}'."
                )

        bone = Bone(name=name, head_pos=head, tail_pos=tail, parent=parent)
        self.bones[name] = bone

        if parent is None:
            self.root_bones.append(bone)

        return bone

    def remove_bone(self, name: str) -> None:
        """Remove a bone and reparent its children to the removed bone's
        parent (or promote them to roots).

        Parameters
        ----------
        name : str
            Name of the bone to remove.

        Raises
        ------
        KeyError
            If the bone does not exist.
        """
        bone = self.bones.pop(name, None)
        if bone is None:
            raise KeyError(f"Bone '{name}' not found in armature '{self.name}'.")

        # Detach from parent.
        if bone.parent is not None:
            bone.parent.children.remove(bone)

        # Reparent children.
        for child in bone.children:
            child.parent = bone.parent
            if bone.parent is not None:
                bone.parent.children.append(child)
            else:
                # Promote to root.
                if child not in self.root_bones:
                    self.root_bones.append(child)

        # Remove from root list if it was a root.
        if bone in self.root_bones:
            self.root_bones.remove(bone)

    def get_bone(self, name: str) -> Optional[Bone]:
        """Look up a bone by name.  Returns ``None`` if not found."""
        return self.bones.get(name)

    def get_chain(self, end_bone_name: str) -> List[Bone]:
        """Return an ordered list of bones from the root ancestor down to
        the named end bone (inclusive).

        Parameters
        ----------
        end_bone_name : str
            Name of the tip bone in the chain.

        Returns
        -------
        list[Bone]
            Bones ordered root → end.  Empty if the bone is not found.
        """
        bone = self.bones.get(end_bone_name)
        if bone is None:
            return []

        chain: List[Bone] = []
        while bone is not None:
            chain.append(bone)
            bone = bone.parent
        chain.reverse()
        return chain

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------

    def solve_ik(
        self,
        end_bone_name: str,
        target_position: np.ndarray,
        pole: Optional[np.ndarray] = None,
        solver: str = "fabrik",
        **kwargs: Any,
    ) -> None:
        """Solve inverse kinematics for a bone chain.

        After solving, bone head / tail positions and transforms are
        updated and :meth:`update_transforms` is called automatically.

        Parameters
        ----------
        end_bone_name : str
            Name of the end-effector bone.
        target_position : np.ndarray, shape (3,)
            Desired world-space position for the end-effector's tail.
        pole : np.ndarray | None
            World-space pole vector (used by ``2bone`` solver to control
            the bend plane).
        solver : str
            One of ``'fabrik'``, ``'ccd'``, ``'2bone'``.
        **kwargs
            Additional keyword arguments forwarded to the solver
            (``tolerances``, ``max_iterations``, ``bend_factor``, etc.).

        Raises
        ------
        ValueError
            If the solver name is unrecognised or the chain is too short.
        """
        chain = self.get_chain(end_bone_name)
        if len(chain) < 2:
            raise ValueError(
                f"IK requires at least 2 bones in the chain; "
                f"got {len(chain)} for '{end_bone_name}'."
            )

        target = np.asarray(target_position, dtype=np.float64)

        # Gather current world-space joint positions.
        # joints[0] = chain root head, joints[i] = chain[i-1].tail
        joints: List[np.ndarray] = [chain[0].world_head]
        for bone in chain:
            joints.append(bone.world_tail)

        # ── Dispatch to the requested solver ──
        if solver == "2bone":
            if len(chain) != 2:
                raise ValueError(
                    f"2-bone IK requires exactly 2 bones in the chain; "
                    f"got {len(chain)}."
                )
            elbow, wrist = solve_2bone_ik(
                joints[0],
                target,
                chain[0].length,
                chain[1].length,
                pole=pole,
                bend_factor=kwargs.get("bend_factor", 0.5),
            )
            solved_joints = [joints[0], elbow, wrist]

        elif solver == "fabrik":
            solved_joints = solve_fabrik(
                joints,
                target,
                tolerances=kwargs.get("tolerances", 0.01),
                max_iterations=kwargs.get("max_iterations", 100),
            )

        elif solver == "ccd":
            solved_joints = solve_ccd(
                joints,
                target,
                tolerances=kwargs.get("tolerances", 0.01),
                max_iterations=kwargs.get("max_iterations", 100),
                angles_limit=kwargs.get("angles_limit", math.pi),
            )

        else:
            raise ValueError(
                f"Unknown IK solver '{solver}'.  "
                f"Choose from 'fabrik', 'ccd', '2bone'."
            )

        # ── Write solved positions back to bones ──
        self._write_ik_chain(chain, solved_joints)
        self.update_transforms()

    def _write_ik_chain(
        self,
        chain: List[Bone],
        joints: List[np.ndarray],
    ) -> None:
        """Update bone head/tail positions from solved joint positions.

        Each *joints[i]* is the world-space head of ``chain[i]``, and
        *joints[i+1]* is the world-space tail.  Positions are
        inverse-transformed into the parent's coordinate system before
        being written to :attr:`Bone.head_pos` / :attr:`Bone.tail_pos`.

        Each bone's :attr:`world_transform` is updated eagerly so that
        subsequent bones in the chain see the correct parent transform.
        """
        for i, bone in enumerate(chain):
            new_world_head = joints[i]
            new_world_tail = joints[i + 1]

            if bone.parent is not None:
                parent_inv = np.linalg.inv(bone.parent.world_transform)
                bone.head_pos = (parent_inv @ np.array([*new_world_head, 1.0]))[:3]
                bone.tail_pos = (parent_inv @ np.array([*new_world_tail, 1.0]))[:3]
            else:
                bone.head_pos = new_world_head.copy()
                bone.tail_pos = new_world_tail.copy()

            bone._rebuild_local_transform()

            # Eagerly update world transform so children see the correct parent.
            if bone.parent is None:
                bone.world_transform = bone.local_transform.copy()
            else:
                bone.world_transform = bone.parent.world_transform @ bone.local_transform

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def update_transforms(self) -> None:
        """Recalculate all bone world transforms from root down.

        Must be called after any structural change (adding / removing
        bones) or after IK solving.
        """
        for root in self.root_bones:
            self._cascade_world(root)

    def _cascade_world(self, bone: Bone) -> None:
        """Depth-first recursion: set world transform then recurse into
        children."""
        if bone.parent is None:
            bone.world_transform = bone.local_transform.copy()
        else:
            bone.world_transform = bone.parent.world_transform @ bone.local_transform

        for child in bone.children:
            self._cascade_world(child)

    # ------------------------------------------------------------------
    # Pose I/O
    # ------------------------------------------------------------------

    def to_pose(self) -> Dict[str, Dict[str, Any]]:
        """Export current bone transforms as a serialisable pose dict.

        Returns
        -------
        dict
            ``{bone_name: {"head_pos": [...], "tail_pos": [...], "roll": float}}``
        """
        pose: Dict[str, Dict[str, Any]] = {}
        for name, bone in self.bones.items():
            pose[name] = {
                "head_pos": bone.head_pos.tolist(),
                "tail_pos": bone.tail_pos.tolist(),
                "roll": bone.roll,
            }
        return pose

    def apply_pose(self, pose_dict: Dict[str, Dict[str, Any]]) -> None:
        """Apply a pose dict (from :meth:`to_pose`) to set bone transforms.

        Only bones present in *pose_dict* are affected; all others retain
        their current state.  After applying, :meth:`update_transforms`
        is called automatically.

        Parameters
        ----------
        pose_dict : dict
            ``{bone_name: {"head_pos": [...], "tail_pos": [...], "roll": float}}``
        """
        for name, data in pose_dict.items():
            bone = self.bones.get(name)
            if bone is None:
                continue
            bone.head_pos = np.array(data["head_pos"], dtype=np.float64)
            bone.tail_pos = np.array(data["tail_pos"], dtype=np.float64)
            bone.roll = float(data.get("roll", bone.roll))
            bone._rebuild_local_transform()
        self.update_transforms()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the entire armature to a JSON-friendly dictionary."""
        return {
            "name": self.name,
            "bones": [bone.to_dict() for bone in self.bones.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Armature":
        """Deserialise an armature from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary produced by :meth:`to_dict`.
        """
        arm = cls(name=data.get("name", "Armature"))

        # First pass: create all bones without parent links.
        bone_map: Dict[str, Bone] = {}
        for bd in data.get("bones", []):
            bone = Bone(
                name=bd["name"],
                head_pos=np.array(bd["head_pos"], dtype=np.float64),
                tail_pos=np.array(bd["tail_pos"], dtype=np.float64),
                parent=None,
            )
            bone.roll = float(bd.get("roll", 0.0))
            bone.use_connect = bool(bd.get("use_connect", False))
            bone_map[bd["name"]] = bone

        # Second pass: relink parents.
        for bd in data.get("bones", []):
            bone = bone_map[bd["name"]]
            parent_name = bd.get("parent_name")
            if parent_name is not None and parent_name in bone_map:
                parent = bone_map[parent_name]
                bone.parent = parent
                parent.children.append(bone)
                bone.use_connect = bool(bd.get("use_connect", True))
            else:
                arm.root_bones.append(bone)

        arm.bones = bone_map
        arm.update_transforms()
        return arm

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def bone_count(self) -> int:
        """Return the total number of bones."""
        return len(self.bones)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Armature(name={self.name!r}, "
            f"bones={self.bone_count()}, "
            f"roots={len(self.root_bones)})"
        )


# ─── Skin Weights ─────────────────────────────────────────────────────────────


class SkinWeights:
    """Vertex skin weights for binding a mesh to an armature.

    Each vertex may be influenced by up to *max_influences* bones.  The
    weights for each influence are stored alongside the corresponding
    bone index.  Weights are normalised per-vertex so they sum to 1.0.

    Parameters
    ----------
    vertex_count : int
        Number of mesh vertices.
    max_influences : int
        Maximum number of bones that may influence a single vertex
        (typically 4 for GPU skinned meshes).

    Attributes
    ----------
    weights : np.ndarray, shape (vertex_count, max_influences)
        Per-vertex, per-influence weight values.
    bone_indices : np.ndarray, shape (vertex_count, max_influences)
        Per-vertex, per-influence bone index.  An index of ``0`` with a
        weight of ``0`` means "no influence".
    """

    def __init__(self, vertex_count: int = 0, max_influences: int = 4) -> None:
        self.max_influences: int = max_influences
        self.weights: np.ndarray = np.zeros(
            (vertex_count, max_influences), dtype=np.float64
        )
        self.bone_indices: np.ndarray = np.zeros(
            (vertex_count, max_influences), dtype=np.int32
        )

    # ------------------------------------------------------------------
    # Weight access
    # ------------------------------------------------------------------

    def set_weight(self, vertex_idx: int, bone_idx: int, weight: float) -> None:
        """Assign *weight* to a specific bone for a vertex.

        The weight is placed in the first slot that is either zero or
        already assigned to *bone_idx*.  If all slots are occupied by
        other bones the assignment is silently dropped.

        Parameters
        ----------
        vertex_idx : int
        bone_idx : int
        weight : float
        """
        if vertex_idx < 0 or vertex_idx >= self.weights.shape[0]:
            raise IndexError(
                f"vertex_idx {vertex_idx} out of range [0, {self.weights.shape[0]})"
            )

        # Reuse an existing slot for the same bone, or find the first
        # zero-weight slot.
        for k in range(self.max_influences):
            if self.bone_indices[vertex_idx, k] == bone_idx:
                self.weights[vertex_idx, k] = weight
                return
            if self.weights[vertex_idx, k] < 1e-12:
                self.bone_indices[vertex_idx, k] = bone_idx
                self.weights[vertex_idx, k] = weight
                return
        # All slots full — drop the assignment silently.

    def get_weights(self, vertex_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (weights, bone_indices) arrays for a single vertex.

        Parameters
        ----------
        vertex_idx : int

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(weights, bone_indices)`` each of shape ``(max_influences,)``.
        """
        return self.weights[vertex_idx].copy(), self.bone_indices[vertex_idx].copy()

    def normalize_weights(self) -> None:
        """Normalise weights per vertex so they sum to 1.0.

        Vertices whose total weight is below ``1e-10`` are left unchanged
        (all-zero influence).
        """
        row_sums = self.weights.sum(axis=1, keepdims=True)
        mask = (row_sums > 1e-10).flatten()
        self.weights[mask] /= row_sums[mask]

    # ------------------------------------------------------------------
    # Skinning
    # ------------------------------------------------------------------

    def apply(
        self,
        vertices: np.ndarray,
        bone_transforms: List[np.ndarray],
    ) -> np.ndarray:
        """Apply linear blend skinning to a vertex array.

        Each vertex is transformed by the weighted sum of its influencing
        bone transforms.

        Parameters
        ----------
        vertices : np.ndarray, shape (N, 3)
            Mesh vertex positions in **bind-pose** space.
        bone_transforms : list[np.ndarray]
            One 4x4 matrix per bone.  Typically these are
            ``bone_world_pose @ inverse(bone_world_bind_pose)`` but the
            caller may supply any transform; this function simply
            performs the weighted blend.

        Returns
        -------
        np.ndarray, shape (N, 3)
            The skinned (deformed) vertex positions.
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        n_verts = vertices.shape[0]
        result = np.zeros((n_verts, 3), dtype=np.float64)

        if not bone_transforms or n_verts == 0:
            return result

        # Stack transforms into a (num_bones, 4, 4) tensor.
        all_transforms = np.array(bone_transforms, dtype=np.float64)
        num_bones = all_transforms.shape[0]

        # Homogeneous vertex coordinates: (N, 4)
        v_homo = np.hstack([vertices, np.ones((n_verts, 1), dtype=np.float64)])

        for k in range(self.max_influences):
            w = self.weights[:, k]  # (N,)
            bi = self.bone_indices[:, k]  # (N,) bone indices

            # Clamp indices to valid range.
            bi_safe = np.clip(bi, 0, num_bones - 1)
            mask = (w > 1e-10) & (bi < num_bones)  # only active influences

            if not np.any(mask):
                continue

            # Gather per-vertex transforms: (N, 4, 4)
            v_transforms = all_transforms[bi_safe]  # (N, 4, 4)

            # Batch matrix-vector multiply: (N, 4, 4) × (N, 4) → (N, 4)
            v_transformed = np.einsum("nij,nj->ni", v_transforms, v_homo)[
                :, :3
            ]  # (N, 3)

            result += (w[:, np.newaxis] * v_transformed) * mask[:, np.newaxis]

        return result

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "vertex_count": int(self.weights.shape[0]),
            "max_influences": self.max_influences,
            "weights": self.weights.tolist(),
            "bone_indices": self.bone_indices.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkinWeights":
        """Deserialise from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary produced by :meth:`to_dict`.
        """
        sw = cls(
            vertex_count=int(data["vertex_count"]),
            max_influences=int(data["max_influences"]),
        )
        sw.weights = np.array(data["weights"], dtype=np.float64)
        sw.bone_indices = np.array(data["bone_indices"], dtype=np.int32)
        return sw

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SkinWeights(vertices={self.weights.shape[0]}, "
            f"max_influences={self.max_influences})"
        )


# ─── Humanoid Factory ────────────────────────────────────────────────────────


def create_humanoid_armature(height: float = 1.8, name: str = "Humanoid") -> Armature:
    """Create a basic humanoid skeleton with standard bone names.

    The skeleton is built in a T-pose at the given *height*.  All
    proportions are scaled uniformly.

    Bone names follow the typical game-engine convention::

        Hips → Spine → Chest → Neck → Head
              ├─ LeftShoulder → LeftArm → LeftForeArm → LeftHand
              ├─ RightShoulder → RightArm → RightForeArm → RightHand
              ├─ LeftUpLeg → LeftLeg → LeftFoot
              └─ RightUpLeg → RightLeg → RightFoot

    Parameters
    ----------
    height : float
        Total character height in metres (default 1.8).
    name : str
        Name for the armature.

    Returns
    -------
    Armature
        A fully constructed humanoid armature with world transforms
        already computed.
    """
    s = height / 1.8  # uniform scale factor

    # All positions defined in world space, then converted to parent-local.
    bone_defs: Dict[str, Tuple[Tuple[float, float, float],
                                Tuple[float, float, float],
                                Optional[str]]] = {
        # Spine chain
        "Hips":          ((0.0,  1.00 * s, 0.0), (0.0,  1.05 * s, 0.0), None),
        "Spine":         ((0.0,  1.05 * s, 0.0), (0.0,  1.25 * s, 0.0), "Hips"),
        "Chest":         ((0.0,  1.25 * s, 0.0), (0.0,  1.42 * s, 0.0), "Spine"),
        "Neck":          ((0.0,  1.42 * s, 0.0), (0.0,  1.52 * s, 0.0), "Chest"),
        "Head":          ((0.0,  1.52 * s, 0.0), (0.0,  1.72 * s, 0.0), "Neck"),
        # Left arm
        "LeftShoulder":  ((0.0,  1.40 * s, 0.0), (-0.19 * s, 1.40 * s, 0.0), "Chest"),
        "LeftArm":       ((-0.19 * s, 1.40 * s, 0.0), (-0.19 * s, 1.10 * s, 0.0), "LeftShoulder"),
        "LeftForeArm":   ((-0.19 * s, 1.10 * s, 0.0), (-0.19 * s, 0.84 * s, 0.0), "LeftArm"),
        "LeftHand":      ((-0.19 * s, 0.84 * s, 0.0), (-0.19 * s, 0.74 * s, 0.0), "LeftForeArm"),
        # Right arm
        "RightShoulder": ((0.0,  1.40 * s, 0.0), (0.19 * s, 1.40 * s, 0.0), "Chest"),
        "RightArm":      ((0.19 * s, 1.40 * s, 0.0), (0.19 * s, 1.10 * s, 0.0), "RightShoulder"),
        "RightForeArm":  ((0.19 * s, 1.10 * s, 0.0), (0.19 * s, 0.84 * s, 0.0), "RightArm"),
        "RightHand":     ((0.19 * s, 0.84 * s, 0.0), (0.19 * s, 0.74 * s, 0.0), "RightForeArm"),
        # Left leg
        "LeftUpLeg":     ((-0.09 * s, 1.00 * s, 0.0), (-0.09 * s, 0.55 * s, 0.0), "Hips"),
        "LeftLeg":       ((-0.09 * s, 0.55 * s, 0.0), (-0.09 * s, 0.10 * s, 0.0), "LeftUpLeg"),
        "LeftFoot":      ((-0.09 * s, 0.10 * s, 0.0), (-0.09 * s, 0.02 * s, 0.12 * s), "LeftLeg"),
        # Right leg
        "RightUpLeg":    ((0.09 * s, 1.00 * s, 0.0), (0.09 * s, 0.55 * s, 0.0), "Hips"),
        "RightLeg":      ((0.09 * s, 0.55 * s, 0.0), (0.09 * s, 0.10 * s, 0.0), "RightUpLeg"),
        "RightFoot":     ((0.09 * s, 0.10 * s, 0.0), (0.09 * s, 0.02 * s, 0.12 * s), "RightLeg"),
    }

    arm = Armature(name=name)

    # Topological sort: parents before children.
    visited: set = set()
    order: List[str] = []

    def _topo(name: str) -> None:
        if name in visited:
            return
        parent_name = bone_defs[name][2]
        if parent_name is not None:
            _topo(parent_name)
        visited.add(name)
        order.append(name)

    for bname in bone_defs:
        _topo(bname)

    # Create bones in topological order so parent world transforms are
    # available when converting world positions to parent-local.
    # Each bone's world_transform is set eagerly so children can rely on it.
    for bname in order:
        w_head, w_tail, parent_name = bone_defs[bname]
        head_world = vec3(*w_head)
        tail_world = vec3(*w_tail)

        if parent_name is None:
            bone = arm.add_bone(bname, head=head_world, tail=tail_world)
            # Root bone: world = local.
            bone.world_transform = bone.local_transform.copy()
        else:
            parent = arm.bones[parent_name]
            parent_inv = np.linalg.inv(parent.world_transform)
            local_head = (parent_inv @ np.array([*head_world, 1.0]))[:3]
            local_tail = (parent_inv @ np.array([*tail_world, 1.0]))[:3]
            bone = arm.add_bone(
                bname, head=local_head, tail=local_tail, parent_name=parent_name
            )
            # Child bone: world = parent.world @ local.
            bone.world_transform = parent.world_transform @ bone.local_transform

    return arm


# ─── Automatic Skin Weight Generation ────────────────────────────────────────


def auto_skin_weights(
    mesh: Any,
    armature: Armature,
    max_distance: float = 1.0,
    max_influences: int = 4,
) -> SkinWeights:
    """Generate automatic skin weights based on proximity to bones.

    For each mesh vertex the Euclidean distance to every bone segment
    (world_head → world_tail) is computed.  Bones within *max_distance*
    are weighted by inverse distance, keeping only the top
    *max_influences* bones per vertex.

    **Fallback**: vertices farther than *max_distance* from all bones are
    assigned to the single nearest bone, ensuring no vertex is left
    unweighted (which would pin it at the origin during skinning).

    Parameters
    ----------
    mesh : Any
        Must expose a ``vertices`` attribute that is an
        ``np.ndarray`` of shape ``(N, 3)``.
    armature : Armature
        The skeleton to bind to.
    max_distance : float
        Maximum influence radius in world units (default 1.0 — safe for
        humanoid characters up to ~2m tall).
    max_influences : int
        Per-vertex bone influence limit.

    Returns
    -------
    SkinWeights
        Normalised skin weights ready for :meth:`SkinWeights.apply`.
    """
    import logging
    _log = logging.getLogger("nexus3d.rigging.armature.auto_skin_weights")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    n_verts = vertices.shape[0]

    # Ensure the armature has up-to-date world transforms.
    armature.update_transforms()

    bone_list = list(armature.bones.values())
    n_bones = len(bone_list)

    # Pre-compute world-space head/tail for each bone.
    bone_heads = np.array([b.world_head for b in bone_list], dtype=np.float64)  # (B, 3)
    bone_tails = np.array([b.world_tail for b in bone_list], dtype=np.float64)  # (B, 3)

    skin = SkinWeights(vertex_count=n_verts, max_influences=max_influences)
    isolated_count = 0

    for vi in range(n_verts):
        v = vertices[vi]

        # Compute distance from vertex to each bone segment.
        dists: List[Tuple[int, float]] = []
        for bi in range(n_bones):
            closest = closest_point_on_segment(v, bone_heads[bi], bone_tails[bi])
            d = float(magnitude(v - closest))
            if d < max_distance:
                dists.append((bi, d))

        if not dists:
            # ── Fallback: assign nearest bone even if beyond max_distance ──
            isolated_count += 1
            nearest_bi = 0
            nearest_d = float("inf")
            for bi in range(n_bones):
                closest = closest_point_on_segment(v, bone_heads[bi], bone_tails[bi])
                d = float(magnitude(v - closest))
                if d < nearest_d:
                    nearest_d = d
                    nearest_bi = bi
            skin.set_weight(vi, nearest_bi, 1.0)
            continue

        # Sort by distance (closest first).
        dists.sort(key=lambda x: x[1])

        # Take the top influences and apply inverse-distance weighting.
        selected = dists[:max_influences]
        inv_dists = [1.0 / (d + 1e-10) for _, d in selected]
        total = sum(inv_dists)
        weights = [w / total for w in inv_dists]

        for k, (bi, _) in enumerate(selected):
            skin.set_weight(vi, bi, weights[k])

    if isolated_count > 0:
        _log.warning(
            "%d / %d vertices had no bones within max_distance=%.1f — "
            "assigned to nearest bone (may cause minor deformation artifacts)",
            isolated_count, n_verts, max_distance,
        )

    skin.normalize_weights()
    return skin
