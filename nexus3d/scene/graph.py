"""Nexus3D Scene Graph.

Provides the hierarchical scene structure for the headless 3D engine.
Every object that exists in the 3D world is represented as a :class:`SceneNode`
attached to a tree rooted at a :class:`Scene`.

Mathematical convention
-----------------------
* Quaternion layout: ``[w, x, y, z]``
* Transform matrices are row-major 4x4 NumPy arrays (:class:`numpy.ndarray`
  with ``dtype=np.float64``).
* The local transform is built as **T · R · S** (translate, then rotate, then
  scale).
* ``look_at`` orients the node's **-Z axis** toward the target (OpenGL
  convention).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from nexus3d.math3d.core import (
    compose_matrix,
    cross,
    dot,
    magnitude,
    mat4_identity,
    normalize,
    normalize_quat,
    quat_identity,
    quat_multiply,
    quat_to_rotation_matrix,
    rotation_matrix_to_quat,
    vec3,
)

# ──────────────────────────────────────────────────────────────────────────────
# SceneNode
# ──────────────────────────────────────────────────────────────────────────────


class SceneNode:
    """A single node in the hierarchical scene graph.

    Each node stores a **local** position, rotation (quaternion), and scale.
    The local 4x4 transform matrix is recomputed automatically whenever any of
    these properties change.  World-space transforms are derived on demand by
    walking up the parent chain.

    Parameters
    ----------
    name:
        Human-readable identifier for the node.  Must be unique among siblings
        when the node is attached to a :class:`Scene`.
    """

    # ── Construction ──────────────────────────────────────────────────────

    def __init__(self, name: str = "Node") -> None:
        self.name: str = name

        # Hierarchy
        self.parent: Optional[SceneNode] = None
        self.children: List[SceneNode] = []

        # Local-space TRS properties
        self._position: np.ndarray = vec3(0.0, 0.0, 0.0)
        self._rotation: np.ndarray = quat_identity()
        self._scale: np.ndarray = vec3(1.0, 1.0, 1.0)

        # Cached composed matrices
        self._local_transform: np.ndarray = mat4_identity()
        self._world_transform: Optional[np.ndarray] = None

        # Visibility flag – can be toggled without removing the node
        self.visible: bool = True

        # Arbitrary user-owned payload (materials, physics bodies, …)
        self.user_data: Dict[str, Any] = {}

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def position(self) -> np.ndarray:
        """Local-space position (read-only copy)."""
        return self._position.copy()

    @property
    def rotation(self) -> np.ndarray:
        """Local-space rotation quaternion ``[w, x, y, z]`` (read-only copy)."""
        return self._rotation.copy()

    @property
    def scale(self) -> np.ndarray:
        """Local-space scale (read-only copy)."""
        return self._scale.copy()

    @property
    def local_transform(self) -> np.ndarray:
        """Composed local 4x4 transform (T · R · S)."""
        return self._local_transform.copy()

    @property
    def world_transform(self) -> np.ndarray:
        """World-space 4x4 transform, computed on demand."""
        return self.get_world_transform()

    # ── Private helpers ───────────────────────────────────────────────────

    def _compose_local_transform(self) -> None:
        """Rebuild the cached 4x4 local transform from TRS properties."""
        self._local_transform = compose_matrix(
            self._position, self._rotation, self._scale
        )
        # Invalidate cached world transform
        self._world_transform = None

    def _invalidate_child_world_transforms(self) -> None:
        """Recursively clear cached world transforms for every descendant."""
        self._world_transform = None
        for child in self.children:
            child._invalidate_child_world_transforms()

    # ── Hierarchy ─────────────────────────────────────────────────────────

    def add_child(self, child: SceneNode) -> None:
        """Attach *child* to this node.

        If the child already has a parent it is first detached.  Cycles are
        rejected with a :class:`ValueError`.

        Raises
        ------
        ValueError
            If attaching *child* would introduce a cycle in the graph.
        """
        if child is self:
            raise ValueError("Cannot add a node as its own child.")

        # Detect cycles by walking up from the proposed parent.
        ancestor: Optional[SceneNode] = self
        while ancestor is not None:
            if ancestor is child:
                raise ValueError(
                    f"Adding '{child.name}' as child of '{self.name}' "
                    "would create a cycle in the scene graph."
                )
            ancestor = ancestor.parent

        # Detach from previous parent.
        if child.parent is not None:
            child.parent.remove_child(child)

        child.parent = self
        self.children.append(child)
        child._invalidate_child_world_transforms()

    def remove_child(self, child: SceneNode) -> None:
        """Detach *child* from this node.

        Raises
        ------
        ValueError
            If *child* is not a direct child of this node.
        """
        if child not in self.children:
            raise ValueError(
                f"'{child.name}' is not a child of '{self.name}'."
            )
        self.children.remove(child)
        child.parent = None
        child._invalidate_child_world_transforms()

    # ── Transform queries ─────────────────────────────────────────────────

    def get_world_transform(self) -> np.ndarray:
        """Return the world-space 4x4 transform of this node.

        The result is computed by walking up the parent chain and multiplying
        all local transforms together (child × parent × grandparent × …).

        Returns
        -------
        numpy.ndarray
            A ``4×4`` matrix in world space.
        """
        if self._world_transform is not None:
            return self._world_transform

        # Gather local transforms from this node up to the root.
        chain: List[np.ndarray] = []
        node: Optional[SceneNode] = self
        while node is not None:
            chain.append(node._local_transform)
            node = node.parent

        # Multiply: root · … · parent · self  (top-down order).
        result = chain[-1]
        for mat in reversed(chain[:-1]):
            result = result @ mat

        self._world_transform = result
        return result

    # ── Absolute transform setters ────────────────────────────────────────

    def set_position(self, pos: Any) -> None:
        """Set the local-space position.

        Parameters
        ----------
        pos:
            An array-like of length 3 ``(x, y, z)``.
        """
        self._position = np.asarray(pos, dtype=np.float64).reshape(3)
        self._compose_local_transform()
        self._invalidate_child_world_transforms()

    def set_rotation(self, q: Any) -> None:
        """Set the local-space rotation quaternion.

        Parameters
        ----------
        q:
            An array-like of length 4 in ``[w, x, y, z]`` order.
        """
        self._rotation = normalize_quat(np.asarray(q, dtype=np.float64).reshape(4))
        self._compose_local_transform()
        self._invalidate_child_world_transforms()

    def set_scale(self, s: Any) -> None:
        """Set the local-space scale.

        Parameters
        ----------
        s:
            An array-like of length 3 or a single scalar.
        """
        s = np.asarray(s, dtype=np.float64)
        if s.ndim == 0:
            s = np.full(3, float(s))
        else:
            s = s.reshape(3)
        self._scale = s
        self._compose_local_transform()
        self._invalidate_child_world_transforms()

    # ── Relative transform operations ─────────────────────────────────────

    def translate(self, delta: Any) -> None:
        """Move the node by *delta* in local space.

        Parameters
        ----------
        delta:
            An array-like of length 3 ``(dx, dy, dz)``.
        """
        delta = np.asarray(delta, dtype=np.float64).reshape(3)
        # Rotate the delta into local space and add to position.
        rot_mat = quat_to_rotation_matrix(self._rotation)
        world_delta = rot_mat @ delta
        self._position += world_delta
        self._compose_local_transform()
        self._invalidate_child_world_transforms()

    def rotate_by(self, q: Any) -> None:
        """Apply an additional rotation (local space).

        The rotation is applied **after** the existing rotation:
        ``new_rotation = existing × q``.

        Parameters
        ----------
        q:
            An array-like of length 4 in ``[w, x, y, z]`` order.
        """
        q = normalize_quat(np.asarray(q, dtype=np.float64).reshape(4))
        self._rotation = normalize_quat(quat_multiply(self._rotation, q))
        self._compose_local_transform()
        self._invalidate_child_world_transforms()

    # ── Look-at ───────────────────────────────────────────────────────────

    def look_at(self, target: Any, up: Optional[Any] = None) -> None:
        """Orient the node so that its **-Z axis** points toward *target*.

        This follows the standard OpenGL / right-hand convention.  The node's
        position is **not** changed – only the rotation is adjusted.

        Parameters
        ----------
        target:
            World-space point to look at (array-like of length 3).
        up:
            World-space up hint.  Defaults to ``+Y``.
        """
        target = np.asarray(target, dtype=np.float64).reshape(3)
        if up is None:
            up = vec3(0.0, 1.0, 0.0)
        else:
            up = np.asarray(up, dtype=np.float64).reshape(3)

        forward = target - self._position
        fwd_len = magnitude(forward)
        if fwd_len < 1e-10:
            return  # Already at target; nothing to do.
        forward = forward / fwd_len

        right = normalize(cross(forward, up))
        if magnitude(right) < 1e-10:
            # forward is parallel to up – pick an arbitrary right vector.
            fallback = vec3(1.0, 0.0, 0.0)
            if abs(dot(forward, fallback)) > 0.99:
                fallback = vec3(0.0, 0.0, 1.0)
            right = normalize(cross(forward, fallback))

        true_up = cross(right, forward)

        # Build a 3×3 rotation matrix with -Z as forward (OpenGL convention).
        # Columns represent where the local axes map to in parent space.
        rot_mat = np.eye(3, dtype=np.float64)
        rot_mat[:, 0] = right        # local +X → right
        rot_mat[:, 1] = true_up      # local +Y → true_up
        rot_mat[:, 2] = -forward     # local +Z → -forward (so -Z → target)

        self._rotation = normalize_quat(rotation_matrix_to_quat(rot_mat))
        self._compose_local_transform()
        self._invalidate_child_world_transforms()

    # ── Tree traversal ────────────────────────────────────────────────────

    def find(self, name: str) -> Optional[SceneNode]:
        """Find the first descendant whose ``name`` matches *name*.

        The search is depth-first and visits this node's children recursively.

        Parameters
        ----------
        name:
            The name to search for.

        Returns
        -------
        SceneNode or None
        """
        for child in self.children:
            if child.name == name:
                return child
            result = child.find(name)
            if result is not None:
                return result
        return None

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the node and its entire subtree to a plain dict.

        Returns
        -------
        dict
            A JSON-serializable dictionary representation.
        """
        return {
            "name": self.name,
            "position": self._position.tolist(),
            "rotation": self._rotation.tolist(),
            "scale": self._scale.tolist(),
            "visible": self.visible,
            "user_data": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.user_data.items()
            },
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SceneNode:
        """Reconstruct a node (and its subtree) from a dict.

        Parameters
        ----------
        data:
            A dictionary previously produced by :meth:`to_dict`.

        Returns
        -------
        SceneNode
        """
        node = cls(name=data.get("name", "Node"))
        node.set_position(data.get("position", [0.0, 0.0, 0.0]))
        node.set_rotation(data.get("rotation", [1.0, 0.0, 0.0, 0.0]))
        node.set_scale(data.get("scale", [1.0, 1.0, 1.0]))
        node.visible = data.get("visible", True)
        node.user_data = data.get("user_data", {})

        for child_data in data.get("children", []):
            child = cls.from_dict(child_data)
            node.add_child(child)

        return node

    # ── Dunder ────────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        parent_name = self.parent.name if self.parent else "None"
        return (
            f"SceneNode(name={self.name!r}, parent={parent_name!r}, "
            f"children={len(self.children)}, visible={self.visible})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Scene
# ──────────────────────────────────────────────────────────────────────────────


class Scene:
    """Root-level container for a 3D scene graph.

    The :class:`Scene` owns a root :class:`SceneNode` and maintains a flat
    name → node lookup dictionary for fast access.  All top-level nodes added
    via :meth:`add_node` become children of the internal root.

    Parameters
    ----------
    name:
        Human-readable name for the scene.
    """

    def __init__(self, name: str = "Scene") -> None:
        self.name: str = name
        self._root: SceneNode = SceneNode("__scene_root__")
        self._nodes: Dict[str, SceneNode] = {}

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def root(self) -> SceneNode:
        """The hidden root node of the scene (read-only)."""
        return self._root

    # ── Node management ───────────────────────────────────────────────────

    def add_node(self, node: SceneNode) -> None:
        """Add a node to the scene.

        The node becomes a child of the scene root.  If a node with the same
        name already exists a :class:`ValueError` is raised.

        Parameters
        ----------
        node:
            The :class:`SceneNode` to add.

        Raises
        ------
        ValueError
            If a node with the same name is already registered.
        """
        if node.name in self._nodes:
            raise ValueError(
                f"A node named '{node.name}' already exists in scene '{self.name}'."
            )
        self._nodes[node.name] = node
        self._root.add_child(node)

    def remove_node(self, name: str) -> None:
        """Remove a node from the scene by name.

        The node is detached from the scene root **and** removed from the
        flat name registry.  All descendants remain attached to the removed
        node.

        Parameters
        ----------
        name:
            Name of the node to remove.

        Raises
        ------
        KeyError
            If no node with *name* is found.
        """
        if name not in self._nodes:
            raise KeyError(f"No node named '{name}' in scene '{self.name}'.")
        node = self._nodes.pop(name)
        self._root.remove_child(node)

    def get_node(self, name: str) -> SceneNode:
        """Retrieve a node by name.

        This only searches the flat registry (top-level children of the scene
        root).  For deeper searches use :meth:`SceneNode.find`.

        Parameters
        ----------
        name:
            Name of the node.

        Returns
        -------
        SceneNode

        Raises
        ------
        KeyError
            If no node with *name* is found.
        """
        if name not in self._nodes:
            raise KeyError(f"No node named '{name}' in scene '{self.name}'.")
        return self._nodes[name]

    # ── Transform propagation ─────────────────────────────────────────────

    def update_world_transforms(self) -> None:
        """Recompute cached world transforms for **every** node in the scene.

        Performs a depth-first traversal from the root, multiplying each
        node's local transform with its parent's world transform.
        """
        self._update_subtree(self._root, mat4_identity())

    def _update_subtree(
        self, node: SceneNode, parent_world: np.ndarray
    ) -> None:
        """Recursively propagate world transforms through the subtree."""
        node._world_transform = parent_world @ node._local_transform
        for child in node.children:
            self._update_subtree(child, node._world_transform)

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the scene to a plain dict.

        Returns
        -------
        dict
            A JSON-serializable dictionary.
        """
        return {
            "name": self.name,
            "nodes": [node.to_dict() for node in self._root.children],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Scene:
        """Reconstruct a scene from a dict.

        Parameters
        ----------
        data:
            A dictionary previously produced by :meth:`to_dict`.

        Returns
        -------
        Scene
        """
        scene = cls(name=data.get("name", "Scene"))
        for node_data in data.get("nodes", []):
            node = SceneNode.from_dict(node_data)
            scene.add_node(node)
        return scene

    # ── Dunder ────────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Scene(name={self.name!r}, nodes={list(self._nodes.keys())})"
        )

    def __len__(self) -> int:
        """Return the number of top-level nodes."""
        return len(self._nodes)
