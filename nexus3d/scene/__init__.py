"""Nexus3D Scene Graph package.

Manages the hierarchical 3D scene tree, node transforms, and scene-level
serialization. The graph is designed for headless (rendering-independent)
use by AI agents.
"""

from nexus3d.scene.graph import SceneNode, Scene

__all__ = ["SceneNode", "Scene"]
