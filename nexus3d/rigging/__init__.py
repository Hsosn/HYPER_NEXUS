"""Nexus3D Rigging — skeletal animation primitives.

Re-exports the core rigging types for convenient top-level imports::

    from nexus3d.rigging import Armature, Bone, SkinWeights
"""

from nexus3d.rigging.armature import (
    Bone,
    Armature,
    SkinWeights,
    create_humanoid_armature,
    auto_skin_weights,
)

__all__ = [
    "Bone",
    "Armature",
    "SkinWeights",
    "create_humanoid_armature",
    "auto_skin_weights",
]
