"""Nexus3D Animation System package.

Re-exports every public class and function from :mod:`nexus3d.animation.anim`
so that users can write::

    from nexus3d.animation import AnimCurve, AnimationClip, AnimMixer
"""

from nexus3d.animation.anim import (
    # Core curve / track / clip
    AnimCurve,
    AnimTrack,
    AnimationClip,
    # Mixing
    AnimMixer,
    # Pose library
    PoseLibrary,
    # Procedural animation
    ProceduralAnimation,
    # Export
    AnimationExporter,
)

__all__ = [
    "AnimCurve",
    "AnimTrack",
    "AnimationClip",
    "AnimMixer",
    "PoseLibrary",
    "ProceduralAnimation",
    "AnimationExporter",
]
