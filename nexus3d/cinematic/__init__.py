"""Nexus3D Cinematic Camera System.

Professional-grade virtual cinematography tools including animated camera paths,
depth of field, motion blur, camera shake, exposure control, and lens profiles.
"""

from .camera import (
    CinematicCamera,
    CameraPath,
    CameraShake,
    DepthOfField,
    MotionBlur,
    CameraPresets,
    Exposure,
    LensProfile,
    PerlinNoise,
    ease_in_out,
    ease_in,
    ease_out,
    ease_in_cubic,
    ease_out_cubic,
    ease_in_out_cubic,
    ease_in_quad,
    ease_out_quad,
    ease_in_out_quad,
)

__all__ = [
    "CinematicCamera",
    "CameraPath",
    "CameraShake",
    "DepthOfField",
    "MotionBlur",
    "CameraPresets",
    "Exposure",
    "LensProfile",
    "PerlinNoise",
    "ease_in_out",
    "ease_in",
    "ease_out",
    "ease_in_cubic",
    "ease_out_cubic",
    "ease_in_out_cubic",
    "ease_in_quad",
    "ease_out_quad",
    "ease_in_out_quad",
]
