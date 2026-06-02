"""
Nexus3D Texture and Material Pipeline
=====================================
Production-grade procedural texture generation, PBR material system,
UV mapping utilities, and image-based lighting for the Nexus3D engine.
"""

from .pipeline import (
    # --- Perlin Noise ---
    PerlinNoise,
    # --- Texture Base ---
    Texture,
    # --- Procedural Textures ---
    NoiseTexture,
    CheckerTexture,
    BrickTexture,
    WoodTexture,
    MarbleTexture,
    GradientTexture,
    VoronoiTexture,
    ImageTexture,
    # --- Material Layering ---
    MaterialLayer,
    LayeredMaterial,
    # --- UV Mapping ---
    UVMapper,
    # --- PBR Material Library ---
    MaterialLibrary,
    # --- HDR Environment / IBL ---
    EnvironmentMap,
)

__all__ = [
    "PerlinNoise",
    "Texture",
    "NoiseTexture",
    "CheckerTexture",
    "BrickTexture",
    "WoodTexture",
    "MarbleTexture",
    "GradientTexture",
    "VoronoiTexture",
    "ImageTexture",
    "MaterialLayer",
    "LayeredMaterial",
    "UVMapper",
    "MaterialLibrary",
    "EnvironmentMap",
]
