"""Nexus3D Rendering Pipeline.

Provides headless rendering of 3D scenes to images with automatic fallback
from PyRender (OSMesa) to a pure-Python software rasterizer, and a
production-grade Monte Carlo path tracer for physically-based global illumination.

Exported classes (rasteriser):
    Material              – PBR material definition with serialisation.
    Light                 – Point, directional, spot, and area light sources.
    Camera                – Virtual camera with perspective / orthographic projection.
    Renderer              – Headless renderer (dual-backend: PyRender / software).
    SoftwareRasterizer    – Pure-Python/NumPy triangle rasterizer fallback.

Exported classes (path tracer):
    PathTracer            – Monte Carlo path tracer with multi-bounce GI.
    PTMaterial            – Physically-based material for the path tracer.
    AreaLight             – Rectangular area light source.
    BVH                   – Bounding Volume Hierarchy acceleration structure.
"""

from nexus3d.rendering.renderer import (
    Camera,
    Light,
    Material,
    Renderer,
    SoftwareRasterizer,
)
from nexus3d.rendering.pathtracer import (
    PathTracer,
    PTMaterial,
    AreaLight,
    BVH,
)

__all__ = [
    # Rasteriser
    "Material",
    "Light",
    "Camera",
    "Renderer",
    "SoftwareRasterizer",
    # Path tracer
    "PathTracer",
    "PTMaterial",
    "AreaLight",
    "BVH",
]
