"""
Product Renderer — Professional Product Visualization Studio
=============================================================
Provides automated studio lighting (3-point + HDR), camera
framing, turntable animation, material assignment, and render
preset management for photorealistic product renders.
"""

from __future__ import annotations

import json
import math
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class RenderQuality(Enum):
    DRAFT = "draft"
    PREVIEW = "preview"
    FINAL = "final"
    PRODUCTION = "production"


class LightingPreset(Enum):
    STUDIO_3POINT = "studio_3point"
    STUDIO_SOFT = "studio_soft"
    RIM = "rim"
    DRAMATIC = "dramatic"
    FLAT = "flat"
    GOLDEN_HOUR = "golden_hour"
    NIGHT = "night"
    HDR_ENVIRONMENT = "hdr_environment"


class CameraFraming(Enum):
    PERSPECTIVE = "perspective"
    ORTHOGRAPHIC = "orthographic"
    ISOMETRIC = "isometric"
    CLOSE_UP = "close_up"
    MACRO = "macro"


QUALITY_SETTINGS = {
    RenderQuality.DRAFT: {"samples": 64, "bounces": 2, "resolution": 512},
    RenderQuality.PREVIEW: {"samples": 256, "bounces": 4, "resolution": 1024},
    RenderQuality.FINAL: {"samples": 1024, "bounces": 8, "resolution": 2048},
    RenderQuality.PRODUCTION: {"samples": 4096, "bounces": 16, "resolution": 4096},
}


@dataclass
class Light:
    """A light source in the studio."""
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = "light"
    light_type: str = "point"       # point | directional | spot | area | hdr
    position: Tuple[float, float, float] = (0.0, 5.0, 0.0)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    size: float = 0.5               # for area lights
    angle: float = math.radians(30)  # for spot lights
    falloff: float = 2.0
    cast_shadows: bool = True
    visible: bool = True


@dataclass
class Camera:
    """Camera setup for product rendering."""
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = "camera"
    position: Tuple[float, float, float] = (3.0, 2.0, 3.0)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov: float = math.radians(45)
    focal_length: float = 50.0      # mm equivalent
    aperture: float = 2.8
    dof_enabled: bool = False
    dof_distance: float = 3.0
    orthographic_scale: float = 2.0

    @property
    def direction(self) -> Tuple[float, float, float]:
        dx = self.target[0] - self.position[0]
        dy = self.target[1] - self.position[1]
        dz = self.target[2] - self.position[2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d == 0:
            return (0.0, 0.0, -1.0)
        return (dx / d, dy / d, dz / d)

    def orbit(self, theta: float, phi: float, radius: float) -> None:
        """Orbit camera around target by spherical coordinates."""
        self.position = (
            self.target[0] + radius * math.sin(theta) * math.cos(phi),
            self.target[1] + radius * math.sin(phi),
            self.target[2] + radius * math.cos(theta) * math.cos(phi),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Studio Setup
# ---------------------------------------------------------------------------

class StudioSetup:
    """Configure studio environment for product rendering."""

    @staticmethod
    def create_studio_3point(center: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> List[Light]:
        """Classic 3-point studio lighting."""
        return [
            Light(name="Key", light_type="area", color=(1.0, 0.95, 0.9),
                  position=(-2.0, 3.0, 2.0), intensity=2.0, size=1.0),
            Light(name="Fill", light_type="area", color=(0.9, 0.95, 1.0),
                  position=(2.0, 1.5, 2.5), intensity=1.0, size=1.0),
            Light(name="Rim", light_type="directional", color=(1.0, 1.0, 1.0),
                  position=(-1.0, 4.0, -3.0), intensity=1.5, size=0.5),
        ]

    @staticmethod
    def create_soft_studio(center: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> List[Light]:
        """Soft diffused studio lighting."""
        return [
            Light(name="Soft_Key", light_type="area", color=(1.0, 1.0, 1.0),
                  position=(-3.0, 4.0, 3.0), intensity=3.0, size=2.0),
            Light(name="Soft_Fill", light_type="area", color=(0.95, 0.95, 1.0),
                  position=(3.0, 3.0, 2.0), intensity=1.5, size=2.0),
            Light(name="Bounce", light_type="area", color=(1.0, 1.0, 1.0),
                  position=(0.0, 1.0, -1.0), intensity=0.5, size=3.0),
        ]

    @staticmethod
    def create_dramatic(center: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> List[Light]:
        """High-contrast dramatic lighting."""
        return [
            Light(name="Key", light_type="spot", color=(1.0, 0.9, 0.7),
                  position=(-3.0, 4.0, 1.0), intensity=5.0,
                  angle=math.radians(20), size=0.3),
            Light(name="Rim_Cold", light_type="spot", color=(0.5, 0.7, 1.0),
                  position=(1.0, 2.0, -3.0), intensity=2.0,
                  angle=math.radians(15), size=0.2),
        ]

    @staticmethod
    def create_golden_hour(center: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> List[Light]:
        """Warm golden hour lighting."""
        return [
            Light(name="Sun", light_type="directional", color=(1.0, 0.7, 0.3),
                  position=(-5.0, 2.0, 3.0), intensity=3.0),
            Light(name="Sky", light_type="directional", color=(0.4, 0.6, 1.0),
                  position=(0.0, 5.0, 0.0), intensity=0.5),
        ]

    PRESETS = {
        LightingPreset.STUDIO_3POINT: create_studio_3point,
        LightingPreset.STUDIO_SOFT: create_soft_studio,
        LightingPreset.DRAMATIC: create_dramatic,
        LightingPreset.GOLDEN_HOUR: create_golden_hour,
    }

    @classmethod
    def get_lights(cls, preset: LightingPreset,
                   center: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> List[Light]:
        if preset in cls.PRESETS:
            return cls.PRESETS[preset](center)
        return cls.create_studio_3point(center)


# ---------------------------------------------------------------------------
# Camera Framing
# ---------------------------------------------------------------------------

class CameraFramer:
    """Automatically frame a product with the camera."""

    @staticmethod
    def auto_frame(product_bounds: Tuple[float, float, float, float, float, float],
                   framing: CameraFraming = CameraFraming.PERSPECTIVE,
                   margin: float = 0.2) -> Camera:
        """Calculate camera position to frame the product bounding box.

        Args:
            product_bounds: (min_x, min_y, min_z, max_x, max_y, max_z)
            framing: Camera framing type
            margin: Fractional margin around the object
        """
        min_x, min_y, min_z, max_x, max_y, max_z = product_bounds
        center = (
            (min_x + max_x) / 2.0,
            (min_y + max_y) / 2.0,
            (min_z + max_z) / 2.0,
        )
        size = max(max_x - min_x, max_y - min_y, max_z - min_z)

        cam = Camera(target=center)

        if framing == CameraFraming.ISOMETRIC:
            dist = size * (1.0 + margin) * 1.5
            cam.position = (center[0] + dist, center[1] + dist * 0.7, center[2] + dist)
            cam.fov = math.radians(30)

        elif framing == CameraFraming.ORTHOGRAPHIC:
            dist = size * (1.0 + margin) * 2.0
            cam.position = (center[0], center[1] + size * 0.5, center[2] + dist)
            cam.orthographic_scale = size * (1.0 + margin)
            cam.fov = 0.0

        elif framing == CameraFraming.CLOSE_UP:
            dist = size * (1.0 + margin * 0.5)
            cam.position = (center[0] + dist * 0.3, center[1] + dist * 0.4, center[2] + dist)
            cam.fov = math.radians(60)

        elif framing == CameraFraming.MACRO:
            dist = size * (1.0 + margin * 0.2)
            cam.position = (center[0], center[1] + size * 0.3, center[2] + dist)
            cam.fov = math.radians(70)
            cam.dof_enabled = True
            cam.dof_distance = dist

        else:  # perspective
            dist = size * (1.0 + margin) * 2.0
            cam.position = (center[0] + dist * 0.4, center[1] + dist * 0.3, center[2] + dist)
            cam.fov = math.radians(45)

        return cam


# ---------------------------------------------------------------------------
# Turntable Animation
# ---------------------------------------------------------------------------

class Turntable:
    """Automated turntable animation for product showcases."""

    def __init__(self, axis: str = "y", speed: float = 360.0,
                 duration: float = 5.0, start_angle: float = 0.0) -> None:
        self.axis = axis
        self.speed = speed           # degrees per second
        self.duration = duration
        self.start_angle = start_angle
        self.elapsed: float = 0.0

    def step(self, dt: float) -> Tuple[float, float, float]:
        """Advance the turntable and return (rx, ry, rz) rotation."""
        self.elapsed += dt
        angle = self.start_angle + self.speed * self.elapsed
        axis_map = {"x": 0, "y": 1, "z": 2}
        rot = [0.0, 0.0, 0.0]
        rot[axis_map.get(self.axis, 1)] = math.radians(angle % 360.0)
        return tuple(rot)

    def reset(self) -> None:
        self.elapsed = 0.0

    def get_progress(self) -> float:
        return min(self.elapsed / self.duration, 1.0) if self.duration > 0 else 1.0


# ---------------------------------------------------------------------------
# Render Pipeline Configuration
# ---------------------------------------------------------------------------

class RenderPipeline:
    """Configure and manage the render pipeline."""

    @staticmethod
    def get_settings(quality: RenderQuality) -> Dict[str, Any]:
        return dict(QUALITY_SETTINGS[quality])

    @staticmethod
    def estimate_time(num_materials: int = 1, num_lights: int = 3,
                      quality: RenderQuality = RenderQuality.FINAL) -> Dict[str, Any]:
        """Estimate render time based on scene complexity."""
        base = {
            RenderQuality.DRAFT: 2,
            RenderQuality.PREVIEW: 15,
            RenderQuality.FINAL: 120,
            RenderQuality.PRODUCTION: 600,
        }
        base_seconds = base[quality]
        material_factor = 1.0 + (num_materials - 1) * 0.2
        light_factor = 1.0 + (num_lights - 3) * 0.15
        total = base_seconds * material_factor * light_factor
        return {
            "estimated_seconds": total,
            "estimated_minutes": total / 60,
            "quality": quality.value,
            "materials": num_materials,
            "lights": num_lights,
        }

    @staticmethod
    def generate_render_preset(name: str = "product_studio",
                                quality: RenderQuality = RenderQuality.FINAL,
                                resolution: Tuple[int, int] = (1920, 1080),
                                denoise: bool = True,
                                denoiser: str = "optix",
                                output_format: str = "png") -> Dict[str, Any]:
        """Generate a complete render preset configuration."""
        qs = QUALITY_SETTINGS[quality]
        return {
            "name": name,
            "quality": quality.value,
            "width": resolution[0],
            "height": resolution[1],
            "samples": qs["samples"],
            "bounces": qs["bounces"],
            "denoise": denoise,
            "denoiser": denoiser,
            "output_format": output_format,
            "color_space": "ACEScg",
            "tone_mapping": "filmic",
            "bloom": True,
            "bloom_intensity": 0.05,
            "vignette": 0.0,
            "dof": False,
            "motion_blur": False,
            "ca": False,  # chromatic aberration
            "adaptive_sampling": True,
            "noise_threshold": 0.02 if quality == RenderQuality.PRODUCTION else 0.05,
        }


# ---------------------------------------------------------------------------
# Virtual Studio Environment
# ---------------------------------------------------------------------------

@dataclass
class VirtualStudio:
    """Complete virtual photography studio."""
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = "studio"
    lights: List[Light] = field(default_factory=list)
    cameras: List[Camera] = field(default_factory=list)
    active_camera: str = ""
    environment_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    environment_intensity: float = 0.5
    show_grid: bool = True
    ground_plane: bool = True
    ground_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    backdrop_color: Tuple[float, float, float] = (0.9, 0.9, 0.9)
    reflection: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "name": self.name,
            "lights": [asdict(l) for l in self.lights],
            "cameras": [c.to_dict() for c in self.cameras],
            "active_camera": self.active_camera,
            "environment": {
                "color": self.environment_color,
                "intensity": self.environment_intensity,
            },
            "ground": {
                "enabled": self.ground_plane,
                "color": self.ground_color,
            },
            "backdrop": self.backdrop_color,
            "reflection": self.reflection,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def create_studio(name: str = "product_studio",
                         lighting_preset: str = "studio_3point",
                         camera_framing: str = "perspective",
                         product_bounds: Optional[Tuple[float, float, float,
                                                        float, float, float]] = None,
                         quality: str = "final") -> Dict[str, Any]:
    """Create a complete product render studio setup."""
    try:
        preset = LightingPreset(lighting_preset)
    except ValueError:
        preset = LightingPreset.STUDIO_3POINT

    try:
        framing = CameraFraming(camera_framing)
    except ValueError:
        framing = CameraFraming.PERSPECTIVE

    studio = VirtualStudio(name=name)
    studio.lights = StudioSetup.get_lights(preset)

    # Auto-frame or default camera
    if product_bounds:
        camera = CameraFramer.auto_frame(product_bounds, framing)
    else:
        camera = CameraFramer.auto_frame((-0.5, -0.5, -0.5, 0.5, 0.5, 0.5), framing)

    studio.cameras = [camera]
    studio.active_camera = camera.uid

    render_settings = RenderPipeline.generate_render_preset(
        quality=RenderQuality(quality),
    )

    return {
        "studio": studio.to_dict(),
        "render_settings": render_settings,
        "estimate": RenderPipeline.estimate_time(
            quality=RenderQuality(quality),
        ),
    }


async def setup_lighting(preset_name: str = "studio_3point",
                          intensity_scale: float = 1.0,
                          color_temp: Optional[Tuple[float, float, float]] = None) -> List[Dict[str, Any]]:
    """Create lighting setup for a product render."""
    try:
        preset = LightingPreset(preset_name)
    except ValueError:
        preset = LightingPreset.STUDIO_3POINT

    lights = StudioSetup.get_lights(preset)
    if intensity_scale != 1.0:
        for light in lights:
            light.intensity *= intensity_scale
    if color_temp:
        for light in lights:
            light.color = tuple(c * t for c, t in zip(light.color, color_temp))

    return [asdict(light) for light in lights]


async def frame_product(bounds: Tuple[float, float, float, float, float, float],
                         framing: str = "perspective",
                         margin: float = 0.2) -> Dict[str, Any]:
    """Auto-frame camera for a product bounding box."""
    try:
        cam_type = CameraFraming(framing)
    except ValueError:
        cam_type = CameraFraming.PERSPECTIVE
    camera = CameraFramer.auto_frame(bounds, cam_type, margin)
    return camera.to_dict()


async def create_turntable(duration: float = 5.0,
                            axis: str = "y",
                            speed: float = 360.0) -> Dict[str, Any]:
    """Create a turntable animation config."""
    tt = Turntable(axis=axis, speed=speed, duration=duration)
    return {"axis": axis, "speed": speed, "duration": duration,
            "start_angle": tt.start_angle}


async def estimate_render_time(num_materials: int = 1,
                                num_lights: int = 3,
                                quality: str = "final") -> Dict[str, Any]:
    """Estimate render time based on scene complexity."""
    return RenderPipeline.estimate_time(
        num_materials=num_materials,
        num_lights=num_lights,
        quality=RenderQuality(quality),
    )


async def get_render_presets() -> List[str]:
    """List all available render quality presets."""
    return [q.value for q in RenderQuality]


async def get_lighting_presets() -> List[str]:
    """List all available lighting presets."""
    return [p.value for p in LightingPreset]


async def get_camera_framings() -> List[str]:
    """List all available camera framing options."""
    return [f.value for f in CameraFraming]


async def orbit_camera(camera_data: Dict[str, Any],
                        theta: float = 0.0,
                        phi: float = 0.0,
                        radius: float = 5.0) -> Dict[str, Any]:
    """Orbit camera around its target."""
    cam = Camera(**{k: v for k, v in camera_data.items()
                    if k in Camera.__dataclass_fields__})
    cam.orbit(theta, phi, radius)
    return cam.to_dict()
