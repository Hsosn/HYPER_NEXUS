"""
Material Lab — Advanced Physically-Based Material System
========================================================
Generates, composites, and optimises PBR material graphs with
displacement, subsurface scattering, anisotropy, clear-coat,
and procedural texture generation. Integrates with the
Nexus3D material pipeline and ML‑driven texture enhancement.
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Types & Constants
# ---------------------------------------------------------------------------

PBR_CHANNELS = (
    "albedo", "roughness", "metallic", "normal",
    "displacement", "ao", "specular", "transmission",
    "subsurface", "clearcoat", "sheen", "anisotropic",
)

QUALITY_PRESETS = {"draft": 256, "low": 512, "medium": 1024, "high": 2048, "ultra": 4096}


class MaterialDomain(Enum):
    STANDARD = "standard"
    SUBSURFACE = "subsurface"
    CLEARCOAT = "clearcoat"
    TRANSMISSION = "transmission"
    HAIR = "hair"
    VOLUME = "volume"


@dataclass
class MaterialNode:
    """A single node in the material graph."""
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    node_type: str = "PBR"          # PBR | MIX | BLEND | NORMAL_MAP | DISPLACEMENT | TEX
    label: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)   # node UIDs
    output: str = "out"


@dataclass
class PBRMaterial:
    """Complete PBR material definition."""
    name: str = "default"
    domain: MaterialDomain = MaterialDomain.STANDARD
    channels: Dict[str, Any] = field(default_factory=dict)
    tiling: Tuple[float, float] = (1.0, 1.0)
    resolution: int = 1024

    albedo: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    roughness: float = 0.5
    metallic: float = 0.0
    normal_strength: float = 1.0
    displacement_scale: float = 0.05
    ior: float = 1.45
    subsurface_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    clearcoat: float = 0.0
    clearcoat_roughness: float = 0.03
    sheen: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    anisotropy: float = 0.0
    anisotropy_rotation: float = 0.0
    emission: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    emission_strength: float = 0.0
    opacity: float = 1.0

    # Procedural texture seeds
    procedural_seed: int = 0
    texture_scale: float = 1.0

    graph_nodes: List[MaterialNode] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PBRMaterial":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def blend(self, other: "PBRMaterial", weight: float = 0.5) -> "PBRMaterial":
        """Linearly blend two materials."""
        def _lerp(a, b, w):
            if isinstance(a, (int, float)):
                return a * (1 - w) + b * w
            if isinstance(a, (tuple, list)):
                return tuple(ai * (1 - w) + bi * w for ai, bi in zip(a, b))
            return b if w > 0.5 else a

        blended = PBRMaterial(name=f"{self.name}_blend_{other.name}")
        for field_name in ("albedo", "roughness", "metallic", "normal_strength",
                           "displacement_scale", "ior", "clearcoat", "clearcoat_roughness",
                           "sheen", "anisotropy", "anisotropy_rotation",
                           "emission", "emission_strength", "opacity",
                           "subsurface_color", "metallic"):
            sv = getattr(self, field_name)
            ov = getattr(other, field_name)
            setattr(blended, field_name, _lerp(sv, ov, weight))
        return blended

    def channel_map(self) -> Dict[str, str]:
        """Return a dict of channel → placeholder texture path."""
        base = f"textures/{self.name}"
        return {
            ch: f"{base}_{ch}.exr"
            for ch in PBR_CHANNELS
        }


# ---------------------------------------------------------------------------
# Procedural Generators
# ---------------------------------------------------------------------------

class ProceduralTextureGenerator:
    """Generate procedural textures using math-based approaches (no GPU)."""

    @staticmethod
    def _fbm(x: float, y: float, octaves: int = 4, seed: int = 0) -> float:
        """Simple value noise FBM."""
        val = 0.0
        amp = 0.5
        freq = 1.0
        r = random.Random(seed)
        for _ in range(octaves):
            v = math.sin(x * freq + r.random() * 6.283) * math.cos(y * freq + r.random() * 6.283)
            val += amp * v
            amp *= 0.5
            freq *= 2.0
        return val * 0.5 + 0.5

    def generate_albedo(self, material: PBRMaterial, width: int, height: int) -> bytes:
        """Generate a simple PNG albedo texture procedurally."""
        import struct
        import zlib

        rng = random.Random(material.procedural_seed)
        pixels = bytearray()
        for y in range(height):
            for x in range(width):
                u = x / width * material.texture_scale
                v = y / height * material.texture_scale
                noise = self._fbm(u, v, octaves=4, seed=material.procedural_seed)
                n2 = self._fbm(u * 2, v * 2, octaves=2, seed=material.procedural_seed + 1)
                r = int(clamp(material.albedo[0] * (0.8 + 0.4 * noise), 0, 1) * 255)
                g = int(clamp(material.albedo[1] * (0.8 + 0.4 * n2), 0, 1) * 255)
                b = int(clamp(material.albedo[2] * (0.7 + 0.6 * (noise * n2)), 0, 1) * 255)
                pixels.extend([r, g, b, 255])

        def _write_png(data, w, h):
            def _chunk(ctype, cdata):
                c = ctype + cdata
                return struct.pack(">I", len(cdata)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

            raw = b""
            for y in range(h):
                raw += b"\x00" + bytes(data[y * w * 4:(y + 1) * w * 4])
            compressed = zlib.compress(raw)
            ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
            return (b"\x89PNG\r\n\x1a\n" +
                    _chunk(b"IHDR", ihdr) +
                    _chunk(b"IDAT", compressed) +
                    _chunk(b"IEND", b""))

        return _write_png(pixels, width, height)

    def generate_normal(self, width: int, height: int, seed: int = 0, strength: float = 1.0) -> bytes:
        """Generate a simple normal map from noise."""
        import struct, zlib
        pixels = bytearray()
        rng = random.Random(seed)
        for y in range(height):
            for x in range(width):
                u = x / width * 4.0
                v = y / height * 4.0
                dx = self._fbm(u + 0.01, v, seed=seed) - self._fbm(u - 0.01, v, seed=seed)
                dy = self._fbm(u, v + 0.01, seed=seed) - self._fbm(u, v - 0.01, seed=seed)
                nx = clamp(-dx * strength, -1, 1)
                ny = clamp(-dy * strength, -1, 1)
                nz = math.sqrt(max(0, 1 - nx * nx - ny * ny))
                pixels.extend([int((nx * 0.5 + 0.5) * 255),
                               int((ny * 0.5 + 0.5) * 255),
                               int(nz * 255), 255])

        def _write_png(data, w, h):
            def _chunk(ctype, cdata):
                c = ctype + cdata
                return struct.pack(">I", len(cdata)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            raw = b""
            for y in range(h):
                raw += b"\x00" + bytes(data[y * w * 4:(y + 1) * w * 4])
            compressed = zlib.compress(raw)
            ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
            return (b"\x89PNG\r\n\x1a\n" +
                    _chunk(b"IHDR", ihdr) +
                    _chunk(b"IDAT", compressed) +
                    _chunk(b"IEND", b""))

        return _write_png(pixels, width, height)

    def generate_roughness(self, material: PBRMaterial, width: int, height: int) -> bytes:
        """Generate a roughness map."""
        import struct, zlib
        base_val = int(clamp(material.roughness, 0, 1) * 255)
        pixels = bytearray()
        for y in range(height):
            for x in range(width):
                u = x / width * material.texture_scale
                v = y / height * material.texture_scale
                noise = self._fbm(u * 3, v * 3, octaves=3, seed=material.procedural_seed + 10)
                val = int(clamp(base_val * (0.5 + noise * 0.5), 0, 255))
                pixels.extend([val, val, val, 255])

        def _write_png(data, w, h):
            def _chunk(ctype, cdata):
                c = ctype + cdata
                return struct.pack(">I", len(cdata)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            raw = b""
            for y in range(h):
                raw += b"\x00" + bytes(data[y * w * 4:(y + 1) * w * 4])
            compressed = zlib.compress(raw)
            ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
            return (b"\x89PNG\r\n\x1a\n" +
                    _chunk(b"IHDR", ihdr) +
                    _chunk(b"IDAT", compressed) +
                    _chunk(b"IEND", b""))

        return _write_png(pixels, width, height)


# ---------------------------------------------------------------------------
# Material Mixer
# ---------------------------------------------------------------------------

class MaterialMixer:
    """Blend, layer, and compose multiple materials."""

    @staticmethod
    def mix(materials: List[PBRMaterial], weights: Optional[List[float]] = None) -> PBRMaterial:
        """Weighted mix of multiple materials."""
        if not materials:
            raise ValueError("At least one material required.")
        if weights is None:
            weights = [1.0 / len(materials)] * len(materials)
        total = sum(weights)
        if total <= 0:
            raise ValueError("Weights must sum to > 0.")
        weights = [w / total for w in weights]

        base = materials[0]
        mixed = PBRMaterial(name=f"mixed_{len(materials)}")
        for field_name in ("albedo", "roughness", "metallic", "normal_strength",
                           "displacement_scale", "ior", "clearcoat", "clearcoat_roughness",
                           "opacity", "emission_strength"):
            svals = [getattr(m, field_name) for m in materials]
            if isinstance(svals[0], (int, float)):
                setattr(mixed, field_name, sum(s * w for s, w in zip(svals, weights)))
            elif isinstance(svals[0], (tuple, list)):
                setattr(mixed, field_name,
                        tuple(sum(s[i] * w for s, w in zip(svals, weights))
                              for i in range(len(svals[0]))))
        return mixed

    @staticmethod
    def layer(base: PBRMaterial, top: PBRMaterial, mask_channel: str = "roughness") -> PBRMaterial:
        """Layer a top material over base using a mask channel."""
        result = PBRMaterial(name=f"{base.name}_layered_{top.name}")
        weight = getattr(top, mask_channel, 0.5)
        weight = clamp(weight, 0, 1)
        for field_name in ("albedo", "roughness", "metallic", "clearcoat", "opacity",
                           "emission_strength", "displacement_scale"):
            bv = getattr(base, field_name)
            tv = getattr(top, field_name)
            if isinstance(bv, (int, float)):
                setattr(result, field_name, bv * (1 - weight) + tv * weight)
            elif isinstance(bv, (tuple, list)):
                setattr(result, field_name,
                        tuple(b * (1 - weight) + t * weight for b, t in zip(bv, tv)))
        return result


# ---------------------------------------------------------------------------
# Material Library Presets
# ---------------------------------------------------------------------------

class MaterialLibrary:
    """Curated library of physically-based material presets."""

    PRESETS: Dict[str, Dict[str, Any]] = {
        "gold": {
            "albedo": (1.0, 0.843, 0.0), "roughness": 0.1, "metallic": 1.0,
            "ior": 0.47, "anisotropy": 0.3,
        },
        "copper": {
            "albedo": (0.955, 0.637, 0.538), "roughness": 0.15, "metallic": 1.0,
            "ior": 1.95, "anisotropy": 0.2,
        },
        "iron": {
            "albedo": (0.562, 0.565, 0.578), "roughness": 0.3, "metallic": 1.0,
        },
        "aluminium": {
            "albedo": (0.913, 0.921, 0.925), "roughness": 0.2, "metallic": 1.0,
        },
        "glass": {
            "albedo": (0.9, 0.9, 1.0), "roughness": 0.0, "metallic": 0.0,
            "ior": 1.45, "transmission": 1.0, "opacity": 0.1,
        },
        "water": {
            "albedo": (0.1, 0.2, 0.3), "roughness": 0.0, "metallic": 0.0,
            "ior": 1.33, "transmission": 0.9, "opacity": 0.2,
        },
        "rubber": {
            "albedo": (0.05, 0.05, 0.05), "roughness": 0.9, "metallic": 0.0,
        },
        "plastic_rough": {
            "albedo": (0.8, 0.1, 0.1), "roughness": 0.7, "metallic": 0.0,
        },
        "plastic_smooth": {
            "albedo": (0.8, 0.1, 0.1), "roughness": 0.05, "metallic": 0.0,
            "clearcoat": 0.3,
        },
        "skin": {
            "albedo": (0.8, 0.6, 0.5), "roughness": 0.4, "metallic": 0.0,
            "subsurface_color": (0.9, 0.4, 0.3), "domain": "subsurface",
        },
        "wood": {
            "albedo": (0.6, 0.4, 0.2), "roughness": 0.6, "metallic": 0.0,
            "anisotropy": 0.5,
        },
        "concrete": {
            "albedo": (0.5, 0.5, 0.5), "roughness": 0.9, "metallic": 0.0,
        },
        "asphalt": {
            "albedo": (0.2, 0.2, 0.22), "roughness": 0.85, "metallic": 0.0,
        },
        "marble": {
            "albedo": (0.9, 0.88, 0.85), "roughness": 0.2, "metallic": 0.0,
        },
        "car_paint": {
            "albedo": (0.9, 0.1, 0.1), "roughness": 0.05, "metallic": 0.0,
            "clearcoat": 1.0, "clearcoat_roughness": 0.02,
        },
    }

    @classmethod
    def get(cls, name: str, **overrides) -> PBRMaterial:
        if name not in cls.PRESETS:
            valid = ", ".join(sorted(cls.PRESETS))
            raise KeyError(f"Unknown preset '{name}'. Valid: {valid}")
        data = dict(cls.PRESETS[name])
        data.update(overrides)
        domain_name = data.pop("domain", "standard")
        mat = PBRMaterial(name=name, domain=MaterialDomain(domain_name))
        for k, v in data.items():
            if hasattr(mat, k):
                setattr(mat, k, v)
        return mat

    @classmethod
    def list_presets(cls) -> List[str]:
        return sorted(cls.PRESETS)


# ---------------------------------------------------------------------------
# Material Optimizer
# ---------------------------------------------------------------------------

class MaterialOptimizer:
    """Reduce texture size, merge channels, and strip unused properties."""

    @staticmethod
    def optimize(mat: PBRMaterial, target_resolution: int = 512,
                 strip_channels: Optional[List[str]] = None) -> PBRMaterial:
        opt = PBRMaterial(
            name=f"{mat.name}_opt",
            resolution=min(mat.resolution, target_resolution),
            procedural_seed=mat.procedural_seed,
            texture_scale=mat.texture_scale,
        )
        strip = set(strip_channels or [])

        if "albedo" not in strip:
            opt.albedo = mat.albedo
        if "roughness" not in strip:
            opt.roughness = mat.roughness
        if "metallic" not in strip:
            opt.metallic = mat.metallic

        # Keep essential properties only
        opt.ior = mat.ior
        opt.normal_strength = mat.normal_strength
        opt.opacity = mat.opacity

        return opt

    @staticmethod
    def merge_channels(materials: List[PBRMaterial],
                       channel: str = "roughness") -> PBRMaterial:
        """Merge a single channel across multiple materials into one."""
        if not materials:
            raise ValueError("Need at least one material.")
        base = materials[0]
        merged = PBRMaterial(name=f"merged_{channel}")
        vals = [getattr(m, channel, 0.5) for m in materials]
        avg = sum(vals) / len(vals)
        setattr(merged, channel, avg)
        return merged


# ---------------------------------------------------------------------------
# ML Texture Enhancement (simulated)
# ---------------------------------------------------------------------------

class TextureEnhancer:
    """ML-driven texture upscaling and enhancement."""

    @staticmethod
    def upscale(width: int, height: int, factor: int = 2) -> Dict[str, Any]:
        """Simulate ML upscale — returns metadata."""
        return {
            "original": (width, height),
            "upscaled": (width * factor, height * factor),
            "factor": factor,
            "method": "esrgan_simulated",
            "duration_ms": random.uniform(50, 200),
        }

    @staticmethod
    def denoise(strength: float = 0.5) -> Dict[str, Any]:
        """Simulate ML denoising info."""
        return {
            "method": "nlm_simulated",
            "strength": strength,
            "duration_ms": random.uniform(20, 80),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_preset_material(preset_name: str, **overrides) -> Dict[str, Any]:
    """Generate a material from a library preset."""
    mat = MaterialLibrary.get(preset_name, **overrides)
    return mat.to_dict()


async def mix_materials(material_dicts: List[Dict[str, Any]],
                        weights: Optional[List[float]] = None) -> Dict[str, Any]:
    """Mix multiple materials into one."""
    mats = [PBRMaterial.from_dict(d) for d in material_dicts]
    mixer = MaterialMixer()
    result = mixer.mix(mats, weights)
    return result.to_dict()


async def layer_materials(base_dict: Dict[str, Any],
                          top_dict: Dict[str, Any],
                          mask_channel: str = "roughness") -> Dict[str, Any]:
    """Layer a top material over a base."""
    base = PBRMaterial.from_dict(base_dict)
    top = PBRMaterial.from_dict(top_dict)
    mixer = MaterialMixer()
    result = mixer.layer(base, top, mask_channel)
    return result.to_dict()


async def optimize_material(mat_dict: Dict[str, Any],
                            target_resolution: int = 512,
                            strip_channels: Optional[List[str]] = None) -> Dict[str, Any]:
    """Optimize a material for performance."""
    mat = PBRMaterial.from_dict(mat_dict)
    opt = MaterialOptimizer.optimize(mat, target_resolution, strip_channels)
    return opt.to_dict()


async def generate_procedural_textures(mat_dict: Dict[str, Any],
                                       width: int = 512,
                                       height: int = 512) -> Dict[str, Any]:
    """Generate procedural PBR texture maps (returns b64 PNG data)."""
    import base64
    mat = PBRMaterial.from_dict(mat_dict)
    gen = ProceduralTextureGenerator()

    albedo_data = gen.generate_albedo(mat, width, height)
    normal_data = gen.generate_normal(width, height, mat.procedural_seed, mat.normal_strength)
    roughness_data = gen.generate_roughness(mat, width, height)

    return {
        "albedo": base64.b64encode(albedo_data).decode(),
        "normal": base64.b64encode(normal_data).decode(),
        "roughness": base64.b64encode(roughness_data).decode(),
        "width": width,
        "height": height,
        "format": "png",
    }


async def list_material_presets() -> List[str]:
    """List all available material library presets."""
    return MaterialLibrary.list_presets()


async def blend_materials(mat_a: Dict[str, Any],
                          mat_b: Dict[str, Any],
                          weight: float = 0.5) -> Dict[str, Any]:
    """Blend two materials with a weight."""
    a = PBRMaterial.from_dict(mat_a)
    b = PBRMaterial.from_dict(mat_b)
    result = a.blend(b, weight)
    return result.to_dict()
