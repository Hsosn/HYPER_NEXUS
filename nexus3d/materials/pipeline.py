"""
Nexus3D Texture and Material Pipeline
=====================================
Production-grade procedural texture generation, PBR material system,
UV mapping utilities, and image-based lighting.

Dependencies: numpy, Pillow (PIL)
"""

from __future__ import annotations

import math
import hashlib
import random
from typing import Tuple, Optional, List, Dict, Any

import numpy as np


# =============================================================================
# Section 1: Perlin Noise Implementation
# =============================================================================

class PerlinNoise:
    """
    Self-contained 3D Perlin gradient noise implementation.

    Provides:
    - 3D gradient noise (smooth, continuous)
    - Fractional Brownian Motion (fBm) with configurable octaves
    - Turbulence function
    - Deterministic seeding via permutation table

    No external noise libraries required.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        rng = random.Random(seed)
        # Build a shuffled permutation table (256 entries, doubled for wrapping)
        perm = list(range(256))
        rng.shuffle(perm)
        self._perm = perm + perm  # duplicate to avoid modulo in inner loop

        # Precompute 12 gradient vectors uniformly distributed on the unit sphere
        # Using Ken Perlin's improved gradient set
        self._gradients = np.array([
            [ 1,  1,  0], [-1,  1,  0], [ 1, -1,  0], [-1, -1,  0],
            [ 1,  0,  1], [-1,  0,  1], [ 1,  0, -1], [-1,  0, -1],
            [ 0,  1,  1], [ 0, -1,  1], [ 0,  1, -1], [ 0, -1, -1],
        ], dtype=np.float64)

    @staticmethod
    def _fade(t: float) -> float:
        """Quintic fade curve: 6t^5 - 15t^4 + 10t^3."""
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation."""
        return a + t * (b - a)

    def _gradient_hash(self, ix: int, iy: int, iz: int) -> np.ndarray:
        """Look up a gradient vector for the given integer lattice point."""
        idx = self._perm[self._perm[self._perm[ix & 255] + (iy & 255)] + (iz & 255)] % 12
        return self._gradients[idx]

    def noise3d(self, x: float, y: float, z: float) -> float:
        """
        Evaluate 3D Perlin gradient noise at (x, y, z).

        Returns a value approximately in the range [-1, 1].
        """
        # Integer lattice coordinates of the surrounding cube
        x0 = int(math.floor(x))
        y0 = int(math.floor(y))
        z0 = int(math.floor(z))
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        # Fractional position within the cube [0, 1)
        xf = x - x0
        yf = y - y0
        zf = z - z0

        # Apply fade curves to fractional components
        u = self._fade(xf)
        v = self._fade(yf)
        w = self._fade(zf)

        # Hash the 8 corners of the cube to get gradient indices
        g000 = self._gradient_hash(x0, y0, z0)
        g100 = self._gradient_hash(x1, y0, z0)
        g010 = self._gradient_hash(x0, y1, z0)
        g110 = self._gradient_hash(x1, y1, z0)
        g001 = self._gradient_hash(x0, y0, z1)
        g101 = self._gradient_hash(x1, y0, z1)
        g011 = self._gradient_hash(x0, y1, z1)
        g111 = self._gradient_hash(x1, y1, z1)

        # Dot product of gradient and distance vector for each corner
        d000 = np.dot(g000, np.array([xf,     yf,     zf    ]))
        d100 = np.dot(g100, np.array([xf - 1, yf,     zf    ]))
        d010 = np.dot(g010, np.array([xf,     yf - 1, zf    ]))
        d110 = np.dot(g110, np.array([xf - 1, yf - 1, zf    ]))
        d001 = np.dot(g001, np.array([xf,     yf,     zf - 1]))
        d101 = np.dot(g101, np.array([xf - 1, yf,     zf - 1]))
        d011 = np.dot(g011, np.array([xf,     yf - 1, zf - 1]))
        d111 = np.dot(g111, np.array([xf - 1, yf - 1, zf - 1]))

        # Trilinear interpolation
        x00 = self._lerp(d000, d100, u)
        x10 = self._lerp(d010, d110, u)
        x01 = self._lerp(d001, d101, u)
        x11 = self._lerp(d011, d111, u)

        y0_ = self._lerp(x00, x10, v)
        y1_ = self._lerp(x01, x11, v)

        return self._lerp(y0_, y1_, w)

    def noise2d(self, x: float, y: float) -> float:
        """Evaluate 2D Perlin noise (implemented via 3D noise with z=0)."""
        return self.noise3d(x, y, 0.0)

    def fbm(self, x: float, y: float, z: float = 0.0,
            octaves: int = 6, lacunarity: float = 2.0,
            gain: float = 0.5) -> float:
        """
        Fractional Brownian Motion — layered noise at multiple frequencies.

        Args:
            x, y, z: Sample coordinates.
            octaves: Number of noise layers.
            lacunarity: Frequency multiplier between layers (typically 2.0).
            gain: Amplitude multiplier between layers (typically 0.5).

        Returns:
            Combined noise value, approximately in [-1, 1].
        """
        value = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_amplitude = 0.0

        for _ in range(octaves):
            value += amplitude * self.noise3d(x * frequency, y * frequency, z * frequency)
            max_amplitude += amplitude
            amplitude *= gain
            frequency *= lacunarity

        # Normalize to approximately [-1, 1]
        return value / max_amplitude if max_amplitude > 0 else 0.0

    def turbulence(self, x: float, y: float, z: float = 0.0,
                   octaves: int = 6, lacunarity: float = 2.0,
                   gain: float = 0.5) -> float:
        """
        Turbulence — absolute-value layered noise. Produces sharp, vein-like
        structures useful for marble, wood, and cloud textures.

        Returns a value approximately in [0, 1].
        """
        value = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_amplitude = 0.0

        for _ in range(octaves):
            value += amplitude * abs(self.noise3d(x * frequency, y * frequency, z * frequency))
            max_amplitude += amplitude
            amplitude *= gain
            frequency *= lacunarity

        return value / max_amplitude if max_amplitude > 0 else 0.0


# =============================================================================
# Section 2: Texture Base Class
# =============================================================================

class Texture:
    """
    Base class for all textures in the Nexus3D material pipeline.

    Subclasses must implement ``sample(u, v)`` which returns an [R, G, B] float
    array in the [0, 1] range for the given UV coordinates.

    Built-in methods:
        - ``to_image(width, height)`` → numpy (H, W, 3) uint8 array
        - ``save(filepath, width, height)`` → writes a PNG via Pillow
        - ``to_dict()`` / ``from_dict()`` for serialization
    """

    def sample(self, u: float, v: float) -> np.ndarray:
        """
        Sample the texture at UV coordinates.

        Args:
            u: Horizontal coordinate [0, 1].
            v: Vertical coordinate [0, 1].

        Returns:
            np.ndarray of shape (3,) with float values in [0, 1] for R, G, B.
        """
        raise NotImplementedError("Subclasses must implement sample(u, v)")

    def to_image(self, width: int = 512, height: int = 512) -> np.ndarray:
        """
        Render the texture to a numpy image array.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            np.ndarray of shape (height, width, 3) with uint8 values [0, 255].
        """
        img = np.zeros((height, width, 3), dtype=np.float64)
        for y in range(height):
            v = y / max(height - 1, 1)
            for x in range(width):
                u = x / max(width - 1, 1)
                img[y, x] = self.sample(u, v)

        # Clamp and convert to uint8
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img

    def save(self, filepath: str, width: int = 512, height: int = 512) -> None:
        """Save the texture as a PNG image using Pillow."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required to save textures. Install with: pip install Pillow")

        img = self.to_image(width, height)
        pil_img = Image.fromarray(img, mode="RGB")
        pil_img.save(filepath)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the texture to a dictionary."""
        return {"type": self.__class__.__name__}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Texture":
        """Deserialize a texture from a dictionary."""
        texture_classes = {
            "NoiseTexture": NoiseTexture,
            "CheckerTexture": CheckerTexture,
            "BrickTexture": BrickTexture,
            "WoodTexture": WoodTexture,
            "MarbleTexture": MarbleTexture,
            "GradientTexture": GradientTexture,
            "VoronoiTexture": VoronoiTexture,
            "ImageTexture": ImageTexture,
        }
        type_name = data.get("type", "")
        texture_cls = texture_classes.get(type_name)
        if texture_cls is None:
            raise ValueError(f"Unknown texture type: {type_name}")
        return texture_cls.from_dict(data)


# =============================================================================
# Section 3: Procedural Texture Generators
# =============================================================================

class NoiseTexture(Texture):
    """
    Perlin/Simplex noise texture with octaves for turbulence.

    Uses fBm (fractional Brownian motion) to produce natural-looking noise.
    """

    def __init__(
        self,
        scale: float = 1.0,
        octaves: int = 6,
        lacunarity: float = 2.0,
        gain: float = 0.5,
        color1: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        color2: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        seed: int = 42,
    ):
        self.scale = scale
        self.octaves = octaves
        self.lacunarity = lacunarity
        self.gain = gain
        self.color1 = np.array(color1, dtype=np.float64)
        self.color2 = np.array(color2, dtype=np.float64)
        self._perlin = PerlinNoise(seed=seed)

    def sample(self, u: float, v: float) -> np.ndarray:
        n = self._perlin.fbm(
            u * self.scale, v * self.scale, 0.0,
            octaves=self.octaves,
            lacunarity=self.lacunarity,
            gain=self.gain,
        )
        # Map from [-1, 1] to [0, 1]
        t = np.clip((n + 1.0) * 0.5, 0.0, 1.0)
        return self.color1 + t * (self.color2 - self.color1)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "scale": self.scale,
            "octaves": self.octaves,
            "lacunarity": self.lacunarity,
            "gain": self.gain,
            "color1": tuple(self.color1.tolist()),
            "color2": tuple(self.color2.tolist()),
            "seed": self.seed if hasattr(self, "seed") else 42,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NoiseTexture":
        return cls(
            scale=data.get("scale", 1.0),
            octaves=data.get("octaves", 6),
            lacunarity=data.get("lacunarity", 2.0),
            gain=data.get("gain", 0.5),
            color1=tuple(data.get("color1", (0, 0, 0))),
            color2=tuple(data.get("color2", (1, 1, 1))),
            seed=data.get("seed", 42),
        )


class CheckerTexture(Texture):
    """Checkerboard pattern with configurable scale."""

    def __init__(
        self,
        scale: float = 4.0,
        color1: Tuple[float, float, float] = (0.2, 0.2, 0.2),
        color2: Tuple[float, float, float] = (0.8, 0.8, 0.8),
    ):
        self.scale = scale
        self.color1 = np.array(color1, dtype=np.float64)
        self.color2 = np.array(color2, dtype=np.float64)

    def sample(self, u: float, v: float) -> np.ndarray:
        # Wrap UVs
        su = (u * self.scale) % 2.0
        sv = (v * self.scale) % 2.0
        # Checker: parity of floor
        if (int(math.floor(su)) + int(math.floor(sv))) % 2 == 0:
            return self.color1.copy()
        return self.color2.copy()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "scale": self.scale,
            "color1": tuple(self.color1.tolist()),
            "color2": tuple(self.color2.tolist()),
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckerTexture":
        return cls(
            scale=data.get("scale", 4.0),
            color1=tuple(data.get("color1", (0.2, 0.2, 0.2))),
            color2=tuple(data.get("color2", (0.8, 0.8, 0.8))),
        )


class BrickTexture(Texture):
    """
    Realistic brick pattern with mortar lines.

    Bricks are arranged in a standard running-bond pattern where every
    other row is offset by half a brick width.
    """

    def __init__(
        self,
        brick_width: float = 0.5,
        brick_height: float = 0.25,
        mortar_width: float = 0.02,
        brick_color: Tuple[float, float, float] = (0.6, 0.2, 0.1),
        mortar_color: Tuple[float, float, float] = (0.7, 0.7, 0.7),
        seed: int = 42,
    ):
        self.brick_width = brick_width
        self.brick_height = brick_height
        self.mortar_width = mortar_width
        self.brick_color = np.array(brick_color, dtype=np.float64)
        self.mortar_color = np.array(mortar_color, dtype=np.float64)
        self._perlin = PerlinNoise(seed=seed)

    def sample(self, u: float, v: float) -> np.ndarray:
        mw = self.mortar_width
        # Total cell dimensions including mortar
        cell_w = self.brick_width + mw
        cell_h = self.brick_height + mw

        # Which row are we in?
        row = math.floor(v / cell_h)
        # Running bond: offset every other row
        offset = (cell_w * 0.5) if (row % 2 != 0) else 0.0

        # Position within the row, accounting for offset
        u_shifted = (u + offset) % (cell_w * 2.0)  # wrap for safety
        if u_shifted > cell_w:
            u_shifted -= cell_w

        # Fractional position within cell
        local_u = u_shifted / cell_w
        local_v = (v / cell_h) - row

        # Mortar check: near edges of the cell
        mortar_u = local_u < (mw / cell_w) or local_u > (1.0 - mw / cell_w)
        mortar_v = local_v < (mw / cell_h) or local_v > (1.0 - mw / cell_h)

        if mortar_u or mortar_v:
            return self.mortar_color.copy()

        # Inside a brick — add subtle color variation via noise
        noise_val = self._perlin.noise2d(u * 15.0, v * 15.0) * 0.05
        color = self.brick_color + noise_val
        return np.clip(color, 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "brick_width": self.brick_width,
            "brick_height": self.brick_height,
            "mortar_width": self.mortar_width,
            "brick_color": tuple(self.brick_color.tolist()),
            "mortar_color": tuple(self.mortar_color.tolist()),
            "seed": self.seed if hasattr(self, "seed") else 42,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrickTexture":
        return cls(
            brick_width=data.get("brick_width", 0.5),
            brick_height=data.get("brick_height", 0.25),
            mortar_width=data.get("mortar_width", 0.02),
            brick_color=tuple(data.get("brick_color", (0.6, 0.2, 0.1))),
            mortar_color=tuple(data.get("mortar_color", (0.7, 0.7, 0.7))),
            seed=data.get("seed", 42),
        )


class WoodTexture(Texture):
    """
    Wood grain texture using concentric ring noise perturbed by turbulence.

    The ring pattern follows the distance from a central axis, then is
    perturbed by Perlin noise to create natural-looking grain.
    """

    def __init__(
        self,
        rings: float = 10.0,
        turbulence: float = 0.05,
        color1: Tuple[float, float, float] = (0.4, 0.2, 0.05),
        color2: Tuple[float, float, float] = (0.7, 0.4, 0.15),
        seed: int = 42,
    ):
        self.rings = rings
        self.turbulence = turbulence
        self.color1 = np.array(color1, dtype=np.float64)
        self.color2 = np.array(color2, dtype=np.float64)
        self._perlin = PerlinNoise(seed=seed)

    def sample(self, u: float, v: float) -> np.ndarray:
        # Distance from the center-left edge of UV space
        dx = u
        dy = (v - 0.5) * 2.0  # Map v to [-1, 1]
        dist = math.sqrt(dx * dx + dy * dy)

        # Add turbulence to the ring coordinate
        turb = self._perlin.turbulence(u * 5.0, v * 5.0, octaves=4) * self.turbulence
        ring_val = math.sin((dist + turb) * self.rings * 2.0 * math.pi)

        # Map ring value to color
        t = np.clip((ring_val + 1.0) * 0.5, 0.0, 1.0)
        return self.color1 + t * (self.color2 - self.color1)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "rings": self.rings,
            "turbulence": self.turbulence,
            "color1": tuple(self.color1.tolist()),
            "color2": tuple(self.color2.tolist()),
            "seed": self.seed if hasattr(self, "seed") else 42,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WoodTexture":
        return cls(
            rings=data.get("rings", 10.0),
            turbulence=data.get("turbulence", 0.05),
            color1=tuple(data.get("color1", (0.4, 0.2, 0.05))),
            color2=tuple(data.get("color2", (0.7, 0.4, 0.15))),
            seed=data.get("seed", 42),
        )


class MarbleTexture(Texture):
    """
    Marble/stone texture using sine waves perturbed by turbulence noise.

    The classic marble look comes from a sine wave of the coordinate,
    distorted by layered noise to create veining patterns.
    """

    def __init__(
        self,
        scale: float = 1.0,
        turbulence: float = 5.0,
        color1: Tuple[float, float, float] = (0.9, 0.9, 0.9),
        color2: Tuple[float, float, float] = (0.2, 0.2, 0.3),
        seed: int = 42,
    ):
        self.scale = scale
        self.turbulence = turbulence
        self.color1 = np.array(color1, dtype=np.float64)
        self.color2 = np.array(color2, dtype=np.float64)
        self._perlin = PerlinNoise(seed=seed)

    def sample(self, u: float, v: float) -> np.ndarray:
        # Turbulence-distorted sine wave
        turb = self._perlin.turbulence(
            u * self.scale, v * self.scale,
            octaves=6, lacunarity=2.0, gain=0.5,
        )
        # Sinusoidal veining pattern
        val = math.sin((u * self.scale + turb * self.turbulence) * math.pi)

        # Map [-1, 1] → [0, 1]
        t = np.clip((val + 1.0) * 0.5, 0.0, 1.0)
        return self.color1 + t * (self.color2 - self.color1)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "scale": self.scale,
            "turbulence": self.turbulence,
            "color1": tuple(self.color1.tolist()),
            "color2": tuple(self.color2.tolist()),
            "seed": self.seed if hasattr(self, "seed") else 42,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarbleTexture":
        return cls(
            scale=data.get("scale", 1.0),
            turbulence=data.get("turbulence", 5.0),
            color1=tuple(data.get("color1", (0.9, 0.9, 0.9))),
            color2=tuple(data.get("color2", (0.2, 0.2, 0.3))),
            seed=data.get("seed", 42),
        )


class GradientTexture(Texture):
    """
    Directional gradient with multiple color stops.

    Supports 'vertical', 'horizontal', 'radial', and 'diagonal' directions.
    Color stops are defined as a list of (position, color) tuples where
    position is in [0, 1]. Positions are automatically sorted.
    """

    DIRECTIONS = ("vertical", "horizontal", "diagonal", "radial")

    def __init__(
        self,
        direction: str = "vertical",
        stops: Optional[List[Tuple[float, Tuple[float, float, float]]]] = None,
    ):
        if direction not in self.DIRECTIONS:
            raise ValueError(f"Direction must be one of {self.DIRECTIONS}, got '{direction}'")
        self.direction = direction
        if stops is None:
            stops = [(0.0, (0, 0, 0)), (0.5, (1, 1, 1)), (1.0, (0, 0, 0))]
        # Sort stops by position
        self.stops = sorted(stops, key=lambda s: s[0])
        # Pre-convert colors to arrays
        self._stop_colors = [np.array(c, dtype=np.float64) for _, c in self.stops]
        self._stop_positions = [p for p, _ in self.stops]

    def _interpolate_stops(self, t: float) -> np.ndarray:
        """Interpolate between color stops for parameter t in [0, 1]."""
        t = np.clip(t, 0.0, 1.0)
        positions = self._stop_positions
        colors = self._stop_colors

        if t <= positions[0]:
            return colors[0].copy()
        if t >= positions[-1]:
            return colors[-1].copy()

        # Find surrounding stops
        for i in range(len(positions) - 1):
            if positions[i] <= t <= positions[i + 1]:
                span = positions[i + 1] - positions[i]
                if span < 1e-9:
                    return colors[i].copy()
                frac = (t - positions[i]) / span
                return colors[i] + frac * (colors[i + 1] - colors[i])

        return colors[-1].copy()

    def sample(self, u: float, v: float) -> np.ndarray:
        if self.direction == "vertical":
            t = v
        elif self.direction == "horizontal":
            t = u
        elif self.direction == "diagonal":
            t = (u + v) * 0.5
        elif self.direction == "radial":
            du = u - 0.5
            dv = v - 0.5
            t = math.sqrt(du * du + dv * dv) * 2.0
        else:
            t = v
        return self._interpolate_stops(t)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "direction": self.direction,
            "stops": [(p, tuple(c.tolist())) for (p, c) in zip(self._stop_positions, self._stop_colors)],
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GradientTexture":
        stops = data.get("stops", [(0.0, (0, 0, 0)), (0.5, (1, 1, 1)), (1.0, (0, 0, 0))])
        # Convert colors back to tuples if needed
        converted_stops = []
        for pos, color in stops:
            converted_stops.append((pos, tuple(color)))
        return cls(direction=data.get("direction", "vertical"), stops=converted_stops)


class VoronoiTexture(Texture):
    """
    Voronoi / Worley cellular texture pattern.

    Generates a pattern based on distance to the nearest randomly-placed
    feature points in a tiled grid.

    cell_type controls the distance metric:
        'distance' — Euclidean distance to nearest cell center
        'f1'       — Feature 1 (closest cell)
        'f2'       — Feature 2 (second closest cell) minus f1
        'cell_id'  — Uses the cell index to create distinct colored cells
    """

    CELL_TYPES = ("distance", "f1", "f2", "cell_id")

    def __init__(
        self,
        scale: float = 10.0,
        cell_type: str = "distance",
        color1: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        color2: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        seed: int = 42,
    ):
        if cell_type not in self.CELL_TYPES:
            raise ValueError(f"cell_type must be one of {self.CELL_TYPES}, got '{cell_type}'")
        self.scale = scale
        self.cell_type = cell_type
        self.color1 = np.array(color1, dtype=np.float64)
        self.color2 = np.array(color2, dtype=np.float64)
        self._seed = seed
        self._rng = random.Random(seed)

        # Pre-generate cell center positions for a large enough grid
        self._cache_size = 64
        self._cells = self._generate_cells()

    def _generate_cells(self) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """Generate random cell centers for integer grid cells."""
        cells = {}
        for ix in range(self._cache_size):
            for iy in range(self._cache_size):
                cells[(ix, iy)] = (self._rng.random(), self._rng.random())
        return cells

    def _get_cell(self, ix: int, iy: int) -> Tuple[float, float]:
        """Get cell center with wrapping."""
        return self._cells[(ix % self._cache_size, iy % self._cache_size)]

    def sample(self, u: float, v: float) -> np.ndarray:
        # Scale UV to cell space
        sx = u * self.scale
        sy = v * self.scale

        # Current cell and fractional offset
        cx = math.floor(sx)
        cy = math.floor(sy)
        fx = sx - cx
        fy = sy - cy

        # Find the distance to the nearest two cell centers (3x3 neighborhood)
        min_dist = float("inf")
        min_dist2 = float("inf")
        nearest_cell = (0, 0)

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = int(cx) + dx, int(cy) + dy
                cell_cx, cell_cy = self._get_cell(nx, ny)
                # Distance from sample point to this cell center
                dist_x = (cell_cx + nx) - sx
                dist_y = (cell_cy + ny) - sy
                dist = math.sqrt(dist_x * dist_x + dist_y * dist_y)

                if dist < min_dist:
                    min_dist2 = min_dist
                    min_dist = dist
                    nearest_cell = (nx, ny)
                elif dist < min_dist2:
                    min_dist2 = dist

        # Compute value based on cell_type
        if self.cell_type == "distance":
            t = min_dist  # Range [0, ~0.7]
        elif self.cell_type == "f1":
            t = min_dist
        elif self.cell_type == "f2":
            t = min_dist2 - min_dist  # Edge-like pattern
        elif self.cell_type == "cell_id":
            # Use cell index to create distinct colored cells
            h = hashlib.md5(f"{nearest_cell[0]},{nearest_cell[1]}".encode()).hexdigest()
            t = int(h[:8], 16) / 0xFFFFFFFF
        else:
            t = min_dist

        # Normalize t to [0, 1]
        if self.cell_type != "cell_id":
            t = np.clip(t * 1.5, 0.0, 1.0)  # Scale factor to use full range

        return self.color1 + t * (self.color2 - self.color1)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "scale": self.scale,
            "cell_type": self.cell_type,
            "color1": tuple(self.color1.tolist()),
            "color2": tuple(self.color2.tolist()),
            "seed": self._seed,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoronoiTexture":
        return cls(
            scale=data.get("scale", 10.0),
            cell_type=data.get("cell_type", "distance"),
            color1=tuple(data.get("color1", (0, 0, 0))),
            color2=tuple(data.get("color2", (1, 1, 1))),
            seed=data.get("seed", 42),
        )


class ImageTexture(Texture):
    """
    Load a texture from an image file (PNG, JPG, BMP, etc.).

    Supports bilinear filtering and UV wrap modes (repeat, clamp, mirror).

    Args:
        filepath: Path to the image file.
        wrap: UV wrapping mode — 'repeat', 'clamp', or 'mirror'.
        filter_mode: Filtering mode — 'nearest' or 'bilinear'.
    """

    WRAP_MODES = ("repeat", "clamp", "mirror")
    FILTER_MODES = ("nearest", "bilinear")

    def __init__(
        self,
        filepath: str,
        wrap: str = "repeat",
        filter_mode: str = "bilinear",
    ):
        if wrap not in self.WRAP_MODES:
            raise ValueError(f"wrap must be one of {self.WRAP_MODES}, got '{wrap}'")
        if filter_mode not in self.FILTER_MODES:
            raise ValueError(f"filter_mode must be one of {self.FILTER_MODES}, got '{filter_mode}'")

        self.filepath = filepath
        self.wrap = wrap
        self.filter_mode = filter_mode
        self._image: Optional[np.ndarray] = None
        self._width: int = 0
        self._height: int = 0
        self._load()

    def _load(self) -> None:
        """Load the image from disk and store as float64 numpy array."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required to load images. Install with: pip install Pillow")

        pil_img = Image.open(self.filepath).convert("RGB")
        self._width, self._height = pil_img.size
        self._image = np.array(pil_img, dtype=np.float64) / 255.0  # (H, W, 3) in [0, 1]

    def _wrap_coord(self, coord: float, max_val: int) -> float:
        """Apply UV wrapping to a coordinate."""
        if self.wrap == "repeat":
            coord = coord % 1.0
            if coord < 0:
                coord += 1.0
        elif self.wrap == "clamp":
            coord = np.clip(coord, 0.0, 1.0)
        elif self.wrap == "mirror":
            coord = abs(coord) % 2.0
            if coord > 1.0:
                coord = 2.0 - coord
        return coord

    def _sample_nearest(self, u: float, v: float) -> np.ndarray:
        """Nearest-neighbor sampling."""
        u = self._wrap_coord(u, self._width)
        v = self._wrap_coord(v, self._height)
        x = int(u * self._width) % self._width
        y = int(v * self._height) % self._height
        return self._image[y, x].copy()

    def _sample_bilinear(self, u: float, v: float) -> np.ndarray:
        """Bilinear interpolated sampling."""
        u = self._wrap_coord(u, self._width)
        v = self._wrap_coord(v, self._height)

        # Continuous pixel coordinates
        px = u * (self._width - 1)
        py = v * (self._height - 1)

        x0 = int(math.floor(px))
        y0 = int(math.floor(py))
        x1 = (x0 + 1) % self._width
        y1 = (y0 + 1) % self._height

        fx = px - x0
        fy = py - y0

        # Bilinear interpolation
        c00 = self._image[y0, x0]
        c10 = self._image[y0, x1]
        c01 = self._image[y1, x0]
        c11 = self._image[y1, x1]

        top = c00 * (1.0 - fx) + c10 * fx
        bottom = c01 * (1.0 - fx) + c11 * fx
        return top * (1.0 - fy) + bottom * fy

    def sample(self, u: float, v: float) -> np.ndarray:
        if self._image is None:
            return np.array([0.0, 0.0, 0.0])
        if self.filter_mode == "nearest":
            return self._sample_nearest(u, v)
        return self._sample_bilinear(u, v)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "filepath": self.filepath,
            "wrap": self.wrap,
            "filter_mode": self.filter_mode,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageTexture":
        return cls(
            filepath=data["filepath"],
            wrap=data.get("wrap", "repeat"),
            filter_mode=data.get("filter_mode", "bilinear"),
        )


# =============================================================================
# Section 4: Material Layering System
# =============================================================================

# Supported blend modes for material layers
BLEND_MODES = ("mix", "multiply", "add", "overlay")


def _blend_value(base: float, layer_val: float, mode: str, opacity: float) -> float:
    """
    Blend a single channel value using the specified blend mode.

    Args:
        base: Base layer value [0, 1].
        layer_val: Layer value [0, 1].
        mode: Blend mode ('mix', 'multiply', 'add', 'overlay').
        opacity: Layer opacity [0, 1].

    Returns:
        Blended value in [0, 1].
    """
    if opacity <= 0.0:
        return base

    if mode == "mix":
        result = base * (1.0 - opacity) + layer_val * opacity
    elif mode == "multiply":
        result = base * (1.0 - opacity) + (base * layer_val) * opacity
    elif mode == "add":
        result = base + layer_val * opacity
    elif mode == "overlay":
        # Overlay: if base < 0.5 → 2*base*layer; else 1 - 2*(1-base)*(1-layer)
        blended = np.where(
            base < 0.5,
            2.0 * base * layer_val,
            1.0 - 2.0 * (1.0 - base) * (1.0 - layer_val),
        )
        result = base * (1.0 - opacity) + blended * opacity
    else:
        result = base * (1.0 - opacity) + layer_val * opacity

    return float(np.clip(result, 0.0, 1.0))


class MaterialLayer:
    """
    A single material layer with blend mode.

    Each layer can carry texture maps for albedo (base color), metallic,
    roughness, and normal. Textures are sampled at UV coordinates.

    Args:
        name: Human-readable layer name.
        base_color_texture: Texture or solid RGB color for the albedo.
        metallic_texture: Texture or float for metallic channel.
        roughness_texture: Texture or float for roughness channel.
        normal_texture: Texture for normal map (used for bump).
        opacity: Layer opacity [0, 1] controlling blend strength.
        blend_mode: How this layer blends with layers below.
    """

    def __init__(
        self,
        name: str,
        base_color_texture=None,
        metallic_texture=None,
        roughness_texture=None,
        normal_texture=None,
        opacity: float = 1.0,
        blend_mode: str = "mix",
    ):
        if blend_mode not in BLEND_MODES:
            raise ValueError(f"blend_mode must be one of {BLEND_MODES}, got '{blend_mode}'")

        self.name = name
        self.opacity = float(np.clip(opacity, 0.0, 1.0))
        self.blend_mode = blend_mode

        # Normalize inputs: if it's not a Texture, make a solid-color texture
        self.base_color_texture = self._ensure_texture(base_color_texture, (0.5, 0.5, 0.5))
        self.metallic_texture = self._ensure_scalar_texture(metallic_texture, 0.0)
        self.roughness_texture = self._ensure_scalar_texture(roughness_texture, 0.5)
        self.normal_texture = normal_texture  # Can be None

    @staticmethod
    def _ensure_texture(value, default_color) -> Texture:
        """Convert a color tuple/list to a GradientTexture if needed."""
        if value is None:
            value = default_color
        if isinstance(value, Texture):
            return value
        # Assume it's a solid color
        return GradientTexture(stops=[(0.0, tuple(value)), (1.0, tuple(value))])

    @staticmethod
    def _ensure_scalar_texture(value, default: float) -> Texture:
        """Convert a scalar to a constant-valued texture (via GradientTexture)."""
        if value is None:
            value = default
        if isinstance(value, Texture):
            return value
        if isinstance(value, (int, float)):
            val = float(value)
            return GradientTexture(stops=[(0.0, (val, val, val)), (1.0, (val, val, val))])
        # Assume a color tuple but use only R channel for scalar
        return GradientTexture(stops=[(0.0, (float(value[0]),) * 3), (1.0, (float(value[0]),) * 3)])

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "opacity": self.opacity,
            "blend_mode": self.blend_mode,
        }
        if isinstance(self.base_color_texture, Texture):
            d["base_color_texture"] = self.base_color_texture.to_dict()
        if isinstance(self.metallic_texture, Texture):
            d["metallic_texture"] = self.metallic_texture.to_dict()
        if isinstance(self.roughness_texture, Texture):
            d["roughness_texture"] = self.roughness_texture.to_dict()
        if self.normal_texture is not None and isinstance(self.normal_texture, Texture):
            d["normal_texture"] = self.normal_texture.to_dict()
        return d


class LayeredMaterial:
    """
    Stacked material layers with configurable blend modes.

    Layers are evaluated bottom-to-top (first added = bottom). Each layer
    is blended onto the accumulated result using its blend mode and opacity.

    Returns a dictionary of PBR parameters:
        - base_color: np.ndarray (3,) — RGB in [0, 1]
        - metallic: float — [0, 1]
        - roughness: float — [0, 1]
        - normal: np.ndarray (3,) — tangent-space normal (if available)
    """

    def __init__(self, name: str = "LayeredMaterial"):
        self.name = name
        self._layers: List[MaterialLayer] = []

    def add_layer(self, layer: MaterialLayer) -> None:
        """Add a layer to the top of the stack."""
        self._layers.append(layer)

    def remove_layer(self, name: str) -> bool:
        """Remove a layer by name. Returns True if found and removed."""
        for i, layer in enumerate(self._layers):
            if layer.name == name:
                self._layers.pop(i)
                return True
        return False

    def get_layer(self, name: str) -> Optional[MaterialLayer]:
        """Get a layer by name."""
        for layer in self._layers:
            if layer.name == name:
                return layer
        return None

    @property
    def layers(self) -> List[MaterialLayer]:
        return list(self._layers)

    def evaluate(self, u: float, v: float) -> Dict[str, Any]:
        """
        Evaluate all layers at UV coordinates, blending bottom-to-top.

        Returns:
            Dict with keys: base_color, metallic, roughness, normal.
        """
        # Start with defaults
        result = {
            "base_color": np.array([0.5, 0.5, 0.5], dtype=np.float64),
            "metallic": 0.0,
            "roughness": 0.5,
            "normal": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }

        for layer in self._layers:
            # Sample layer textures
            layer_color = layer.base_color_texture.sample(u, v)
            layer_metallic = layer.metallic_texture.sample(u, v)[0]  # Use R channel
            layer_roughness = layer.roughness_texture.sample(u, v)[0]  # Use R channel

            # Blend each channel
            new_color = np.array([
                _blend_value(result["base_color"][0], layer_color[0], layer.blend_mode, layer.opacity),
                _blend_value(result["base_color"][1], layer_color[1], layer.blend_mode, layer.opacity),
                _blend_value(result["base_color"][2], layer_color[2], layer.blend_mode, layer.opacity),
            ])
            result["base_color"] = new_color
            result["metallic"] = _blend_value(result["metallic"], layer_metallic, layer.blend_mode, layer.opacity)
            result["roughness"] = _blend_value(result["roughness"], layer_roughness, layer.blend_mode, layer.opacity)

            # Normal maps are handled differently — they modify the surface normal
            if layer.normal_texture is not None:
                normal_sample = layer.normal_texture.sample(u, v)
                # Convert from [0,1] to [-1,1] range
                tangent_normal = normal_sample * 2.0 - 1.0
                # Normalize
                norm = np.linalg.norm(tangent_normal)
                if norm > 1e-6:
                    tangent_normal /= norm
                # Blend normals using opacity
                result["normal"] = (
                    result["normal"] * (1.0 - layer.opacity) + tangent_normal * layer.opacity
                )
                norm = np.linalg.norm(result["normal"])
                if norm > 1e-6:
                    result["normal"] /= norm

        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "layers": [layer.to_dict() for layer in self._layers],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayeredMaterial":
        mat = cls(name=data.get("name", "LayeredMaterial"))
        for layer_data in data.get("layers", []):
            layer = MaterialLayer(
                name=layer_data.get("name", "unnamed"),
                opacity=layer_data.get("opacity", 1.0),
                blend_mode=layer_data.get("blend_mode", "mix"),
            )
            if "base_color_texture" in layer_data:
                layer.base_color_texture = Texture.from_dict(layer_data["base_color_texture"])
            if "metallic_texture" in layer_data:
                layer.metallic_texture = Texture.from_dict(layer_data["metallic_texture"])
            if "roughness_texture" in layer_data:
                layer.roughness_texture = Texture.from_dict(layer_data["roughness_texture"])
            if "normal_texture" in layer_data:
                layer.normal_texture = Texture.from_dict(layer_data["normal_texture"])
            mat.add_layer(layer)
        return mat


# =============================================================================
# Section 5: UV Mapping Tools
# =============================================================================

class UVMapper:
    """
    Automatic UV generation utilities for meshes.

    Provides static methods for common UV projection strategies used in
    3D modeling and rendering. All methods accept vertex and normal arrays
    and return UV coordinates as a numpy array of shape (N, 2).

    Degenerate normals (zero-length) are handled gracefully by defaulting
    to the dominant axis.
    """

    @staticmethod
    def planar_projection(
        vertices: np.ndarray,
        normals: Optional[np.ndarray] = None,
        axis: str = "z",
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Generate UVs using planar projection.

        Projects vertices onto a 2D plane perpendicular to the given axis.

        Args:
            vertices: np.ndarray of shape (N, 3) — mesh vertex positions.
            normals: Optional np.ndarray of shape (N, 3) — vertex normals (unused
                     for planar projection, accepted for API consistency).
            axis: Projection axis — 'x', 'y', or 'z'.
            scale: UV scale factor.

        Returns:
            np.ndarray of shape (N, 2) with UV coordinates.
        """
        axis = axis.lower()
        if axis not in ("x", "y", "z"):
            raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")

        n = vertices.shape[0]
        uvs = np.zeros((n, 2), dtype=np.float64)

        if axis == "x":
            uvs[:, 0] = vertices[:, 1] * scale  # y → u
            uvs[:, 1] = vertices[:, 2] * scale  # z → v
        elif axis == "y":
            uvs[:, 0] = vertices[:, 0] * scale  # x → u
            uvs[:, 1] = vertices[:, 2] * scale  # z → v
        elif axis == "z":
            uvs[:, 0] = vertices[:, 0] * scale  # x → u
            uvs[:, 1] = vertices[:, 1] * scale  # y → v

        return uvs

    @staticmethod
    def box_projection(
        vertices: np.ndarray,
        normals: Optional[np.ndarray] = None,
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Generate UVs using box (triplanar) projection.

        For each vertex, the dominant normal axis is determined and the
        vertex is projected onto the corresponding plane. This produces
        the best results for architectural and hard-surface meshes.

        If normals are not provided or are degenerate, the dominant axis
        is computed from the vertex position relative to the centroid.

        Args:
            vertices: np.ndarray of shape (N, 3).
            normals: Optional np.ndarray of shape (N, 3).
            scale: UV scale factor.

        Returns:
            np.ndarray of shape (N, 2).
        """
        n = vertices.shape[0]
        uvs = np.zeros((n, 2), dtype=np.float64)

        # Use normals if provided, otherwise derive from position offset
        if normals is None:
            centroid = np.mean(vertices, axis=0)
            normals = vertices - centroid
            # Normalize
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            normals = normals / norms

        # Handle degenerate normals
        norms = np.linalg.norm(normals, axis=1)
        degenerate = norms < 1e-8
        if np.any(degenerate):
            # For degenerate normals, use the z-axis
            normals[degenerate] = np.array([0.0, 0.0, 1.0])
            norms[degenerate] = 1.0
        normals = normals / norms[:, np.newaxis]

        # Absolute normal components to determine dominant axis
        abs_normals = np.abs(normals)

        for i in range(n):
            ax, ay, az = abs_normals[i]

            if ax >= ay and ax >= az:
                # X-dominant: project onto YZ plane
                uvs[i, 0] = vertices[i, 1] * scale
                uvs[i, 1] = vertices[i, 2] * scale
            elif ay >= ax and ay >= az:
                # Y-dominant: project onto XZ plane
                uvs[i, 0] = vertices[i, 0] * scale
                uvs[i, 1] = vertices[i, 2] * scale
            else:
                # Z-dominant: project onto XY plane
                uvs[i, 0] = vertices[i, 0] * scale
                uvs[i, 1] = vertices[i, 1] * scale

        return uvs

    @staticmethod
    def spherical_projection(
        vertices: np.ndarray,
        center: Optional[np.ndarray] = None,
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Generate UVs using spherical projection.

        Maps each vertex to UV by computing the spherical coordinates
        relative to a center point. Best for objects that are roughly
        spherical (planets, heads, etc.).

        Args:
            vertices: np.ndarray of shape (N, 3).
            center: Center point for the sphere. If None, uses the centroid.
            scale: UV scale factor.

        Returns:
            np.ndarray of shape (N, 2).
        """
        if center is None:
            center = np.mean(vertices, axis=0)

        # Offset from center
        delta = vertices - center
        # Convert to spherical coordinates
        x, y, z = delta[:, 0], delta[:, 1], delta[:, 2]
        r = np.linalg.norm(delta, axis=1)
        r = np.maximum(r, 1e-8)  # Avoid division by zero

        # u = atan2(z, x) / (2π)  →  [0, 1]
        # v = asin(y / r) / π + 0.5  →  [0, 1]
        u = (np.arctan2(z, x) / (2.0 * np.pi)) % 1.0
        v = np.arcsin(np.clip(y / r, -1.0, 1.0)) / np.pi + 0.5

        uvs = np.stack([u, v], axis=1) * scale
        return uvs

    @staticmethod
    def cylindrical_projection(
        vertices: np.ndarray,
        axis: str = "y",
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Generate UVs using cylindrical projection.

        Wraps UVs around the specified axis. Best for cylindrical objects
        (pillars, pipes, limbs, etc.).

        Args:
            vertices: np.ndarray of shape (N, 3).
            axis: The cylinder axis — 'x', 'y', or 'z'.
            scale: UV scale factor.

        Returns:
            np.ndarray of shape (N, 2).
        """
        axis = axis.lower()
        if axis not in ("x", "y", "z"):
            raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")

        n = vertices.shape[0]
        uvs = np.zeros((n, 2), dtype=np.float64)

        # Choose the two radial axes
        if axis == "x":
            a1, a2 = vertices[:, 1], vertices[:, 2]
            height = vertices[:, 0]
        elif axis == "y":
            a1, a2 = vertices[:, 0], vertices[:, 2]
            height = vertices[:, 1]
        else:  # z
            a1, a2 = vertices[:, 0], vertices[:, 1]
            height = vertices[:, 2]

        # u = atan2(a2, a1) / (2π)
        u = (np.arctan2(a2, a1) / (2.0 * np.pi)) % 1.0

        # v = normalized height
        h_min = np.min(height)
        h_max = np.max(height)
        h_range = h_max - h_min
        if h_range < 1e-8:
            v = np.full(n, 0.5)
        else:
            v = (height - h_min) / h_range

        uvs[:, 0] = u * scale
        uvs[:, 1] = v * scale
        return uvs


# =============================================================================
# Section 6: PBR Material Library
# =============================================================================

class MaterialLibrary:
    """
    Pre-built PBR materials matching real-world physical values.

    All material parameters follow standard PBR conventions:
        - albedo: Base RGB color (linear sRGB, [0, 1])
        - metallic: Metalness [0, 1]
        - roughness: Surface roughness [0, 1]
        - ior: Index of refraction (for dielectrics)
        - transmission: Transmittance [0, 1] (0 = opaque, 1 = fully transparent)
        - sss_radius: Subsurface scattering radius in mm (R, G, B)

    Values are sourced from established PBR references including:
        - Filament material system (Google)
        - Blender Principled BSDF defaults
        - Mitsuba/warp renderers
        - Measured real-world data
    """

    # Internal database of materials
    _MATERIALS: Dict[str, Dict[str, Any]] = {
        # ---- Metals ----
        "gold": {
            "albedo": (1.000, 0.766, 0.336),
            "metallic": 1.0,
            "roughness": 0.25,
            "ior": 0.470,         # Complex IOR (n,k) — n component
            "transmission": 0.0,
            "sss_radius": (0.0, 0.0, 0.0),
            "description": "Polished gold — warm yellow metallic reflection.",
        },
        "copper": {
            "albedo": (0.955, 0.638, 0.538),
            "metallic": 1.0,
            "roughness": 0.30,
            "ior": 0.620,
            "transmission": 0.0,
            "sss_radius": (0.0, 0.0, 0.0),
            "description": "Polished copper — reddish metallic reflection.",
        },
        "aluminum": {
            "albedo": (0.913, 0.922, 0.924),
            "metallic": 1.0,
            "roughness": 0.30,
            "ior": 1.500,
            "transmission": 0.0,
            "sss_radius": (0.0, 0.0, 0.0),
            "description": "Brushed aluminum — neutral silver reflection.",
        },
        "iron": {
            "albedo": (0.560, 0.570, 0.580),
            "metallic": 1.0,
            "roughness": 0.55,
            "ior": 2.940,
            "transmission": 0.0,
            "sss_radius": (0.0, 0.0, 0.0),
            "description": "Raw iron — dark gray metallic surface.",
        },
        "chrome": {
            "albedo": (0.549, 0.556, 0.554),
            "metallic": 1.0,
            "roughness": 0.05,
            "ior": 3.110,
            "transmission": 0.0,
            "sss_radius": (0.0, 0.0, 0.0),
            "description": "Mirror-polished chrome — very high reflectivity.",
        },
        "silver": {
            "albedo": (0.972, 0.960, 0.915),
            "metallic": 1.0,
            "roughness": 0.20,
            "ior": 0.180,
            "transmission": 0.0,
            "sss_radius": (0.0, 0.0, 0.0),
            "description": "Polished silver — bright neutral metallic reflection.",
        },
        "titanium": {
            "albedo": (0.542, 0.497, 0.449),
            "metallic": 1.0,
            "roughness": 0.35,
            "ior": 2.160,
            "transmission": 0.0,
            "sss_radius": (0.0, 0.0, 0.0),
            "description": "Titanium — medium gray with slight warm tint.",
        },

        # ---- Dielectrics (transparent / refractive) ----
        "glass": {
            "albedo": (1.000, 1.000, 1.000),
            "metallic": 0.0,
            "roughness": 0.02,
            "ior": 1.520,
            "transmission": 1.0,
            "sss_radius": (0.0, 0.0, 0.0),
            "description": "Clear glass — nearly perfect transparency with slight reflection.",
        },
        "water": {
            "albedo": (0.800, 0.860, 0.920),
            "metallic": 0.0,
            "roughness": 0.05,
            "ior": 1.333,
            "transmission": 0.95,
            "sss_radius": (0.0, 0.0, 0.0),
            "description": "Water — slight blue tint, high transmission.",
        },
        "diamond": {
            "albedo": (1.000, 1.000, 1.000),
            "metallic": 0.0,
            "roughness": 0.00,
            "ior": 2.417,
            "transmission": 1.0,
            "sss_radius": (0.0, 0.0, 0.0),
            "description": "Diamond — very high IOR, perfect transparency.",
        },
        "amber": {
            "albedo": (0.850, 0.600, 0.200),
            "metallic": 0.0,
            "roughness": 0.15,
            "ior": 1.550,
            "transmission": 0.80,
            "sss_radius": (5.0, 3.0, 1.0),
            "description": "Amber — warm translucent gemstone with subsurface scattering.",
        },
        "ice": {
            "albedo": (0.900, 0.950, 1.000),
            "metallic": 0.0,
            "roughness": 0.10,
            "ior": 1.310,
            "transmission": 0.90,
            "sss_radius": (3.0, 3.0, 3.0),
            "description": "Ice — translucent with subtle blue tint.",
        },

        # ---- Diffuse (non-metallic, opaque) ----
        "rubber": {
            "albedo": (0.050, 0.050, 0.050),
            "metallic": 0.0,
            "roughness": 0.90,
            "ior": 1.519,
            "transmission": 0.0,
            "sss_radius": (0.0, 0.0, 0.0),
            "description": "Black rubber — very dark, very rough, non-reflective.",
        },
        "plastic": {
            "albedo": (0.300, 0.300, 0.300),
            "metallic": 0.0,
            "roughness": 0.40,
            "ior": 1.460,
            "transmission": 0.0,
            "sss_radius": (0.0, 0.0, 0.0),
            "description": "Generic matte plastic — medium gray, low sheen.",
        },
        "red_plastic": {
            "albedo": (0.600, 0.040, 0.040),
            "metallic": 0.0,
            "roughness": 0.35,
            "ior": 1.460,
            "transmission": 0.0,
            "sss_radius": (0.0, 0.0, 0.0),
            "description": "Red plastic — classic shiny red surface.",
        },
        "leather": {
            "albedo": (0.340, 0.160, 0.080),
            "metallic": 0.0,
            "roughness": 0.70,
            "ior": 1.500,
            "transmission": 0.0,
            "sss_radius": (0.5, 0.3, 0.2),
            "description": "Brown leather — dark, rough, slight subsurface.",
        },
        "fabric": {
            "albedo": (0.350, 0.220, 0.180),
            "metallic": 0.0,
            "roughness": 0.95,
            "ior": 1.450,
            "transmission": 0.0,
            "sss_radius": (0.0, 0.0, 0.0),
            "description": "Woven fabric — very rough, no specular highlights.",
        },
        "concrete": {
            "albedo": (0.550, 0.530, 0.500),
            "metallic": 0.0,
            "roughness": 0.85,
            "ior": 1.500,
            "transmission": 0.0,
            "sss_radius": (0.0, 0.0, 0.0),
            "description": "Raw concrete — medium gray, very rough.",
        },
        "marble_white": {
            "albedo": (0.920, 0.910, 0.890),
            "metallic": 0.0,
            "roughness": 0.20,
            "ior": 1.490,
            "transmission": 0.0,
            "sss_radius": (2.0, 2.0, 2.0),
            "description": "White marble — polished stone with subsurface translucency.",
        },
        "sandstone": {
            "albedo": (0.760, 0.680, 0.520),
            "metallic": 0.0,
            "roughness": 0.80,
            "ior": 1.500,
            "transmission": 0.0,
            "sss_radius": (0.0, 0.0, 0.0),
            "description": "Sandstone — warm sandy color, rough.",
        },
        "clay": {
            "albedo": (0.640, 0.450, 0.320),
            "metallic": 0.0,
            "roughness": 0.90,
            "ior": 1.500,
            "transmission": 0.0,
            "sss_radius": (1.0, 0.8, 0.6),
            "description": "Raw clay — brownish-orange, very matte with slight SSS.",
        },
        "ceramic": {
            "albedo": (0.950, 0.930, 0.880),
            "metallic": 0.0,
            "roughness": 0.25,
            "ior": 1.500,
            "transmission": 0.0,
            "sss_radius": (0.0, 0.0, 0.0),
            "description": "White ceramic — glossy smooth surface.",
        },

        # ---- Subsurface Scattering Materials ----
        "skin": {
            "albedo": (0.780, 0.570, 0.440),
            "metallic": 0.0,
            "roughness": 0.55,
            "ior": 1.400,
            "transmission": 0.30,
            "sss_radius": (3.6, 1.6, 0.8),
            "description": "Human skin — warm tone, strong subsurface scattering.",
        },
        "wax": {
            "albedo": (0.950, 0.850, 0.550),
            "metallic": 0.0,
            "roughness": 0.40,
            "ior": 1.430,
            "transmission": 0.50,
            "sss_radius": (4.0, 2.0, 1.0),
            "description": "Candle wax — translucent, warm glow under light.",
        },
        "jade": {
            "albedo": (0.200, 0.520, 0.250),
            "metallic": 0.0,
            "roughness": 0.30,
            "ior": 1.620,
            "transmission": 0.60,
            "sss_radius": (5.0, 5.0, 3.0),
            "description": "Jade — green translucent gemstone with deep SSS.",
        },
        "milk": {
            "albedo": (0.960, 0.950, 0.920),
            "metallic": 0.0,
            "roughness": 0.60,
            "ior": 1.350,
            "transmission": 0.70,
            "sss_radius": (8.0, 5.0, 3.0),
            "description": "Milk — highly scattering, creamy white.",
        },
        "fruit_flesh": {
            "albedo": (0.900, 0.300, 0.200),
            "metallic": 0.0,
            "roughness": 0.45,
            "ior": 1.350,
            "transmission": 0.40,
            "sss_radius": (6.0, 2.0, 1.0),
            "description": "Fruit flesh (e.g., watermelon) — red/orange translucent.",
        },
        "ketchup": {
            "albedo": (0.600, 0.020, 0.010),
            "metallic": 0.0,
            "roughness": 0.50,
            "ior": 1.390,
            "transmission": 0.50,
            "sss_radius": (2.0, 0.4, 0.2),
            "description": "Ketchup — deep red with strong subsurface scattering.",
        },

        # ---- Special ----
        "emissive_white": {
            "albedo": (1.000, 1.000, 1.000),
            "metallic": 0.0,
            "roughness": 0.50,
            "ior": 1.500,
            "transmission": 0.0,
            "sss_radius": (0.0, 0.0, 0.0),
            "emissive": (1.0, 1.0, 1.0),
            "emissive_intensity": 5.0,
            "description": "White emissive surface (light source).",
        },
    }

    @classmethod
    def get(cls, name: str) -> Dict[str, Any]:
        """
        Look up a pre-built material by name.

        Args:
            name: Material name (case-insensitive).

        Returns:
            Dict of material parameters (albedo, metallic, roughness, etc.).

        Raises:
            KeyError: If the material name is not found.
        """
        key = name.lower()
        if key not in cls._MATERIALS:
            available = ", ".join(sorted(cls._MATERIALS.keys()))
            raise KeyError(
                f"Material '{name}' not found. Available materials: {available}"
            )
        # Return a copy to prevent mutation of the internal database
        import copy
        return copy.deepcopy(cls._MATERIALS[key])

    @classmethod
    def list_materials(cls) -> List[str]:
        """Return a sorted list of all available material names."""
        return sorted(cls._MATERIALS.keys())

    @classmethod
    def register(cls, name: str, params: Dict[str, Any]) -> None:
        """
        Register a custom material in the library.

        Args:
            name: Material name.
            params: Dict with keys: albedo, metallic, roughness, ior, transmission, sss_radius.
        """
        required = {"albedo", "metallic", "roughness", "ior", "transmission", "sss_radius"}
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Material params missing required keys: {missing}")
        cls._MATERIALS[name.lower()] = params


# =============================================================================
# Section 7: HDR Environment / Image-Based Lighting
# =============================================================================

class EnvironmentMap:
    """
    Procedural HDR environment map for image-based lighting.

    Generates a simple analytical sky model that can be sampled for
    ambient lighting, reflections, and direct sun illumination.

    Supported types:
        - 'gradient': Smooth sky-to-horizon-to-ground gradient with optional sun.
        - 'solid': Uniform color environment.
        - 'studio': Neutral studio lighting setup with bright upper hemisphere.

    Args:
        env_type: Environment type ('gradient', 'solid', 'studio').
        sun_direction: Normalized direction TO the sun (3-tuple). None disables the sun.
        sun_color: RGB color of the sun.
        sky_color: Zenith sky color.
        ground_color: Ground plane color.
        sun_intensity: Brightness multiplier for the sun disk.
        sky_exposure: Overall exposure multiplier for the sky.
    """

    ENV_TYPES = ("gradient", "solid", "studio")

    def __init__(
        self,
        env_type: str = "gradient",
        sun_direction: Optional[Tuple[float, float, float]] = None,
        sun_color: Tuple[float, float, float] = (1.0, 1.0, 0.9),
        sky_color: Tuple[float, float, float] = (0.4, 0.6, 1.0),
        ground_color: Tuple[float, float, float] = (0.2, 0.2, 0.2),
        sun_intensity: float = 5.0,
        sky_exposure: float = 1.0,
    ):
        if env_type not in self.ENV_TYPES:
            raise ValueError(f"env_type must be one of {self.ENV_TYPES}, got '{env_type}'")

        self.env_type = env_type
        self.sun_color = np.array(sun_color, dtype=np.float64)
        self.sky_color = np.array(sky_color, dtype=np.float64)
        self.ground_color = np.array(ground_color, dtype=np.float64)
        self.sun_intensity = sun_intensity
        self.sky_exposure = sky_exposure

        if sun_direction is not None:
            sd = np.array(sun_direction, dtype=np.float64)
            norm = np.linalg.norm(sd)
            if norm < 1e-8:
                raise ValueError("sun_direction must be a non-zero vector")
            self.sun_direction = sd / norm
        else:
            self.sun_direction = None

    def sample(self, direction: np.ndarray) -> np.ndarray:
        """
        Sample the environment color for a given world-space direction.

        Args:
            direction: Normalized direction vector (3,).

        Returns:
            np.ndarray of shape (3,) — RGB radiance values (HDR).
        """
        direction = np.asarray(direction, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return np.array([0.0, 0.0, 0.0])
        direction = direction / norm

        y = direction[1]  # Up component

        if self.env_type == "gradient":
            color = self._sample_gradient(direction, y)
        elif self.env_type == "solid":
            color = self.sky_color.copy()
        elif self.env_type == "studio":
            color = self._sample_studio(direction, y)
        else:
            color = self.sky_color.copy()

        # Add sun if present
        if self.sun_direction is not None:
            color = color + self._sample_sun(direction)

        return color * self.sky_exposure

    def _sample_gradient(self, direction: np.ndarray, y: float) -> np.ndarray:
        """Sample the gradient sky model (sky → horizon → ground)."""
        if y > 0.0:
            # Sky hemisphere: interpolate from horizon to zenith
            t = y  # 0 at horizon, 1 at zenith
            # Use a power curve for more natural sky darkening at zenith
            horizon_color = self.sky_color * 0.7
            zenith_color = self.sky_color * 1.2
            color = horizon_color * (1.0 - t) + zenith_color * t
        else:
            # Ground hemisphere
            t = -y  # 0 at horizon, 1 at nadir
            horizon_color = self.ground_color * 1.2
            ground_color = self.ground_color * 0.5
            color = horizon_color * (1.0 - t) + ground_color * t

        return color

    def _sample_studio(self, direction: np.ndarray, y: float) -> np.ndarray:
        """Sample a neutral studio lighting environment."""
        if y > 0.0:
            # Bright upper hemisphere for soft studio illumination
            t = y
            base = np.array([0.4, 0.4, 0.4])
            bright = np.array([0.8, 0.8, 0.85])
            color = base * (1.0 - t) + bright * t
            # Slight brightening from the upper-front for key light feel
            key_boost = max(0.0, direction[2]) * 0.3  # Front direction (z)
            color = color + key_boost
        else:
            # Dimmer ground bounce
            t = -y
            base = np.array([0.2, 0.2, 0.2])
            dark = np.array([0.1, 0.1, 0.1])
            color = base * (1.0 - t) + dark * t

        return color

    def _sample_sun(self, direction: np.ndarray) -> np.ndarray:
        """Sample the sun disk contribution."""
        if self.sun_direction is None:
            return np.array([0.0, 0.0, 0.0])

        # Cosine of angle between direction and sun
        cos_angle = np.clip(np.dot(direction, self.sun_direction), -1.0, 1.0)
        angle = math.acos(cos_angle)

        # Sun disk angular radius: ~0.53 degrees (32 arcminutes)
        sun_radius = math.radians(0.53)
        # Soft glow around the sun (limb darkening approximation)
        glow_radius = math.radians(5.0)

        if angle < sun_radius:
            # Inside the sun disk — full brightness with limb darkening
            limb = math.sqrt(max(0.0, 1.0 - (angle / sun_radius) ** 2))
            return self.sun_color * self.sun_intensity * limb
        elif angle < glow_radius:
            # Sun glow / atmospheric scatter
            t = (angle - sun_radius) / (glow_radius - sun_radius)
            # Smooth falloff
            falloff = (1.0 - t) ** 3
            return self.sun_color * self.sun_intensity * 0.15 * falloff
        else:
            # Distant atmospheric scattering (Rayleigh-like)
            scatter = max(0.0, cos_angle) ** 8 * 0.02
            return self.sky_color * self.sun_intensity * scatter

    def importance_sample(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Importance sample the environment for Monte Carlo integration.

        Uses a simple two-strategy approach:
        1. With 30% probability, sample near the sun direction (if present).
        2. With 70% probability, sample the sky dome uniformly by cosine.

        Returns:
            Tuple of (direction, color, pdf):
                - direction: np.ndarray (3,) — sampled direction
                - color: np.ndarray (3,) — radiance at that direction
                - pdf: float — probability density of the sample
        """
        rng = np.random.default_rng()

        # Strategy 1: Sample near sun
        if self.sun_direction is not None and rng.random() < 0.3:
            # Perturb sun direction with a small Gaussian spread
            perturbation = rng.normal(0, 0.05, size=3)
            direction = self.sun_direction + perturbation
            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                direction = self.sun_direction.copy()
            else:
                direction = direction / norm
            color = self.sample(direction)
            # Approximate PDF for sun region
            sun_solid_angle = 2.0 * math.pi * (1.0 - math.cos(math.radians(5.0)))
            pdf = 0.3 / sun_solid_angle
            return direction, color, pdf

        # Strategy 2: Cosine-weighted hemisphere sampling
        # We sample a direction with PDF proportional to cos(theta)
        u1 = rng.random()
        u2 = rng.random()

        # Uniform direction on the sphere
        phi = 2.0 * math.pi * u1
        cos_theta = 2.0 * u2 - 1.0  # [-1, 1]
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))

        direction = np.array([
            sin_theta * math.cos(phi),
            cos_theta,
            sin_theta * math.sin(phi),
        ])

        color = self.sample(direction)
        # Uniform sphere PDF = 1 / (4π)
        pdf = 0.7 / (4.0 * math.pi)

        return direction, color, pdf

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the environment map to a dictionary."""
        d = {
            "type": "EnvironmentMap",
            "env_type": self.env_type,
            "sun_color": tuple(self.sun_color.tolist()),
            "sky_color": tuple(self.sky_color.tolist()),
            "ground_color": tuple(self.ground_color.tolist()),
            "sun_intensity": self.sun_intensity,
            "sky_exposure": self.sky_exposure,
        }
        if self.sun_direction is not None:
            d["sun_direction"] = tuple(self.sun_direction.tolist())
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvironmentMap":
        """Deserialize an environment map from a dictionary."""
        return cls(
            env_type=data.get("env_type", "gradient"),
            sun_direction=tuple(data["sun_direction"]) if "sun_direction" in data else None,
            sun_color=tuple(data.get("sun_color", (1.0, 1.0, 0.9))),
            sky_color=tuple(data.get("sky_color", (0.4, 0.6, 1.0))),
            ground_color=tuple(data.get("ground_color", (0.2, 0.2, 0.2))),
            sun_intensity=data.get("sun_intensity", 5.0),
            sky_exposure=data.get("sky_exposure", 1.0),
        )
