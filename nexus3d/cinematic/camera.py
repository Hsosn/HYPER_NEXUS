"""Nexus3D Cinematic Camera System.

Professional-grade virtual cinematography engine providing physically accurate
camera models, animated spline paths, depth-of-field simulation, motion blur,
procedural camera shake, exposure control, and realistic lens profiles.

All optics math follows real thin-lens equations. All exposure calculations
match the standard exposure triangle (aperture, shutter, ISO).
"""

import math
import json
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Union

import numpy as np

# ── Import math3d utilities (graceful fallback if not available) ──────────────
try:
    from nexus3d.math3d.core import (
        vec3, normalize, lerp, distance, cross, dot, magnitude,
        mat4_look_at, mat4_perspective, mat4_identity,
        quat, quat_identity, quat_slerp, quat_multiply,
        quat_from_axis_angle, quat_to_rotation_matrix,
        quat_to_euler, euler_to_quat,
        catmull_rom_spline, bezier_cubic,
    )
    _HAS_MATH3D = True
except ImportError:
    _HAS_MATH3D = False


# ══════════════════════════════════════════════════════════════════════════════
# Easing Functions
# ══════════════════════════════════════════════════════════════════════════════

def ease_in(t: float) -> float:
    """Quadratic ease-in: t^2."""
    return t * t


def ease_out(t: float) -> float:
    """Quadratic ease-out: 1-(1-t)^2."""
    return 1.0 - (1.0 - t) ** 2


def ease_in_out(t: float) -> float:
    """Quadratic ease-in-out (smooth step approximation)."""
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0


def ease_in_cubic(t: float) -> float:
    """Cubic ease-in: t^3."""
    return t * t * t


def ease_out_cubic(t: float) -> float:
    """Cubic ease-out: 1-(1-t)^3."""
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in-out."""
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0


def ease_in_quad(t: float) -> float:
    """Quadratic ease-in."""
    return t * t


def ease_out_quad(t: float) -> float:
    """Quadratic ease-out."""
    return 1.0 - (1.0 - t) ** 2


def ease_in_out_quad(t: float) -> float:
    """Quadratic ease-in-out."""
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0


# ══════════════════════════════════════════════════════════════════════════════
# Perlin Noise (Self-contained for camera shake)
# ══════════════════════════════════════════════════════════════════════════════

class PerlinNoise:
    """Classic Perlin noise generator for procedural camera shake.

    Implements improved Perlin noise with multiple octaves for natural-looking
    randomness.  Used internally by CameraShake; can also be used standalone.
    """

    def __init__(self, seed: int = 42):
        """Initialise the noise generator with a deterministic seed."""
        rng = np.random.RandomState(seed)
        self._perm = np.arange(256, dtype=np.int32)
        rng.shuffle(self._perm)
        self._perm = np.concatenate([self._perm, self._perm])

        # Pre-compute 2-D gradient vectors (12 directions as per Ken Perlin)
        self._grads = np.array([
            [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
            [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
            [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
        ], dtype=np.float64)

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _fade(t: float) -> float:
        """6t^5 - 15t^4 + 10t^3  (improved Perlin fade curve)."""
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + t * (b - a)

    def _grad3(self, hash_val: int, x: float, y: float, z: float) -> float:
        g = self._grads[hash_val % 12]
        return g[0] * x + g[1] * y + g[2] * z

    # ── Public API ─────────────────────────────────────────────────────────

    def noise3d(self, x: float, y: float, z: float) -> float:
        """Sample 3-D Perlin noise in the range approximately [-1, 1]."""
        X = int(math.floor(x)) & 255
        Y = int(math.floor(y)) & 255
        Z = int(math.floor(z)) & 255
        xf = x - math.floor(x)
        yf = y - math.floor(y)
        zf = z - math.floor(z)
        u = self._fade(xf)
        v = self._fade(yf)
        w = self._fade(zf)

        A  = self._perm[X] + Y
        AA = self._perm[A] + Z
        AB = self._perm[A + 1] + Z
        B  = self._perm[X + 1] + Y
        BA = self._perm[B] + Z
        BB = self._perm[B + 1] + Z

        return self._lerp(
            self._lerp(
                self._lerp(self._grad3(self._perm[AA],     xf,     yf,     zf),
                           self._grad3(self._perm[BA],     xf - 1, yf,     zf), u),
                self._lerp(self._grad3(self._perm[AB],     xf,     yf - 1, zf),
                           self._grad3(self._perm[BB],     xf - 1, yf - 1, zf), u),
                v),
            self._lerp(
                self._lerp(self._grad3(self._perm[AA + 1], xf,     yf,     zf - 1),
                           self._grad3(self._perm[BA + 1], xf - 1, yf,     zf - 1), u),
                self._lerp(self._grad3(self._perm[AB + 1], xf,     yf - 1, zf - 1),
                           self._grad3(self._perm[BB + 1], xf - 1, yf - 1, zf - 1), u),
                v),
            w)

    def fractal(self, x: float, y: float, z: float, octaves: int = 4,
                persistence: float = 0.5, lacunarity: float = 2.0) -> float:
        """Fractal Brownian Motion (layered noise) for richer detail."""
        value = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_val = 0.0
        for _ in range(octaves):
            value += self.noise3d(x * frequency, y * frequency, z * frequency) * amplitude
            max_val += amplitude
            amplitude *= persistence
            frequency *= lacunarity
        return value / max_val


# ══════════════════════════════════════════════════════════════════════════════
# Lens Profiles
# ══════════════════════════════════════════════════════════════════════════════

class LensProfile:
    """Optical characteristics for common photographic focal lengths.

    Each profile stores the nominal focal length, typical field-of-view
    on a full-frame sensor (36 x 24 mm), radial distortion coefficients
    (Brown–Conrady model: k1, k2, k3, p1, p2), and typical max aperture.
    """

    PROFILES: Dict[str, Dict[str, Any]] = {
        "14mm_ultrawide": {
            "focal_length": 14.0,
            "fov_deg": 114.0,
            "distortion": {"k1": -0.12, "k2": 0.04, "k3": -0.006, "p1": 0.0, "p2": 0.0},
            "max_aperture": 2.8,
            "min_focus_distance": 0.28,
            "weight_g": 780,
            "description": "Ultra-wide angle, strong barrel distortion",
        },
        "24mm_wide": {
            "focal_length": 24.0,
            "fov_deg": 84.0,
            "distortion": {"k1": -0.03, "k2": 0.008, "k3": -0.001, "p1": 0.0002, "p2": -0.0002},
            "max_aperture": 1.4,
            "min_focus_distance": 0.25,
            "weight_g": 645,
            "description": "Wide-angle, mild barrel distortion",
        },
        "35mm_standard": {
            "focal_length": 35.0,
            "fov_deg": 63.4,
            "distortion": {"k1": -0.005, "k2": 0.001, "k3": 0.0, "p1": 0.0001, "p2": -0.0001},
            "max_aperture": 1.4,
            "min_focus_distance": 0.25,
            "weight_g": 620,
            "description": "Standard wide, near-zero distortion",
        },
        "50mm_normal": {
            "focal_length": 50.0,
            "fov_deg": 46.8,
            "distortion": {"k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0},
            "max_aperture": 1.2,
            "min_focus_distance": 0.45,
            "weight_g": 580,
            "description": "'Nifty fifty', near-rectilinear, natural perspective",
        },
        "85mm_portrait": {
            "focal_length": 85.0,
            "fov_deg": 28.8,
            "distortion": {"k1": 0.001, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0},
            "max_aperture": 1.4,
            "min_focus_distance": 0.85,
            "weight_g": 950,
            "description": "Portrait telephoto, very slight pincushion",
        },
        "135mm_telephoto": {
            "focal_length": 135.0,
            "fov_deg": 18.2,
            "distortion": {"k1": 0.002, "k2": -0.001, "k3": 0.0, "p1": 0.0, "p2": 0.0},
            "max_aperture": 2.0,
            "min_focus_distance": 0.88,
            "weight_g": 905,
            "description": "Short telephoto, excellent subject isolation",
        },
        "200mm_telephoto": {
            "focal_length": 200.0,
            "fov_deg": 12.3,
            "distortion": {"k1": 0.003, "k2": -0.001, "k3": 0.0002, "p1": 0.0, "p2": 0.0},
            "max_aperture": 2.8,
            "min_focus_distance": 1.9,
            "weight_g": 2090,
            "description": "Telephoto, strong compression effect",
        },
    }

    @staticmethod
    def get(name: str) -> Dict[str, Any]:
        """Return a lens profile by name.  Raises KeyError if not found."""
        if name not in LensProfile.PROFILES:
            available = ", ".join(sorted(LensProfile.PROFILES.keys()))
            raise KeyError(f"Unknown lens profile '{name}'. Available: {available}")
        return dict(LensProfile.PROFILES[name])

    @staticmethod
    def apply_distortion(uv: np.ndarray, lens_name: str, strength: float = 1.0) -> np.ndarray:
        """Apply radial (barrel / pincushion) distortion to UV coordinates.

        Parameters
        ----------
        uv : np.ndarray
            Nx2 array of UV coordinates in range [-1, 1] (normalised from centre).
        lens_name : str
            Key into LensProfile.PROFILES.
        strength : float
            Multiplier on the distortion coefficients (0 = no distortion, 1 = full).

        Returns
        -------
        np.ndarray
            Distorted UV coordinates, same shape as input.
        """
        profile = LensProfile.get(lens_name)
        d = profile["distortion"]
        k1 = d["k1"] * strength
        k2 = d["k2"] * strength
        k3 = d["k3"] * strength
        p1 = d["p1"] * strength
        p2 = d["p2"] * strength

        uv = np.asarray(uv, dtype=np.float64)
        if uv.ndim == 1:
            uv = uv.reshape(1, 2)

        x = uv[:, 0]
        y = uv[:, 1]
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        # Radial component
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        result = np.column_stack([x * radial, y * radial])

        # Tangential component
        xy = x * y
        result[:, 0] += 2.0 * p1 * xy + p2 * (r2 + 2.0 * x * x)
        result[:, 1] += p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * xy

        return result.squeeze()


# ══════════════════════════════════════════════════════════════════════════════
# Aspect Ratio Presets
# ══════════════════════════════════════════════════════════════════════════════

ASPECT_RATIOS = {
    "16:9": 16.0 / 9.0,
    "2.39:1": 2.39,
    "4:3": 4.0 / 3.0,
    "1:1": 1.0,
    "9:16": 9.0 / 16.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# Depth of Field
# ══════════════════════════════════════════════════════════════════════════════

class DepthOfField:
    """Thin-lens depth-of-field model for realistic bokeh simulation.

    All distances are in **metres**, focal length in **millimetres**,
    circle of confusion in **millimetres**.

    Key equations (thin-lens model):
        Hyperfocal  H = f² / (N · c)  +  f
        Near plane  Dn = s · (H - f) / (H + s - 2·f)
        Far plane   Df = s · (H - f) / (H - s)        (or ∞)
        CoC at d    c(d) = |A · f² · (s - d)| / (N · d · (s - f))

    where f = focal_length (mm), N = f-number, c = max acceptable CoC (mm),
    s = focus distance (mm), A = pupil magnification (≈1 for symmetric lens).
    """

    def __init__(self, focal_length: float = 50.0, aperture: float = 2.8,
                 focus_distance: float = 10.0, sensor_width: float = 36.0,
                 coc: float = 0.03):
        self.focal_length = focal_length        # mm
        self.aperture = aperture                # f-number (N)
        self.focus_distance = focus_distance    # metres
        self.sensor_width = sensor_width        # mm
        self.coc = coc                          # circle of confusion, mm

    # ── Derived quantities ─────────────────────────────────────────────────

    def hyperfocal_distance(self) -> float:
        """Hyperfocal distance in metres.

        H = f² / (N · c) where f is in mm, result converted to metres.
        """
        f = self.focal_length
        N = self.aperture
        c = self.coc
        H_mm = (f * f) / (N * c)
        return H_mm / 1000.0  # metres

    def near_plane(self) -> float:
        """Near focus plane distance in metres."""
        f_mm = self.focal_length
        s_mm = self.focus_distance * 1000.0
        H_mm = self.hyperfocal_distance() * 1000.0
        denom = H_mm + s_mm - 2.0 * f_mm
        if abs(denom) < 1e-10:
            return 0.0
        Dn_mm = s_mm * (H_mm - f_mm) / denom
        return max(Dn_mm / 1000.0, 0.0)

    def far_plane(self) -> float:
        """Far focus plane distance in metres.  Returns ``float('inf')`` when
        the focus distance is at or beyond the hyperfocal distance."""
        f_mm = self.focal_length
        s_mm = self.focus_distance * 1000.0
        H_mm = self.hyperfocal_distance() * 1000.0
        denom = H_mm - s_mm
        if denom <= 1e-10:
            return float('inf')
        Df_mm = s_mm * (H_mm - f_mm) / denom
        if Df_mm < 0:
            return float('inf')
        return Df_mm / 1000.0

    def blur_amount(self, distance: float) -> float:
        """Circle of confusion (mm) at *distance* metres from camera.

        Positive = in front of focus plane, negative = behind.
        Magnitude represents blur radius on sensor.
        """
        f_mm = self.focal_length
        N = self.aperture
        s_mm = self.focus_distance * 1000.0
        d_mm = distance * 1000.0

        if d_mm < 1e-6 or abs(s_mm - f_mm) < 1e-10:
            return 0.0

        coc_mm = abs((f_mm * f_mm) * (s_mm - d_mm)) / (N * d_mm * (s_mm - f_mm))
        return coc_mm

    def get_dof_passes(self) -> Dict[str, float]:
        """Return near / far blur amounts for post-processing passes.

        Returns ``{'near_blur': float, 'far_blur': float}`` in mm of CoC
        evaluated at the near and far clip distances (assumed 0.3 m and
        1000 m respectively).
        """
        return {
            "near_blur": self.blur_amount(0.3),
            "far_blur": self.blur_amount(1000.0),
            "near_plane": self.near_plane(),
            "far_plane": self.far_plane(),
            "total_dof": self.far_plane() - self.near_plane(),
        }

    def set_aperture(self, f_stop: float):
        """Set the f-number (aperture)."""
        self.aperture = max(0.5, f_stop)

    def set_focus(self, distance_meters: float):
        """Set the focus distance in metres."""
        self.focus_distance = max(0.01, distance_meters)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "focal_length": self.focal_length,
            "aperture": self.aperture,
            "focus_distance": self.focus_distance,
            "sensor_width": self.sensor_width,
            "coc": self.coc,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Exposure
# ══════════════════════════════════════════════════════════════════════════════

class Exposure:
    """Camera exposure matching the real exposure triangle.

    Exposure Value (EV) at ISO 100:
        EV = log₂(N² / t)

    The APEX system relation:
        Ev = Av + Tv = Sv + Bv
    where  Av = log₂(N²),  Tv = -log₂(t),  Sv = log₂(ISO/100),  Bv = log₂(B)

    Exposure time from shutter angle:
        t = shutter_angle / 360 / fps
    """

    # Standard f-stop sequence
    F_STOPS = [0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0, 32.0]
    # Common ISO values
    ISO_VALUES = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

    def __init__(self, aperture: float = 2.8, shutter_time: float = 1.0 / 48.0,
                 iso: float = 800.0):
        self.aperture = aperture        # f-number
        self.shutter_time = shutter_time  # seconds
        self.iso = iso

    def get_ev(self) -> float:
        """Exposure Value at ISO 100: EV100 = log2(N² / t)."""
        if self.shutter_time <= 0:
            return float('inf')
        N2 = self.aperture ** 2
        return math.log2(N2 / self.shutter_time)

    def get_ev_at_iso(self) -> float:
        """Exposure Value adjusted for current ISO.

        EV_actual = EV100 + log2(ISO / 100)
        """
        return self.get_ev() - math.log2(self.iso / 100.0)

    def get_exposure_compensation(self, target_ev: float) -> float:
        """EV compensation needed to reach *target_ev* (in stops).

        Positive = need more exposure (brighten).
        Negative = need less exposure (darken).
        """
        return target_ev - self.get_ev_at_iso()

    def auto_exposure(self, scene_luminance: float) -> Dict[str, Any]:
        """Suggest camera settings for a given average scene luminance (cd/m²).

        Uses the APEX system:  Bv = log2(B * K / ISO)
        where K = 12.5 (reflected-light calibration constant).

        Returns dict with suggested aperture, shutter_time, and iso, keeping
        the current shutter angle / fps constraints in mind.
        """
        K = 12.5  # Reflected-light calibration constant
        # Target Bv so that the scene luminance maps to mid-grey
        # For a "correctly exposed" scene: EV = Bv + Sv
        # We want to choose Av and Tv such that Av + Tv = target_ev

        Sv = math.log2(self.iso / 100.0)
        Bv = math.log2(max(scene_luminance, 0.001) * K / self.iso)
        target_ev = Bv + Sv

        # Try to keep the current aperture and adjust shutter
        suggested_aperture = self.aperture
        Av = math.log2(self.aperture ** 2)
        Tv_needed = target_ev - Av
        suggested_shutter = 2.0 ** (-Tv_needed)

        # Clamp shutter to reasonable range (1/8000 to 1s)
        suggested_shutter = max(1.0 / 8000.0, min(suggested_shutter, 1.0))

        # If shutter can't compensate enough, suggest ISO change
        residual_ev = target_ev - (Av - math.log2(suggested_shutter))
        suggested_iso = self.iso * (2.0 ** residual_ev)
        suggested_iso = max(50.0, min(25600.0, suggested_iso))

        # Snap to nearest standard values
        suggested_aperture = self._snap_to_list(suggested_aperture, self.F_STOPS)
        suggested_iso = self._snap_to_list(suggested_iso, self.ISO_VALUES)

        return {
            "aperture": suggested_aperture,
            "shutter_time": suggested_shutter,
            "iso": suggested_iso,
            "ev100": target_ev + math.log2(self.iso / 100.0) - Sv,
            "scene_luminance": scene_luminance,
        }

    @staticmethod
    def ev_to_settings(ev: float, iso: float = 800.0,
                       preferred_aperture: float = 2.8) -> Dict[str, Any]:
        """Convert a target EV100 to suggested aperture / shutter / ISO.

        Tries to honour *preferred_aperture* and adjusts shutter first,
        then ISO if shutter is out of range.
        """
        # Adjust EV for ISO: ev_actual = ev100 - log2(iso/100)
        ev_actual = ev - math.log2(iso / 100.0)

        # Av = log2(N²) => N = sqrt(2^Av)
        Av = math.log2(preferred_aperture ** 2)
        Tv = ev_actual - Av  # Tv = -log2(t)
        shutter = 2.0 ** (-Tv)
        shutter = max(1.0 / 8000.0, min(shutter, 1.0))

        return {
            "aperture": preferred_aperture,
            "shutter_time": shutter,
            "iso": iso,
            "ev100": ev,
        }

    @staticmethod
    def _snap_to_list(value: float, candidates: List[float]) -> float:
        """Return the candidate closest to *value*."""
        return min(candidates, key=lambda c: abs(c - value))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aperture": self.aperture,
            "shutter_time": self.shutter_time,
            "iso": self.iso,
            "ev100": self.get_ev(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Motion Blur
# ══════════════════════════════════════════════════════════════════════════════

class MotionBlur:
    """Camera-based motion blur from shutter angle.

    Shutter angle models a rotating disc shutter (as on a film camera):
        exposure_time = (shutter_angle / 360) / fps

    A shutter angle of 180° at 24 fps gives the standard 1/48 s exposure.
    Wider angles (e.g. 270°) produce more blur; narrower angles (e.g. 90°)
    produce a crisper image with a strobe-y look.
    """

    def __init__(self, shutter_angle: float = 180.0, fps: float = 24.0):
        self.shutter_angle = np.clip(shutter_angle, 1.0, 360.0)
        self.fps = max(1.0, fps)

    def get_shutter_time(self) -> float:
        """Exposure time in seconds from shutter angle and fps."""
        return (self.shutter_angle / 360.0) / self.fps

    def calculate_exposure_time(self) -> float:
        """Alias for get_shutter_time()."""
        return self.get_shutter_time()

    def get_blur_transforms(self, camera: Any, dt: float,
                            subframes: int = 8) -> List[np.ndarray]:
        """Generate sub-frame transform matrices for motion blur.

        Uses the camera's velocity to extrapolate positions across the
        exposure window and returns a list of 4×4 view matrices.

        Parameters
        ----------
        camera : CinematicCamera
            Must expose ``position``, ``target``, ``up``, ``velocity``.
        dt : float
            Frame time (1/fps).
        subframes : int
            Number of sub-frames to sample within the exposure.

        Returns
        -------
        list of np.ndarray
            4×4 view matrices for each sub-frame.
        """
        transforms = []
        shutter = self.get_shutter_time()
        # The exposure opens (shutter_angle/360)/2 before the frame centre
        # and closes the same amount after.
        half_exposure = shutter / 2.0

        pos = np.array(camera.position, dtype=np.float64)
        tar = np.array(camera.target, dtype=np.float64)
        up = np.array(camera.up, dtype=np.float64)
        vel = np.array(getattr(camera, 'velocity', [0, 0, 0]), dtype=np.float64)

        for i in range(subframes):
            t_offset = -half_exposure + (shutter * i / max(subframes - 1, 1))
            # Map time offset to a sub-frame fraction
            frac = t_offset / max(dt, 1e-10)
            sub_pos = pos + vel * t_offset
            sub_tar = tar + vel * t_offset
            transforms.append(mat4_look_at(sub_pos, sub_tar, up))
        return transforms


# ══════════════════════════════════════════════════════════════════════════════
# Camera Shake
# ══════════════════════════════════════════════════════════════════════════════

class CameraShake:
    """Procedural camera shake driven by multi-octave Perlin noise.

    Presets
    -------
    handheld   – Subtle, organic wander like a handheld camera (~0.02 m rms).
    earthquake – Violent low-frequency shaking (~0.5 m rms).
    breathing  – Gentle rhythmic sway (~0.005 m rms).
    wind       – Directional low-frequency oscillation (~0.03 m rms).
    vehicle    – Combined engine vibration frequencies (~0.04 m rms).
    """

    _PRESETS = {
        "handheld": {
            "intensity": 0.02,
            "frequency": 3.0,
            "octaves": 4,
            "persistence": 0.5,
            "lacunarity": 2.0,
            "roll_intensity": 0.005,  # radians
        },
        "earthquake": {
            "intensity": 0.5,
            "frequency": 2.0,
            "octaves": 5,
            "persistence": 0.7,
            "lacunarity": 2.2,
            "roll_intensity": 0.05,
        },
        "breathing": {
            "intensity": 0.005,
            "frequency": 0.3,
            "octaves": 2,
            "persistence": 0.3,
            "lacunarity": 2.0,
            "roll_intensity": 0.001,
        },
        "wind": {
            "intensity": 0.03,
            "frequency": 1.5,
            "octaves": 3,
            "persistence": 0.4,
            "lacunarity": 2.0,
            "roll_intensity": 0.003,
        },
        "vehicle": {
            "intensity": 0.04,
            "frequency": 8.0,
            "octaves": 4,
            "persistence": 0.6,
            "lacunarity": 3.0,
            "roll_intensity": 0.002,
        },
    }

    def __init__(self, intensity: float = 1.0, frequency: float = 5.0,
                 noise_octaves: int = 4, seed: int = 42):
        self._noise = PerlinNoise(seed=seed)
        self._noise2 = PerlinNoise(seed=seed + 7)  # second noise source for roll
        self.intensity = intensity        # metres
        self.frequency = frequency        # Hz
        self.noise_octaves = noise_octaves
        self.persistence = 0.5
        self.lacunarity = 2.0
        self.roll_intensity = 0.005       # radians

    def set_preset(self, name: str):
        """Apply a named shake preset."""
        if name not in self._PRESETS:
            available = ", ".join(sorted(self._PRESETS.keys()))
            raise ValueError(f"Unknown shake preset '{name}'. Available: {available}")
        p = self._PRESETS[name]
        self.intensity = p["intensity"]
        self.frequency = p["frequency"]
        self.noise_octaves = p["octaves"]
        self.persistence = p["persistence"]
        self.lacunarity = p["lacunarity"]
        self.roll_intensity = p["roll_intensity"]

    def get_offset(self, time: float) -> np.ndarray:
        """Return ``[dx, dy, dz, roll_offset]`` at *time*.

        Each axis samples a different slice of 3-D noise to decorrelate
        the three translation channels.
        """
        t = time * self.frequency
        n = self._noise
        n2 = self._noise2

        dx = n.fractal(t, 0.0, 0.0, self.noise_octaves,
                        self.persistence, self.lacunarity) * self.intensity
        dy = n.fractal(0.0, t, 0.0, self.noise_octaves,
                        self.persistence, self.lacunarity) * self.intensity
        dz = n.fractal(0.0, 0.0, t, self.noise_octaves,
                        self.persistence, self.lacunarity) * self.intensity
        roll = n2.fractal(t, 10.0, 10.0, self.noise_octaves,
                          self.persistence, self.lacunarity) * self.roll_intensity

        return np.array([dx, dy, dz, roll], dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Camera Path (Spline Animation)
# ══════════════════════════════════════════════════════════════════════════════

class _Waypoint:
    """Internal representation of a single camera keyframe."""

    __slots__ = ("position", "time", "look_at", "fov", "roll",
                 "aperture", "focus_distance", "easing")

    def __init__(self, position, time, look_at=None, fov=None,
                 roll=0.0, aperture=None, focus_distance=None,
                 easing=None):
        self.position = np.asarray(position, dtype=np.float64)
        self.time = float(time)
        self.look_at = np.asarray(look_at, dtype=np.float64) if look_at is not None else None
        self.fov = float(fov) if fov is not None else None
        self.roll = float(roll)
        self.aperture = float(aperture) if aperture is not None else None
        self.focus_distance = float(focus_distance) if focus_distance is not None else None
        self.easing = easing or ease_in_out

    def to_dict(self):
        d: Dict[str, Any] = {"position": self.position.tolist(), "time": self.time}
        if self.look_at is not None:
            d["look_at"] = self.look_at.tolist()
        if self.fov is not None:
            d["fov"] = self.fov
        d["roll"] = self.roll
        if self.aperture is not None:
            d["aperture"] = self.aperture
        if self.focus_distance is not None:
            d["focus_distance"] = self.focus_distance
        if self.easing is not None:
            d["easing"] = self.easing.__name__
        return d


def _catmull_rom_segment(p0, p1, p2, p3, t):
    """Evaluate one segment of a Catmull-Rom spline at parameter *t* ∈ [0, 1]."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _catmull_rom_scalar(v0, v1, v2, v3, t):
    """Catmull-Rom interpolation for a scalar value."""
    return float(_catmull_rom_segment(
        np.array([v0]), np.array([v1]), np.array([v2]), np.array([v3]), t
    )[0])


class CameraPath:
    """Animated camera path using spline interpolation through waypoints.

    The camera travels along a smooth Catmull-Rom (or linear, or Bézier) path.
    All camera parameters (position, look_at, fov, roll, aperture, focus_distance)
    are interpolated between keyframes.

    Usage
    -----
    >>> path = CameraPath("hero_shot")
    >>> path.add_waypoint([0, 2, 5], 0.0, look_at=[0, 1, 0], fov=35)
    >>> path.add_waypoint([5, 3, 5], 5.0, look_at=[5, 1, 0], fov=24)
    >>> pose = path.get_pose(2.5)
    """

    def __init__(self, name: str = "CameraPath"):
        self.name = name
        self.waypoints: List[_Waypoint] = []
        self._interpolation = "catmull_rom"
        self._default_fov = 50.0
        self._default_aperture = 2.8
        self._default_focus = 10.0

    # ── Public API ─────────────────────────────────────────────────────────

    def add_waypoint(self, position, time, look_at=None, fov=None,
                     roll=0.0, aperture=None, focus_distance=None,
                     easing=None):
        """Append a keyframe to the path.

        Parameters
        ----------
        position : array-like, shape (3,)
            Camera position in world space.
        time : float
            Time in seconds at which the camera reaches this waypoint.
        look_at : array-like or None
            If given, the camera looks at this world-space point at this keyframe.
        fov : float or None
            Field of view in degrees at this keyframe.
        roll : float
            Camera roll (Dutch tilt) in radians.
        aperture : float or None
            F-number at this keyframe.
        focus_distance : float or None
            Focus distance in metres at this keyframe.
        easing : callable or None
            Easing function for the segment leading *to* this waypoint.
        """
        wp = _Waypoint(position=position, time=time, look_at=look_at,
                       fov=fov, roll=roll, aperture=aperture,
                       focus_distance=focus_distance, easing=easing)
        self.waypoints.append(wp)
        self.waypoints.sort(key=lambda w: w.time)

    def set_interpolation(self, mode: str = "catmull_rom"):
        """Set the interpolation mode for the entire path.

        Modes: ``'linear'``, ``'catmull_rom'``, ``'bezier'``.
        """
        if mode not in ("linear", "catmull_rom", "bezier"):
            raise ValueError(f"Unknown interpolation mode '{mode}'. "
                             f"Choose from: linear, catmull_rom, bezier")
        self._interpolation = mode

    def get_duration(self) -> float:
        """Total path duration in seconds."""
        if not self.waypoints:
            return 0.0
        return self.waypoints[-1].time - self.waypoints[0].time

    def get_pose(self, time: float) -> Dict[str, Any]:
        """Evaluate all camera parameters at *time*.

        Returns a dict with keys:
        ``position, target, up, fov, roll, aperture, focus_distance, time``
        """
        if not self.waypoints:
            raise RuntimeError("CameraPath has no waypoints")

        # Clamp to path range
        t_start = self.waypoints[0].time
        t_end = self.waypoints[-1].time
        time = max(t_start, min(time, t_end))

        # Find surrounding waypoints
        idx = self._find_segment(time)
        if idx is None:
            # Exactly on a waypoint
            wp = self.waypoints[0] if time <= t_start else self.waypoints[-1]
            return self._waypoint_to_pose(wp)

        w0 = self.waypoints[idx]
        w1 = self.waypoints[idx + 1]
        seg_duration = w1.time - w0.time
        if seg_duration < 1e-10:
            return self._waypoint_to_pose(w1)

        raw_t = (time - w0.time) / seg_duration

        # Apply easing from the incoming waypoint
        easing_fn = w1.easing or ease_in_out
        t = float(np.clip(easing_fn(raw_t), 0.0, 1.0))

        # Interpolate position
        position = self._interpolate_param(
            "position", idx, t, as_vector=True)

        # Interpolate look_at target (or compute from orientation)
        has_look_targets = (w0.look_at is not None and w1.look_at is not None)
        if has_look_targets:
            target = self._interpolate_param(
                "look_at", idx, t, as_vector=True)
        else:
            # Derive target from path tangent
            tangent = self._get_path_tangent(idx, t)
            if np.linalg.norm(tangent) > 1e-10:
                target = position + normalize(tangent)
            else:
                target = position + np.array([0.0, 0.0, -1.0])

        # Compute up vector with roll
        forward = normalize(target - position)
        world_up = np.array([0.0, 1.0, 0.0])
        right = normalize(cross(forward, world_up))
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0.0, 0.0, 1.0])
            right = normalize(cross(forward, world_up))
        up = normalize(cross(right, forward))

        # Apply roll rotation around forward axis
        roll = self._interpolate_scalar("roll", idx, t)
        if abs(roll) > 1e-8:
            cos_r, sin_r = math.cos(roll), math.sin(roll)
            up_rotated = up * cos_r + right * sin_r
            up = normalize(up_rotated)

        # Interpolate scalar parameters
        fov = self._interpolate_scalar("fov", idx, t, default=self._default_fov)
        aperture = self._interpolate_scalar("aperture", idx, t,
                                            default=self._default_aperture)
        focus_dist = self._interpolate_scalar("focus_distance", idx, t,
                                              default=self._default_focus)

        return {
            "position": position,
            "target": target,
            "up": up,
            "fov": fov,
            "roll": roll,
            "aperture": aperture,
            "focus_distance": focus_dist,
            "time": time,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the camera path to a dictionary."""
        return {
            "name": self.name,
            "interpolation": self._interpolation,
            "default_fov": self._default_fov,
            "default_aperture": self._default_aperture,
            "default_focus": self._default_focus,
            "waypoints": [wp.to_dict() for wp in self.waypoints],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraPath":
        """Deserialize a camera path from a dictionary."""
        path = cls(name=data.get("name", "CameraPath"))
        path.set_interpolation(data.get("interpolation", "catmull_rom"))
        path._default_fov = data.get("default_fov", 50.0)
        path._default_aperture = data.get("default_aperture", 2.8)
        path._default_focus = data.get("default_focus", 10.0)

        _easing_map = {
            "ease_in": ease_in, "ease_out": ease_out,
            "ease_in_out": ease_in_out, "ease_in_cubic": ease_in_cubic,
            "ease_out_cubic": ease_out_cubic, "ease_in_out_cubic": ease_in_out_cubic,
        }

        for wp_data in data.get("waypoints", []):
            easing_name = wp_data.get("easing")
            easing_fn = _easing_map.get(easing_name) if easing_name else None
            path.add_waypoint(
                position=wp_data["position"],
                time=wp_data["time"],
                look_at=wp_data.get("look_at"),
                fov=wp_data.get("fov"),
                roll=wp_data.get("roll", 0.0),
                aperture=wp_data.get("aperture"),
                focus_distance=wp_data.get("focus_distance"),
                easing=easing_fn,
            )
        return path

    # ── Internal helpers ──────────────────────────────────────────────────

    def _find_segment(self, time: float) -> Optional[int]:
        """Return the index of the segment containing *time*, or None."""
        for i in range(len(self.waypoints) - 1):
            if self.waypoints[i].time <= time <= self.waypoints[i + 1].time:
                return i
        return None

    def _waypoint_to_pose(self, wp: _Waypoint) -> Dict[str, Any]:
        """Convert a single waypoint to a full pose dict."""
        pos = wp.position.copy()
        if wp.look_at is not None:
            tar = wp.look_at.copy()
        else:
            tar = pos + np.array([0.0, 0.0, -1.0])

        forward = normalize(tar - pos)
        world_up = np.array([0.0, 1.0, 0.0])
        right = normalize(cross(forward, world_up))
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0.0, 0.0, 1.0])
            right = normalize(cross(forward, world_up))
        up = normalize(cross(right, forward))

        if abs(wp.roll) > 1e-8:
            cos_r, sin_r = math.cos(wp.roll), math.sin(wp.roll)
            up = normalize(up * cos_r + right * sin_r)

        return {
            "position": pos,
            "target": tar,
            "up": up,
            "fov": wp.fov if wp.fov is not None else self._default_fov,
            "roll": wp.roll,
            "aperture": wp.aperture if wp.aperture is not None else self._default_aperture,
            "focus_distance": wp.focus_distance if wp.focus_distance is not None else self._default_focus,
            "time": wp.time,
        }

    def _get_control_points(self, idx: int, attr: str):
        """Get 4 control points for Catmull-Rom around segment *idx*."""
        wps = self.waypoints
        n = len(wps)

        def _val(i):
            v = getattr(wps[i], attr)
            return v if v is not None else np.zeros(3)

        # Extend at boundaries by duplicating edge points
        i0 = max(idx - 1, 0)
        i1 = idx
        i2 = min(idx + 1, n - 1)
        i3 = min(idx + 2, n - 1)

        p0 = np.asarray(_val(i0), dtype=np.float64)
        p1 = np.asarray(_val(i1), dtype=np.float64)
        p2 = np.asarray(_val(i2), dtype=np.float64)
        p3 = np.asarray(_val(i3), dtype=np.float64)

        # Catmull-Rom phantom endpoints
        if idx == 0:
            p0 = 2.0 * p1 - p2
        if idx + 1 >= n - 1:
            p3 = 2.0 * p2 - p1

        return p0, p1, p2, p3

    def _interpolate_param(self, attr: str, idx: int, t: float,
                           as_vector: bool = False) -> np.ndarray:
        """Interpolate a vector parameter using the selected method."""
        if self._interpolation == "linear":
            v0 = np.asarray(getattr(self.waypoints[idx], attr), dtype=np.float64)
            v1 = np.asarray(getattr(self.waypoints[idx + 1], attr), dtype=np.float64)
            return v0 + (v1 - v0) * t
        elif self._interpolation == "catmull_rom":
            p0, p1, p2, p3 = self._get_control_points(idx, attr)
            return _catmull_rom_segment(p0, p1, p2, p3, t)
        elif self._interpolation == "bezier":
            p0, p1, p2, p3 = self._get_control_points(idx, attr)
            return bezier_cubic(p0, p1, p2, p3, t)
        else:
            raise RuntimeError(f"Unknown interpolation: {self._interpolation}")

    def _interpolate_scalar(self, attr: str, idx: int, t: float,
                            default: Optional[float] = None) -> float:
        """Interpolate a scalar parameter using the selected method."""
        v0_raw = getattr(self.waypoints[idx], attr)
        v1_raw = getattr(self.waypoints[idx + 1], attr)
        v0 = v0_raw if v0_raw is not None else default
        v1 = v1_raw if v1_raw is not None else default

        if v0 is None and v1 is None:
            return default if default is not None else 0.0
        if v0 is None:
            return float(v1)
        if v1 is None:
            return float(v0)

        if self._interpolation == "linear":
            return float(v0) + (float(v1) - float(v0)) * t
        elif self._interpolation in ("catmull_rom", "bezier"):
            # For scalars, fetch scalar control points
            wps = self.waypoints
            n = len(wps)
            vals = []
            for i in [max(idx - 1, 0), idx, min(idx + 1, n - 1), min(idx + 2, n - 1)]:
                v = getattr(wps[i], attr)
                vals.append(float(v) if v is not None else float(default if default is not None else 0.0))
            if idx == 0:
                vals[0] = 2.0 * vals[1] - vals[2]
            if idx + 1 >= n - 1:
                vals[3] = 2.0 * vals[2] - vals[1]
            return _catmull_rom_scalar(vals[0], vals[1], vals[2], vals[3], t)
        else:
            raise RuntimeError(f"Unknown interpolation: {self._interpolation}")

    def _get_path_tangent(self, idx: int, t: float) -> np.ndarray:
        """Approximate the tangent (first derivative) of the path at (idx, t)."""
        dt = 0.001
        t0 = max(0.0, t - dt)
        t1 = min(1.0, t + dt)
        if t1 - t0 < 1e-10:
            return np.zeros(3)
        p0 = self._interpolate_param("position", idx, t0)
        p1 = self._interpolate_param("position", idx, t1)
        return normalize(p1 - p0)


# ══════════════════════════════════════════════════════════════════════════════
# Cinematic Camera
# ══════════════════════════════════════════════════════════════════════════════

class CinematicCamera:
    """Professional cinematic virtual camera.

    Provides physically accurate optics (field-of-view linked to focal length
    and sensor size), depth-of-field, motion blur, and camera shake.
    """

    def __init__(self, name: str = "Camera"):
        self.name = name

        # ── Transform ─────────────────────────────────────────────────────
        self._position = np.array([0.0, 2.0, 5.0], dtype=np.float64)
        self._target = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self._up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        # ── Optics ────────────────────────────────────────────────────────
        self.sensor_width = 36.0   # mm (full-frame)
        self.sensor_height = 24.0  # mm (full-frame)
        self._focal_length = 50.0  # mm
        self._fov = self._compute_fov()
        self._aperture = 2.8       # f-number
        self._focus_distance = 10.0  # metres

        # ── Exposure ──────────────────────────────────────────────────────
        self._shutter_angle = 180.0  # degrees
        self._iso = 800.0

        # ── Resolution ────────────────────────────────────────────────────
        self.resolution = (1920, 1080)

        # ── Clipping ──────────────────────────────────────────────────────
        self.near_clip = 0.1
        self.far_clip = 1000.0

        # ── Aspect ratio ──────────────────────────────────────────────────
        self._aspect_ratio_name = "16:9"

        # ── Circle of confusion for DoF ───────────────────────────────────
        self.circle_of_confusion = 0.03  # mm

        # ── Subsystems ────────────────────────────────────────────────────
        self._shake = CameraShake(intensity=0.0, frequency=5.0)
        self._shake_enabled = False

        # ── Velocity tracking (for motion blur) ──────────────────────────
        self._prev_position = self._position.copy()
        self.velocity = np.zeros(3, dtype=np.float64)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def position(self) -> np.ndarray:
        return self._position.copy()

    @position.setter
    def position(self, value):
        self._position = np.asarray(value, dtype=np.float64)

    @property
    def target(self) -> np.ndarray:
        return self._target.copy()

    @target.setter
    def target(self, value):
        self._target = np.asarray(value, dtype=np.float64)

    @property
    def up(self) -> np.ndarray:
        return self._up.copy()

    @up.setter
    def up(self, value):
        self._up = np.asarray(value, dtype=np.float64)

    @property
    def fov(self) -> float:
        """Field of view in degrees (horizontal, derived from focal length)."""
        return self._fov

    @fov.setter
    def fov(self, value_deg: float):
        """Set FOV in degrees; updates focal length accordingly."""
        self._fov = float(np.clip(value_deg, 1.0, 179.0))
        self._focal_length = self._compute_focal_length()

    @property
    def focal_length(self) -> float:
        """Focal length in millimetres."""
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value_mm: float):
        """Set focal length in mm; updates FOV accordingly."""
        self._focal_length = max(1.0, float(value_mm))
        self._fov = self._compute_fov()

    @property
    def aperture(self) -> float:
        return self._aperture

    @aperture.setter
    def aperture(self, f_number: float):
        self._aperture = max(0.5, float(f_number))

    @property
    def focus_distance(self) -> float:
        return self._focus_distance

    @focus_distance.setter
    def focus_distance(self, metres: float):
        self._focus_distance = max(0.01, float(metres))

    @property
    def shutter_angle(self) -> float:
        return self._shutter_angle

    @shutter_angle.setter
    def shutter_angle(self, degrees: float):
        self._shutter_angle = float(np.clip(degrees, 1.0, 360.0))

    @property
    def iso(self) -> float:
        return self._iso

    @iso.setter
    def iso(self, value: float):
        self._iso = max(25.0, float(value))

    @property
    def aspect_ratio(self) -> float:
        """Current width / height aspect ratio."""
        return ASPECT_RATIOS.get(self._aspect_ratio_name,
                                 self.resolution[0] / self.resolution[1])

    @property
    def shake(self) -> CameraShake:
        """Access the camera shake subsystem."""
        return self._shake

    # ── Optics helpers ────────────────────────────────────────────────────

    def _compute_fov(self) -> float:
        """FOV (horizontal, degrees) from focal length and sensor width."""
        return 2.0 * math.degrees(math.atan(self.sensor_width /
                                              (2.0 * self._focal_length)))

    def _compute_focal_length(self) -> float:
        """Focal length (mm) from FOV and sensor width."""
        half_fov_rad = math.radians(self._fov / 2.0)
        return self.sensor_width / (2.0 * math.tan(half_fov_rad))

    def set_aspect_ratio(self, name: str):
        """Set aspect ratio by preset name (e.g. '16:9', '2.39:1')."""
        if name not in ASPECT_RATIOS:
            raise ValueError(f"Unknown aspect ratio '{name}'. "
                             f"Available: {', '.join(ASPECT_RATIOS.keys())}")
        self._aspect_ratio_name = name

    def set_resolution(self, width: int, height: int):
        """Set the render resolution in pixels."""
        self.resolution = (width, height)

    # ── Shake ─────────────────────────────────────────────────────────────

    def enable_shake(self, preset: str = "handheld"):
        """Enable camera shake with a named preset."""
        self._shake.set_preset(preset)
        self._shake_enabled = True

    def disable_shake(self):
        """Disable camera shake."""
        self._shake_enabled = False

    def apply_shake(self, time: float):
        """Apply shake offset to the camera's position and roll."""
        if not self._shake_enabled:
            return
        offset = self._shake.get_offset(time)
        self._position = self._position + offset[:3]
        # Apply roll around the view axis
        if abs(offset[3]) > 1e-8:
            forward = normalize(self._target - self._position)
            right = normalize(cross(forward, self._up))
            cos_r = math.cos(offset[3])
            sin_r = math.sin(offset[3])
            new_up = self._up * cos_r + right * sin_r
            self._up = normalize(new_up)

    # ── Matrix builders ───────────────────────────────────────────────────

    def view_matrix(self) -> np.ndarray:
        """4×4 view (look-at) matrix."""
        return mat4_look_at(self._position, self._target, self._up)

    def projection_matrix(self) -> np.ndarray:
        """4×4 perspective projection matrix."""
        return mat4_perspective(self._fov, self.aspect_ratio,
                                self.near_clip, self.far_clip)

    # ── Subsystem accessors ───────────────────────────────────────────────

    def get_dof(self) -> DepthOfField:
        """Return a DepthOfField instance matching current camera settings."""
        return DepthOfField(
            focal_length=self._focal_length,
            aperture=self._aperture,
            focus_distance=self._focus_distance,
            sensor_width=self.sensor_width,
            coc=self.circle_of_confusion,
        )

    def get_exposure(self) -> Exposure:
        """Return an Exposure instance matching current camera settings."""
        fps = 24.0  # default
        shutter_time = (self._shutter_angle / 360.0) / fps
        return Exposure(aperture=self._aperture, shutter_time=shutter_time,
                        iso=self._iso)

    def get_motion_blur(self, fps: float = 24.0) -> MotionBlur:
        """Return a MotionBlur instance matching current camera settings."""
        return MotionBlur(shutter_angle=self._shutter_angle, fps=fps)

    # ── Update (call each frame) ──────────────────────────────────────────

    def update(self, dt: float, time: float):
        """Update camera state.  Call once per frame.

        Parameters
        ----------
        dt : float
            Frame delta time (seconds).
        time : float
            Elapsed time (seconds) since start — used for shake noise.
        """
        # Track velocity for motion blur
        if dt > 1e-10:
            self.velocity = (self._position - self._prev_position) / dt
        self._prev_position = self._position.copy()

        # Apply shake
        self.apply_shake(time)

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "position": self._position.tolist(),
            "target": self._target.tolist(),
            "up": self._up.tolist(),
            "focal_length": self._focal_length,
            "fov": self._fov,
            "aperture": self._aperture,
            "focus_distance": self._focus_distance,
            "shutter_angle": self._shutter_angle,
            "iso": self._iso,
            "resolution": list(self.resolution),
            "near_clip": self.near_clip,
            "far_clip": self.far_clip,
            "aspect_ratio": self._aspect_ratio_name,
            "sensor_width": self.sensor_width,
            "sensor_height": self.sensor_height,
            "circle_of_confusion": self.circle_of_confusion,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Camera Presets
# ══════════════════════════════════════════════════════════════════════════════

class CameraPresets:
    """Pre-configured camera animation presets for common cinematic shots.

    Every static method returns a :class:`CameraPath` that can be evaluated
    at any time with :meth:`CameraPath.get_pose`.
    """

    @staticmethod
    def dolly_zoom(subject_distance: float = 10.0, fov_start: float = 24.0,
                   fov_end: float = 85.0, duration: float = 5.0,
                   num_waypoints: int = 32) -> CameraPath:
        """Vertigo / dolly-zoom (Hitchcock) effect.

        The camera dollies toward the subject while zooming out so that the
        subject stays the same apparent size but the background stretches.

        Uses the relation:  distance / focal_length = constant  to maintain
        subject magnification while changing perspective compression.
        """
        path = CameraPath("dolly_zoom")
        fl_start = 36.0 / (2.0 * math.tan(math.radians(fov_start / 2.0)))
        fl_end = 36.0 / (2.0 * math.tan(math.radians(fov_end / 2.0)))
        # Keep  subject_magnification = fl / (subject_distance - fl) constant
        mag = fl_start / (subject_distance * 1000.0 - fl_start)  # distance in mm
        # At each step: distance_mm = fl + fl / mag  =>  distance_m = (fl + fl/mag)/1000

        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            t_ease = ease_in_out(t)
            fl = fl_start + (fl_end - fl_start) * t_ease
            dist_mm = fl + fl / mag
            dist_m = dist_mm / 1000.0
            fov = 2.0 * math.degrees(math.atan(36.0 / (2.0 * fl)))
            path.add_waypoint(
                position=[0.0, 1.7, dist_m],
                time=t * duration,
                look_at=[0.0, 1.7, 0.0],
                fov=fov,
                easing=ease_in_out,
            )
        return path

    @staticmethod
    def orbit(subject_position: Any = None, radius: float = 5.0,
              height: float = 2.0, duration: float = 10.0,
              start_angle: float = 0.0, end_angle: float = 360.0,
              num_waypoints: int = 64) -> CameraPath:
        """Smooth orbit shot around a subject.

        Parameters
        ----------
        subject_position : array-like or None
            Centre of orbit.  Defaults to origin.
        radius : float
            Orbit radius in metres.
        height : float
            Camera height in metres.
        duration : float
            Total orbit time in seconds.
        start_angle : float
            Starting angle in degrees.
        end_angle : float
            Ending angle in degrees.
        """
        if subject_position is None:
            subject_position = [0.0, 0.0, 0.0]
        subject = np.asarray(subject_position, dtype=np.float64)

        path = CameraPath("orbit")
        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            angle_deg = start_angle + (end_angle - start_angle) * ease_in_out(t)
            angle_rad = math.radians(angle_deg)
            px = subject[0] + radius * math.sin(angle_rad)
            pz = subject[2] + radius * math.cos(angle_rad)
            py = subject[1] + height
            path.add_waypoint(
                position=[px, py, pz],
                time=t * duration,
                look_at=subject.tolist(),
                fov=50.0,
                easing=ease_in_out,
            )
        return path

    @staticmethod
    def tracking_shot(start_pos: Any, end_pos: Any, height: float = 1.5,
                      look_ahead: float = 0.0, duration: float = 5.0,
                      num_waypoints: int = 32) -> CameraPath:
        """Dolly / tracking shot moving from *start_pos* to *end_pos*.

        Parameters
        ----------
        start_pos, end_pos : array-like (3,)
            Horizontal (XZ) start and end positions.
        height : float
            Camera height (Y) in metres.
        look_ahead : float
            How far ahead of the current position to look (0 = look forward
            along track, > 0 = look further ahead).
        duration : float
            Shot duration in seconds.
        """
        start = np.asarray(start_pos, dtype=np.float64)
        end = np.asarray(end_pos, dtype=np.float64)
        direction = normalize(end - start)
        perp = np.cross(direction, np.array([0, 1, 0]))

        path = CameraPath("tracking_shot")
        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            t_ease = ease_in_out(t)
            pos = start + (end - start) * t_ease
            pos[1] = height  # maintain constant height
            look_target = pos + direction * (1.0 + look_ahead)
            look_target[1] = height - 0.1
            path.add_waypoint(
                position=pos.tolist(),
                time=t * duration,
                look_at=look_target.tolist(),
                fov=50.0,
                easing=ease_in_out,
            )
        return path

    @staticmethod
    def crane_shot(start_pos: Any, start_height: float,
                   end_pos: Any, end_height: float,
                   duration: float = 5.0,
                   num_waypoints: int = 32) -> CameraPath:
        """Crane / jib shot — vertical + horizontal camera movement.

        The camera moves from (*start_pos*, *start_height*) to (*end_pos*,
        *end_height*) with a parabolic vertical profile that eases in and out.
        """
        start = np.asarray(start_pos, dtype=np.float64)
        end = np.asarray(end_pos, dtype=np.float64)

        path = CameraPath("crane_shot")
        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            t_ease = ease_in_out(t)
            pos_h = start + (end - start) * t_ease
            h = start_height + (end_height - start_height) * t_ease
            pos = np.array([pos_h[0], h, pos_h[2]])
            look_target = (start + end) / 2.0
            look_target[1] = (start_height + end_height) / 2.0
            path.add_waypoint(
                position=pos.tolist(),
                time=t * duration,
                look_at=look_target.tolist(),
                fov=50.0,
                easing=ease_in_out,
            )
        return path

    @staticmethod
    def reveal(start_pos: Any, end_pos: Any, subject: Any,
               duration: float = 3.0,
               num_waypoints: int = 32) -> CameraPath:
        """Camera reveal — the camera moves from a hidden position to reveal
        the subject.

        Initially the camera looks away (or past) the subject; as it moves to
        *end_pos* the subject gradually enters frame.
        """
        start = np.asarray(start_pos, dtype=np.float64)
        end = np.asarray(end_pos, dtype=np.float64)
        subj = np.asarray(subject, dtype=np.float64)

        path = CameraPath("reveal")
        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            t_ease = ease_in_out_cubic(t)
            pos = start + (end - start) * t_ease
            # Interpolate look-at: start looking past subject, end at subject
            beyond = subj + normalize(subj - start) * 10.0
            look = beyond + (subj - beyond) * t_ease
            path.add_waypoint(
                position=pos.tolist(),
                time=t * duration,
                look_at=look.tolist(),
                fov=50.0,
                easing=ease_in_out_cubic,
            )
        return path

    @staticmethod
    def handheld_follow(subject_path: CameraPath, intensity: float = 0.5,
                        offset: Any = None) -> CameraPath:
        """Handheld camera that follows a subject path with subtle shake.

        Generates waypoints by sampling *subject_path* and adding a
        small offset (simulating the operator's shoulder position).

        Parameters
        ----------
        subject_path : CameraPath
            Path the subject is following.
        intensity : float
            Base shake intensity (metres).
        offset : array-like or None
            Constant offset from subject position (e.g. over-the-shoulder).
        """
        if offset is None:
            offset = np.array([0.0, 0.3, -2.0])
        offset = np.asarray(offset, dtype=np.float64)

        result = CameraPath("handheld_follow")
        duration = subject_path.get_duration()
        num = 48
        noise = PerlinNoise(seed=137)

        for i in range(num + 1):
            t = i / num * duration
            pose = subject_path.get_pose(t)
            pos = np.array(pose["position"]) + offset
            # Add subtle positional wander
            pos += np.array([
                noise.fractal(t * 2.0, 0.0, 0.0, 3, 0.5) * intensity,
                noise.fractal(0.0, t * 2.0, 0.0, 3, 0.5) * intensity * 0.5,
                noise.fractal(0.0, 0.0, t * 2.0, 3, 0.5) * intensity,
            ])
            tar = np.array(pose["position"])
            roll = noise.fractal(t * 1.5, 5.0, 5.0, 2, 0.3) * 0.008 * intensity
            result.add_waypoint(
                position=pos.tolist(),
                time=t,
                look_at=tar.tolist(),
                fov=pose.get("fov", 35.0),
                roll=roll,
            )
        return result

    @staticmethod
    def push_in(start_pos: Any, end_pos: Any,
                start_fov: float = 35.0, end_fov: float = 24.0,
                duration: float = 3.0,
                num_waypoints: int = 32) -> CameraPath:
        """Slow push-in with a subtle zoom change (dramatic effect).

        Commonly used to build tension: the camera physically moves closer
        while the lens also zooms in slightly, creating an accelerating
        sense of proximity.
        """
        start = np.asarray(start_pos, dtype=np.float64)
        end = np.asarray(end_pos, dtype=np.float64)

        path = CameraPath("push_in")
        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            t_ease = ease_in_out_cubic(t)
            pos = start + (end - start) * t_ease
            fov = start_fov + (end_fov - start_fov) * t_ease
            path.add_waypoint(
                position=pos.tolist(),
                time=t * duration,
                look_at=end.tolist(),
                fov=fov,
                aperture=2.0 + 0.8 * t_ease,  # gradually open aperture
                easing=ease_in_out_cubic,
            )
        return path
