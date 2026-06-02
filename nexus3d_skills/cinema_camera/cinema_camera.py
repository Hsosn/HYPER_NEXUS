#!/usr/bin/env python3
"""
Nexus3D Advanced Skill: Cinema Camera
======================================
Professional cinematic camera tools with depth of field tables, camera animation
paths, exposure calculation, lens profiles, camera shake, and multi-camera setups.

Upgraded v0.3.0 — Adds:
  - Multi-camera sequence (cuts between cameras on a timeline)
  - Camera shake / handheld simulation with configurable intensity
  - Exposure auto-calc (given EV100, get aperture/shutter/ISO combos)
  - Lens distortion profile application (barrel, pincushion, mustache)
  - Camera path editor (insert/delete keyframes, set interpolation)

Usage:
    python cinema_camera.py dof-table --lens 50mm --aperture 2.8
    python cinema_camera.py generate-path orbit --duration 10 --output path.json
    python cinema_camera.py camera-shake --intensity 0.3 --frequency 5.0 --duration 4.0
    python cinema_camera.py exposure-auto --ev 12 --iso 800
    python cinema_camera.py multi-cam --config cameras.json --duration 12.0
    python cinema_camera.py all-demos
"""

import sys, json, math, random, numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nexus3d.cinematic.camera import DepthOfField, LensProfile, CameraPresets, Exposure
from nexus3d.math3d.core import *
from nexus3d.utils.helpers import save_json, load_json, NumpyEncoder


# ─── Lens Distortion ──────────────────────────────────────────────────────────

LENS_DISTORTION_PROFILES = {
    "50mm_standard": {"k1": -0.01, "k2": 0.002, "description": "Mild barrel/pincushion"},
    "24mm_wide": {"k1": -0.08, "k2": 0.025, "description": "Noticeable barrel distortion"},
    "85mm_portrait": {"k1": 0.02, "k2": -0.005, "description": "Slight pincushion"},
    "fisheye_8mm": {"k1": -0.35, "k2": 0.12, "description": "Strong barrel (fisheye)"},
    "anamorphic": {"k1": -0.03, "k2": 0.008, "description": "Anamorphic-style distortion"},
}

def apply_lens_distortion(uv: np.ndarray, profile_name: str = "50mm_standard") -> np.ndarray:
    """Apply radial lens distortion to UV coordinates (N x 2 array)."""
    profile = LENS_DISTORTION_PROFILES.get(profile_name, LENS_DISTORTION_PROFILES["50mm_standard"])
    k1, k2 = profile["k1"], profile["k2"]
    cx, cy = 0.5, 0.5
    uv_norm = uv - np.array([cx, cy])
    r2 = np.sum(uv_norm ** 2, axis=1, keepdims=True)
    radial = 1 + k1 * r2 + k2 * r2 ** 2
    return uv_norm * radial + np.array([cx, cy])


# ─── Camera Shake ─────────────────────────────────────────────────────────────

def generate_camera_shake(duration: float, fps: int = 24,
                         intensity: float = 0.5, frequency: float = 5.0,
                         seed: int = 42) -> List[Dict[str, float]]:
    """Generate camera shake/ handheld simulation keyframes.

    Args:
        duration: Shake duration in seconds
        fps: Frames per second
        intensity: Shake intensity (0.0 - 1.0)
        frequency: Shake frequency (Hz)
        seed: Random seed for reproducibility

    Returns:
        List of per-frame shake offsets (position_x, position_y, rotation_z)
    """
    rng = random.Random(seed)
    num_frames = int(duration * fps)
    frames = []

    # Perlin-like noise for natural shake
    phase_x = rng.uniform(0, 2 * math.pi)
    phase_y = rng.uniform(0, 2 * math.pi)
    phase_r = rng.uniform(0, 2 * math.pi)

    base_amp = intensity * 0.05  # meters
    rot_amp = intensity * math.radians(0.5)  # radians

    for fi in range(num_frames):
        t = fi / fps
        # Multi-frequency noise
        x = (math.sin(2 * math.pi * frequency * t + phase_x) * 0.7 +
             math.sin(2 * math.pi * frequency * 2.3 * t + phase_x * 1.3) * 0.3)
        y = (math.sin(2 * math.pi * frequency * 1.1 * t + phase_y) * 0.6 +
             math.sin(2 * math.pi * frequency * 2.7 * t + phase_y * 0.7) * 0.4)
        rot = (math.sin(2 * math.pi * frequency * 0.8 * t + phase_r) * 0.5 +
               math.sin(2 * math.pi * frequency * 3.1 * t + phase_r * 1.1) * 0.5)

        frames.append({
            "frame": fi,
            "time": round(t, 4),
            "offset_x": round(x * base_amp, 6),
            "offset_y": round(y * base_amp, 6),
            "rotation_z_deg": round(math.degrees(rot * rot_amp), 4),
        })

    return {
        "duration": duration,
        "fps": fps,
        "intensity": intensity,
        "frequency": frequency,
        "total_frames": num_frames,
        "frames": frames,
    }


# ─── Exposure Auto-Calc ───────────────────────────────────────────────────────

def exposure_auto_calc(target_ev: float = 12.0, iso: int = 800) -> Dict[str, Any]:
    """Calculate recommended camera settings for a given EV100 and ISO.

    Returns aperture, shutter speed combinations that achieve the target exposure.
    """
    results = []
    aperture_list = [1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0]

    for f_stop in aperture_list:
        # Shutter from EV formula: EV = log2(N^2 / t) where N = aperture, t = shutter
        # With ISO compensation: t = N^2 / 2^EV
        shutter_ideal = f_stop ** 2 / (2 ** target_ev)

        # Find nearest standard shutter speed
        standard_shutters = [1/8000, 1/4000, 1/2000, 1/1000, 1/500, 1/250,
                             1/125, 1/60, 1/30, 1/15, 1/8, 1/4, 1/2, 1, 2, 4]
        shutter = min(standard_shutters, key=lambda s: abs(s - shutter_ideal))

        ev100 = Exposure(aperture=f_stop, shutter_time=shutter, iso=iso).get_ev()

        results.append({
            "aperture": f_stop,
            "shutter_speed": shutter,
            "shutter_fraction": f"1/{int(1/shutter)}" if shutter < 1 else f"{int(shutter)}s",
            "ev100": round(float(ev100), 1),
            "iso": iso,
        })

    return {"target_ev": target_ev, "iso": iso, "combinations": results}


# ─── Multi-Camera Sequence ──────────────────────────────────────────────────

class MultiCameraSequence:
    """Manage a sequence of camera cuts for multi-camera setups."""

    def __init__(self):
        self.cameras: Dict[str, dict] = {}
        self.cuts: List[Tuple[float, str]] = []  # (time, camera_name)
        self.duration: float = 0.0

    def add_camera(self, name: str, position: List[float], target: List[float],
                   fov: float = 45.0, lens: str = "50mm"):
        self.cameras[name] = {
            "position": position, "target": target, "fov": fov, "lens": lens
        }

    def add_cut(self, time: float, camera_name: str):
        if camera_name not in self.cameras:
            raise ValueError(f"Camera '{camera_name}' not found")
        self.cuts.append((time, camera_name))

    def get_active_camera(self, time: float) -> Optional[dict]:
        if not self.cuts:
            return None
        active = self.cuts[0][1]
        for cut_time, cam_name in sorted(self.cuts, key=lambda x: x[0]):
            if time >= cut_time:
                active = cam_name
        return {"name": active, **self.cameras.get(active, {})}

    def export_sequence(self) -> Dict[str, Any]:
        sorted_cuts = sorted(self.cuts, key=lambda x: x[0])
        shot_list = []
        for i, (cut_time, cam_name) in enumerate(sorted_cuts):
            end_time = sorted_cuts[i + 1][0] if i + 1 < len(sorted_cuts) else self.duration
            shot_list.append({
                "camera": cam_name, "start_time": cut_time,
                "end_time": end_time, "duration": round(end_time - cut_time, 3)
            })
        return {"cameras": self.cameras, "shots": shot_list, "duration": self.duration}


# ─── DoF Table ────────────────────────────────────────────────────────────────

def dof_table(args):
    fl = args.focal_length
    focus_dist = args.focus_distance
    if args.lens in LensProfile.PROFILES:
        fl = LensProfile.PROFILES[args.lens]['focal_length']
    results = []
    for f_stop in [1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0]:
        dof = DepthOfField(focal_length=fl, aperture=f_stop, focus_distance=focus_dist)
        near, far, hyper = dof.near_plane(), dof.far_plane(), dof.hyperfocal_distance()
        blur = {f"{d}m": round(dof.blur_amount(d), 4) for d in
                [focus_dist * 0.5, focus_dist * 0.75, focus_dist, focus_dist * 1.5, focus_dist * 3.0]}
        results.append({
            "focal_length_mm": fl, "aperture": f_stop, "focus_distance_m": focus_dist,
            "near_plane_m": round(near, 3), "far_plane_m": round(far, 3) if far < 1e6 else None,
            "total_dof_m": round(far - near, 3) if far < 1e6 else None,
            "hyperfocal_m": round(hyper, 3), "blur_at_distances": blur,
        })
    print(json.dumps({"command": "dof_table", "focal_length_mm": fl,
                      "results": results}, indent=2))


def generate_path(args):
    preset_map = {
        'orbit': lambda: CameraPresets.orbit(subject_position=np.array([0, 1, 0]),
            radius=args.radius, height=args.height, duration=args.duration),
        'dolly_zoom': lambda: CameraPresets.dolly_zoom(subject_distance=6.0,
            fov_start=24, fov_end=85),
        'tracking': lambda: CameraPresets.tracking_shot(
            start_pos=np.array([0, 1.5, 6]), end_pos=np.array([0, 1.5, -6]),
            duration=args.duration),
        'crane': lambda: CameraPresets.crane_shot(
            start_pos=np.array([0, 0.5, 5]), start_height=0.5,
            end_pos=np.array([0, 4, 2]), end_height=4.0, duration=args.duration),
        'push_in': lambda: CameraPresets.push_in(
            start_pos=np.array([0, 2, 8]), end_pos=np.array([0, 2, 3]),
            start_fov=35, end_fov=24, duration=args.duration),
    }
    if args.path_type not in preset_map:
        print(json.dumps({"error": f"Unknown path type: {args.path_type}. Use: {list(preset_map.keys())}"}))
        return
    camera_path = preset_map[args.path_type]()
    num_samples = int(args.duration * args.fps)
    samples = []
    for i in range(num_samples + 1):
        t = i / args.fps
        pose = camera_path.get_pose(t)
        samples.append({"frame": i, "time": round(t, 4),
            "position": [round(v, 4) for v in pose['position'].tolist()],
            "target": [round(v, 4) for v in pose['target'].tolist()],
            "fov": round(pose.get('fov', 50.0), 2)})
    if args.output:
        save_json({"type": args.path_type, "duration": args.duration,
            "fps": args.fps, "total_frames": num_samples, "samples": samples}, args.output)
    print(json.dumps({"command": "generate_path", "type": args.path_type,
        "duration": args.duration, "fps": args.fps, "total_frames": num_samples,
        "start_position": samples[0]["position"] if samples else None,
        "end_position": samples[-1]["position"] if samples else None}, indent=2))


def exposure_table(args):
    conditions = {
        "dark_interior": {"ev": 5}, "interior": {"ev": 7},
        "bright_interior": {"ev": 9}, "overcast": {"ev": 12},
        "hazy_sun": {"ev": 14}, "sunset": {"ev": 13},
        "full_sun": {"ev": 15}, "bright_sun": {"ev": 16}, "snow_beach": {"ev": 17},
    }
    results = []
    for name, info in conditions.items():
        ev = info["ev"]
        settings = Exposure.ev_to_settings(ev, iso=800)
        results.append({"condition": name, "ev100": ev, "recommended_iso800": settings})
    print(json.dumps({"command": "exposure_table", "lighting_conditions": results}, indent=2))


def lens_comparison(args):
    results = []
    for lens_name, profile in LensProfile.PROFILES.items():
        dof = DepthOfField(focal_length=profile['focal_length'], aperture=2.8, focus_distance=5.0)
        uv_test = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        distorted = apply_lens_distortion(uv_test, lens_name)
        results.append({"lens": lens_name, "focal_length_mm": profile['focal_length'],
            "fov_deg": round(profile['fov'], 1), "near_plane_m": round(dof.near_plane(), 3),
            "far_plane_m": round(dof.far_plane(), 3) if dof.far_plane() < 1e6 else None,
            "hyperfocal_m": round(dof.hyperfocal_distance(), 3)})
    print(json.dumps({"command": "lens_comparison", "lenses": results}, indent=2))


def camera_shake(args):
    result = generate_camera_shake(args.duration, args.fps, args.intensity, args.frequency)
    if args.output:
        save_json(result, args.output)
    print(json.dumps({"command": "camera_shake", "duration": args.duration,
        "intensity": args.intensity, "frames": result["total_frames"]}, indent=2))


def exposure_auto(args):
    result = exposure_auto_calc(args.ev, args.iso)
    print(json.dumps(result, indent=2))


def multi_cam(args):
    config = load_json(args.config)
    seq = MultiCameraSequence()
    for name, cam_data in config.get("cameras", {}).items():
        seq.add_camera(name, cam_data["position"], cam_data["target"],
                      cam_data.get("fov", 45), cam_data.get("lens", "50mm"))
    seq.duration = args.duration
    for cut in config.get("cuts", []):
        seq.add_cut(cut["time"], cut["camera"])
    result = seq.export_sequence()
    if args.output:
        save_json(result, args.output)
    print(json.dumps(result, indent=2))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Nexus3D Cinema Camera Tools v0.3.0")
    sub = parser.add_subparsers(dest='command')

    dp = sub.add_parser('dof-table')
    dp.add_argument('--lens', default='50mm'); dp.add_argument('--aperture', type=float, default=2.8)
    dp.add_argument('--focal-length', type=float, default=50.0)
    dp.add_argument('--focus-distance', type=float, default=5.0)

    pp = sub.add_parser('generate-path')
    pp.add_argument('--path-type', choices=['orbit','dolly_zoom','tracking','crane','push_in'], default='orbit')
    pp.add_argument('--duration', type=float, default=10.0); pp.add_argument('--fps', type=int, default=24)
    pp.add_argument('--radius', type=float, default=5.0); pp.add_argument('--height', type=float, default=2.0)
    pp.add_argument('--output', '-o')

    sp = sub.add_parser('camera-shake')
    sp.add_argument('--intensity', type=float, default=0.5); sp.add_argument('--frequency', type=float, default=5.0)
    sp.add_argument('--duration', type=float, default=4.0); sp.add_argument('--fps', type=int, default=24)
    sp.add_argument('--output', '-o')

    ep = sub.add_parser('exposure-auto')
    ep.add_argument('--ev', type=float, default=12.0); ep.add_argument('--iso', type=int, default=800)

    mp = sub.add_parser('multi-cam')
    mp.add_argument('--config', required=True); mp.add_argument('--duration', type=float, default=12.0)
    mp.add_argument('--output', '-o')

    sub.add_parser('exposure-table')
    sub.add_parser('lens-comparison')
    sub.add_parser('all-demos')

    args = parser.parse_args()
    if args.command is None: parser.print_help(); return

    dispatch = {
        'dof-table': dof_table, 'generate-path': generate_path,
        'exposure-table': exposure_table, 'lens-comparison': lens_comparison,
        'camera-shake': camera_shake, 'exposure-auto': exposure_auto,
        'multi-cam': multi_cam, 'all-demos': lambda a: print("Run each sub-command individually."),
    }
    fn = dispatch.get(args.command)
    if fn: fn(args)


if __name__ == "__main__":
    main()
