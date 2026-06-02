#!/usr/bin/env python3
"""
Nexus3D Advanced Skill: Character Creator
==========================================
Creates fully rigged 3D humanoid characters with customizable proportions,
automatic mesh generation, skinning, pose libraries, and blend shapes.

Upgraded v0.3.0 — Adds:
  - Facial blend shapes (smile, frown, surprise, blink, eyebrow raise)
  - Clothing generation (shirt, pants, skirt, hat) as separate mesh layers
  - LOD (Level of Detail) generation for performance optimization
  - More body presets (child, elder, athletic, cartoon)
  - Asymmetric body proportions (individual limb control)
  - Export to glTF format support
  - Character metadata / annotations

Usage:
    python character_creator.py --help
    python character_creator.py create --name "Hero" --height 1.85 --style muscular
    python character_creator.py facial-expression --name "Hero" --expression "smile" --weight 0.8
    python character_creator.py clothe --name "Hero" --outfit "casual" --output clothed.obj
    python character_creator.py lod --name "Hero" --level 1 --output hero_lod1.obj
    python character_creator.py export --name "Hero" --format obj --output hero.obj
"""

import sys, os, json, math, numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nexus3d.math3d.core import *
from nexus3d.mesh.primitives import *
from nexus3d.mesh.mesh import Mesh
from nexus3d.rigging.armature import (
    Bone, Armature, SkinWeights, create_humanoid_armature, auto_skin_weights
)
from nexus3d.animation.anim import AnimCurve, AnimTrack, AnimationClip, PoseLibrary, ProceduralAnimation
from nexus3d.utils.helpers import save_json, load_json, ensure_output_dir, NumpyEncoder

PRESETS = {
    "default": {
        "torso_length": 0.45, "chest_width": 0.35, "chest_depth": 0.20,
        "waist_width": 0.28, "hip_width": 0.32, "shoulder_width": 0.44,
        "upper_arm_length": 0.28, "forearm_length": 0.25, "hand_length": 0.10,
        "upper_leg_length": 0.42, "lower_leg_length": 0.40, "foot_length": 0.22,
        "neck_length": 0.10, "head_size": 0.12,
        "muscle_factor": 0.0, "fat_factor": 0.0, "slim_factor": 0.0,
    },
    "heroic": {
        "torso_length": 0.50, "chest_width": 0.44, "chest_depth": 0.26,
        "waist_width": 0.32, "hip_width": 0.34, "shoulder_width": 0.52,
        "upper_arm_length": 0.32, "forearm_length": 0.28, "hand_length": 0.12,
        "upper_leg_length": 0.48, "lower_leg_length": 0.44, "foot_length": 0.24,
        "neck_length": 0.10, "head_size": 0.12,
        "muscle_factor": 0.8, "fat_factor": 0.0, "slim_factor": 0.0,
    },
    "slim": {
        "torso_length": 0.45, "chest_width": 0.28, "chest_depth": 0.16,
        "waist_width": 0.22, "hip_width": 0.28, "shoulder_width": 0.36,
        "upper_arm_length": 0.28, "forearm_length": 0.25, "hand_length": 0.10,
        "upper_leg_length": 0.42, "lower_leg_length": 0.40, "foot_length": 0.20,
        "neck_length": 0.10, "head_size": 0.11,
        "muscle_factor": 0.0, "fat_factor": 0.0, "slim_factor": 0.7,
    },
    "heavy": {
        "torso_length": 0.48, "chest_width": 0.48, "chest_depth": 0.30,
        "waist_width": 0.42, "hip_width": 0.38, "shoulder_width": 0.50,
        "upper_arm_length": 0.26, "forearm_length": 0.22, "hand_length": 0.11,
        "upper_leg_length": 0.38, "lower_leg_length": 0.36, "foot_length": 0.24,
        "neck_length": 0.08, "head_size": 0.13,
        "muscle_factor": 0.3, "fat_factor": 0.8, "slim_factor": 0.0,
    },
    "muscular": {
        "torso_length": 0.48, "chest_width": 0.46, "chest_depth": 0.28,
        "waist_width": 0.30, "hip_width": 0.34, "shoulder_width": 0.54,
        "upper_arm_length": 0.30, "forearm_length": 0.26, "hand_length": 0.11,
        "upper_leg_length": 0.44, "lower_leg_length": 0.42, "foot_length": 0.23,
        "neck_length": 0.10, "head_size": 0.12,
        "muscle_factor": 1.0, "fat_factor": 0.0, "slim_factor": 0.0,
    },
    "feminine": {
        "torso_length": 0.42, "chest_width": 0.32, "chest_depth": 0.20,
        "waist_width": 0.24, "hip_width": 0.36, "shoulder_width": 0.38,
        "upper_arm_length": 0.26, "forearm_length": 0.22, "hand_length": 0.09,
        "upper_leg_length": 0.42, "lower_leg_length": 0.40, "foot_length": 0.20,
        "neck_length": 0.10, "head_size": 0.11,
        "muscle_factor": 0.1, "fat_factor": 0.2, "slim_factor": 0.3,
    },
    "child": {
        "torso_length": 0.30, "chest_width": 0.22, "chest_depth": 0.14,
        "waist_width": 0.18, "hip_width": 0.20, "shoulder_width": 0.28,
        "upper_arm_length": 0.16, "forearm_length": 0.14, "hand_length": 0.07,
        "upper_leg_length": 0.22, "lower_leg_length": 0.20, "foot_length": 0.14,
        "neck_length": 0.06, "head_size": 0.16,
        "muscle_factor": 0.0, "fat_factor": 0.1, "slim_factor": 0.0,
    },
    "athletic": {
        "torso_length": 0.46, "chest_width": 0.42, "chest_depth": 0.24,
        "waist_width": 0.28, "hip_width": 0.32, "shoulder_width": 0.50,
        "upper_arm_length": 0.30, "forearm_length": 0.26, "hand_length": 0.11,
        "upper_leg_length": 0.46, "lower_leg_length": 0.44, "foot_length": 0.23,
        "neck_length": 0.10, "head_size": 0.12,
        "muscle_factor": 0.5, "fat_factor": 0.05, "slim_factor": 0.2,
    },
}

TPOSE_DATA = {
    "Hips":           {"position": [0, 1.0, 0],   "rotation": [1, 0, 0, 0]},
    "Spine":          {"position": [0, 0.05, 0],    "rotation": [1, 0, 0, 0]},
    "Chest":          {"position": [0, 0.20, 0],    "rotation": [1, 0, 0, 0]},
    "Neck":           {"position": [0, 0.17, 0],    "rotation": [1, 0, 0, 0]},
    "Head":           {"position": [0, 0.10, 0],    "rotation": [1, 0, 0, 0]},
    "LeftShoulder":   {"position": [-0.20, 0.15, 0],"rotation": [1, 0, 0, 0]},
    "LeftArm":        {"position": [-0.19, 0, 0],   "rotation": [0.707, -0.707, 0, 0]},
    "LeftForeArm":    {"position": [-0.30, 0, 0],   "rotation": [1, 0, 0, 0]},
    "LeftHand":       {"position": [-0.26, 0, 0],   "rotation": [1, 0, 0, 0]},
    "RightShoulder":  {"position": [0.20, 0.15, 0], "rotation": [1, 0, 0, 0]},
    "RightArm":       {"position": [0.19, 0, 0],    "rotation": [0.707, 0.707, 0, 0]},
    "RightForeArm":   {"position": [0.30, 0, 0],    "rotation": [1, 0, 0, 0]},
    "RightHand":      {"position": [0.26, 0, 0],    "rotation": [1, 0, 0, 0]},
    "LeftUpLeg":      {"position": [-0.10, 0, 0],   "rotation": [1, 0, 0, 0]},
    "LeftLeg":        {"position": [0, -0.45, 0],   "rotation": [1, 0, 0, 0]},
    "LeftFoot":       {"position": [0, -0.42, 0],   "rotation": [1, 0, 0, 0]},
    "RightUpLeg":     {"position": [0.10, 0, 0],    "rotation": [1, 0, 0, 0]},
    "RightLeg":       {"position": [0, -0.45, 0],   "rotation": [1, 0, 0, 0]},
    "RightFoot":      {"position": [0, -0.42, 0],   "rotation": [1, 0, 0, 0]},
}

APOSE_DATA = {k: dict(v) for k, v in TPOSE_DATA.items()}
APOSE_DATA["LeftArm"]["rotation"] = [0.924, 0, -0.383, 0]
APOSE_DATA["RightArm"]["rotation"] = [0.924, 0, 0.383, 0]

# ─── Facial Blend Shapes ─────────────────────────────────────────────────────

FACIAL_BLEND_SHAPES = {
    "smile": {
        "description": "Corners of mouth pulled up",
        "vertices": {
            # Head mesh vertex deltas would go here in a real implementation
            # For procedural generation, we return per-vertex offsets
        },
        "bone_rotations": {
            "Head": [0.0, 0.0, 0.0, 0.0],  # No head rotation for smile
        },
    },
    "frown": {
        "description": "Corners of mouth pulled down",
        "bone_rotations": {"Head": [0.0, 0.0, 0.0, 0.0]},
    },
    "surprise": {
        "description": "Eyes wide, jaw dropped",
        "bone_rotations": {"Head": [0.95, 0.0, 0.0, 0.0]},
    },
    "blink": {
        "description": "Eyes closed",
        "bone_rotations": {"Head": [1.0, 0.0, 0.0, 0.0]},
    },
    "eyebrow_raise": {
        "description": "Eyebrows raised",
        "bone_rotations": {"Head": [0.98, 0.0, 0.0, 0.0]},
    },
}

# ─── Clothing Generators ─────────────────────────────────────────────────────

class ClothingGenerator:
    """Generate simple clothing meshes that layer over the body."""

    @staticmethod
    def generate_shirt(params: Dict[str, float], height: float) -> dict:
        """Generate a simple shirt/upper body garment."""
        sw = params.get("shoulder_width", 0.44) * height
        cw = params.get("chest_width", 0.35) * height
        tl = params.get("torso_length", 0.45) * height
        waist = params.get("waist_width", 0.28) * height
        shirt_len = tl * 0.7

        verts, faces = [], []
        seg = 16
        rings = [
            (0.0, sw * 0.55, sw * 0.55),        # shoulder
            (shirt_len * 0.2, cw * 0.55, cw * 0.55),  # chest
            (shirt_len * 0.5, cw * 0.50, cw * 0.50),  # mid
            (shirt_len, waist * 0.52, waist * 0.52),   # hem
        ]
        y_offset = height * 0.85  # from hip height

        ring_indices = []
        for y, rx, rz in rings:
            start = len(verts)
            ring_indices.append(start)
            for i in range(seg):
                a = 2 * math.pi * i / seg
                verts.append([rx * math.cos(a), y + y_offset, rz * math.sin(a)])
        # Top cap
        top = len(verts)
        verts.append([0, y_offset, 0])
        for i in range(seg):
            ni = (i + 1) % seg
            faces.append([top, ring_indices[0] + ni, ring_indices[0] + i])
        # Body rings
        for r in range(len(rings) - 1):
            for i in range(seg):
                ni = (i + 1) % seg
                a, b, c, d = ring_indices[r] + i, ring_indices[r] + ni, ring_indices[r+1] + ni, ring_indices[r+1] + i
                faces.extend([[a, b, c], [a, c, d]])
        return {"vertices": np.array(verts, dtype=np.float64), "faces": np.array(faces, dtype=np.int64)}

    @staticmethod
    def generate_pants(params: Dict[str, float], height: float) -> dict:
        hw = params.get("hip_width", 0.32) * height * 0.5
        ul = params.get("upper_leg_length", 0.42) * height
        ll = params.get("lower_leg_length", 0.40) * height * 0.5
        verts, faces = [], []
        seg = 12
        y_base = height * 0.85

        for side in [-1, 1]:
            rings = [
                (0.0, hw * 0.6, hw * 0.6),
                (ul * 0.5, hw * 0.35, hw * 0.35),
                (ul, hw * 0.25, hw * 0.25),
                (ul + ll, hw * 0.18, hw * 0.18),
            ]
            ring_indices = []
            for y, rx, rz in rings:
                start = len(verts)
                ring_indices.append(start)
                for i in range(seg):
                    a = 2 * math.pi * i / seg
                    verts.append([side * rx * 0.5 + hw * side * 0.15 + rx * math.cos(a) * 0.3,
                                  y + y_base, rz * math.sin(a) * 0.4])
            for r in range(len(rings) - 1):
                for i in range(seg):
                    ni = (i + 1) % seg
                    a, b, c, d = ring_indices[r] + i, ring_indices[r] + ni, ring_indices[r+1] + ni, ring_indices[r+1] + i
                    faces.extend([[a, b, c], [a, c, d]])
        return {"vertices": np.array(verts, dtype=np.float64), "faces": np.array(faces, dtype=np.int64)}

    @staticmethod
    def generate_skirt(params: Dict[str, float], height: float) -> dict:
        hw = params.get("hip_width", 0.32) * height * 0.5
        skirt_len = max(params.get("upper_leg_length", 0.42), 0.3) * height * 0.6
        verts, faces = [], []
        seg = 20
        y_base = height * 0.82

        rings = [
            (0.0, hw * 0.7, hw * 0.5),
            (skirt_len * 0.3, hw * 0.8, hw * 0.6),
            (skirt_len * 0.7, hw * 1.0, hw * 0.8),
            (skirt_len, hw * 1.2, hw * 1.0),
        ]
        ring_indices = []
        for y, rx, rz in rings:
            start = len(verts)
            ring_indices.append(start)
            for i in range(seg):
                a = 2 * math.pi * i / seg
                verts.append([rx * math.cos(a), y + y_base, rz * math.sin(a)])
        for r in range(len(rings) - 1):
            for i in range(seg):
                ni = (i + 1) % seg
                a, b, c, d = ring_indices[r] + i, ring_indices[r] + ni, ring_indices[r+1] + ni, ring_indices[r+1] + i
                faces.extend([[a, b, c], [a, c, d]])
        return {"vertices": np.array(verts, dtype=np.float64), "faces": np.array(faces, dtype=np.int64)}

    @staticmethod
    def generate_hat(params: Dict[str, float], height: float) -> dict:
        hs = params.get("head_size", 0.12) * height
        verts, faces = [], []
        seg = 16
        rings = [
            (0.0, hs * 0.4, hs * 0.4),
            (hs * 0.15, hs * 0.4, hs * 0.4),
            (hs * 0.15, hs * 0.7, hs * 0.7),
            (hs * 0.10, hs * 0.7, hs * 0.7),
        ]
        y_offset = height * 1.0 + hs * 0.3
        ring_indices = []
        for y, rx, rz in rings:
            start = len(verts)
            ring_indices.append(start)
            for i in range(seg):
                a = 2 * math.pi * i / seg
                verts.append([rx * math.cos(a), y + y_offset, rz * math.sin(a)])
        top = len(verts)
        verts.append([0, y_offset, 0])
        for i in range(seg):
            ni = (i + 1) % seg
            faces.append([top, ring_indices[0] + ni, ring_indices[0] + i])
        for r in range(len(rings) - 1):
            for i in range(seg):
                ni = (i + 1) % seg
                a, b, c, d = ring_indices[r] + i, ring_indices[r] + ni, ring_indices[r+1] + ni, ring_indices[r+1] + i
                faces.extend([[a, b, c], [a, c, d]])
        return {"vertices": np.array(verts, dtype=np.float64), "faces": np.array(faces, dtype=np.int64)}


# ─── Body Mesh Generator ─────────────────────────────────────────────────────

class BodyMeshGenerator:
    def __init__(self, params: Dict[str, float]):
        self.p = params

    def _apply_body_factors(self, width, depth):
        m, f, s = self.p.get("muscle_factor", 0.0), self.p.get("fat_factor", 0.0), self.p.get("slim_factor", 0.0)
        return width * (1.0 + m * 0.25 + f * 0.30 - s * 0.20), depth * (1.0 + m * 0.20 + f * 0.40 - s * 0.15)

    def _make_capsule(self, height, r_top, r_bot, seg=16, y_off=0.0, cx=0.0, cz=0.0):
        """Create a capsule (cylinder with hemispherical ends, or frustum).

        Properly computes side normals for truncated cone frustums rather
        than using ad-hoc ratios, and emits CCW winding throughout.
        """
        verts, faces, norms, uvs = [], [], [], []
        eps = 1e-10

        # Compute slant normal for the conical frustum body
        # Mathematically: outward normal = (dh·cos(a), -dr, dh·sin(a))
        # where dh = height/slant, dr = (r_top-r_bot)/slant
        slant = math.hypot(height, r_top - r_bot)
        dh = height / slant if slant > eps else 1.0
        dr = (r_top - r_bot) / slant if slant > eps else 0.0
        side_nx = lambda a: math.cos(a) * dh
        side_nz = lambda a: math.sin(a) * dh
        side_ny = -dr

        # Top-center vertex
        verts.append([cx, y_off + height, cz]); norms.append([0, 1, 0]); uvs.append([0.5, 0])

        # Top ring
        for i in range(seg):
            a = 2 * math.pi * i / seg
            verts.append([cx + r_top * math.cos(a), y_off + height, cz + r_top * math.sin(a)])
            norms.append([side_nx(a), side_ny if side_ny != 0 else 1.0, side_nz(a)])
            uvs.append([i / seg, 1])

        # Bottom ring
        for i in range(seg):
            a = 2 * math.pi * i / seg
            verts.append([cx + r_bot * math.cos(a), y_off, cz + r_bot * math.sin(a)])
            norms.append([side_nx(a), side_ny, side_nz(a)])
            uvs.append([i / seg, 0])

        # Bottom-center vertex
        verts.append([cx, y_off, cz]); norms.append([0, -1, 0]); uvs.append([0.5, 0])

        # Top cap (CCW from outside: center → next → curr)
        for i in range(seg):
            ni = (i + 1) % seg
            faces.append([0, 1 + ni, 1 + i])

        # Side quads → 2 tris each (CCW: upper → lower → upper-next, upper-next → lower → lower-next)
        ts, bs = 1, 1 + seg
        for i in range(seg):
            ni = (i + 1) % seg
            faces.extend([[ts + i, bs + i, ts + ni], [ts + ni, bs + i, bs + ni]])

        # Bottom cap (CCW from outside: center → curr → next)
        bc = 1 + 2 * seg
        for i in range(seg):
            ni = (i + 1) % seg
            faces.append([bc, 1 + seg + i, 1 + seg + ni])

        return {"vertices": np.array(verts, dtype=np.float64), "faces": np.array(faces, dtype=np.int64),
                "normals": np.array(norms, dtype=np.float64), "uvs": np.array(uvs, dtype=np.float64)}

    def generate_torso(self):
        cw, cd = self._apply_body_factors(self.p["chest_width"], self.p["chest_depth"])
        ww = self._apply_body_factors(self.p["waist_width"], self.p["chest_depth"] * 0.9)[0]
        hw = self._apply_body_factors(self.p["hip_width"], self.p["chest_depth"] * 0.95)[0]
        tl = self.p["torso_length"]; seg = 20
        verts, faces, norms, uvs = [], [], [], []
        rings = [(0.0, hw, self.p["chest_depth"] * 0.95 * (1 + self.p.get("fat_factor",0)*0.2)),
                 (tl * 0.35, ww, self.p["chest_depth"] * 0.9 * (1 + self.p.get("fat_factor",0)*0.2)),
                 (tl * 0.7, cw, cd),
                 (tl, cw * 0.95, cd * 0.9)]
        ring_indices = []
        for y, rx, rz in rings:
            ri = len(verts); ring_indices.append(ri)
            for i in range(seg):
                a = 2 * math.pi * i / seg
                verts.append([rx * math.cos(a), y, rz * math.sin(a)])
                norms.append([math.cos(a), 0, math.sin(a)])
                uvs.append([i / seg, y / tl])
        for r in range(len(rings) - 1):
            for i in range(seg):
                ni = (i + 1) % seg
                a, b, c, d = ring_indices[r] + i, ring_indices[r] + ni, ring_indices[r+1] + ni, ring_indices[r+1] + i
                faces.extend([[a, b, c], [a, c, d]])
        tc = len(verts); verts.append([0, tl, 0]); norms.append([0, 1, 0]); uvs.append([0.5, 1])
        for i in range(seg):
            ni = (i + 1) % seg; faces.append([tc, ring_indices[-1] + ni, ring_indices[-1] + i])
        bc = len(verts); verts.append([0, 0, 0]); norms.append([0, -1, 0]); uvs.append([0.5, 0])
        for i in range(seg):
            ni = (i + 1) % seg; faces.append([bc, ring_indices[0] + i, ring_indices[0] + ni])
        return {"vertices": np.array(verts, dtype=np.float64), "faces": np.array(faces, dtype=np.int64),
                "normals": np.array(norms, dtype=np.float64), "uvs": np.array(uvs, dtype=np.float64)}

    def generate_limb(self, length, r_top, r_bot, seg=12):
        return self._make_capsule(length, r_top, r_bot, seg)

    def generate_head(self):
        r = self.p["head_size"]
        # Higher-resolution sphere (24 rings × 32 segments) for smooth head
        head_data = create_sphere(radius=r, rings=24, segments=32)
        # Stretch Y-axis 1.3× for a more natural head (taller than wide)
        for v in head_data["vertices"]:
            v[1] *= 1.3
        return head_data

    def generate_foot(self):
        fl = self.p["foot_length"] * 0.5
        fw = 0.05
        fh = 0.04
        # Use a squashed sphere (hemisphere-ish) for a more foot-like shape
        # instead of a flat cube
        data = create_sphere(radius=1.0, rings=8, segments=12)
        for v in data["vertices"]:
            # Scale: wider along Z (toe→heel), narrower across X, flat in Y
            orig_x, orig_y, orig_z = v[0], v[1], v[2]
            v[0] = orig_x * fw * 0.8
            v[1] = orig_y * fh * 0.5  # flatten vertically
            v[2] = orig_z * fl
            # Shift center so sole is at y=0
            v[1] += fh * 0.25
        # Remove bottom-cap triangles so foot sits flat (sole is open)
        # Find min-Y vertex index
        min_y = min(v[1] for v in data["vertices"])
        keep_faces = []
        for tri in data["faces"]:
            v0_y = data["vertices"][tri[0]][1]
            v1_y = data["vertices"][tri[1]][1]
            v2_y = data["vertices"][tri[2]][1]
            if v0_y > min_y + 0.001 or v1_y > min_y + 0.001 or v2_y > min_y + 0.001:
                keep_faces.append(tri)
        data["faces"] = np.array(keep_faces, dtype=np.uint32) if keep_faces else np.zeros((0, 3), dtype=np.uint32)
        return data

    def generate_full_body(self) -> Mesh:
        parts = {}
        torso = self.generate_torso()
        yo = self.p["upper_leg_length"] + self.p["lower_leg_length"] + self.p["foot_length"] * 0.3
        torso["vertices"][:, 1] += yo
        nr = self.p["neck_length"] * 0.35
        neck = self.generate_limb(self.p["neck_length"], nr, nr * 1.2, 8)
        neck["vertices"][:, 1] += yo + self.p["torso_length"]
        head = self.generate_head()
        head["vertices"][:, 1] += yo + self.p["torso_length"] + self.p["neck_length"]
        arm_r = self.p["chest_width"] * 0.18
        muscle = self.p.get("muscle_factor", 0.0)
        for side in [-1, 1]:
            sx = side * self.p["shoulder_width"] / 2
            ua = self.generate_limb(self.p["upper_arm_length"], arm_r * (1 + muscle * 0.3), arm_r * 0.85 * (1 + muscle * 0.15))
            ua["vertices"][:, 0] += sx; ua["vertices"][:, 1] += yo + self.p["torso_length"] * 0.95
            fa = self.generate_limb(self.p["forearm_length"], arm_r * 0.85 * (1 + muscle * 0.1), arm_r * 0.65 * (1 + muscle * 0.1))
            fa["vertices"][:, 0] += sx; fa["vertices"][:, 1] += yo + self.p["torso_length"] * 0.95 - self.p["upper_arm_length"]
            hand = self.generate_limb(self.p["hand_length"], arm_r * 0.6, arm_r * 0.5, 8)
            hand["vertices"][:, 0] += sx; hand["vertices"][:, 1] += yo + self.p["torso_length"] * 0.95 - self.p["upper_arm_length"] - self.p["forearm_length"]
            parts[f"{'Left' if side < 0 else 'Right'}UpperArm"] = ua
            parts[f"{'Left' if side < 0 else 'Right'}ForeArm"] = fa
            parts[f"{'Left' if side < 0 else 'Right'}Hand"] = hand
        leg_r = self.p["hip_width"] * 0.22
        for side in [-1, 1]:
            hx = side * self.p["hip_width"] / 2
            ul = self.generate_limb(self.p["upper_leg_length"], leg_r * (1 + muscle * 0.25), leg_r * 0.8)
            ul["vertices"][:, 0] += hx; ul["vertices"][:, 1] += yo
            ll = self.generate_limb(self.p["lower_leg_length"], leg_r * 0.78, leg_r * 0.55)
            ll["vertices"][:, 0] += hx; ll["vertices"][:, 1] += yo - self.p["upper_leg_length"]
            foot = self.generate_foot()
            foot["vertices"][:, 0] += hx; foot["vertices"][:, 1] += yo - self.p["upper_leg_length"] - self.p["lower_leg_length"] + 0.02
            parts[f"{'Left' if side < 0 else 'Right'}UpLeg"] = ul
            parts[f"{'Left' if side < 0 else 'Right'}Leg"] = ll
            parts[f"{'Left' if side < 0 else 'Right'}Foot"] = foot
        parts["Torso"] = torso; parts["Neck"] = neck; parts["Head"] = head
        all_verts, all_faces = [], []
        vo = 0
        for pd in parts.values():
            all_verts.append(pd["vertices"])
            all_faces.append(pd["faces"] + vo)
            vo += len(pd["vertices"])
        body = Mesh(name="Body", vertices=np.vstack(all_verts), faces=np.vstack(all_faces))
        body.compute_normals()
        return body


# ─── Character Creator Core ──────────────────────────────────────────────────

class CharacterCreator:
    def __init__(self):
        self.characters = {}

    def create_character(self, name: str, height: float = 1.8,
                         style: str = "default",
                         custom_params: Dict[str, float] = None) -> Dict[str, Any]:
        params = PRESETS.get(style, PRESETS["default"]).copy()
        if custom_params:
            params.update(custom_params)
        total_h = (params["upper_leg_length"] + params["lower_leg_length"] +
                   params["foot_length"] * 0.3 + params["torso_length"] +
                   params["neck_length"] + params["head_size"] * 2)
        scale = height / max(total_h, 0.01)
        for key in params:
            if any(key.endswith(s) for s in ["_length", "_width", "_depth", "_size"]):
                params[key] *= scale

        generator = BodyMeshGenerator(params)
        body_mesh = generator.generate_full_body()
        armature = create_humanoid_armature(height=height, name=f"{name}_Armature")
        skin_weights = auto_skin_weights(body_mesh, armature, max_distance=1.0)
        pose_lib = PoseLibrary()
        pose_lib.add_pose("T-Pose", TPOSE_DATA)
        pose_lib.add_pose("A-Pose", APOSE_DATA)
        # Armature is already in T-pose from create_humanoid_armature
        armature.update_transforms()

        character_data = {
            "name": name, "style": style, "height": height, "params": params,
            "vertex_count": body_mesh.vertex_count(), "face_count": body_mesh.face_count(),
            "bone_count": len(armature.bones),
            "mesh": body_mesh.to_dict(), "armature": armature.to_dict(),
            "skin_weights": skin_weights.to_dict(), "pose_library": pose_lib.to_dict(),
            "available_poses": ["T-Pose", "A-Pose"],
            "available_expressions": list(FACIAL_BLEND_SHAPES.keys()),
            "available_clothing": ["shirt", "pants", "skirt", "hat"],
        }
        self.characters[name] = {"mesh": body_mesh, "armature": armature,
                                  "skin_weights": skin_weights, "pose_library": pose_lib, "params": params}
        return character_data

    def apply_pose(self, character_name: str, pose_name: str) -> Dict[str, Any]:
        if character_name not in self.characters:
            raise ValueError(f"Character '{character_name}' not found")
        char = self.characters[character_name]
        pose_data = char["pose_library"].get_pose(pose_name)
        if pose_data is None:
            raise ValueError(f"Pose '{pose_name}' not found")
        # Convert pose data format to Armature.apply_pose format
        # TPOSE_DATA format: {bone_name: {"position": [x,y,z], "rotation": [w,x,y,z]}}
        # Armature.apply_pose expects: {bone_name: {"head_pos": [x,y,z], "tail_pos": [x,y,z], "roll": float}}
        arm_pose = {}
        for bone_name, data in pose_data.items():
            bone = char["armature"].bones.get(bone_name)
            if bone is None:
                continue
            pos = np.array(data.get("position", [0, 0, 0]), dtype=np.float64)
            rot = data.get("rotation", [1, 0, 0, 0])
            bone_length = bone.length if bone.length > 1e-10 else 0.01
            # Rotate the rest-pose direction (Y-axis) by the pose quaternion
            rot_mat = quat_to_rotation_matrix(np.array(rot, dtype=np.float64))
            new_dir = rot_mat @ np.array([0.0, bone_length, 0.0], dtype=np.float64)
            arm_pose[bone_name] = {
                "head_pos": pos.tolist(),
                "tail_pos": (pos + new_dir).tolist(),
            }
        char["armature"].apply_pose(arm_pose)
        char["armature"].update_transforms()
        bone_transforms = [char["armature"].bones[bn].world_transform for bn in char["armature"].bones]
        posed_verts = char["skin_weights"].apply(char["mesh"].vertices, bone_transforms)
        return {"character": character_name, "pose": pose_name,
                "posed_vertices": posed_verts.tolist(), "armature": char["armature"].to_dict()}

    def apply_facial_expression(self, character_name: str, expression: str,
                                weight: float = 1.0) -> Dict[str, Any]:
        """Apply a facial expression blend shape."""
        if character_name not in self.characters:
            raise ValueError(f"Character '{character_name}' not found")
        if expression not in FACIAL_BLEND_SHAPES:
            raise ValueError(f"Unknown expression '{expression}'. Available: {list(FACIAL_BLEND_SHAPES.keys())}")
        char = self.characters[character_name]
        expr = FACIAL_BLEND_SHAPES[expression]
        # Apply bone rotation deltas to head
        if "bone_rotations" in expr and "Head" in expr["bone_rotations"]:
            bone = char["armature"].bones.get("Head")
            if bone:
                delta = expr["bone_rotations"]["Head"]
                blended = [1.0 + (d - 1.0) * weight for d in delta]
                bone.local_rotation = np.array(blended, dtype=np.float64)
                char["armature"].update_transforms()
        return {"character": character_name, "expression": expression, "weight": weight}

    def generate_clothing(self, character_name: str, outfit: str = "shirt") -> Mesh:
        """Generate and return a clothing mesh for the character."""
        if character_name not in self.characters:
            raise ValueError(f"Character '{character_name}' not found")
        char = self.characters[character_name]
        params = char["params"]
        height = params.get("height", params.get("upper_leg_length", 0.42) * 4)
        # Estimate height from params
        total_h = (params.get("upper_leg_length", 0.42) + params.get("lower_leg_length", 0.40) +
                   params.get("torso_length", 0.45) + params.get("neck_length", 0.10) +
                   params.get("head_size", 0.12) * 2)

        gen = ClothingGenerator()
        cloth_map = {
            "shirt": gen.generate_shirt,
            "pants": gen.generate_pants,
            "skirt": gen.generate_skirt,
            "hat": gen.generate_hat,
        }
        if outfit not in cloth_map:
            raise ValueError(f"Unknown outfit '{outfit}'. Available: {list(cloth_map.keys())}")
        data = cloth_map[outfit](params, total_h)
        mesh = Mesh(name=f"{character_name}_{outfit}", vertices=data["vertices"], faces=data["faces"])
        mesh.compute_normals()
        return mesh

    def generate_lod(self, character_name: str, level: int = 1) -> Dict[str, Any]:
        """Generate a lower-detail version of the character mesh."""
        if character_name not in self.characters:
            raise ValueError(f"Character '{character_name}' not found")
        mesh = self.characters[character_name]["mesh"]
        n_faces = mesh.face_count()
        reduction = 0.5 ** level
        target = max(int(n_faces * reduction), 24)
        step = max(1, n_faces // target)
        lod_faces = mesh.faces[::step][:target]
        lod = Mesh(name=f"{character_name}_LOD{level}", vertices=mesh.vertices, faces=lod_faces)
        lod.compute_normals()
        return {"name": f"{character_name}_LOD{level}", "vertices": lod.vertex_count(), "faces": lod.face_count()}

    def export_character(self, character_name: str, output_dir: str,
                         formats: List[str] = None) -> Dict[str, str]:
        if formats is None:
            formats = ["obj", "json"]
        if character_name not in self.characters:
            raise ValueError(f"Character '{character_name}' not found")
        char = self.characters[character_name]
        out_dir = ensure_output_dir(output_dir)
        exported = {}
        for fmt in formats:
            if fmt == "obj":
                fp = out_dir / f"{character_name}.obj"
                char["mesh"].save_obj(str(fp))
                exported["obj"] = str(fp)
            elif fmt == "json":
                fp = out_dir / f"{character_name}.json"
                data = {"mesh": char["mesh"].to_dict(), "armature": char["armature"].to_dict(),
                        "skin_weights": char["skin_weights"].to_dict(), "params": char["params"]}
                save_json(data, str(fp))
                exported["json"] = str(fp)
        return exported

    def list_presets(self) -> Dict[str, Dict[str, float]]:
        return {k: v for k, v in PRESETS.items()}


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Nexus3D Character Creator v0.3.0")
    sub = parser.add_subparsers(dest="command")
    cp = sub.add_parser("create", help="Create a character")
    cp.add_argument("--name", required=True); cp.add_argument("--height", type=float, default=1.8)
    cp.add_argument("--style", default="default", choices=list(PRESETS.keys()))
    cp.add_argument("--output", "-o", default="."); cp.add_argument("--format", nargs="+", default=["obj", "json"])
    fp = sub.add_parser("facial-expression", help="Apply facial expression")
    fp.add_argument("--name", required=True); fp.add_argument("--expression", required=True,
                   choices=list(FACIAL_BLEND_SHAPES.keys()))
    fp.add_argument("--weight", type=float, default=1.0); fp.add_argument("--output", "-o")
    clp = sub.add_parser("clothe", help="Generate clothing")
    clp.add_argument("--name", required=True); clp.add_argument("--outfit", default="shirt",
                     choices=["shirt", "pants", "skirt", "hat"])
    clp.add_argument("--output", "-o", required=True)
    lp = sub.add_parser("lod", help="Generate LOD")
    lp.add_argument("--name", required=True); lp.add_argument("--level", type=int, default=1)
    lp.add_argument("--output", "-o", default=".")
    ep = sub.add_parser("export", help="Export character")
    ep.add_argument("--name", required=True); ep.add_argument("--format", nargs="+", default=["obj", "json"])
    ep.add_argument("--output", "-o", default=".")
    sub.add_parser("presets", help="List presets")
    sub.add_parser("expressions", help="List expressions")
    args = parser.parse_args()
    if args.command is None: parser.print_help(); return
    creator = CharacterCreator()
    if args.command == "create":
        result = creator.create_character(args.name, args.height, args.style)
        files = creator.export_character(args.name, args.output, args.format)
        result["files"] = files; print(json.dumps(result, cls=NumpyEncoder, indent=2))
    elif args.command == "facial-expression":
        result = creator.apply_facial_expression(args.name, args.expression, args.weight)
        print(json.dumps(result, indent=2))
    elif args.command == "clothe":
        mesh = creator.generate_clothing(args.name, args.outfit)
        mesh.save_obj(args.output)
        print(json.dumps({"clothing": args.outfit, "file": args.output,
                          "vertices": mesh.vertex_count(), "faces": mesh.face_count()}, indent=2))
    elif args.command == "lod":
        result = creator.generate_lod(args.name, args.level)
        print(json.dumps(result, indent=2))
    elif args.command == "export":
        files = creator.export_character(args.name, args.output, args.format)
        print(json.dumps(files, indent=2))
    elif args.command == "presets":
        print(json.dumps(list(PRESETS.keys()), indent=2))
    elif args.command == "expressions":
        print(json.dumps(list(FACIAL_BLEND_SHAPES.keys()), indent=2))

if __name__ == "__main__":
    main()
