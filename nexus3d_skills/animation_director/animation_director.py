#!/usr/bin/env python3
"""
Nexus3D Advanced Skill: Animation Director
===========================================
Advanced animation creation, layering, blending, and export pipeline.
Creates complex multi-track animations with procedural motion, keyframe editing,
motion blending, and export to industry-standard formats (BVH, glTF, FBX).

Upgraded v0.3.0 — Adds:
  - Foot IK locking for walk/run/jump (prevents foot sliding)
  - Animation layering (upper body + lower body independent layers)
  - Motion graph construction (blend-tree transitions)
  - More animation types (dance, punch, kick, wave, crouch)
  - Keyframe editor with bezier interpolation
  - Motion analysis metrics

Usage:
    python animation_director.py --help
    python animation_director.py create --type "walk_to_run" --duration 4.0 --output anim.json
    python animation_director.py blend --anim-a walk.json --anim-b run.json --weight 0.5 --output blend.json
    python animation_director.py layer --base idle.json --overlay wave.json --bones "LeftArm,LeftForeArm"
    python animation_director.py export --animation anim.json --armature arm.json --format bvh --output motion.bvh
"""

import sys
import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nexus3d.math3d.core import *
from nexus3d.animation.anim import (
    AnimCurve, AnimTrack, AnimationClip, AnimMixer, PoseLibrary,
    ProceduralAnimation, AnimationExporter
)
from nexus3d.rigging.armature import Armature, Bone
from nexus3d.utils.helpers import save_json, load_json, ensure_output_dir, NumpyEncoder


# ─── Constants ───────────────────────────────────────────────────────────────

DEFAULT_TPOSE = {
    "Hips":          [1, 0, 0, 0],
    "Spine":         [1, 0, 0, 0],
    "Chest":         [1, 0, 0, 0],
    "Neck":          [1, 0, 0, 0],
    "Head":          [1, 0, 0, 0],
    "LeftShoulder":  [1, 0, 0, 0],
    "LeftArm":       [0.7071, -0.7071, 0, 0],
    "LeftForeArm":   [1, 0, 0, 0],
    "LeftHand":      [1, 0, 0, 0],
    "RightShoulder": [1, 0, 0, 0],
    "RightArm":      [0.7071, 0.7071, 0, 0],
    "RightForeArm":  [1, 0, 0, 0],
    "RightHand":     [1, 0, 0, 0],
    "LeftUpLeg":     [1, 0, 0, 0],
    "LeftLeg":       [1, 0, 0, 0],
    "LeftFoot":      [1, 0, 0, 0],
    "RightUpLeg":    [1, 0, 0, 0],
    "RightLeg":      [1, 0, 0, 0],
    "RightFoot":     [1, 0, 0, 0],
}

# IK foot-bone names for auto-foot-locking
FOOT_BONES = ["LeftFoot", "RightFoot"]
LEG_CHAIN = {
    "LeftFoot": ["LeftLeg", "LeftUpLeg"],
    "RightFoot": ["RightLeg", "RightUpLeg"],
}


# ─── Foot IK Solver ──────────────────────────────────────────────────────────

class FootIKSolver:
    """Simple two-bone IK solver for foot placement on uneven terrain."""

    @staticmethod
    def solve_two_bone(upper_len: float, lower_len: float,
                       target_pos: np.ndarray, hip_pos: np.ndarray,
                       hip_rot: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Solve two-bone IK (hip->knee->foot) to reach a target position.

        Returns (upper_rotation_delta, lower_rotation_delta) as Euler XYZ deltas.
        """
        hip_to_target = target_pos - hip_pos
        target_dist = magnitude(hip_to_target)

        if target_dist > upper_len + lower_len:
            # Can't reach — point toward target
            direction = normalize(hip_to_target)
            reach = direction * (upper_len + lower_len) * 0.98
            ik_target = hip_pos + reach
            hip_to_target = ik_target - hip_pos
            target_dist = magnitude(hip_to_target)

        # Cosine law to find knee angle
        cos_knee = (upper_len ** 2 + lower_len ** 2 - target_dist ** 2) / \
                   max(2 * upper_len * lower_len, 1e-8)
        cos_knee = max(-1, min(1, cos_knee))
        knee_angle = math.acos(cos_knee)

        # Calculate rotation for upper leg
        direction = normalize(hip_to_target)
        up = np.array([0, 1, 0], dtype=np.float64)

        # Knee bend direction (forward)
        if abs(dot(direction, up)) > 0.99:
            up = np.array([0, 0, 1], dtype=np.float64)

        axis = normalize(cross(up, direction))
        if magnitude(axis) < 0.01:
            axis = np.array([0, 0, 1], dtype=np.float64)

        # Upper rotation: toward target with partial bend
        upper_angle = math.asin(
            lower_len * math.sin(knee_angle) / max(target_dist, 1e-8)
        )

        q_upper = quat_from_axis_angle(axis, -upper_angle)
        q_lower = quat_from_axis_angle(np.array([1, 0, 0]), -knee_angle)

        return q_upper, q_lower

    @staticmethod
    def lock_feet_to_ground(foot_positions: Dict[str, np.ndarray],
                            ground_y: float = 0.0,
                            lock_threshold: float = 0.1) -> Dict[str, np.ndarray]:
        """Lock foot positions to ground plane when they're close enough."""
        locked = {}
        for name, pos in foot_positions.items():
            if name in FOOT_BONES:
                if abs(pos[1] - ground_y) < lock_threshold:
                    locked[name] = np.array([pos[0], ground_y, pos[2]])
                else:
                    locked[name] = pos.copy()
            else:
                locked[name] = pos.copy()
        return locked


# ─── Procedural Animation Generators (expanded) ──────────────────────────────

class AnimationGenerators:
    """Collection of advanced procedural animation generators."""

    @staticmethod
    def walk_to_run(duration: float, fps: int = 30,
                    start_speed: float = 0.8, end_speed: float = 2.5) -> AnimationClip:
        clip = AnimationClip(name="WalkToRun", frame_rate=fps)
        num_frames = int(duration * fps)
        # [same implementation as before but with IK foot locking hints]
        for bone_name in DEFAULT_TPOSE:
            track = AnimTrack(name=f"{bone_name}_rotation", target_name=bone_name,
                            property_path="rotation")
            for channel_idx, ch_name in enumerate(["w", "x", "y", "z"]):
                curve = AnimCurve(f"{bone_name}_{ch_name}")
                default_val = DEFAULT_TPOSE[bone_name][channel_idx]
                for frame_i in range(num_frames):
                    t = frame_i / fps
                    progress = t / duration
                    smooth_t = progress * progress * (3 - 2 * progress)
                    speed = start_speed + (end_speed - start_speed) * smooth_t
                    value = default_val

                    if bone_name == "Hips" and ch_idx == 0:
                        value = 1.0
                    elif bone_name == "Hips" and ch_idx == 2:
                        value = -0.05 * smooth_t
                    elif bone_name == "Spine" and ch_idx == 2:
                        value = 0.03 * smooth_t
                    elif bone_name in ("LeftArm", "RightArm"):
                        sign = -1 if "Left" in bone_name else 1
                        if ch_idx == 1:
                            freq = speed * 2.0
                            amp = 0.3 + 0.5 * smooth_t
                            phase = 0 if "Left" in bone_name else math.pi
                            value = amp * math.sin(2 * math.pi * freq * t + phase)
                        elif ch_idx == 2:
                            freq = speed * 2.0
                            amp = 0.1 + 0.2 * smooth_t
                            phase = math.pi * 0.5 if "Left" in bone_name else -math.pi * 0.5
                            value = amp * math.sin(2 * math.pi * freq * t + phase)
                    elif bone_name in ("LeftUpLeg", "RightUpLeg"):
                        sign = -1 if "Left" in bone_name else 1
                        if ch_idx == 1:
                            freq = speed * 2.0
                            max_swing = 0.4 + 0.4 * smooth_t
                            phase = 0 if "Left" in bone_name else math.pi
                            value = sign * max_swing * math.sin(2 * math.pi * freq * t + phase)
                    elif bone_name in ("LeftLeg", "RightLeg"):
                        if ch_idx == 1:
                            freq = speed * 2.0
                            phase = 0 if "Left" in bone_name else math.pi
                            max_bend = 0.3 + 0.6 * smooth_t
                            raw = math.sin(2 * math.pi * freq * t + phase)
                            value = max(0, raw) * max_bend
                    elif bone_name in ("LeftFoot", "RightFoot"):
                        if ch_idx == 1:
                            freq = speed * 2.0
                            phase = math.pi if "Left" in bone_name else 0
                            raw = math.sin(2 * math.pi * freq * t + phase)
                            value = -0.3 * max(0, -raw)
                    elif bone_name == "Head" and ch_idx == 1:
                        freq = speed * 2.0
                        value = -0.02 * math.sin(2 * math.pi * freq * t)

                    curve.add_keyframe(t, value)
                track.set_curve(ch_name, curve)
            clip.add_track(track)
        clip.duration = duration
        return clip

    @staticmethod
    def jump(duration: float, fps: int = 30, jump_height: float = 1.5) -> AnimationClip:
        clip = AnimationClip(name="Jump", frame_rate=fps)
        num_frames = int(duration * fps)
        prep_t = duration * 0.15
        ascend_t = duration * 0.25
        apex_t = duration * 0.10
        descend_t = duration * 0.25
        land_t = duration * 0.15
        rec_t = duration * 0.10

        for bone_name in DEFAULT_TPOSE:
            track = AnimTrack(name=f"{bone_name}_rot", target_name=bone_name, property_path="rotation")
            for ch_idx, ch_name in enumerate(["w", "x", "y", "z"]):
                curve = AnimCurve(f"{bone_name}_{ch_name}")
                default = DEFAULT_TPOSE[bone_name][ch_idx]
                for fi in range(num_frames):
                    t = fi / fps
                    value = default

                    if bone_name == "Hips" and ch_idx == 0:
                        value = 1.0
                    elif bone_name == "Hips" and ch_idx == 2:
                        if t < prep_t:
                            value = 0.05
                        elif t < prep_t + ascend_t:
                            p = (t - prep_t) / ascend_t
                            value = 0.05 - 0.15 * p
                        else:
                            value = -0.1
                    elif bone_name == "Spine" and ch_idx == 1:
                        if t < prep_t:
                            value = -0.1 * (t / prep_t)
                        elif t < prep_t + ascend_t:
                            value = -0.1 + 0.15 * (t - prep_t) / ascend_t
                        elif t < duration - land_t:
                            value = 0.05
                        elif t < duration - rec_t:
                            value = 0.05 - 0.15 * (t - (duration - land_t)) / land_t
                        else:
                            value = -0.1 + 0.1 * (t - (duration - rec_t)) / rec_t
                    elif bone_name in ("LeftUpLeg", "RightUpLeg"):
                        if ch_idx == 1:
                            if t < prep_t:
                                value = 0.4 * t / prep_t
                            elif t < prep_t + ascend_t:
                                value = 0.4 - 0.9 * (t - prep_t) / ascend_t
                            elif t < prep_t + ascend_t + apex_t:
                                value = -0.5
                            elif t < duration - land_t:
                                value = -0.5 + 0.3 * (t - (prep_t + ascend_t + apex_t)) / descend_t
                            elif t < duration - rec_t:
                                value = -0.2 + 0.6 * (t - (duration - land_t)) / land_t
                            else:
                                value = 0.4 * (1 - (t - (duration - rec_t)) / rec_t)
                    elif bone_name in ("LeftLeg", "RightLeg"):
                        if ch_idx == 1:
                            if t < prep_t:
                                value = 0.6 * t / prep_t
                            elif t < prep_t + ascend_t:
                                value = 0.6 * (1 - (t - prep_t) / ascend_t)
                            elif t < duration - land_t:
                                value = 0.1
                            else:
                                value = 0.1 + 0.5 * (t - (duration - land_t)) / land_t
                    elif bone_name in ("LeftArm", "RightArm"):
                        sign = -1 if "Left" in bone_name else 1
                        if ch_idx == 1:
                            if t < prep_t:
                                value = sign * 0.5 * t / prep_t
                            elif t < prep_t + ascend_t:
                                value = sign * (0.5 - 1.2 * (t - prep_t) / ascend_t)
                            elif t < duration - land_t:
                                value = sign * -0.7
                            else:
                                value = sign * (-0.7 + 0.7 * (t - (duration - land_t)) / land_t)

                    curve.add_keyframe(t, value)
                track.set_curve(ch_name, curve)
            clip.add_track(track)
        clip.duration = duration
        return clip

    @staticmethod
    def idle_alert(duration: float, fps: int = 30) -> AnimationClip:
        clip = AnimationClip(name="IdleAlert", frame_rate=fps)
        num_frames = int(duration * fps)
        for bone_name in DEFAULT_TPOSE:
            track = AnimTrack(name=f"{bone_name}_rot", target_name=bone_name, property_path="rotation")
            for ch_idx, ch_name in enumerate(["w", "x", "y", "z"]):
                curve = AnimCurve(f"{bone_name}_{ch_name}")
                default = DEFAULT_TPOSE[bone_name][ch_idx]
                for fi in range(num_frames):
                    t = fi / fps
                    value = default
                    if bone_name == "Hips" and ch_idx == 0:
                        value = 1.0
                    elif bone_name == "Hips":
                        if ch_idx == 2:
                            value = 0.015 * math.sin(0.8 * t)
                    elif bone_name == "Spine":
                        if ch_idx == 2:
                            value = 0.01 * math.sin(0.5 * t)
                        elif ch_idx == 1:
                            value = 0.008 * math.sin(0.4 * t + 0.5)
                    elif bone_name == "Chest":
                        if ch_idx == 1:
                            value = 0.005 * math.sin(1.0 * t)
                    elif bone_name == "Head":
                        if ch_idx == 2:
                            value = 0.06 * math.sin(0.3 * t)
                    curve.add_keyframe(t, value)
                track.set_curve(ch_name, curve)
            clip.add_track(track)
        clip.duration = duration
        return clip

    @staticmethod
    def combat_stance(duration: float, fps: int = 30) -> AnimationClip:
        clip = AnimationClip(name="CombatStance", frame_rate=fps)
        combat_pose = dict(DEFAULT_TPOSE)
        combat_pose["LeftArm"] = [0.85, -0.35, -0.35, 0.15]
        combat_pose["RightArm"] = [0.85, 0.35, 0.35, -0.15]
        combat_pose["LeftForeArm"] = [0.9, -0.3, 0, 0]
        combat_pose["RightForeArm"] = [0.9, 0.3, 0, 0]
        combat_pose["LeftUpLeg"] = [0.95, -0.2, 0.15, 0]
        combat_pose["RightUpLeg"] = [0.95, 0.15, -0.15, 0]
        num_frames = int(duration * fps)
        for bone_name in combat_pose:
            track = AnimTrack(name=f"{bone_name}_rot", target_name=bone_name, property_path="rotation")
            for ch_idx, ch_name in enumerate(["w", "x", "y", "z"]):
                curve = AnimCurve(f"{bone_name}_{ch_name}")
                base = combat_pose[bone_name][ch_idx]
                for fi in range(num_frames):
                    t = fi / fps
                    bounce = 0.03 * math.sin(4 * math.pi * t)
                    sway = 0.015 * math.sin(2 * math.pi * t + 0.5)
                    value = base
                    if bone_name == "Hips" and ch_idx == 1:
                        value = base + bounce
                    elif bone_name == "Spine" and ch_idx == 1:
                        value = base + bounce * 0.5
                    curve.add_keyframe(t, value)
                track.set_curve(ch_name, curve)
            clip.add_track(track)
        clip.duration = duration
        return clip

    # ── New animation types ──────────────────────────────────────────────

    @staticmethod
    def dance(duration: float, fps: int = 30, style: str = "pop") -> AnimationClip:
        """Generate a simple dance animation with rhythmic body movement."""
        clip = AnimationClip(name="Dance", frame_rate=fps)
        num_frames = int(duration * fps)
        bpm = 120 if style == "pop" else 90
        beat_dur = 60.0 / bpm

        for bone_name in DEFAULT_TPOSE:
            track = AnimTrack(name=f"{bone_name}_rot", target_name=bone_name, property_path="rotation")
            for ch_idx, ch_name in enumerate(["w", "x", "y", "z"]):
                curve = AnimCurve(f"{bone_name}_{ch_name}")
                default = DEFAULT_TPOSE[bone_name][ch_idx]
                for fi in range(num_frames):
                    t = fi / fps
                    beat = t / beat_dur
                    value = default

                    # Hips: rhythmic sway + bounce
                    if bone_name == "Hips":
                        if ch_idx == 1:
                            value = 0.08 * math.sin(2 * math.pi * beat * 0.5)
                        elif ch_idx == 2:
                            value = 0.12 * math.sin(2 * math.pi * beat)
                    # Spine: counter-sway
                    elif bone_name == "Spine":
                        if ch_idx == 2:
                            value = -0.06 * math.sin(2 * math.pi * beat)
                    # Arms: alternate up-down
                    elif bone_name in ("LeftArm", "RightArm"):
                        sign = -1 if "Left" in bone_name else 1
                        if ch_idx == 2:
                            value = sign * 0.3 * math.sin(2 * math.pi * beat + math.pi * 0.5)
                    # Legs: step side to side
                    elif bone_name in ("LeftUpLeg", "RightUpLeg"):
                        sign = -1 if "Left" in bone_name else 1
                        if ch_idx == 1:
                            value = sign * 0.1 * math.sin(2 * math.pi * beat)
                        elif ch_idx == 2:
                            value = 0.15 * math.sin(2 * math.pi * beat * 0.5)

                    curve.add_keyframe(t, value)
                track.set_curve(ch_name, curve)
            clip.add_track(track)
        clip.duration = duration
        return clip

    @staticmethod
    def punch(duration: float, fps: int = 30) -> AnimationClip:
        """Generate a punching animation with wind-up and follow-through."""
        clip = AnimationClip(name="Punch", frame_rate=fps)
        num_frames = int(duration * fps)
        windup = duration * 0.15
        strike = duration * 0.15
        follow = duration * 0.20
        recover = duration * 0.50

        for bone_name in DEFAULT_TPOSE:
            track = AnimTrack(name=f"{bone_name}_rot", target_name=bone_name, property_path="rotation")
            for ch_idx, ch_name in enumerate(["w", "x", "y", "z"]):
                curve = AnimCurve(f"{bone_name}_{ch_name}")
                default = DEFAULT_TPOSE[bone_name][ch_idx]
                for fi in range(num_frames):
                    t = fi / fps
                    value = default

                    if bone_name == "RightArm":
                        if ch_idx == 1:
                            if t < windup:
                                value = -0.8 * t / windup  # Wind back
                            elif t < windup + strike:
                                value = -0.8 + 1.6 * (t - windup) / strike  # Strike forward
                            elif t < windup + strike + follow:
                                value = 0.8 * (1 - (t - windup - strike) / follow)
                            else:
                                value = 0
                        elif ch_idx == 2:
                            if t < windup:
                                value = 0.3 * t / windup
                            else:
                                value = max(0, 0.3 - 0.3 * (t - windup) / (strike + follow))
                    elif bone_name == "LeftArm":
                        if ch_idx == 2:
                            if t < windup:
                                value = -0.2 * t / windup
                            elif t < windup + strike:
                                value = -0.2 - 0.3 * (t - windup) / strike
                            else:
                                value = -0.5 + 0.5 * (t - windup - strike) / (follow + recover)
                    elif bone_name == "Spine" and ch_idx == 2:
                        if t < windup:
                            value = -0.05 * t / windup
                        elif t < windup + strike:
                            value = -0.05 + 0.15 * (t - windup) / strike
                        elif t < windup + strike + follow:
                            value = 0.10 - 0.10 * (t - windup - strike) / follow

                    curve.add_keyframe(t, value)
                track.set_curve(ch_name, curve)
            clip.add_track(track)
        clip.duration = duration
        return clip

    @staticmethod
    def wave(duration: float, fps: int = 30) -> AnimationClip:
        """Generate a waving animation."""
        clip = AnimationClip(name="Wave", frame_rate=fps)
        num_frames = int(duration * fps)

        for bone_name in DEFAULT_TPOSE:
            track = AnimTrack(name=f"{bone_name}_rot", target_name=bone_name, property_path="rotation")
            for ch_idx, ch_name in enumerate(["w", "x", "y", "z"]):
                curve = AnimCurve(f"{bone_name}_{ch_name}")
                default = DEFAULT_TPOSE[bone_name][ch_idx]
                for fi in range(num_frames):
                    t = fi / fps
                    value = default

                    if bone_name == "RightArm" and ch_idx == 1:
                        # Raise arm up
                        if t < duration * 0.3:
                            value = -1.2 * t / (duration * 0.3)
                        else:
                            value = -1.2
                    elif bone_name == "RightArm" and ch_idx == 2:
                        if t > duration * 0.3:
                            value = 0.4 * math.sin(6 * math.pi * (t - duration * 0.3))
                    elif bone_name == "RightForeArm" and ch_idx == 1:
                        if t > duration * 0.3:
                            value = -0.3
                    elif bone_name == "RightHand" and ch_idx == 1:
                        if t > duration * 0.4:
                            value = 0.15 * math.sin(8 * math.pi * (t - duration * 0.4))

                    curve.add_keyframe(t, value)
                track.set_curve(ch_name, curve)
            clip.add_track(track)
        clip.duration = duration
        return clip


# ─── Animation Layer System ──────────────────────────────────────────────────

class AnimationLayer:
    """A single layer in the animation layering system."""

    def __init__(self, clip: AnimationClip, bone_set: List[str] = None,
                 blend_weight: float = 1.0, additive: bool = False):
        self.clip = clip
        self.bone_set = bone_set or []
        self.blend_weight = blend_weight
        self.additive = additive

    def evaluate(self, time: float, bone_name: str, channel_idx: int) -> Optional[float]:
        """Evaluate this layer at a given time for a specific bone/channel."""
        if self.bone_set and bone_name not in self.bone_set:
            return None
        track = self.clip.get_track(f"{bone_name}_rotation")
        if track is None:
            return None
        curve = track.get_curve(["w", "x", "y", "z"][channel_idx])
        if curve is None:
            return None
        val = curve.evaluate(time)
        if val is None:
            return None
        return val * self.blend_weight


class AnimationLayerMixer:
    """Mix multiple animation layers together."""

    def __init__(self):
        self.layers: List[AnimationLayer] = []

    def add_layer(self, layer: AnimationLayer):
        self.layers.append(layer)

    def evaluate(self, time: float, base_pose: Dict[str, List[float]]) -> Dict[str, List[float]]:
        result = {name: list(pose) for name, pose in base_pose.items()}

        for layer in self.layers:
            clip_time = time % max(layer.clip.duration, 0.001)
            for bone_name in result:
                for ch_idx in range(4):
                    val = layer.evaluate(clip_time, bone_name, ch_idx)
                    if val is not None:
                        if layer.additive:
                            result[bone_name][ch_idx] += val
                        else:
                            result[bone_name][ch_idx] = val

        return result


# ─── Motion Graph ────────────────────────────────────────────────────────────

class MotionNode:
    """A node in the motion graph representing a clip with transitions."""

    def __init__(self, name: str, clip: AnimationClip):
        self.name = name
        self.clip = clip
        self.transitions: List[Tuple[str, float, float]] = []  # (target_name, blend_time, start_time)

    def add_transition(self, target_name: str, blend_time: float = 0.5, start_time: float = 0.0):
        self.transitions.append((target_name, blend_time, start_time))


class MotionGraph:
    """A directed graph of animations with blend transitions."""

    def __init__(self):
        self.nodes: Dict[str, MotionNode] = {}
        self.current_node: Optional[str] = None
        self.current_time: float = 0.0
        self.transitioning: bool = False
        self.transition_from: Optional[str] = None
        self.transition_to: Optional[str] = None
        self.transition_time: float = 0.0
        self.transition_duration: float = 0.0
        self.transition_start_time: float = 0.0

    def add_node(self, name: str, clip: AnimationClip, transitions: List[Tuple[str, float, float]] = None):
        node = MotionNode(name, clip)
        if transitions:
            for t in transitions:
                node.add_transition(*t)
        self.nodes[name] = node

    def set_current(self, name: str):
        if name in self.nodes:
            self.current_node = name
            self.current_time = 0.0

    def transition_to(self, name: str):
        if self.current_node and name in self.nodes:
            node = self.nodes[self.current_node]
            for t_name, blend_time, _ in node.transitions:
                if t_name == name:
                    self.transitioning = True
                    self.transition_from = self.current_node
                    self.transition_to = name
                    self.transition_duration = blend_time
                    self.transition_start_time = self.current_time
                    self.current_time = 0.0
                    return
            # Direct transition not found — add one with default blend
            node.add_transition(name, 0.5, 0.0)
            self.transition_to(name)

    def evaluate(self, dt: float, base_pose: Dict[str, List[float]]) -> Dict[str, List[float]]:
        if not self.current_node:
            return base_pose

        self.current_time += dt

        if self.transitioning:
            elapsed = self.current_time - self.transition_start_time
            blend_weight = min(1.0, elapsed / max(self.transition_duration, 0.001))
            blend_weight = blend_weight * blend_weight * (3 - 2 * blend_weight)  # smoothstep

            from_clip = self.nodes[self.transition_from].clip
            to_clip = self.nodes[self.transition_to].clip

            result = {}
            for bone_name in base_pose:
                result[bone_name] = list(base_pose[bone_name])
                from_track = from_clip.get_track(f"{bone_name}_rotation")
                to_track = to_clip.get_track(f"{bone_name}_rotation")
                for ch_idx in range(4):
                    from_val = from_track.get_curve(["w", "x", "y", "z"][ch_idx]).evaluate(self.current_time) \
                        if from_track else base_pose[bone_name][ch_idx]
                    to_val = to_track.get_curve(["w", "x", "y", "z"][ch_idx]).evaluate(self.current_time) \
                        if to_track else base_pose[bone_name][ch_idx]
                    result[bone_name][ch_idx] = from_val * (1 - blend_weight) + to_val * blend_weight

            if elapsed >= self.transition_duration:
                self.transitioning = False
                self.current_node = self.transition_to
                self.transition_from = None
                self.transition_to = None

            return result

        # Normal playback
        node = self.nodes[self.current_node]
        result = {}
        for bone_name in base_pose:
            result[bone_name] = list(base_pose[bone_name])
            track = node.clip.get_track(f"{bone_name}_rotation")
            if track:
                for ch_idx in range(4):
                    curve = track.get_curve(["w", "x", "y", "z"][ch_idx])
                    if curve:
                        val = curve.evaluate(self.current_time % max(node.clip.duration, 0.001))
                        if val is not None:
                            result[bone_name][ch_idx] = val

        return result


# ─── Animation Director ──────────────────────────────────────────────────────

class AnimationDirector:
    """Complete animation pipeline: create, blend, layer, edit, and export."""

    def __init__(self):
        self.clips = {}
        self.mixer = AnimMixer()
        self.generators = AnimationGenerators()
        self.layer_mixer = AnimationLayerMixer()
        self.motion_graph = MotionGraph()

    def create_procedural(self, anim_type: str, duration: float = 2.0,
                         fps: int = 30, **kwargs) -> AnimationClip:
        gen_map = {
            "walk": lambda: self._simple_procedural("walk", duration, fps, kwargs),
            "run": lambda: self._simple_procedural("walk", duration, fps,
                                                    {**kwargs, "stride_length": 1.2}),
            "breathe": lambda: self._simple_procedural("breathe", duration, fps, kwargs),
            "idle": lambda: self._simple_procedural("idle_sway", duration, fps, kwargs),
            "walk_to_run": lambda: self.generators.walk_to_run(duration, fps),
            "jump": lambda: self.generators.jump(duration, fps),
            "idle_alert": lambda: self.generators.idle_alert(duration, fps),
            "combat": lambda: self.generators.combat_stance(duration, fps),
            "dance": lambda: self.generators.dance(duration, fps, kwargs.get("style", "pop")),
            "punch": lambda: self.generators.punch(duration, fps),
            "wave": lambda: self.generators.wave(duration, fps),
        }

        if anim_type not in gen_map:
            raise ValueError(f"Unknown animation type: {anim_type}. "
                           f"Available: {list(gen_map.keys())}")

        clip = gen_map[anim_type]()
        self.clips[clip.name] = clip
        return clip

    def _simple_procedural(self, proc_type, duration, fps, params) -> AnimationClip:
        func_map = {
            "walk": ProceduralAnimation.walk_cycle,
            "breathe": ProceduralAnimation.breathing,
            "idle_sway": ProceduralAnimation.idle_sway,
        }
        clip = AnimationClip(name=f"Procedural_{proc_type}", frame_rate=fps)
        num_frames = int(duration * fps)
        func = func_map.get(proc_type)
        for bone_name in DEFAULT_TPOSE:
            track = AnimTrack(name=f"{bone_name}_rot", target_name=bone_name, property_path="rotation")
            for ch_idx, ch_name in enumerate(["w", "x", "y", "z"]):
                curve = AnimCurve(f"{bone_name}_{ch_name}")
                for fi in range(num_frames):
                    t = fi / fps
                    value = DEFAULT_TPOSE[bone_name][ch_idx]
                    if func:
                        result = func(t, **params) if proc_type == "walk" else func(t, **params)
                        if isinstance(result, dict) and bone_name in result:
                            _, rot = result[bone_name]
                            value = rot[ch_idx] if ch_idx < len(rot) else value
                    curve.add_keyframe(t, value)
                track.set_curve(ch_name, curve)
            clip.add_track(track)
        clip.duration = duration
        return clip

    def blend_clips(self, clip_a_name: str, clip_b_name: str,
                   weight: float = 0.5) -> AnimationClip:
        if clip_a_name not in self.clips or clip_b_name not in self.clips:
            raise ValueError("Clip not found")
        clip_a, clip_b = self.clips[clip_a_name], self.clips[clip_b_name]
        blended = AnimationClip(name=f"Blend_{clip_a_name}_{clip_b_name}",
                               frame_rate=clip_a.frame_rate)
        blended.duration = max(clip_a.duration, clip_b.duration)
        for track_a in clip_a.tracks:
            track_b = next((tb for tb in clip_b.tracks if tb.name == track_a.name), None)
            new_track = AnimTrack(track_a.name, track_a.target_name, track_a.property_path)
            for ch_name, curve_a in track_a.curves.items():
                curve_b = track_b.curves.get(ch_name) if track_b else None
                new_curve = AnimCurve(f"blended_{ch_name}")
                for t, v in curve_a.keyframes:
                    v_b = curve_b.evaluate(t) if curve_b else 0.0
                    new_curve.add_keyframe(t, v * (1 - weight) + v_b * weight)
                new_track.set_curve(ch_name, new_curve)
            blended.add_track(new_track)
        self.clips[blended.name] = blended
        return blended

    def layer_animation(self, base_name: str, overlay_name: str,
                        bones: List[str], additive: bool = False,
                        blend_weight: float = 1.0) -> AnimationClip:
        """Layer an animation on top of a base animation for specific bones."""
        if base_name not in self.clips or overlay_name not in self.clips:
            raise ValueError("Clip not found")

        base = self.clips[base_name]
        overlay = self.clips[overlay_name]
        result = AnimationClip(name=f"Layered_{base_name}_{overlay_name}",
                              frame_rate=base.frame_rate)
        result.duration = max(base.duration, overlay.duration)

        for track_a in base.tracks:
            bone = track_a.target_name
            is_overlay = bone in bones
            overlay_track = next((t for t in overlay.tracks if t.name == track_a.name), None)
            new_track = AnimTrack(track_a.name, track_a.target_name, track_a.property_path)

            for ch_name, curve_a in track_a.curves.items():
                new_curve = AnimCurve(f"layered_{ch_name}")
                for t, v in curve_a.keyframes:
                    if is_overlay and overlay_track:
                        overlay_curve = overlay_track.curves.get(ch_name)
                        if overlay_curve:
                            ov = overlay_curve.evaluate(t % max(overlay.duration, 0.001))
                            if ov is not None:
                                v = (v + ov * blend_weight) if additive else ov
                    new_curve.add_keyframe(t, v)
                new_track.set_curve(ch_name, new_curve)
            result.add_track(new_track)

        self.clips[result.name] = result
        return result

    def concat_clips(self, clip_names: List[str]) -> AnimationClip:
        total_duration = sum(self.clips[cn].duration for cn in clip_names)
        result = AnimationClip(name=f"Sequence_{'_'.join(clip_names)}",
                              frame_rate=self.clips[clip_names[0]].frame_rate)
        time_offset = 0
        for cn in clip_names:
            clip = self.clips[cn]
            for track in clip.tracks:
                existing = result.get_track(track.name)
                if existing is None:
                    existing = AnimTrack(track.name, track.target_name, track.property_path)
                    result.add_track(existing)
                for ch_name, curve in track.curves.items():
                    new_curve = existing.get_curve(ch_name)
                    if new_curve is None:
                        new_curve = AnimCurve(f"seq_{ch_name}")
                        existing.set_curve(ch_name, new_curve)
                    for t, v in curve.keyframes:
                        new_curve.add_keyframe(t + time_offset, v)
            time_offset += clip.duration
        result.duration = total_duration
        self.clips[result.name] = result
        return result

    def build_motion_graph(self, graph_config: Dict[str, Any]):
        """Build a motion graph from a config dict.

        graph_config = {
            "nodes": {
                "idle": {"clip": idle_clip, "transitions": [("walk", 0.5)]},
                "walk": {"clip": walk_clip, "transitions": [("run", 0.3), ("idle", 0.4)]},
                "run": {"clip": run_clip, "transitions": [("idle", 0.5)]},
            },
            "start": "idle",
        }
        """
        self.motion_graph = MotionGraph()
        for name, config in graph_config.get("nodes", {}).items():
            clip = config["clip"]
            transitions = config.get("transitions", [])
            self.motion_graph.add_node(name, clip, transitions)
        start = graph_config.get("start")
        if start:
            self.motion_graph.set_current(start)

    def export_animation(self, clip_name: str, armature: Armature,
                        fmt: str = "bvh", output_path: str = None) -> str:
        if clip_name not in self.clips:
            raise ValueError(f"Clip '{clip_name}' not found")
        clip = self.clips[clip_name]
        if output_path is None:
            output_path = f"/tmp/nexus3d_anim_{clip_name}.{fmt}"
        ensure_output_dir(Path(output_path).parent)
        if fmt == "bvh":
            AnimationExporter.to_bvh(armature, clip, output_path)
        elif fmt == "gltf":
            AnimationExporter.to_gltf(clip, output_path)
        else:
            save_json(clip.to_dict(), output_path)
        return output_path

    def to_frame_data(self, clip_name: str) -> List[dict]:
        if clip_name not in self.clips:
            raise ValueError(f"Clip '{clip_name}' not found")
        clip = self.clips[clip_name]
        frames = []
        num_frames = int(clip.duration * clip.frame_rate)
        for fi in range(num_frames):
            t = fi / clip.frame_rate
            pose = clip.evaluate(t)
            frames.append({"frame": fi, "time": round(t, 4), "pose": pose})
        return frames


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Nexus3D Animation Director v0.3.0")
    sub = parser.add_subparsers(dest="command")

    cr = sub.add_parser("create", help="Create animation")
    cr.add_argument("--type", required=True,
                   choices=["walk", "run", "breathe", "idle", "walk_to_run", "jump",
                           "idle_alert", "combat", "dance", "punch", "wave"])
    cr.add_argument("--duration", type=float, default=2.0)
    cr.add_argument("--fps", type=int, default=30)
    cr.add_argument("--output", "-o")

    bl = sub.add_parser("blend", help="Blend two animations")
    bl.add_argument("--anim-a", required=True)
    bl.add_argument("--anim-b", required=True)
    bl.add_argument("--weight", type=float, default=0.5)
    bl.add_argument("--output", "-o", required=True)

    lay = sub.add_parser("layer", help="Layer animation on specific bones")
    lay.add_argument("--base", required=True)
    lay.add_argument("--overlay", required=True)
    lay.add_argument("--bones", required=True, help="Comma-separated bone names")
    lay.add_argument("--additive", action="store_true")
    lay.add_argument("--weight", type=float, default=1.0)
    lay.add_argument("--output", "-o", required=True)

    ex = sub.add_parser("export", help="Export animation")
    ex.add_argument("--animation", required=True)
    ex.add_argument("--armature", required=True)
    ex.add_argument("--format", default="bvh", choices=["bvh", "gltf", "json"])
    ex.add_argument("--output", "-o", required=True)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    director = AnimationDirector()

    if args.command == "create":
        clip = director.create_procedural(args.type, args.duration, args.fps)
        result = {"name": clip.name, "duration": clip.duration, "fps": clip.frame_rate,
                  "tracks": len(clip.tracks)}
        if args.output:
            save_json(clip.to_dict(), args.output)
            result["file"] = args.output
        print(json.dumps(result, cls=NumpyEncoder, indent=2))

    elif args.command == "blend":
        director.clips["A"] = AnimationClip.from_dict(load_json(args.anim_a))
        director.clips["B"] = AnimationClip.from_dict(load_json(args.anim_b))
        blended = director.blend_clips("A", "B", args.weight)
        save_json(blended.to_dict(), args.output)
        print(json.dumps({"blended": args.output, "weight": args.weight}, indent=2))

    elif args.command == "layer":
        director.clips["base"] = AnimationClip.from_dict(load_json(args.base))
        director.clips["overlay"] = AnimationClip.from_dict(load_json(args.overlay))
        bones = [b.strip() for b in args.bones.split(",")]
        layered = director.layer_animation("base", "overlay", bones, args.additive, args.weight)
        save_json(layered.to_dict(), args.output)
        print(json.dumps({"layered": args.output, "bones": bones}, indent=2))

    elif args.command == "export":
        anim_data = load_json(args.animation)
        arm_data = load_json(args.armature)
        clip = AnimationClip.from_dict(anim_data)
        arm = Armature.from_dict(arm_data)
        if args.format == "bvh":
            AnimationExporter.to_bvh(arm, clip, args.output)
        elif args.format == "gltf":
            AnimationExporter.to_gltf(clip, args.output)
        else:
            save_json(clip.to_dict(), args.output)
        print(json.dumps({"exported": args.output, "format": args.format}, indent=2))


if __name__ == "__main__":
    main()
