"""
Motion Pipeline — Advanced Motion Synthesis & Retargeting
==========================================================
Provides a complete motion data pipeline: loading, blending,
retargeting, motion graph traversal, and procedural footstep
IK. Designed for the Nexus3D animation system.
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

import numpy as np

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class JointType(Enum):
    ROOT = "root"
    SPINE = "spine"
    HEAD = "head"
    ARM_L = "arm_left"
    ARM_R = "arm_right"
    HAND_L = "hand_left"
    HAND_R = "hand_right"
    LEG_L = "leg_left"
    LEG_R = "leg_right"
    FOOT_L = "foot_left"
    FOOT_R = "foot_right"
    TOE_L = "toe_left"
    TOE_R = "toe_right"
    FINGER = "finger"
    ACCESSORY = "accessory"


@dataclass
class Joint:
    name: str = ""
    parent: Optional[str] = None
    joint_type: JointType = JointType.SPINE
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # quat
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class Keyframe:
    time: float = 0.0
    positions: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    rotations: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)
    scales: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)


@dataclass
class MotionClip:
    """A single motion clip with keyframe data."""
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = "clip"
    duration: float = 1.0
    fps: int = 30
    joints: Dict[str, Joint] = field(default_factory=dict)
    keyframes: List[Keyframe] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    loop: bool = False

    @property
    def num_frames(self) -> int:
        return len(self.keyframes)

    @property
    def frame_time(self) -> float:
        return 1.0 / self.fps

    def get_frame(self, index: int) -> Optional[Keyframe]:
        if 0 <= index < len(self.keyframes):
            return self.keyframes[index]
        return None

    def sample(self, t: float) -> Optional[Keyframe]:
        """Sample the clip at time t (clamped to duration)."""
        if not self.keyframes:
            return None
        t = max(0.0, min(t, self.duration))
        frac = t / self.duration
        idx = int(frac * (len(self.keyframes) - 1))
        return self.keyframes[min(idx, len(self.keyframes) - 1)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "name": self.name,
            "duration": self.duration,
            "fps": self.fps,
            "num_frames": self.num_frames,
            "tags": self.tags,
            "loop": self.loop,
            "joints": {n: asdict(j) for n, j in self.joints.items()},
        }


@dataclass
class MotionGraphNode:
    """A node in the motion graph representing a clip or transition."""
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    clip_uid: str = ""
    name: str = ""
    transitions: List[Tuple[str, float]] = field(default_factory=list)  # (target_uid, cost)
    is_exit: bool = False


# ---------------------------------------------------------------------------
# Skeleton & Retargeting
# ---------------------------------------------------------------------------

@dataclass
class Skeleton:
    """Hierarchical skeleton definition."""
    name: str = "skeleton"
    joints: Dict[str, Joint] = field(default_factory=dict)
    root_joint: str = "hips"

    def get_children(self, joint_name: str) -> List[str]:
        return [n for n, j in self.joints.items() if j.parent == joint_name]

    def traverse(self, order: str = "bfs") -> List[str]:
        """BFS or DFS traversal."""
        visited = []
        queue = [self.root_joint]
        while queue:
            j = queue.pop(0 if order == "bfs" else -1)
            if j not in visited and j in self.joints:
                visited.append(j)
                queue.extend(self.get_children(j))
        return visited


class Retargeter:
    """Retarget motion from one skeleton to another using joint mapping."""

    COMMON_MAP = {
        "hips": "hips", "spine": "spine", "chest": "chest",
        "neck": "neck", "head": "head",
        "left_shoulder": "left_shoulder", "right_shoulder": "right_shoulder",
        "left_upper_arm": "left_upper_arm", "right_upper_arm": "right_upper_arm",
        "left_lower_arm": "left_lower_arm", "right_lower_arm": "right_lower_arm",
        "left_hand": "left_hand", "right_hand": "right_hand",
        "left_upper_leg": "left_upper_leg", "right_upper_leg": "right_upper_leg",
        "left_lower_leg": "left_lower_leg", "right_lower_leg": "right_lower_leg",
        "left_foot": "left_foot", "right_foot": "right_foot",
        "left_toe": "left_toe", "right_toe": "right_toe",
    }

    @staticmethod
    def retarget(clip: MotionClip, source_skeleton: Skeleton,
                 target_skeleton: Skeleton,
                 joint_map: Optional[Dict[str, str]] = None) -> MotionClip:
        """Retarget a motion clip from source to target skeleton."""
        if joint_map is None:
            joint_map = Retargeter.COMMON_MAP

        # Build mapping of common joints
        src_to_tgt = {}
        for src_j, tgt_j in joint_map.items():
            if src_j in source_skeleton.joints and tgt_j in target_skeleton.joints:
                src_to_tgt[src_j] = tgt_j

        new_clip = MotionClip(
            name=f"{clip.name}_retargeted",
            duration=clip.duration,
            fps=clip.fps,
            loop=clip.loop,
            tags=clip.tags + ["retargeted"],
            joints={n: Joint(name=n) for n in target_skeleton.joints},
        )

        for kf in clip.keyframes:
            nkf = Keyframe(time=kf.time)
            for src_j, tgt_j in src_to_tgt.items():
                if src_j in kf.positions:
                    nkf.positions[tgt_j] = kf.positions[src_j]
                if src_j in kf.rotations:
                    nkf.rotations[tgt_j] = kf.rotations[src_j]
            new_clip.keyframes.append(nkf)

        return new_clip


# ---------------------------------------------------------------------------
# Motion Blending
# ---------------------------------------------------------------------------

class MotionBlender:
    """Blend multiple motion clips together."""

    @staticmethod
    def cross_fade(clip_a: MotionClip, clip_b: MotionClip,
                   fade_duration: float = 0.2) -> MotionClip:
        """Cross-fade from clip_a to clip_b."""
        if not clip_a.keyframes or not clip_b.keyframes:
            raise ValueError("Both clips must have keyframes.")

        fps = max(clip_a.fps, clip_b.fps)
        fade_frames = max(1, int(fade_duration * fps))
        total_duration = clip_a.duration + clip_b.duration
        result = MotionClip(name=f"{clip_a.name}_to_{clip_b.name}",
                            duration=total_duration, fps=fps)

        # First clip full
        for kf in clip_a.keyframes:
            result.keyframes.append(kf)

        # Fade region
        for i in range(1, fade_frames + 1):
            t = i / (fade_frames + 1)
            akf = clip_a.keyframes[-1]
            bkf = clip_b.keyframes[0]
            nkf = Keyframe(time=(clip_a.duration + (i / fps)))
            for jn in set(list(akf.positions.keys()) + list(bkf.positions.keys())):
                if jn in akf.positions and jn in bkf.positions:
                    pa, pb = akf.positions[jn], bkf.positions[jn]
                    nkf.positions[jn] = tuple(
                        a * (1 - t) + b * t for a, b in zip(pa, pb)
                    )
                elif jn in akf.positions:
                    nkf.positions[jn] = akf.positions[jn]

                if jn in akf.rotations and jn in bkf.rotations:
                    ra, rb = akf.rotations[jn], bkf.rotations[jn]
                    nkf.rotations[jn] = _slerp(ra, rb, t)
                elif jn in akf.rotations:
                    nkf.rotations[jn] = akf.rotations[jn]
            result.keyframes.append(nkf)

        # Second clip full
        for kf in clip_b.keyframes:
            nkf = Keyframe(time=kf.time + clip_a.duration)
            nkf.positions = dict(kf.positions)
            nkf.rotations = dict(kf.rotations)
            result.keyframes.append(nkf)

        return result

    @staticmethod
    def additive_blend(base: MotionClip, overlay: MotionClip,
                       weight: float = 0.5) -> MotionClip:
        """Additively blend an overlay motion onto a base."""
        if not base.keyframes:
            return overlay
        result = MotionClip(name=f"{base.name}_add_{overlay.name}",
                            duration=max(base.duration, overlay.duration),
                            fps=base.fps)
        for i, kf in enumerate(base.keyframes):
            nkf = Keyframe(time=kf.time, positions=dict(kf.positions),
                           rotations=dict(kf.rotations))
            okf = overlay.get_frame(min(i, len(overlay.keyframes) - 1))
            if okf:
                for jn in set(list(kf.positions.keys()) + list(okf.positions.keys())):
                    if jn in okf.positions and jn not in kf.positions:
                        nkf.positions[jn] = okf.positions[jn]
                    elif jn in kf.positions and jn in okf.positions:
                        dp = tuple((b - a) * weight for a, b in
                                   zip(kf.positions[jn], okf.positions[jn]))
                        nkf.positions[jn] = tuple(a + d for a, d in
                                                  zip(kf.positions[jn], dp))
            result.keyframes.append(nkf)
        return result


# ---------------------------------------------------------------------------
# Motion Graph
# ---------------------------------------------------------------------------

class MotionGraph:
    """Traversable motion graph for seamless character animation."""

    def __init__(self) -> None:
        self.nodes: Dict[str, MotionGraphNode] = {}
        self.clips: Dict[str, MotionClip] = {}

    def add_clip(self, clip: MotionClip) -> str:
        node = MotionGraphNode(clip_uid=clip.uid, name=clip.name)
        self.nodes[node.uid] = node
        self.clips[clip.uid] = clip
        return node.uid

    def add_transition(self, from_node: str, to_node: str,
                       cost: float = 0.0) -> None:
        if from_node in self.nodes and to_node in self.nodes:
            self.nodes[from_node].transitions.append((to_node, cost))

    def find_path(self, start_node: str, end_node: str) -> List[str]:
        """Simple Dijkstra-based path through the graph."""
        import heapq
        pq = [(0.0, start_node, [])]
        visited = set()
        while pq:
            cost, current, path = heapq.heappop(pq)
            if current in visited:
                continue
            visited.add(current)
            path = path + [current]
            if current == end_node:
                return path
            for neighbor, edge_cost in self.nodes[current].transitions:
                if neighbor not in visited:
                    heapq.heappush(pq, (cost + edge_cost, neighbor, path))
        return []

    def random_walk(self, steps: int = 10) -> List[str]:
        """Random walk through the graph."""
        if not self.nodes:
            return []
        current = random.choice(list(self.nodes.keys()))
        path = [current]
        for _ in range(steps - 1):
            trans = self.nodes[current].transitions
            if not trans:
                break
            current = random.choice([t for t, _ in trans])
            path.append(current)
        return path

    def synthesize(self, path: List[str]) -> Optional[MotionClip]:
        """Synthesize a motion clip from a path through the graph."""
        if not path:
            return None
        clips = [self.clips.get(self.nodes[n].clip_uid) for n in path if n in self.nodes]
        clips = [c for c in clips if c and c.keyframes]
        if not clips:
            return None
        result = clips[0]
        for i in range(1, len(clips)):
            result = MotionBlender.cross_fade(result, clips[i], fade_duration=0.15)
        result.name = "synthesized"
        return result


# ---------------------------------------------------------------------------
# Procedural Foot IK
# ---------------------------------------------------------------------------

class FootIK:
    """Simple two-bone foot IK solver."""

    @staticmethod
    def solve(hip_pos: Tuple[float, float, float],
              knee_pos: Tuple[float, float, float],
              target_pos: Tuple[float, float, float],
              upper_len: float = 0.4, lower_len: float = 0.4) -> Dict[str, Tuple[float, float, float]]:
        """Solve two-bone IK for a leg."""
        import numpy as np

        hip = np.array(hip_pos, dtype=np.float64)
        knee = np.array(knee_pos, dtype=np.float64)
        target = np.array(target_pos, dtype=np.float64)

        # Calculate new knee position
        h_to_t = target - hip
        dist_ht = np.linalg.norm(h_to_t)
        if dist_ht > upper_len + lower_len:
            # Target too far — extend fully
            direction = h_to_t / dist_ht
            new_knee = hip + direction * upper_len
            new_foot = hip + direction * (upper_len + lower_len)
        else:
            # Law of cosines
            cos_theta = (upper_len ** 2 + dist_ht ** 2 - lower_len ** 2) / \
                        (2 * upper_len * dist_ht)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            theta = math.acos(cos_theta)

            htd = h_to_t / dist_ht
            up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            perp = np.cross(htd, up)
            if np.linalg.norm(perp) < 1e-6:
                perp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            perp = perp / np.linalg.norm(perp)

            rotation_axis = np.cross(htd, perp)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

            # Rodrigues rotation
            kx, ky, kz = rotation_axis
            c, s = math.cos(theta), math.sin(theta)
            R = np.array([
                [c + kx * kx * (1 - c), kx * ky * (1 - c) - kz * s, kx * kz * (1 - c) + ky * s],
                [ky * kx * (1 - c) + kz * s, c + ky * ky * (1 - c), ky * kz * (1 - c) - kx * s],
                [kz * kx * (1 - c) - ky * s, kz * ky * (1 - c) + kx * s, c + kz * kz * (1 - c)],
            ])
            rotated_dir = R @ htd
            new_knee = hip + rotated_dir * upper_len

            # Foot position
            k_to_t = target - new_knee
            ktt_len = np.linalg.norm(k_to_t)
            if ktt_len > 1e-6:
                new_foot = new_knee + (k_to_t / ktt_len) * min(ktt_len, lower_len)
            else:
                new_foot = target

        return {
            "knee": tuple(new_knee.tolist()),
            "foot": tuple(new_foot.tolist()),
        }


# ---------------------------------------------------------------------------
# Procedural Motion Generation
# ---------------------------------------------------------------------------

class ProceduralMotion:
    """Generate basic procedural motions (walk, run, idle)."""

    @staticmethod
    def generate_walk(duration: float = 2.0, fps: int = 30,
                      stride: float = 0.6, height: float = 1.7) -> MotionClip:
        """Generate a simple walk cycle."""
        clip = MotionClip(name="walk", duration=duration, fps=fps, loop=True)
        joints = ["hips", "left_foot", "right_foot", "left_knee", "right_knee", "head"]
        for j in joints:
            clip.joints[j] = Joint(name=j)

        for i in range(int(duration * fps)):
            t = i / fps
            phase = (t / duration) * 2 * math.pi
            kf = Keyframe(time=t)

            # Root sway
            kf.positions["hips"] = (
                0.05 * math.sin(phase * 2),
                height + 0.02 * math.sin(phase),
                0.0,
            )
            # Foot positions
            kf.positions["left_foot"] = (
                stride * 0.5 * math.sin(phase),
                0.02 * max(0, math.sin(phase + math.pi)),
                0.1 * math.cos(phase),
            )
            kf.positions["right_foot"] = (
                stride * 0.5 * math.sin(phase + math.pi),
                0.02 * max(0, math.sin(phase)),
                -0.1 * math.cos(phase),
            )
            # Head bob
            kf.positions["head"] = (
                0.0,
                height + 1.0 + 0.01 * math.sin(phase * 2),
                0.0,
            )
            clip.keyframes.append(kf)

        return clip

    @staticmethod
    def generate_idle(duration: float = 3.0, fps: int = 30) -> MotionClip:
        """Generate a subtle idle breathing motion."""
        clip = MotionClip(name="idle", duration=duration, fps=fps, loop=True)
        for j in ["hips", "head", "chest"]:
            clip.joints[j] = Joint(name=j)

        for i in range(int(duration * fps)):
            t = i / fps
            phase = (t / duration) * 2 * math.pi
            kf = Keyframe(time=t)
            breath = 0.005 * math.sin(phase * 2)
            sway = 0.003 * math.sin(phase)
            kf.positions["hips"] = (sway * 0.5, breath, 0.0)
            kf.positions["chest"] = (sway, breath * 2, 0.0)
            kf.positions["head"] = (sway * 0.3, breath * 1.5 + 1.6, 0.0)
            clip.keyframes.append(kf)

        return clip


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slerp(qa: Tuple[float, float, float, float],
           qb: Tuple[float, float, float, float],
           t: float) -> Tuple[float, float, float, float]:
    """Spherical linear interpolation between two quaternions."""
    a = np.array(qa, dtype=np.float64)
    b = np.array(qb, dtype=np.float64)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        result = a + t * (b - a)
        return tuple((result / np.linalg.norm(result)).tolist())
    theta = math.acos(dot)
    result = (math.sin((1 - t) * theta) * a + math.sin(t * theta) * b) / math.sin(theta)
    return tuple(result.tolist())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def create_motion_clip(name: str, duration: float = 2.0,
                              fps: int = 30, loop: bool = False) -> Dict[str, Any]:
    """Create an empty motion clip."""
    clip = MotionClip(name=name, duration=duration, fps=fps, loop=loop)
    return clip.to_dict()


async def blend_motions(clip_a: Dict[str, Any], clip_b: Dict[str, Any],
                        method: str = "cross_fade",
                        fade_duration: float = 0.2,
                        weight: float = 0.5) -> Dict[str, Any]:
    """Blend two motion clips together."""
    ca = MotionClip(**{k: v for k, v in clip_a.items() if k in
                       ("name", "duration", "fps", "keyframes", "tags", "loop")})
    cb = MotionClip(**{k: v for k, v in clip_b.items() if k in
                       ("name", "duration", "fps", "keyframes", "tags", "loop")})

    # Reconstruct keyframes
    ca.keyframes = [Keyframe(**kf) if isinstance(kf, dict) else kf for kf in clip_a.get("keyframes", [])]
    cb.keyframes = [Keyframe(**kf) if isinstance(kf, dict) else kf for kf in clip_b.get("keyframes", [])]

    blender = MotionBlender()
    if method == "cross_fade":
        result = blender.cross_fade(ca, cb, fade_duration)
    elif method == "additive":
        result = blender.additive_blend(ca, cb, weight)
    else:
        raise ValueError(f"Unknown blend method: {method}")
    return result.to_dict()


async def generate_procedural_motion(motion_type: str = "walk",
                                      duration: float = 2.0, fps: int = 30,
                                      **kwargs) -> Dict[str, Any]:
    """Generate a procedural motion (walk, idle)."""
    pm = ProceduralMotion()
    if motion_type == "walk":
        clip = pm.generate_walk(duration, fps, stride=kwargs.get("stride", 0.6))
    elif motion_type == "idle":
        clip = pm.generate_idle(duration, fps)
    else:
        raise ValueError(f"Unknown motion type: {motion_type}")
    return clip.to_dict()


async def solve_foot_ik(hip: Tuple[float, float, float],
                         knee: Tuple[float, float, float],
                         target: Tuple[float, float, float],
                         upper_len: float = 0.4,
                         lower_len: float = 0.4) -> Dict[str, Any]:
    """Solve two-bone IK for a leg."""
    ik = FootIK()
    return ik.solve(hip, knee, target, upper_len, lower_len)


async def retarget_motion(clip_dict: Dict[str, Any],
                           source_joints: Dict[str, Any],
                           target_joints: Dict[str, Any],
                           joint_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Retarget a motion clip from source to target skeleton."""
    clip = MotionClip(name=clip_dict.get("name", "clip"),
                      keyframes=[Keyframe(**kf) if isinstance(kf, dict) else kf
                                 for kf in clip_dict.get("keyframes", [])])
    src_skel = Skeleton(joints={n: Joint(**j) if isinstance(j, dict) else j
                                for n, j in source_joints.items()})
    tgt_skel = Skeleton(joints={n: Joint(**j) if isinstance(j, dict) else j
                                for n, j in target_joints.items()})
    r = Retargeter()
    result = r.retarget(clip, src_skel, tgt_skel, joint_map)
    return result.to_dict()
