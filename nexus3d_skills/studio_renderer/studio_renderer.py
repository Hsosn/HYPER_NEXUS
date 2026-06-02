"""
Studio Renderer — Batch Rendering & Compositing Pipeline
=========================================================
Manages render jobs, batch processing, AOV output, compositing
layers, and post-processing effects for the Nexus3D rendering
pipeline. Designed to work alongside the Product Renderer.
"""

from __future__ import annotations

import asyncio
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
# Types
# ---------------------------------------------------------------------------

class RenderStatus(Enum):
    QUEUED = "queued"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AOVType(Enum):
    BEAUTY = "beauty"
    ALBEDO = "albedo"
    NORMAL = "normal"
    DEPTH = "depth"
    POSITION = "position"
    UV = "uv"
    VELOCITY = "velocity"
    ID_OBJECT = "id_object"
    ID_MATERIAL = "id_material"
    AO = "ao"
    DIFFUSE = "diffuse"
    SPECULAR = "specular"
    TRANSMISSION = "transmission"
    VOLUME = "volume"
    MIST = "mist"


class CompositeMode(Enum):
    OVER = "over"
    ADD = "add"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    DODGE = "dodge"
    BURN = "burn"
    DIFFERENCE = "difference"
    SOFT_LIGHT = "soft_light"
    HARD_LIGHT = "hard_light"


class PostEffect(Enum):
    BLOOM = "bloom"
    VIGNETTE = "vignette"
    COLOR_GRADE = "color_grade"
    LENS_DISTORTION = "lens_distortion"
    CHROMATIC_ABERRATION = "chromatic_aberration"
    FILM_GRAIN = "film_grain"
    SHARPEN = "sharpen"
    DENOISE = "denoise"
    TONE_MAP = "tone_map"
    EXPOSURE = "exposure"
    WHITE_BALANCE = "white_balance"


# ---------------------------------------------------------------------------
# Render Job
# ---------------------------------------------------------------------------

@dataclass
class RenderJob:
    """A single render job in the queue."""
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = "render"
    status: RenderStatus = RenderStatus.QUEUED
    priority: int = 0
    scene_uid: str = ""
    camera_uid: str = ""
    width: int = 1920
    height: int = 1080
    samples: int = 256
    bounces: int = 8
    aovs: List[str] = field(default_factory=lambda: ["beauty"])
    seed: int = 0
    output_path: str = "renders/"
    output_format: str = "png"
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_ms: Optional[float] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


# ---------------------------------------------------------------------------
# Render Queue
# ---------------------------------------------------------------------------

class RenderQueue:
    """Manages a queue of render jobs with priority support."""

    def __init__(self, max_concurrent: int = 2) -> None:
        self.jobs: Dict[str, RenderJob] = {}
        self.queue: List[str] = []  # ordered UIDs by priority
        self.active: List[str] = []
        self.completed: List[str] = []
        self.max_concurrent = max_concurrent
        self._lock = asyncio.Lock()

    async def add_job(self, job: RenderJob) -> str:
        async with self._lock:
            if job.uid in self.jobs:
                raise ValueError(f"Job {job.uid} already exists.")
            job.created_at = time.time()
            self.jobs[job.uid] = job
            self.queue.append(job.uid)
            # Sort by priority (higher = first)
            self.queue.sort(key=lambda uid: self.jobs[uid].priority, reverse=True)
        return job.uid

    async def get_next(self) -> Optional[RenderJob]:
        async with self._lock:
            if len(self.active) >= self.max_concurrent:
                return None
            while self.queue:
                uid = self.queue.pop(0)
                if uid in self.jobs and self.jobs[uid].status == RenderStatus.QUEUED:
                    self.active.append(uid)
                    job = self.jobs[uid]
                    job.status = RenderStatus.RENDERING
                    job.started_at = time.time()
                    return job
        return None

    async def complete_job(self, uid: str, success: bool = True) -> None:
        async with self._lock:
            if uid not in self.jobs:
                return
            job = self.jobs[uid]
            job.status = RenderStatus.COMPLETED if success else RenderStatus.FAILED
            job.completed_at = time.time()
            if job.started_at:
                job.duration_ms = (job.completed_at - job.started_at) * 1000
            if uid in self.active:
                self.active.remove(uid)
            self.completed.append(uid)

    async def update_progress(self, uid: str, progress: float) -> None:
        async with self._lock:
            if uid in self.jobs:
                self.jobs[uid].progress = max(0.0, min(1.0, progress))

    async def cancel_job(self, uid: str) -> bool:
        async with self._lock:
            if uid not in self.jobs:
                return False
            job = self.jobs[uid]
            if job.status in (RenderStatus.COMPLETED, RenderStatus.FAILED):
                return False
            job.status = RenderStatus.CANCELLED
            if uid in self.active:
                self.active.remove(uid)
            if uid in self.queue:
                self.queue.remove(uid)
            return True

    async def get_status(self, uid: str) -> Optional[RenderStatus]:
        job = self.jobs.get(uid)
        return job.status if job else None

    async def get_queue_state(self) -> Dict[str, Any]:
        async with self._lock:
            return {
                "queued": len(self.queue),
                "active": len(self.active),
                "completed": len(self.completed),
                "total": len(self.jobs),
                "max_concurrent": self.max_concurrent,
            }

    async def list_jobs(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        jobs = []
        for job in self.jobs.values():
            if status_filter and job.status.value != status_filter:
                continue
            jobs.append(job.to_dict())
        return sorted(jobs, key=lambda j: j.get("created_at", 0), reverse=True)


# ---------------------------------------------------------------------------
# AOV (Arbitrary Output Variable) Manager
# ---------------------------------------------------------------------------

class AOVManager:
    """Manage render AOVs and their configurations."""

    AOV_DEFAULTS = {
        AOVType.BEAUTY: {"channels": ["rgb"], "depth": 8, "denoise": True},
        AOVType.ALBEDO: {"channels": ["rgb"], "depth": 16, "denoise": False},
        AOVType.NORMAL: {"channels": ["rgb"], "depth": 16, "denoise": False},
        AOVType.DEPTH: {"channels": ["r"], "depth": 32, "denoise": False},
        AOVType.POSITION: {"channels": ["rgb"], "depth": 32, "denoise": False},
        AOVType.AO: {"channels": ["r"], "depth": 8, "denoise": True},
        AOVType.DIFFUSE: {"channels": ["rgb"], "depth": 16, "denoise": True},
        AOVType.SPECULAR: {"channels": ["rgb"], "depth": 16, "denoise": True},
        AOVType.VELOCITY: {"channels": ["rg"], "depth": 16, "denoise": False},
    }

    @classmethod
    def get_aov_config(cls, aov_name: str) -> Optional[Dict[str, Any]]:
        try:
            aov = AOVType(aov_name)
            return cls.AOV_DEFAULTS.get(aov)
        except ValueError:
            return None

    @classmethod
    def list_available_aovs(cls) -> List[str]:
        return [a.value for a in AOVType]

    @classmethod
    def generate_aov_output_paths(cls, job: RenderJob, base_path: str = "") -> Dict[str, str]:
        """Generate output paths for each AOV in a job."""
        paths = {}
        for aov_name in job.aovs:
            ext = "exr" if aov_name in ("depth", "position", "velocity") else job.output_format
            filename = f"{job.name}_{aov_name}.{ext}"
            path = str(Path(base_path or job.output_path) / filename)
            paths[aov_name] = path
        return paths


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

@dataclass
class CompositeLayer:
    """A single compositing layer."""
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = "layer"
    aov_source: str = "beauty"
    blend_mode: CompositeMode = CompositeMode.OVER
    opacity: float = 1.0
    color_overlay: Optional[Tuple[float, float, float]] = None
    mask_aov: Optional[str] = None
    visible: bool = True
    order: int = 0


class Compositor:
    """Compositing engine for combining AOVs into final images."""

    @staticmethod
    def blend_pixel(base: Tuple[float, float, float],
                    overlay: Tuple[float, float, float],
                    mode: CompositeMode) -> Tuple[float, float, float]:
        """Blend two pixels using the specified mode."""
        b = list(base)
        o = list(overlay)

        if mode == CompositeMode.OVER:
            return tuple(o[i] * overlay_opacity + b[i] * (1 - overlay_opacity)
                        for i in range(3))

        elif mode == CompositeMode.ADD:
            return tuple(min(1.0, b[i] + o[i]) for i in range(3))

        elif mode == CompositeMode.MULTIPLY:
            return tuple(b[i] * o[i] for i in range(3))

        elif mode == CompositeMode.SCREEN:
            return tuple(1.0 - (1.0 - b[i]) * (1.0 - o[i]) for i in range(3))

        elif mode == CompositeMode.DODGE:
            return tuple(min(1.0, b[i] / max(0.001, 1.0 - o[i])) for i in range(3))

        elif mode == CompositeMode.BURN:
            return tuple(1.0 - min(1.0, (1.0 - b[i]) / max(0.001, o[i])) for i in range(3))

        elif mode == CompositeMode.DIFFERENCE:
            return tuple(abs(b[i] - o[i]) for i in range(3))

        elif mode == CompositeMode.SOFT_LIGHT:
            return tuple(
                (2 * b[i] * o[i] + b[i] * b[i] * (1 - 2 * o[i])) if o[i] < 0.5
                else (2 * b[i] * (1 - o[i]) + math.sqrt(b[i]) * (2 * o[i] - 1))
                for i in range(3)
            )

        elif mode == CompositeMode.HARD_LIGHT:
            return tuple(
                (2 * b[i] * o[i]) if o[i] < 0.5
                else (1 - 2 * (1 - b[i]) * (1 - o[i]))
                for i in range(3)
            )

        return tuple(o)

    @staticmethod
    def apply_color_overlay(color: Tuple[float, float, float],
                            opacity: float) -> Dict[str, Any]:
        return {"color": color, "opacity": opacity}

    @staticmethod
    def create_comp(node: ...) -> ...:
        return node


# ---------------------------------------------------------------------------
# Post-Processing
# ---------------------------------------------------------------------------

class PostProcessor:
    """Apply post-processing effects to rendered images."""

    @staticmethod
    def apply_bloom(intensity: float = 0.5,
                    radius: float = 0.05,
                    threshold: float = 0.8) -> Dict[str, Any]:
        """Configure bloom effect."""
        return {
            "effect": "bloom",
            "intensity": intensity,
            "radius": radius,
            "threshold": threshold,
        }

    @staticmethod
    def apply_vignette(intensity: float = 0.3,
                       falloff: float = 1.5) -> Dict[str, Any]:
        """Configure vignette effect."""
        return {
            "effect": "vignette",
            "intensity": intensity,
            "falloff": falloff,
        }

    @staticmethod
    def apply_color_grade(exposure: float = 0.0,
                          contrast: float = 1.0,
                          saturation: float = 1.0,
                          gamma: float = 1.0,
                          shadows: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                          midtones: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                          highlights: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Dict[str, Any]:
        """Configure color grading."""
        return {
            "effect": "color_grade",
            "exposure": exposure,
            "contrast": contrast,
            "saturation": saturation,
            "gamma": gamma,
            "shadows": shadows,
            "midtones": midtones,
            "highlights": highlights,
        }

    @staticmethod
    def apply_chromatic_aberration(strength: float = 0.01) -> Dict[str, Any]:
        """Configure chromatic aberration."""
        return {
            "effect": "chromatic_aberration",
            "strength": strength,
        }

    @staticmethod
    def apply_film_grain(intensity: float = 0.05,
                         size: float = 1.0) -> Dict[str, Any]:
        """Configure film grain."""
        return {
            "effect": "film_grain",
            "intensity": intensity,
            "size": size,
        }

    @staticmethod
    def apply_sharpen(amount: float = 0.5,
                      radius: float = 1.0) -> Dict[str, Any]:
        """Configure sharpen effect."""
        return {
            "effect": "sharpen",
            "amount": amount,
            "radius": radius,
        }

    @staticmethod
    def build_pipeline(effects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build a post-processing pipeline from effects list."""
        return effects


# ---------------------------------------------------------------------------
# Batch Render Manager
# ---------------------------------------------------------------------------

class BatchRenderer:
    """Coordinate batch rendering across multiple scenes/jobs."""

    def __init__(self) -> None:
        self.queue = RenderQueue()
        self.batches: Dict[str, List[str]] = {}  # batch_uid -> [job_uids]
        self.render_results: Dict[str, Dict[str, Any]] = {}

    async def create_batch(self, jobs: List[RenderJob],
                            name: str = "batch") -> str:
        """Create a batch of render jobs."""
        batch_uid = uuid.uuid4().hex[:12]
        job_uids = []
        for job in jobs:
            uid = await self.queue.add_job(job)
            job_uids.append(uid)
        self.batches[batch_uid] = job_uids
        return batch_uid

    async def process_batch(self, batch_uid: str) -> Dict[str, Any]:
        """Process all jobs in a batch."""
        job_uids = self.batches.get(batch_uid, [])
        results = {}
        for uid in job_uids:
            job = await self.queue.get_next()
            if job:
                # Simulate render
                await asyncio.sleep(random.uniform(0.1, 0.5))
                success = random.random() > 0.05  # 95% success
                await self.queue.complete_job(uid, success)
                final_job = self.queue.jobs.get(uid)
                results[uid] = {
                    "status": final_job.status.value if final_job else "unknown",
                    "duration_ms": final_job.duration_ms if final_job else 0,
                }
        return {
            "batch_uid": batch_uid,
            "num_jobs": len(job_uids),
            "results": results,
        }

    async def get_batch_status(self, batch_uid: str) -> Dict[str, Any]:
        job_uids = self.batches.get(batch_uid, [])
        statuses = {}
        for uid in job_uids:
            s = await self.queue.get_status(uid)
            statuses[uid] = s.value if s else "unknown"
        return {
            "batch_uid": batch_uid,
            "jobs": statuses,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def create_render_job(name: str = "render",
                             width: int = 1920,
                             height: int = 1080,
                             samples: int = 256,
                             bounces: int = 8,
                             aovs: Optional[List[str]] = None,
                             priority: int = 0,
                             output_format: str = "png") -> Dict[str, Any]:
    """Create a render job configuration."""
    job = RenderJob(
        name=name,
        width=width,
        height=height,
        samples=samples,
        bounces=bounces,
        aovs=aovs or ["beauty"],
        priority=priority,
        output_format=output_format,
    )
    return job.to_dict()


async def list_aov_types() -> List[str]:
    """List all available AOV types."""
    return AOVManager.list_available_aovs()


async def get_aov_config(aov_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a specific AOV."""
    return AOVManager.get_aov_config(aov_name)


async def generate_aov_paths(job_data: Dict[str, Any]) -> Dict[str, str]:
    """Generate output paths for each AOV in a job."""
    job = RenderJob(**{k: v for k, v in job_data.items()
                       if k in RenderJob.__dataclass_fields__})
    return AOVManager.generate_aov_output_paths(job)


async def build_post_pipeline(effects: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Build a post-processing pipeline with configured effects."""
    pp = PostProcessor()
    pipeline = []
    effect_builders = {
        "bloom": lambda: pp.apply_bloom(),
        "vignette": lambda: pp.apply_vignette(),
        "color_grade": lambda: pp.apply_color_grade(),
        "chromatic_aberration": lambda: pp.apply_chromatic_aberration(),
        "film_grain": lambda: pp.apply_film_grain(),
        "sharpen": lambda: pp.apply_sharpen(),
        "denoise": lambda: {"effect": "denoise", "method": "optix"},
        "tone_map": lambda: {"effect": "tone_map", "method": "filmic", "exposure": 1.0},
        "exposure": lambda: {"effect": "exposure", "value": 0.0},
        "white_balance": lambda: {"effect": "white_balance", "temperature": 6500},
    }
    for ef in (effects or []):
        builder = effect_builders.get(ef)
        if builder:
            pipeline.append(builder())
    return pipeline


async def create_render_layer(name: str = "layer",
                               aov_source: str = "beauty",
                               blend_mode: str = "over",
                               opacity: float = 1.0) -> Dict[str, Any]:
    """Create a compositing layer."""
    try:
        mode = CompositeMode(blend_mode)
    except ValueError:
        mode = CompositeMode.OVER
    layer = CompositeLayer(name=name, aov_source=aov_source,
                           blend_mode=mode, opacity=opacity)
    return asdict(layer)


async def estimate_batch_time(num_frames: int = 30,
                               samples_per_frame: int = 256,
                               estimated_seconds_per_sample: float = 0.01) -> Dict[str, Any]:
    """Estimate batch render time."""
    seconds_per_frame = samples_per_frame * estimated_seconds_per_sample
    total_seconds = num_frames * seconds_per_frame
    return {
        "num_frames": num_frames,
        "samples_per_frame": samples_per_frame,
        "estimated_seconds": total_seconds,
        "estimated_minutes": total_seconds / 60,
        "estimated_hours": total_seconds / 3600,
    }
