"""Nexus3D Animation System — curves, clips, mixing, procedural animation, and export.

Provides the complete animation pipeline for the headless 3D engine:

* :class:`AnimCurve` — single time-to-value animation curve with multiple
  interpolation modes and extrapolation.
* :class:`AnimTrack` — named group of curves driving a single property on a
  target (e.g. the X/Y/Z components of a position).
* :class:`AnimationClip` — a complete animation composed of multiple tracks.
* :class:`AnimMixer` — blends and cross-fades multiple clips in real time.
* :class:`PoseLibrary` — stores static poses and generates smooth transitions.
* :class:`ProceduralAnimation` — stateless generators for common procedural
  motions (walk, idle, breathing, spring-damper, …).
* :class:`AnimationExporter` — serialise clips to glTF, BVH, FBX, or video.

Quaternion convention throughout is ``[w, x, y, z]``.
"""

from __future__ import annotations

import bisect
import copy
import json
import math
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from nexus3d.math3d.core import (
    bezier_cubic,
    catmull_rom_spline,
    lerp,
    normalize_quat,
    quat_from_axis_angle,
    quat_identity,
    quat_multiply,
    quat_slerp,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _ensure_value(value: Any) -> Union[float, np.ndarray]:
    """Coerce *value* to a numpy array (or leave scalars alone) so that
    arithmetic works uniformly inside curve evaluation."""
    if isinstance(value, np.ndarray):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.asarray(value, dtype=np.float64)


def _catmull_rom_point(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    t: float,
) -> np.ndarray:
    """Evaluate a Catmull-Rom segment at parameter *t* ∈ [0, 1].

    The segment passes through *p1* (t = 0) and *p2* (t = 1).
    """
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


# ─── AnimCurve ────────────────────────────────────────────────────────────────


class AnimCurve:
    """A single animation curve that maps time → value.

    Keyframes are stored as an internally-sorted list of ``(time, value)``
    tuples.  Each keyframe may optionally carry *tangent_in* and *tangent_out*
    handles used by the cubic Bezier interpolation mode.

    Parameters
    ----------
    name : str
        Human-readable name for the curve.
    interpolation : str
        Default interpolation mode.  One of ``'linear'``, ``'step'``,
        ``'bezier'``, ``'catmull_rom'``.
    pre_extrapolation : str
        Behaviour before the first keyframe.  One of ``'constant'``,
        ``'repeat'``, ``'repeat_offset'``, ``'bounce'``.
    post_extrapolation : str
        Behaviour after the last keyframe (same options as
        *pre_extrapolation*).
    """

    _INTERP_MODES = frozenset({"linear", "step", "bezier", "catmull_rom"})
    _EXTRAP_MODES = frozenset({"constant", "repeat", "repeat_offset", "bounce"})

    def __init__(
        self,
        name: str = "Curve",
        interpolation: str = "linear",
        pre_extrapolation: str = "constant",
        post_extrapolation: str = "constant",
    ) -> None:
        if interpolation not in self._INTERP_MODES:
            raise ValueError(
                f"Invalid interpolation mode {interpolation!r}; "
                f"choose from {sorted(self._INTERP_MODES)}"
            )
        if pre_extrapolation not in self._EXTRAP_MODES:
            raise ValueError(
                f"Invalid pre_extrapolation mode {pre_extrapolation!r}"
            )
        if post_extrapolation not in self._EXTRAP_MODES:
            raise ValueError(
                f"Invalid post_extrapolation mode {post_extrapolation!r}"
            )

        self.name: str = name
        self.interpolation: str = interpolation
        self.pre_extrapolation: str = pre_extrapolation
        self.post_extrapolation: str = post_extrapolation

        # (time, value, tangent_in, tangent_out)
        self._keyframes: List[Tuple[float, Any, Optional[float], Optional[float]]] = []

    # ------------------------------------------------------------------
    # Keyframe management
    # ------------------------------------------------------------------

    def add_keyframe(
        self,
        time: float,
        value: Any,
        tangent_in: Optional[float] = None,
        tangent_out: Optional[float] = None,
    ) -> None:
        """Add a keyframe at *time* with *value*.

        If a keyframe already exists at *time* it is replaced.  Tangents are
        only used when ``interpolation == 'bezier'``.

        Parameters
        ----------
        time : float
            Keyframe time in seconds.
        value : float | np.ndarray
            The value at this keyframe.
        tangent_in : float | None
            Incoming tangent (derivative at keyframe from the left).
        tangent_out : float | None
            Outgoing tangent (derivative at keyframe to the right).
        """
        # Coerce value to numpy for consistent arithmetic downstream.
        value = _ensure_value(value)

        # Insert in sorted order, replacing existing key at same time.
        idx = bisect.bisect_left(self._times, time)
        if idx < len(self._keyframes) and abs(self._keyframes[idx][0] - time) < 1e-9:
            self._keyframes[idx] = (time, value, tangent_in, tangent_out)
        else:
            self._keyframes.insert(idx, (time, value, tangent_in, tangent_out))

    def remove_keyframe(self, time: float) -> bool:
        """Remove the keyframe closest to *time* (within 1e-9 tolerance).

        Returns ``True`` if a keyframe was removed, ``False`` otherwise.
        """
        idx = bisect.bisect_left(self._times, time)
        if idx < len(self._keyframes) and abs(self._keyframes[idx][0] - time) < 1e-9:
            self._keyframes.pop(idx)
            return True
        return False

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, time: float) -> Any:
        """Evaluate the curve at *time*, handling extrapolation.

        Returns the interpolated (or extrapolated) value.  For curves with
        array values the return type is ``np.ndarray``; for scalar curves a
        ``float`` is returned.

        Parameters
        ----------
        time : float
            Evaluation time in seconds.

        Returns
        -------
        float | np.ndarray
        """
        n = len(self._keyframes)
        if n == 0:
            return 0.0
        if n == 1:
            return self._keyframes[0][1]

        t_start = self._keyframes[0][0]
        t_end = self._keyframes[-1][0]
        duration = t_end - t_start

        if duration < 1e-12:
            return self._keyframes[0][1]

        # Apply pre-extrapolation
        if time < t_start:
            if self.pre_extrapolation == "constant":
                return self._keyframes[0][1]
            elif self.pre_extrapolation == "repeat":
                time = t_start + ((time - t_start) % duration)
            elif self.pre_extrapolation == "repeat_offset":
                n_repeats = math.ceil((t_start - time) / duration)
                time = time + n_repeats * duration
            elif self.pre_extrapolation == "bounce":
                t_local = t_start - time
                n_bounces = int(t_local / duration)
                t_remain = t_local - n_bounces * duration
                if n_bounces % 2 == 0:
                    time = t_end - t_remain
                else:
                    time = t_start + t_remain
        elif time > t_end:
            if self.post_extrapolation == "constant":
                return self._keyframes[-1][1]
            elif self.post_extrapolation == "repeat":
                time = t_start + ((time - t_start) % duration)
            elif self.post_extrapolation == "repeat_offset":
                n_repeats = int((time - t_end) / duration)
                time = t_start + (time - t_start - n_repeats * duration)
            elif self.post_extrapolation == "bounce":
                t_local = time - t_end
                n_bounces = int(t_local / duration)
                t_remain = t_local - n_bounces * duration
                if n_bounces % 2 == 0:
                    time = t_end - t_remain
                else:
                    time = t_start + t_remain

        # Clamp to valid range after extrapolation
        time = np.clip(time, t_start, t_end)

        # Find surrounding keyframes
        idx = bisect.bisect_right(self._times, time) - 1
        idx = max(0, min(idx, n - 2))

        t0, v0, ti0, to0 = self._keyframes[idx]
        t1, v1, ti1, to1 = self._keyframes[idx + 1]

        seg_duration = t1 - t0
        if seg_duration < 1e-12:
            return v0

        t = (time - t0) / seg_duration  # normalised [0, 1]

        if self.interpolation == "linear":
            return self._evaluate_linear(t0, v0, t1, v1, t)
        elif self.interpolation == "step":
            return self._evaluate_step(v0, t)
        elif self.interpolation == "bezier":
            return self._evaluate_bezier(t0, v0, t1, v1, t, ti0, to1)
        elif self.interpolation == "catmull_rom":
            return self._evaluate_catmull_rom(self._times, self._values, idx, t)
        else:
            return self._evaluate_linear(t0, v0, t1, v1, t)

    # ------------------------------------------------------------------
    # Interpolation helpers
    # ------------------------------------------------------------------

    def _evaluate_linear(
        self, t0: float, v0: Any, t1: float, v1: Any, t: float
    ) -> Any:
        """Standard linear interpolation between two keyframe values."""
        return lerp(
            np.asarray(v0, dtype=np.float64),
            np.asarray(v1, dtype=np.float64),
            t,
        )

    def _evaluate_step(self, v0: Any, t: float) -> Any:
        """Step (hold) interpolation — returns the left keyframe value."""
        return v0

    def _evaluate_bezier(
        self,
        t0: float,
        v0: Any,
        t1: float,
        v1: Any,
        t: float,
        tan_in: Optional[float],
        tan_out: Optional[float],
    ) -> Any:
        """Cubic Bezier interpolation using tangent handles.

        Tangents define the control points:

        * P0 = (0, v0)
        * P1 = (1/3, v0 + tan_out * dt / 3)
        * P2 = (2/3, v1 - tan_in * dt / 3)
        * P3 = (1, v1)

        where *dt* = ``t1 - t0``.
        """
        dt = t1 - t0
        if dt < 1e-12:
            return v0

        v0 = np.asarray(v0, dtype=np.float64)
        v1 = np.asarray(v1, dtype=np.float64)

        # Default tangent = 0 if not provided
        _tan_out = tan_out if tan_out is not None else 0.0
        _tan_in = tan_in if tan_in is not None else 0.0

        p0 = v0
        p1 = v0 + np.asarray(_tan_out, dtype=np.float64) * (dt / 3.0)
        p2 = v1 - np.asarray(_tan_in, dtype=np.float64) * (dt / 3.0)
        p3 = v1

        return bezier_cubic(p0, p1, p2, p3, t)

    def _evaluate_catmull_rom(
        self,
        times: List[float],
        values: List[Any],
        idx: int,
        t: float,
    ) -> Any:
        """Catmull-Rom interpolation using the four surrounding points.

        The value at parameter *t* ∈ [0, 1] between keyframes *idx* and
        *idx + 1* is computed from keyframes *idx − 1*, *idx*, *idx + 1*,
        and *idx + 2*.  Boundary points are duplicated when the required
        neighbours are out of range.
        """
        n = len(values)

        # Gather four surrounding points with mirroring at boundaries.
        def _safe(i: int) -> np.ndarray:
            i = max(0, min(i, n - 1))
            return np.asarray(values[i], dtype=np.float64)

        p0 = _safe(idx - 1)
        p1 = _safe(idx)
        p2 = _safe(idx + 1)
        p3 = _safe(idx + 2)

        return _catmull_rom_point(p0, p1, p2, p3, t)

    # ------------------------------------------------------------------
    # Properties & utilities
    # ------------------------------------------------------------------

    @property
    def _times(self) -> List[float]:
        return [kf[0] for kf in self._keyframes]

    @property
    def _values(self) -> List[Any]:
        return [kf[1] for kf in self._keyframes]

    @property
    def keyframe_count(self) -> int:
        """Return the number of keyframes."""
        return len(self._keyframes)

    def get_length(self) -> float:
        """Return the time span of the curve (last keyframe − first).

        Returns ``0.0`` if the curve has fewer than two keyframes.
        """
        if len(self._keyframes) < 2:
            return 0.0
        return self._keyframes[-1][0] - self._keyframes[0][0]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the curve to a JSON-friendly dictionary."""
        kfs = []
        for t, v, ti, to_ in self._keyframes:
            kf: Dict[str, Any] = {
                "time": float(t),
                "value": v.tolist() if isinstance(v, np.ndarray) else float(v),
            }
            if ti is not None:
                kf["tangent_in"] = float(ti)
            if to_ is not None:
                kf["tangent_out"] = float(to_)
            kfs.append(kf)
        return {
            "name": self.name,
            "interpolation": self.interpolation,
            "pre_extrapolation": self.pre_extrapolation,
            "post_extrapolation": self.post_extrapolation,
            "keyframes": kfs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnimCurve":
        """Deserialise a curve from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary produced by :meth:`to_dict`.
        """
        curve = cls(
            name=data.get("name", "Curve"),
            interpolation=data.get("interpolation", "linear"),
            pre_extrapolation=data.get("pre_extrapolation", "constant"),
            post_extrapolation=data.get("post_extrapolation", "constant"),
        )
        for kf in data.get("keyframes", []):
            val = kf["value"]
            if isinstance(val, list):
                val = np.array(val, dtype=np.float64)
            curve.add_keyframe(
                time=float(kf["time"]),
                value=val,
                tangent_in=kf.get("tangent_in"),
                tangent_out=kf.get("tangent_out"),
            )
        return curve

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AnimCurve(name={self.name!r}, "
            f"keyframes={self.keyframe_count}, "
            f"interpolation={self.interpolation!r})"
        )


# ─── AnimTrack ────────────────────────────────────────────────────────────────


class AnimTrack:
    """A named track containing multiple curves (e.g. for X, Y, Z).

    Each channel maps to a single :class:`AnimCurve`.  A track is associated
    with a *target* object (identified by *target_name*) and a *property_path*
    (e.g. ``'position'`` or ``'rotation'``).

    Parameters
    ----------
    name : str
        Track identifier.
    target_name : str
        Name of the animated object.
    property_path : str
        Property being animated (e.g. ``'position'``, ``'scale'``).
    blend_mode : str
        How this track combines with others during mixing.  One of
        ``'replace'``, ``'add'``, ``'multiply'``.
    """

    _BLEND_MODES = frozenset({"replace", "add", "multiply"})

    def __init__(
        self,
        name: str = "Track",
        target_name: str = "",
        property_path: str = "",
        blend_mode: str = "replace",
    ) -> None:
        if blend_mode not in self._BLEND_MODES:
            raise ValueError(
                f"Invalid blend_mode {blend_mode!r}; "
                f"choose from {sorted(self._BLEND_MODES)}"
            )
        self.name: str = name
        self.target_name: str = target_name
        self.property_path: str = property_path
        self.blend_mode: str = blend_mode
        self.curves: Dict[str, AnimCurve] = {}
        self.enabled: bool = True
        self.weight: float = 1.0

    # ------------------------------------------------------------------
    # Curve management
    # ------------------------------------------------------------------

    def set_curve(self, channel: str, curve: AnimCurve) -> None:
        """Attach an :class:`AnimCurve` to a named *channel*.

        Parameters
        ----------
        channel : str
            Channel name (e.g. ``'x'``, ``'y'``, ``'z'``).
        curve : AnimCurve
            The curve to bind.
        """
        self.curves[channel] = curve

    def get_curve(self, channel: str) -> Optional[AnimCurve]:
        """Return the curve bound to *channel*, or ``None``."""
        return self.curves.get(channel)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, time: float) -> Dict[str, Any]:
        """Evaluate all channels at *time*.

        Returns
        -------
        dict[str, float | np.ndarray]
            Mapping of channel name to evaluated value.
        """
        if not self.enabled or self.weight < 1e-10:
            return {}
        return {ch: curve.evaluate(time) for ch, curve in self.curves.items()}

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the track to a JSON-friendly dictionary."""
        return {
            "name": self.name,
            "target_name": self.target_name,
            "property_path": self.property_path,
            "blend_mode": self.blend_mode,
            "enabled": self.enabled,
            "weight": self.weight,
            "curves": {ch: curve.to_dict() for ch, curve in self.curves.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnimTrack":
        """Deserialise a track from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary produced by :meth:`to_dict`.
        """
        track = cls(
            name=data.get("name", "Track"),
            target_name=data.get("target_name", ""),
            property_path=data.get("property_path", ""),
            blend_mode=data.get("blend_mode", "replace"),
        )
        track.enabled = bool(data.get("enabled", True))
        track.weight = float(data.get("weight", 1.0))
        for ch, curve_data in data.get("curves", {}).items():
            track.set_curve(ch, AnimCurve.from_dict(curve_data))
        return track

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AnimTrack(name={self.name!r}, target={self.target_name!r}, "
            f"channels={sorted(self.curves)}, weight={self.weight})"
        )


# ─── AnimationClip ────────────────────────────────────────────────────────────


class AnimationClip:
    """A complete animation clip with multiple tracks.

    A clip represents a self-contained animation asset — for example a
    ``'Walk'`` clip, a ``'Run'`` clip, or a ``'Jump'`` clip.  Each track
    inside the clip animates a specific property on a specific object.

    Parameters
    ----------
    name : str
        Human-readable clip name.
    frame_rate : float
        Playback frame rate in frames per second (default 30).
    loop : bool
        Whether the clip loops when it reaches the end.
    """

    def __init__(
        self,
        name: str = "Animation",
        frame_rate: float = 30.0,
        loop: bool = False,
    ) -> None:
        self.name: str = name
        self.tracks: Dict[str, AnimTrack] = {}
        self.frame_rate: float = frame_rate
        self.loop: bool = loop
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    # ------------------------------------------------------------------
    # Track management
    # ------------------------------------------------------------------

    def add_track(self, track: AnimTrack) -> None:
        """Add an :class:`AnimTrack` to the clip.

        If a track with the same name already exists it is replaced.

        Parameters
        ----------
        track : AnimTrack
        """
        self.tracks[track.name] = track
        self._update_duration()

    def remove_track(self, name: str) -> bool:
        """Remove a track by name.

        Returns ``True`` if the track was found and removed.
        """
        if name in self.tracks:
            del self.tracks[name]
            self._update_duration()
            return True
        return False

    def get_track(self, name: str) -> Optional[AnimTrack]:
        """Return the track named *name*, or ``None``."""
        return self.tracks.get(name)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, time: float) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Evaluate all tracks at *time*.

        Returns a nested dictionary:

        ``{target_name: {property_path: {channel: value}}}``

        Parameters
        ----------
        time : float
            Evaluation time in seconds.

        Returns
        -------
        dict
        """
        # Handle looping
        clip_len = self.get_length()
        if clip_len > 0 and self.loop:
            time = self.start_time + ((time - self.start_time) % clip_len)

        result: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for track in self.tracks.values():
            channels = track.evaluate(time)
            if not channels:
                continue
            tgt = track.target_name or track.name
            prop = track.property_path or "value"
            result.setdefault(tgt, {}).setdefault(prop, {}).update(channels)
        return result

    def get_length(self) -> float:
        """Return the clip duration in seconds.

        This is ``end_time − start_time``.
        """
        return max(0.0, self.end_time - self.start_time)

    def _update_duration(self) -> None:
        """Recalculate *start_time* and *end_time* from all curves."""
        if not self.tracks:
            self.start_time = 0.0
            self.end_time = 0.0
            return

        all_starts: List[float] = []
        all_ends: List[float] = []
        for track in self.tracks.values():
            for curve in track.curves.values():
                if curve._times:
                    all_starts.append(curve._times[0])
                    all_ends.append(curve._times[-1])

        if all_starts:
            self.start_time = min(all_starts)
            self.end_time = max(all_ends)
        else:
            self.start_time = 0.0
            self.end_time = 0.0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the clip to a JSON-friendly dictionary."""
        return {
            "name": self.name,
            "frame_rate": self.frame_rate,
            "loop": self.loop,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "tracks": {n: t.to_dict() for n, t in self.tracks.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnimationClip":
        """Deserialise a clip from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary produced by :meth:`to_dict`.
        """
        clip = cls(
            name=data.get("name", "Animation"),
            frame_rate=float(data.get("frame_rate", 30.0)),
            loop=bool(data.get("loop", False)),
        )
        clip.start_time = float(data.get("start_time", 0.0))
        clip.end_time = float(data.get("end_time", 0.0))
        for name, track_data in data.get("tracks", {}).items():
            clip.add_track(AnimTrack.from_dict(track_data))
        return clip

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AnimationClip(name={self.name!r}, "
            f"tracks={len(self.tracks)}, "
            f"duration={self.get_length():.3f}s)"
        )


# ─── AnimMixer ────────────────────────────────────────────────────────────────


class AnimMixer:
    """Mixes / blends multiple animation clips together.

    Each *layer* is a tuple ``(clip_name, weight, offset_time, blend_mode)``.
    Layers are evaluated at *global_time* and blended from lowest-weight to
    highest so that the layer with the highest weight dominates.

    Parameters
    ----------
    clips : dict or None
        Initial set of clips (``{name: AnimationClip}``).
    """

    def __init__(
        self,
        clips: Optional[Dict[str, AnimationClip]] = None,
    ) -> None:
        self.clips: Dict[str, AnimationClip] = dict(clips or {})
        # Active layers: list of (clip_name, weight, offset_time, blend_mode)
        self.active_layers: List[Tuple[str, float, float, str]] = []
        # Pending crossfades: list of (from_clip, to_clip, start_time, duration)
        self._crossfades: List[Tuple[str, str, float, float]] = []

    # ------------------------------------------------------------------
    # Clip management
    # ------------------------------------------------------------------

    def add_clip(self, clip: AnimationClip) -> None:
        """Register a clip with the mixer.

        Parameters
        ----------
        clip : AnimationClip
        """
        self.clips[clip.name] = clip

    def play(
        self,
        clip_name: str,
        weight: float = 1.0,
        offset: float = 0.0,
        blend_mode: str = "replace",
    ) -> None:
        """Activate a clip for playback.

        Parameters
        ----------
        clip_name : str
            Name of the registered clip.
        weight : float
            Blending weight (0–1).
        offset : float
            Time offset in seconds.
        blend_mode : str
            One of ``'replace'``, ``'add'``, ``'multiply'``.
        """
        if clip_name not in self.clips:
            raise KeyError(f"Clip {clip_name!r} is not registered with the mixer.")
        # Remove existing entry for this clip to avoid duplicates.
        self.active_layers = [
            layer for layer in self.active_layers if layer[0] != clip_name
        ]
        self.active_layers.append((clip_name, weight, offset, blend_mode))

    def stop(self, clip_name: str) -> None:
        """Deactivate a clip."""
        self.active_layers = [
            layer for layer in self.active_layers if layer[0] != clip_name
        ]

    def crossfade(
        self,
        from_clip: str,
        to_clip: str,
        duration: float = 0.5,
        start_time: Optional[float] = None,
    ) -> None:
        """Schedule a crossfade between two clips.

        The *from_clip* fades out while *to_clip* fades in over *duration*
        seconds starting at *start_time*.  If *start_time* is ``None`` the
        crossfade starts immediately.

        Parameters
        ----------
        from_clip : str
            Name of the clip to fade out.
        to_clip : str
            Name of the clip to fade in.
        duration : float
            Crossfade duration in seconds.
        start_time : float | None
            Global time at which the crossfade begins.
        """
        if start_time is None:
            # Caller must track global time; we use -1 as a sentinel that
            # gets resolved on the first evaluate() call.
            start_time = -1.0
        self._crossfades.append((from_clip, to_clip, start_time, duration))
        # Ensure both clips are playing.
        self.play(from_clip, weight=1.0)
        self.play(to_clip, weight=0.0)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, global_time: float
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Evaluate and blend all active layers at *global_time*.

        Returns a blended pose data dictionary:

        ``{target_name: {property_path: {channel: value}}}``

        Parameters
        ----------
        global_time : float
            Current engine time in seconds.

        Returns
        -------
        dict
        """
        # Process crossfades — update weights.
        self._process_crossfades(global_time)

        if not self.active_layers:
            return {}

        # Collect per-layer evaluations.
        layer_results: List[Tuple[float, str, Dict[str, Dict[str, Dict[str, Any]]]]] = []
        for clip_name, weight, offset, blend_mode in self.active_layers:
            clip = self.clips.get(clip_name)
            if clip is None:
                continue
            eval_time = global_time - offset
            pose = clip.evaluate(eval_time)
            layer_results.append((weight, blend_mode, pose))

        # Sort layers by weight ascending so the highest-weight layer is
        # processed last and dominates the result.
        layer_results.sort(key=lambda x: x[0])

        blended: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for weight, blend_mode, pose in layer_results:
            if weight < 1e-10:
                continue
            for target, props in pose.items():
                for prop, channels in props.items():
                    for ch, val in channels.items():
                        current = (
                            blended.setdefault(target, {})
                            .setdefault(prop, {})
                            .get(ch)
                        )
                        if current is None:
                            blended[target][prop][ch] = val
                        else:
                            blended[target][prop][ch] = self._blend_values(
                                current, val, weight, blend_mode
                            )

        return blended

    def _process_crossfades(self, global_time: float) -> None:
        """Update active layer weights for any running crossfades."""
        remaining: List[Tuple[str, str, float, float]] = []
        for from_clip, to_clip, start_time, duration in self._crossfades:
            if start_time < 0:
                # Resolve sentinel on first call.
                start_time = global_time
                remaining.append((from_clip, to_clip, start_time, duration))

            elapsed = global_time - start_time
            if elapsed < 0:
                remaining.append((from_clip, to_clip, start_time, duration))
                continue

            t = min(1.0, elapsed / duration) if duration > 1e-10 else 1.0
            # Smooth step for natural easing.
            t_smooth = t * t * (3.0 - 2.0 * t)

            w_from = 1.0 - t_smooth
            w_to = t_smooth

            # Update weights on the active layers.
            for i, (cn, w, off, bm) in enumerate(self.active_layers):
                if cn == from_clip:
                    self.active_layers[i] = (cn, w_from, off, bm)
                elif cn == to_clip:
                    self.active_layers[i] = (cn, w_to, off, bm)

            if t < 1.0:
                remaining.append((from_clip, to_clip, start_time, duration))
            else:
                # Crossfade complete — stop the outgoing clip.
                self.stop(from_clip)

        self._crossfades = remaining

    @staticmethod
    def _blend_values(
        val_a: Any,
        val_b: Any,
        weight: float,
        blend_mode: str,
    ) -> Any:
        """Blend two values based on *blend_mode*.

        Parameters
        ----------
        val_a : float | np.ndarray
            Base (already-accumulated) value.
        val_b : float | np.ndarray
            Incoming layer value.
        weight : float
            Weight of the incoming layer (0–1).
        blend_mode : str
            ``'replace'`` — weighted lerp; ``'add'`` — additive; ``'multiply'`` — multiplicative.
        """
        a = np.asarray(val_a, dtype=np.float64)
        b = np.asarray(val_b, dtype=np.float64)

        if blend_mode == "replace":
            return a * (1.0 - weight) + b * weight
        elif blend_mode == "add":
            return a + b * weight
        elif blend_mode == "multiply":
            return a * (1.0 + (b - 1.0) * weight)
        else:
            return lerp(a, b, weight)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the mixer state to a JSON-friendly dictionary."""
        return {
            "clips": {n: c.to_dict() for n, c in self.clips.items()},
            "active_layers": [
                {"clip_name": cn, "weight": w, "offset": off, "blend_mode": bm}
                for cn, w, off, bm in self.active_layers
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnimMixer":
        """Deserialise a mixer from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary produced by :meth:`to_dict`.
        """
        clips = {
            n: AnimationClip.from_dict(cd) for n, cd in data.get("clips", {}).items()
        }
        mixer = cls(clips=clips)
        for layer in data.get("active_layers", []):
            mixer.active_layers.append(
                (
                    layer["clip_name"],
                    float(layer["weight"]),
                    float(layer["offset"]),
                    layer.get("blend_mode", "replace"),
                )
            )
        return mixer

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AnimMixer(clips={len(self.clips)}, "
            f"active={len(self.active_layers)})"
        )


# ─── PoseLibrary ──────────────────────────────────────────────────────────────


class PoseLibrary:
    """Library of static poses that can be applied to armatures.

    Poses are stored as dictionaries mapping bone names to their transform
    data.  The library can also generate smooth :class:`AnimCurve`-based
    transition animations between any two stored poses.

    Parameters
    ----------
    poses : dict or None
        Initial set of poses (``{name: pose_dict}``).
    """

    def __init__(
        self,
        poses: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.poses: Dict[str, Dict[str, Any]] = dict(poses or {})
        # Transitions: (pose_a, pose_b) -> AnimCurve list
        self.transitions: Dict[Tuple[str, str], List[AnimCurve]] = {}

    # ------------------------------------------------------------------
    # Pose management
    # ------------------------------------------------------------------

    def add_pose(self, name: str, pose_data: Dict[str, Any]) -> None:
        """Register a named pose.

        Parameters
        ----------
        name : str
            Pose identifier.
        pose_data : dict
            ``{bone_name: {channel: value, ...}, ...}``.
        """
        self.poses[name] = copy.deepcopy(pose_data)

    def get_pose(self, name: str) -> Optional[Dict[str, Any]]:
        """Return a copy of the named pose, or ``None``."""
        pose = self.poses.get(name)
        return copy.deepcopy(pose) if pose is not None else None

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def create_transition(
        self,
        pose_a: str,
        pose_b: str,
        duration: float = 0.5,
    ) -> AnimationClip:
        """Create a transition animation between two poses.

        Generates an :class:`AnimationClip` that smoothly interpolates every
        bone channel from *pose_a* to *pose_b* over *duration* seconds using
        cubic (catmull_rom) interpolation for smoothness.

        Parameters
        ----------
        pose_a : str
            Source pose name.
        pose_b : str
            Destination pose name.
        duration : float
            Transition length in seconds.

        Returns
        -------
        AnimationClip
            A clip with one track per bone.

        Raises
        ------
        KeyError
            If either pose name is not in the library.
        """
        pa = self.poses.get(pose_a)
        pb = self.poses.get(pose_b)
        if pa is None:
            raise KeyError(f"Source pose {pose_a!r} not found in library.")
        if pb is None:
            raise KeyError(f"Destination pose {pose_b!r} not found in library.")

        # Collect all bone names from both poses.
        all_bones = sorted(set(pa.keys()) | set(pb.keys()))

        clip = AnimationClip(
            name=f"Transition_{pose_a}_to_{pose_b}",
            frame_rate=30.0,
            loop=False,
        )

        for bone_name in all_bones:
            data_a = pa.get(bone_name, {})
            data_b = pb.get(bone_name, {})

            track = AnimTrack(
                name=bone_name,
                target_name=bone_name,
                property_path="pose",
                blend_mode="replace",
            )

            # All channel names across both poses for this bone.
            all_channels = sorted(set(data_a.keys()) | set(data_b.keys()))

            for ch in all_channels:
                val_a = data_a.get(ch, 0.0)
                val_b = data_b.get(ch, 0.0)
                val_a = np.asarray(val_a, dtype=np.float64)
                val_b = np.asarray(val_b, dtype=np.float64)

                curve = AnimCurve(
                    name=f"{bone_name}_{ch}",
                    interpolation="catmull_rom",
                )
                curve.add_keyframe(0.0, val_a)
                # Insert an intermediate keyframe for smoother catmull_rom.
                curve.add_keyframe(duration * 0.5, (val_a + val_b) * 0.5)
                curve.add_keyframe(duration, val_b)

                track.set_curve(ch, curve)

            clip.add_track(track)

        # Cache the transition curves for potential reuse.
        curves: List[AnimCurve] = []
        for track in clip.tracks.values():
            curves.extend(track.curves.values())
        self.transitions[(pose_a, pose_b)] = curves

        return clip

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the pose library to a JSON-friendly dictionary."""
        poses_out: Dict[str, Any] = {}
        for name, pose in self.poses.items():
            serialised: Dict[str, Any] = {}
            for bone, channels in pose.items():
                serialised[bone] = {}
                for ch, val in channels.items():
                    if isinstance(val, np.ndarray):
                        serialised[bone][ch] = val.tolist()
                    else:
                        serialised[bone][ch] = val
            poses_out[name] = serialised
        return {"poses": poses_out}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoseLibrary":
        """Deserialise a pose library from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary produced by :meth:`to_dict`.
        """
        lib = cls()
        for name, pose in data.get("poses", {}).items():
            lib.add_pose(name, pose)
        return lib

    def __repr__(self) -> str:  # pragma: no cover
        return f"PoseLibrary(poses={len(self.poses)})"


# ─── ProceduralAnimation ─────────────────────────────────────────────────────


class ProceduralAnimation:
    """Procedural animation generators — no keyframes needed.

    All methods are stateless ``@staticmethod`` functions that return
    instantaneous values or per-bone pose data based on the current *time*.
    """

    # ------------------------------------------------------------------
    # Wave generators
    # ------------------------------------------------------------------

    @staticmethod
    def sine_wave(
        time: float,
        frequency: float = 1.0,
        amplitude: float = 1.0,
        phase: float = 0.0,
    ) -> float:
        """Generate a sine wave value.

        Parameters
        ----------
        time : float
            Current time in seconds.
        frequency : float
            Oscillations per second (Hz).
        amplitude : float
            Peak amplitude.
        phase : float
            Phase offset in radians.

        Returns
        -------
        float
        """
        return float(amplitude * math.sin(2.0 * math.pi * frequency * time + phase))

    @staticmethod
    def noise(
        time: float,
        frequency: float = 1.0,
        amplitude: float = 1.0,
        octaves: int = 4,
        seed: float = 0.0,
    ) -> float:
        """Simple value noise with octaves (fractal Brownian motion).

        Uses a hash-based pseudo-random function for deterministic output.

        Parameters
        ----------
        time : float
            Current time in seconds.
        frequency : float
            Base frequency.
        amplitude : float
            Base amplitude.
        octaves : int
            Number of noise octaves (detail layers).
        seed : float
            Random seed offset.

        Returns
        -------
        float
        """
        total = 0.0
        current_amp = amplitude
        current_freq = frequency
        max_amp = 0.0

        for _ in range(octaves):
            total += ProceduralAnimation._value_noise(time * current_freq + seed) * current_amp
            max_amp += current_amp
            current_amp *= 0.5  # persistence
            current_freq *= 2.0  # lacunarity

        if max_amp > 1e-10:
            total /= max_amp

        return float(total * amplitude)

    @staticmethod
    def _value_noise(x: float) -> float:
        """Deterministic value noise for a single scalar input.

        Maps the input to a pseudo-random value in [-1, 1] using a hash.
        """
        # Simple hash-based noise
        xi = int(math.floor(x))
        frac = x - xi

        # Smooth interpolation
        t = frac * frac * (3.0 - 2.0 * frac)

        # Hash the integer positions
        def _hash(n: int) -> float:
            n = ((n << 13) ^ n) & 0x7FFFFFFF
            return 1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7FFFFFFF) / 1073741824.0

        v0 = _hash(xi)
        v1 = _hash(xi + 1)

        return v0 + t * (v1 - v0)

    # ------------------------------------------------------------------
    # Humanoid procedural motions
    # ------------------------------------------------------------------

    @staticmethod
    def walk_cycle(
        time: float,
        stride_length: float = 0.6,
        stride_height: float = 0.1,
        cycle_time: float = 1.0,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Generate walk cycle pose data for a humanoid.

        Returns per-bone pose data as:

        ``{bone_name: (position_offset: ndarray(3,), rotation_quat: ndarray(4,))}``

        Only the bones involved in walking are animated: hips, legs, arms,
        and spine.

        Parameters
        ----------
        time : float
            Current time in seconds.
        stride_length : float
            Forward distance per step (metres).
        stride_height : float
            Maximum foot lift height (metres).
        cycle_time : float
            Duration of a full walk cycle (both feet) in seconds.

        Returns
        -------
        dict
        """
        phase = (time % cycle_time) / cycle_time  # [0, 1)
        t = phase * 2.0 * math.pi

        poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Root bob (vertical oscillation)
        bob = stride_height * abs(math.sin(t))
        poses["Hips"] = (
            np.array([0.0, -bob * 0.5, 0.0], dtype=np.float64),
            quat_identity(),
        )

        # Left leg — forward during 0..0.5, back during 0.5..1.0
        left_swing = math.sin(t)
        left_lift = max(0.0, math.sin(t)) * stride_height
        left_hip_rot = quat_from_axis_angle(
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            left_swing * 0.4,
        )
        left_knee_rot = quat_from_axis_angle(
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            -max(0.0, left_swing) * 0.6,
        )
        poses["LeftUpLeg"] = (
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            left_hip_rot,
        )
        poses["LeftLeg"] = (
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            left_knee_rot,
        )
        poses["LeftFoot"] = (
            np.array([0.0, 0.0, left_lift], dtype=np.float64),
            quat_from_axis_angle(
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
                -left_lift * 3.0,
            ),
        )

        # Right leg — opposite phase
        right_swing = math.sin(t + math.pi)
        right_lift = max(0.0, math.sin(t + math.pi)) * stride_height
        right_hip_rot = quat_from_axis_angle(
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            right_swing * 0.4,
        )
        right_knee_rot = quat_from_axis_angle(
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            -max(0.0, right_swing) * 0.6,
        )
        poses["RightUpLeg"] = (
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            right_hip_rot,
        )
        poses["RightLeg"] = (
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            right_knee_rot,
        )
        poses["RightFoot"] = (
            np.array([0.0, 0.0, right_lift], dtype=np.float64),
            quat_from_axis_angle(
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
                -right_lift * 3.0,
            ),
        )

        # Arms swing in opposition to legs
        left_arm_swing = math.sin(t + math.pi) * 0.3
        right_arm_swing = math.sin(t) * 0.3
        poses["LeftArm"] = (
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            quat_from_axis_angle(
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
                left_arm_swing,
            ),
        )
        poses["RightArm"] = (
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            quat_from_axis_angle(
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
                right_arm_swing,
            ),
        )

        # Spine twist
        spine_rot = quat_from_axis_angle(
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            math.sin(t) * 0.05,
        )
        poses["Spine"] = (
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            spine_rot,
        )

        return poses

    @staticmethod
    def breathing(
        time: float,
        frequency: float = 0.25,
        amplitude: float = 0.02,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Generate breathing animation for chest / spine bones.

        Parameters
        ----------
        time : float
            Current time in seconds.
        frequency : float
            Breaths per second (Hz).
        amplitude : float
            Maximum chest expansion offset (metres).

        Returns
        -------
        dict
            ``{bone_name: (position_offset, rotation_quat)}``
        """
        t = time * 2.0 * math.pi * frequency
        breath = math.sin(t)

        scale_x = 1.0 + breath * amplitude * 0.5
        scale_z = 1.0 + breath * amplitude * 0.3

        poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Chest expansion
        poses["Chest"] = (
            np.array([
                0.0,
                breath * amplitude * 0.1,
                0.0,
            ], dtype=np.float64),
            quat_from_axis_angle(
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
                -breath * amplitude * 0.2,
            ),
        )

        # Spine follows with reduced motion
        poses["Spine"] = (
            np.array([
                0.0,
                breath * amplitude * 0.05,
                0.0,
            ], dtype=np.float64),
            quat_identity(),
        )

        # Slight shoulder rise
        poses["LeftShoulder"] = (
            np.array([0.0, breath * amplitude * 0.15, 0.0], dtype=np.float64),
            quat_identity(),
        )
        poses["RightShoulder"] = (
            np.array([0.0, breath * amplitude * 0.15, 0.0], dtype=np.float64),
            quat_identity(),
        )

        return poses

    @staticmethod
    def idle_sway(
        time: float,
        frequency: float = 0.5,
        amplitude: float = 0.01,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Generate subtle idle sway animation.

        A slow, organic-looking sway of the hips, spine, and head to keep a
        character looking alive when otherwise stationary.

        Parameters
        ----------
        time : float
            Current time in seconds.
        frequency : float
            Sway frequency in Hz.
        amplitude : float
            Maximum displacement in metres.

        Returns
        -------
        dict
            ``{bone_name: (position_offset, rotation_quat)}``
        """
        t = time * 2.0 * math.pi * frequency

        poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Hips sway side-to-side and slightly front-to-back
        hip_offset = np.array([
            math.sin(t * 0.7) * amplitude,
            math.sin(t * 1.4) * amplitude * 0.3,
            math.cos(t * 0.5) * amplitude * 0.5,
        ], dtype=np.float64)
        hip_rot = quat_from_axis_angle(
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            math.sin(t * 0.7) * amplitude * 2.0,
        )
        poses["Hips"] = (hip_offset, hip_rot)

        # Spine counter-rotates to keep upper body more stable
        spine_rot = quat_from_axis_angle(
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            -math.sin(t * 0.7) * amplitude * 1.0,
        )
        poses["Spine"] = (np.zeros(3, dtype=np.float64), spine_rot)

        # Head looks slightly around
        head_rot = quat_from_axis_angle(
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            math.sin(t * 0.3) * amplitude * 3.0,
        )
        head_rot = normalize_quat(
            quat_multiply(
                head_rot,
                quat_from_axis_angle(
                    np.array([1.0, 0.0, 0.0], dtype=np.float64),
                    math.sin(t * 0.4 + 1.0) * amplitude * 1.5,
                ),
            )
        )
        poses["Head"] = (np.zeros(3, dtype=np.float64), head_rot)

        # Arms hang with subtle sway
        for side, sign in [("Left", 1.0), ("Right", -1.0)]:
            arm_rot = quat_from_axis_angle(
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
                sign * math.sin(t * 0.6 + side.count("L") * 0.5) * amplitude * 1.5,
            )
            poses[f"{side}Arm"] = (np.zeros(3, dtype=np.float64), arm_rot)

        return poses

    @staticmethod
    def look_at_target(
        head_bone_name: str,
        target_position: np.ndarray,
        head_position: np.ndarray,
        up: Optional[np.ndarray] = None,
        smoothing: float = 0.1,
    ) -> Tuple[str, np.ndarray]:
        """Generate head rotation to look at a target position.

        Returns a tuple ``(bone_name, rotation_quat)`` that should be applied
        as a local-space rotation to the head bone.

        Parameters
        ----------
        head_bone_name : str
            Name of the head bone.
        target_position : array-like, shape (3,)
            World-space position to look at.
        head_position : array-like, shape (3,)
            Current world-space position of the head bone.
        up : array-like or None
            Up vector (defaults to Y-up).
        smoothing : float
            Smoothstep exponent — higher values create a softer onset
            at the extremes of rotation range.

        Returns
        -------
        tuple[str, np.ndarray]
            ``(bone_name, rotation_quat)`` with quaternion [w, x, y, z].
        """
        target = np.asarray(target_position, dtype=np.float64)
        head = np.asarray(head_position, dtype=np.float64)
        if up is None:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            up = np.asarray(up, dtype=np.float64)

        direction = target - head
        dist = np.linalg.norm(direction)

        if dist < 1e-10:
            return (head_bone_name, quat_identity())

        direction /= dist

        # Default forward is -Z (common convention).
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        dot_val = float(np.clip(np.dot(forward, direction), -1.0, 1.0))

        if dot_val > 1.0 - 1e-6:
            return (head_bone_name, quat_identity())
        if dot_val < -1.0 + 1e-6:
            return (head_bone_name, quat_from_axis_angle(up, math.pi))

        axis = np.cross(forward, direction)
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-10:
            return (head_bone_name, quat_identity())
        axis /= axis_len

        angle = math.acos(dot_val)

        # Apply smoothing via smoothstep-like dampening.
        max_angle = math.pi * 0.8  # limit head rotation range
        if angle > max_angle:
            t = max_angle / angle
            angle = max_angle
            # Smooth the direction towards the limited angle.
            direction = forward + t * (direction - forward)
            direction /= np.linalg.norm(direction)

        if smoothing > 0:
            # Ease the rotation: small rotations are linear, large ones are dampened.
            factor = 1.0 - math.exp(-smoothing * angle)
            angle *= factor

        rot = quat_from_axis_angle(axis, angle)
        return (head_bone_name, normalize_quat(rot))

    @staticmethod
    def spring_damper(
        time: float,
        target_value: float,
        current_value: float,
        stiffness: float = 100.0,
        damping: float = 10.0,
        mass: float = 1.0,
    ) -> float:
        """Spring-damper physics for organic motion.

        Computes the response of a damped harmonic oscillator driven towards
        *target_value*.  This is not a true time-stepping simulation; instead
        it evaluates the critically-damped spring analytically for a single
        frame step (dt = 1/60).

        Parameters
        ----------
        time : float
            Current time (unused in the analytical form but kept for API
            consistency; the caller is expected to call this every frame and
            pass the updated *current_value*).
        target_value : float
            The equilibrium / target value.
        current_value : float
            The current value being driven.
        stiffness : float
            Spring constant (N/m equivalent).
        damping : float
            Damping coefficient.
        mass : float
            Mass of the driven object.

        Returns
        -------
        float
            The next value after one timestep of the spring.
        """
        dt = 1.0 / 60.0  # assume 60 fps timestep

        # Spring force: F = -k * (x - x_target)
        displacement = current_value - target_value
        spring_force = -stiffness * displacement

        # Damping force: F = -c * v (estimated velocity from expected change)
        # Use a simple implicit estimate.
        velocity = -spring_force * dt / mass if mass > 1e-10 else 0.0
        damping_force = -damping * velocity

        # Total acceleration
        total_force = spring_force + damping_force
        acceleration = total_force / mass if mass > 1e-10 else 0.0

        # Integrate
        new_velocity = velocity + acceleration * dt
        new_value = current_value + new_velocity * dt

        return float(new_value)


# ─── AnimationExporter ────────────────────────────────────────────────────────


class AnimationExporter:
    """Export animations to various industry-standard formats.

    All methods are ``@staticmethod`` functions that write files to disk.
    """

    # ------------------------------------------------------------------
    # glTF
    # ------------------------------------------------------------------

    @staticmethod
    def to_gltf(clip: AnimationClip, filepath: str) -> None:
        """Export animation clip as a simplified glTF animation.

        Produces a valid glTF 2.0 JSON file with an ``animations`` array
        containing samplers and channels for every animated track.

        Parameters
        ----------
        clip : AnimationClip
            The animation clip to export.
        filepath : str
            Destination file path (should end in ``.gltf``).
        """
        duration = clip.get_length()
        if duration < 1e-10:
            raise ValueError("Cannot export an empty animation clip.")

        frame_count = max(2, int(math.ceil(duration * clip.frame_rate)))
        frame_times = np.linspace(0.0, duration, frame_count).tolist()

        gltf: Dict[str, Any] = {
            "asset": {"version": "2.0", "generator": "Nexus3D"},
            "scenes": [{"nodes": [0]}],
            "nodes": [{"name": "AnimatedNode"}],
            "animations": [],
        }

        accessors: List[Dict[str, Any]] = []
        buffer_views: List[Dict[str, Any]] = []
        buffer_data = bytearray()
        buffer_byte_offset = 0

        anim: Dict[str, Any] = {"channels": [], "samplers": []}

        def _append_accessor(data_list: List[float], component_type: int = 5126, count: int = 0, type_str: str = "SCALAR") -> int:
            """Append an accessor for raw float data."""
            nonlocal buffer_byte_offset
            byte_len = len(data_list) * 4
            bv_idx = len(buffer_views)
            buffer_views.append({
                "buffer": 0,
                "byteOffset": buffer_byte_offset,
                "byteLength": byte_len,
                "target": 34962,  # ARRAY_BUFFER
            })
            buffer_data.extend(struct.pack(f"<{len(data_list)}f", *data_list))
            buffer_byte_offset += byte_len

            acc_idx = len(accessors)
            accessors.append({
                "bufferView": bv_idx,
                "byteOffset": 0,
                "componentType": component_type,
                "count": count or len(data_list),
                "type": type_str,
            })
            return acc_idx

        # Build samplers / channels for each track's curves.
        for track in clip.tracks.values():
            for channel_name, curve in track.curves.items():
                # Sample the curve at each frame.
                values: List[float] = []
                for ft in frame_times:
                    v = curve.evaluate(ft)
                    if isinstance(v, np.ndarray):
                        values.extend(v.tolist())
                    else:
                        values.append(float(v))

                # Determine type
                num_components = len(values) // frame_count
                if num_components == 1:
                    type_str = "SCALAR"
                elif num_components == 3:
                    type_str = "VEC3"
                elif num_components == 4:
                    type_str = "VEC4"
                else:
                    type_str = "SCALAR"

                input_acc = _append_accessor(frame_times, count=frame_count, type_str="SCALAR")
                output_acc = _append_accessor(values, count=frame_count, type_str=type_str)

                sampler_idx = len(anim["samplers"])
                anim["samplers"].append({
                    "input": input_acc,
                    "output": output_acc,
                    "interpolation": "LINEAR",
                })

                # glTF path: "translation", "rotation", "scale", "weights"
                path_map = {
                    "position": "translation",
                    "rotation": "rotation",
                    "scale": "scale",
                }
                gltf_path = "translation"  # default
                if track.property_path in path_map:
                    gltf_path = path_map[track.property_path]

                anim["channels"].append({
                    "sampler": sampler_idx,
                    "target": {
                        "node": 0,
                        "path": gltf_path,
                    },
                })

        gltf["animations"].append(anim)
        gltf["accessors"] = accessors
        gltf["bufferViews"] = buffer_views
        gltf["buffers"] = [{"byteLength": len(buffer_data)}]

        # Write binary buffer alongside glTF JSON.
        import os
        base, ext = os.path.splitext(filepath)
        bin_path = base + ".bin"

        gltf_json = json.dumps(gltf, indent=2)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(gltf_json)
        with open(bin_path, "wb") as f:
            f.write(buffer_data)

    # ------------------------------------------------------------------
    # BVH
    # ------------------------------------------------------------------

    @staticmethod
    def to_bvh(armature: Any, clip: AnimationClip, filepath: str) -> None:
        """Export animation as BVH motion capture file.

        BVH (Biovision Hierarchy) is a standard mocap format readable by
        Blender, Maya, MotionBuilder, and other DCC tools.

        Parameters
        ----------
        armature : Armature
            The skeleton to bake the animation onto.
        clip : AnimationClip
            The animation clip to export.
        filepath : str
            Destination file path (should end in ``.bvh``).
        """
        duration = clip.get_length()
        if duration < 1e-10:
            raise ValueError("Cannot export an empty animation clip.")

        frame_count = max(1, int(math.ceil(duration * clip.frame_rate)))
        frame_dt = 1.0 / clip.frame_rate

        # Build ordered bone list using the armature's hierarchy.
        bone_order: List[Any] = []
        visited: set = set()

        def _walk(bone: Any) -> None:
            if bone.name in visited:
                return
            visited.add(bone.name)
            bone_order.append(bone)
            for child in bone.children:
                _walk(child)

        for root in armature.root_bones:
            _walk(root)

        # ── HIERARCHY section ──
        lines: List[str] = []
        lines.append("HIERARCHY")

        def _write_joint(bone: Any, indent: int) -> None:
            prefix = "  " * indent
            if indent == 0:
                lines.append(f"{prefix}ROOT {bone.name}")
            else:
                lines.append(f"{prefix}JOINT {bone.name}")
            lines.append(f"{prefix}{{")

            # OFFSET
            offset = bone.head_pos if bone.parent is None else bone.head_pos
            lines.append(
                f"{prefix}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}"
            )

            # CHANNELS
            if bone.parent is None:
                # Root: position + rotation (6 channels)
                lines.append(
                    f"{prefix}  CHANNELS 6 "
                    "Xposition Yposition Zposition "
                    "Zrotation Xrotation Yrotation"
                )
            else:
                # Child joints: rotation only (3 channels)
                lines.append(
                    f"{prefix}  CHANNELS 3 Zrotation Xrotation Yrotation"
                )

            # Recurse into children
            for child in bone.children:
                _write_joint(child, indent + 1)

            # End Site
            lines.append(f"{prefix}  End Site")
            lines.append(f"{prefix}  {{")
            tail = bone.tail_pos - bone.head_pos
            lines.append(
                f"{prefix}    OFFSET {tail[0]:.6f} {tail[1]:.6f} {tail[2]:.6f}"
            )
            lines.append(f"{prefix}  }}")
            lines.append(f"{prefix}}}")

        for root in armature.root_bones:
            _write_joint(root, 0)

        # ── MOTION section ──
        lines.append("MOTION")
        lines.append(f"Frames: {frame_count}")
        lines.append(f"Frame Time: {frame_dt:.6f}")

        for frame_idx in range(frame_count):
            t = frame_idx * frame_dt
            pose = clip.evaluate(t)
            frame_values: List[float] = []

            for bone in bone_order:
                # Look up bone data in the pose
                bone_data = pose.get(bone.name, {}).get("pose", {})

                if bone.parent is None:
                    # Root bone: position X Y Z then rotation Z X Y
                    pos = bone_data.get("position", bone.head_pos)
                    if isinstance(pos, np.ndarray):
                        frame_values.extend(pos.tolist())
                    else:
                        frame_values.extend([
                            float(bone.head_pos[0]),
                            float(bone.head_pos[1]),
                            float(bone.head_pos[2]),
                        ])
                    # Convert quaternion rotation to Euler ZXY (BVH convention)
                    rot = bone_data.get("rotation", np.array([0.0, 0.0, 0.0]))
                    if isinstance(rot, np.ndarray) and rot.shape == (4,):
                        eulers = _quat_to_euler_zxy(rot)
                        frame_values.extend([
                            math.degrees(eulers[2]),  # Z
                            math.degrees(eulers[0]),  # X
                            math.degrees(eulers[1]),  # Y
                        ])
                    else:
                        frame_values.extend([0.0, 0.0, 0.0])
                else:
                    # Non-root: rotation only (Z X Y)
                    rot = bone_data.get("rotation", np.array([0.0, 0.0, 0.0]))
                    if isinstance(rot, np.ndarray) and rot.shape == (4,):
                        eulers = _quat_to_euler_zxy(rot)
                        frame_values.extend([
                            math.degrees(eulers[2]),  # Z
                            math.degrees(eulers[0]),  # X
                            math.degrees(eulers[1]),  # Y
                        ])
                    else:
                        frame_values.extend([0.0, 0.0, 0.0])

            lines.append(" ".join(f"{v:.6f}" for v in frame_values))

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # FBX keyframes
    # ------------------------------------------------------------------

    @staticmethod
    def to_fbx_keyframes(clip: AnimationClip) -> Dict[str, Dict[str, Any]]:
        """Generate FBX-compatible keyframe data.

        Returns a dictionary structured for FBX import:

        ``{track_name: {channel: {"times": [...], "values": [...]}}}``

        The output is not a complete FBX file (which is a complex binary
        format), but rather the keyframe data that can be mapped into an
        FBX document by a downstream writer.

        Parameters
        ----------
        clip : AnimationClip

        Returns
        -------
        dict
        """
        duration = clip.get_length()
        if duration < 1e-10:
            return {}

        frame_count = max(2, int(math.ceil(duration * clip.frame_rate)))
        result: Dict[str, Dict[str, Any]] = {}

        for track in clip.tracks.values():
            track_key = f"{track.target_name}.{track.property_path}" if track.target_name else track.name
            result[track_key] = {}

            for channel_name, curve in track.curves.items():
                times: List[float] = []
                values: List[float] = []

                for i in range(frame_count):
                    t = i / clip.frame_rate
                    v = curve.evaluate(t)
                    times.append(t)
                    if isinstance(v, np.ndarray):
                        values.extend(v.tolist())
                    else:
                        values.append(float(v))

                result[track_key][channel_name] = {
                    "times": times,
                    "values": values,
                    "interpolation": curve.interpolation,
                }

        return result

    # ------------------------------------------------------------------
    # Video
    # ------------------------------------------------------------------

    @staticmethod
    def frames_to_video(
        frame_images: Sequence[Any],
        output_path: str,
        fps: int = 30,
    ) -> None:
        """Write a sequence of frame images to a video file.

        Parameters
        ----------
        frame_images : sequence
            List of PIL images or numpy arrays (H, W, 3) uint8.
        output_path : str
            Destination file path (extension determines codec, e.g.
            ``.mp4``, ``.gif``, ``.webm``).
        fps : int
            Frames per second.

        Raises
        ------
        ImportError
            If ``imageio`` (or ``imageio-ffmpeg``) is not installed.
        RuntimeError
            If no frames are provided.
        """
        try:
            import imageio.v2 as iio
        except ImportError:
            raise ImportError(
                "imageio is required for video export.  "
                "Install it with:  pip install imageio imageio-ffmpeg"
            )

        if not frame_images:
            raise RuntimeError("No frames provided for video export.")

        # Convert numpy arrays if needed.
        frames = []
        for img in frame_images:
            if isinstance(img, np.ndarray):
                # Ensure uint8 RGB.
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                frames.append(img)
            else:
                # Assume PIL image — convert to numpy.
                frames.append(np.array(img))

        writer = iio.get_writer(output_path, fps=fps)
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()


# ─── BVH Helper ───────────────────────────────────────────────────────────────


def _quat_to_euler_zxy(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w,x,y,z] to Euler angles (X, Y, Z) using the
    ZXY rotation order required by BVH.

    Returns an array of three angles in radians: [X, Y, Z].
    """
    q = normalize_quat(q)
    w, x, y, z = q

    # ZXY Euler angles from quaternion.
    # Derived from rotation matrix decomposition.
    # R = Rz * Rx * Ry
    sin_x = 2.0 * (w * x - y * z)
    sin_x = np.clip(sin_x, -1.0, 1.0)
    cos_x = 1.0 - 2.0 * (x * x + z * z)

    x_angle = math.asin(sin_x)

    sin_z = 2.0 * (w * z + x * y)
    sin_z = np.clip(sin_z, -1.0, 1.0)
    z_angle = math.asin(sin_z)

    sin_y_cos_x = 2.0 * (w * y + x * z)
    cos_y_cos_x = 1.0 - 2.0 * (y * y + z * z)

    if abs(cos_x) > 1e-6:
        y_angle = math.atan2(sin_y_cos_x, cos_y_cos_x)
    else:
        y_angle = math.atan2(-2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))

    return np.array([x_angle, y_angle, z_angle], dtype=np.float64)
