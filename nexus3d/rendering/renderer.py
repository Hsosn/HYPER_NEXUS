from __future__ import annotations

"""Nexus3D Headless Rendering Pipeline.

Renders 3D scenes to images without any display server, using either
PyRender (OSMesa) for GPU-quality output or a pure-Python software
rasterizer as a fallback when OpenGL is unavailable.
"""

# ── MUST be set before any OpenGL-related import ───────────────────────
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
# ────────────────────────────────────────────────────────────────────────

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nexus3d.math3d.core import (
    mat4_identity,
    mat4_look_at,
    mat4_perspective,
    mat4_rotate_x,
    mat4_rotate_y,
    mat4_rotate_z,
    mat4_translate,
    vec3,
    normalize,
    dot,
    cross,
    reflect,
    quat_to_rotation_matrix,
    compute_bounding_box,
    hex_to_rgb,
    fresnel_schlick,
)

logger = logging.getLogger(__name__)

# Lazy PyRender imports – guarded so the module loads even without OpenGL.
_pyrender_available = False
try:
    import pyrender  # noqa: E402 – after env var is set
    _pyrender_available = True
except Exception:
    logger.info("PyRender / OSMesa not available; software rasterizer will be used.")

_pil_available = False
try:
    from PIL import Image  # noqa: E402
    _pil_available = True
except ImportError:
    logger.info("Pillow not available; image saving will be limited.")

_imageio_available = False
try:
    import imageio  # noqa: E402
    _imageio_available = True
except ImportError:
    logger.info("imageio not available; video export will be disabled.")


# ═══════════════════════════════════════════════════════════════════════════════
# Material
# ═══════════════════════════════════════════════════════════════════════════════

class Material:
    """Physically-Based Rendering (PBR) material definition.

    Attributes:
        name: Human-readable label.
        base_color: RGBA base colour as a float32 array (0-1 per channel).
        metallic: Metallic factor in [0, 1].
        roughness: Roughness factor in [0, 1].
        emissive: RGB emissive colour as a float32 array.
        emissive_strength: Multiplier for the emissive contribution.
        normal_map: File-path to a normal-map texture, or *None*.
        alpha_mode: One of ``'opaque'``, ``'mask'``, ``'blend'``.
        alpha_cutoff: Threshold for alpha masking (used when *alpha_mode* is
            ``'mask'``).
        double_sided: When *True* both face-windings are rendered.
        ior: Index of refraction (relevant for transmission/glass).
        transmission: Transmission factor in [0, 1] (glass).
    """

    def __init__(self, name: str = "Material") -> None:
        self.name: str = name
        self.base_color: np.ndarray = np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32)
        self.metallic: float = 0.0
        self.roughness: float = 0.5
        self.emissive: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.emissive_strength: float = 1.0
        self.normal_map: Optional[str] = None
        self.alpha_mode: str = "opaque"
        self.alpha_cutoff: float = 0.5
        self.double_sided: bool = False
        self.ior: float = 1.5
        self.transmission: float = 0.0

    # ── Convenience constructors ───────────────────────────────────────────

    @classmethod
    def from_color(cls, hex_color: str = "#ffffff", name: str = "Material") -> "Material":
        """Create a simple diffuse material from a hex colour string."""
        mat = cls(name=name)
        r, g, b = hex_to_rgb(hex_color)
        mat.base_color = np.array([r, g, b, 1.0], dtype=np.float32)
        mat.metallic = 0.0
        mat.roughness = 0.8
        return mat

    @classmethod
    def standard_pbr(
        cls,
        base_color: str = "#ffffff",
        metallic: float = 0.0,
        roughness: float = 0.5,
        name: str = "PBR",
    ) -> "Material":
        """Create a standard PBR material with tuned parameters."""
        mat = cls(name=name)
        r, g, b = hex_to_rgb(base_color)
        mat.base_color = np.array([r, g, b, 1.0], dtype=np.float32)
        mat.metallic = float(np.clip(metallic, 0.0, 1.0))
        mat.roughness = float(np.clip(roughness, 0.0, 1.0))
        return mat

    @classmethod
    def glass(
        cls,
        transmission: float = 0.9,
        ior: float = 1.5,
        roughness: float = 0.0,
        name: str = "Glass",
    ) -> "Material":
        """Create a glass / transmission material."""
        mat = cls(name=name)
        mat.base_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        mat.metallic = 0.0
        mat.roughness = float(np.clip(roughness, 0.0, 1.0))
        mat.transmission = float(np.clip(transmission, 0.0, 1.0))
        mat.ior = float(ior)
        mat.alpha_mode = "blend"
        return mat

    @classmethod
    def emissive(
        cls,
        color: str = "#ffffff",
        strength: float = 1.0,
        name: str = "Emissive",
    ) -> "Material":
        """Create an emissive / self-illuminating material."""
        mat = cls(name=name)
        r, g, b = hex_to_rgb(color)
        mat.emissive = np.array([r, g, b], dtype=np.float32)
        mat.emissive_strength = float(strength)
        mat.base_color = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return mat

    # ── Serialisation ───────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-data dictionary suitable for JSON / pickle."""
        return {
            "name": self.name,
            "base_color": self.base_color.tolist(),
            "metallic": self.metallic,
            "roughness": self.roughness,
            "emissive": self.emissive.tolist(),
            "emissive_strength": self.emissive_strength,
            "normal_map": self.normal_map,
            "alpha_mode": self.alpha_mode,
            "alpha_cutoff": self.alpha_cutoff,
            "double_sided": self.double_sided,
            "ior": self.ior,
            "transmission": self.transmission,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Material":
        """Reconstruct a :class:`Material` from a dictionary."""
        mat = cls(name=data.get("name", "Material"))
        if "base_color" in data:
            mat.base_color = np.array(data["base_color"], dtype=np.float32)
        mat.metallic = float(data.get("metallic", 0.0))
        mat.roughness = float(data.get("roughness", 0.5))
        if "emissive" in data:
            mat.emissive = np.array(data["emissive"], dtype=np.float32)
        mat.emissive_strength = float(data.get("emissive_strength", 1.0))
        mat.normal_map = data.get("normal_map")
        mat.alpha_mode = data.get("alpha_mode", "opaque")
        mat.alpha_cutoff = float(data.get("alpha_cutoff", 0.5))
        mat.double_sided = bool(data.get("double_sided", False))
        mat.ior = float(data.get("ior", 1.5))
        mat.transmission = float(data.get("transmission", 0.0))
        return mat

    def __repr__(self) -> str:
        return (
            f"Material(name={self.name!r}, "
            f"metallic={self.metallic:.2f}, roughness={self.roughness:.2f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Light
# ═══════════════════════════════════════════════════════════════════════════════

class Light:
    """A light source in the scene.

    Attributes:
        name: Human-readable label.
        light_type: One of ``'point'``, ``'directional'``, ``'spot'``,
            ``'area'``.
        color: RGB colour as a float32 array (0-1).
        intensity: Luminous intensity multiplier.
        position: World-space position (relevant for point / spot).
        direction: World-space direction (relevant for directional / spot).
        range: Effective range in world units (for point / spot).
        spot_angle: Full cone half-angle in **radians** (for spot only).
        cast_shadows: Whether this light generates a shadow map.
        shadow_map_size: Resolution of the shadow depth texture.
    """

    def __init__(self, name: str = "Light", light_type: str = "point") -> None:
        self.name: str = name
        self.light_type: str = light_type
        self.color: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.intensity: float = 1.0
        self.position: np.ndarray = vec3(0.0, 5.0, 0.0)
        self.direction: np.ndarray = vec3(0.0, -1.0, 0.0)
        self.range: float = 10.0
        self.spot_angle: float = math.radians(45.0)
        self.cast_shadows: bool = False
        self.shadow_map_size: int = 1024

    # ── Convenience constructors ───────────────────────────────────────────

    @classmethod
    def directional(
        cls,
        direction: Optional[np.ndarray] = None,
        color: str = "#ffffff",
        intensity: float = 1.0,
        name: str = "Sun",
    ) -> "Light":
        """Create a directional light (e.g. sunlight)."""
        light = cls(name=name, light_type="directional")
        light.direction = normalize(direction if direction is not None else vec3(-0.5, -1.0, -0.3))
        r, g, b = hex_to_rgb(color)
        light.color = np.array([r, g, b], dtype=np.float32)
        light.intensity = float(intensity)
        light.cast_shadows = True
        return light

    @classmethod
    def point(
        cls,
        position: Optional[np.ndarray] = None,
        color: str = "#ffffff",
        intensity: float = 1.0,
        range: float = 10.0,
        name: str = "PointLight",
    ) -> "Light":
        """Create a point light that radiates uniformly in all directions."""
        light = cls(name=name, light_type="point")
        light.position = position if position is not None else vec3(0.0, 5.0, 0.0)
        r, g, b = hex_to_rgb(color)
        light.color = np.array([r, g, b], dtype=np.float32)
        light.intensity = float(intensity)
        light.range = float(range)
        light.cast_shadows = True
        return light

    @classmethod
    def spot(
        cls,
        position: Optional[np.ndarray] = None,
        direction: Optional[np.ndarray] = None,
        color: str = "#ffffff",
        intensity: float = 1.0,
        spot_angle: float = 45.0,
        range: float = 10.0,
        name: str = "SpotLight",
    ) -> "Light":
        """Create a spot (cone) light."""
        light = cls(name=name, light_type="spot")
        light.position = position if position is not None else vec3(0.0, 5.0, 0.0)
        light.direction = normalize(direction if direction is not None else vec3(0.0, -1.0, 0.0))
        r, g, b = hex_to_rgb(color)
        light.color = np.array([r, g, b], dtype=np.float32)
        light.intensity = float(intensity)
        light.spot_angle = math.radians(spot_angle)
        light.range = float(range)
        light.cast_shadows = True
        return light

    # ── Serialisation ───────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-data dictionary."""
        return {
            "name": self.name,
            "light_type": self.light_type,
            "color": self.color.tolist(),
            "intensity": self.intensity,
            "position": self.position.tolist(),
            "direction": self.direction.tolist(),
            "range": self.range,
            "spot_angle": self.spot_angle,
            "cast_shadows": self.cast_shadows,
            "shadow_map_size": self.shadow_map_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Light":
        """Reconstruct a :class:`Light` from a dictionary."""
        light = cls(name=data.get("name", "Light"), light_type=data.get("light_type", "point"))
        if "color" in data:
            light.color = np.array(data["color"], dtype=np.float32)
        light.intensity = float(data.get("intensity", 1.0))
        if "position" in data:
            light.position = np.array(data["position"], dtype=np.float64)
        if "direction" in data:
            light.direction = np.array(data["direction"], dtype=np.float64)
        light.range = float(data.get("range", 10.0))
        light.spot_angle = float(data.get("spot_angle", math.radians(45.0)))
        light.cast_shadows = bool(data.get("cast_shadows", False))
        light.shadow_map_size = int(data.get("shadow_map_size", 1024))
        return light

    def __repr__(self) -> str:
        return (
            f"Light(name={self.name!r}, type={self.light_type!r}, "
            f"intensity={self.intensity:.2f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Camera
# ═══════════════════════════════════════════════════════════════════════════════

class Camera:
    """A virtual camera for rendering.

    Attributes:
        name: Human-readable label.
        position: World-space camera origin.
        target: Look-at target position in world space.
        up: Up-vector hint (default +Y).
        fov: Vertical field-of-view in **degrees** (perspective mode).
        aspect: Width / height ratio (updated automatically by
            :meth:`set_resolution`).
        near: Near clipping plane distance.
        far: Far clipping plane distance.
        ortho: If *True*, use an orthographic projection instead.
        ortho_size: Half-extent of the orthographic frustum (world units).
        resolution: ``(width, height)`` in pixels.
        background_color: ``(R, G, B)`` tuple, each in 0-1.
    """

    def __init__(self, name: str = "Camera") -> None:
        self.name: str = name
        self.position: np.ndarray = vec3(0.0, 2.0, 5.0)
        self.target: np.ndarray = vec3(0.0, 0.0, 0.0)
        self.up: np.ndarray = vec3(0.0, 1.0, 0.0)
        self.fov: float = 50.0
        self.aspect: float = 16.0 / 9.0
        self.near: float = 0.1
        self.far: float = 1000.0
        self.ortho: bool = False
        self.ortho_size: float = 10.0
        self.resolution: Tuple[int, int] = (1920, 1080)
        self.background_color: Tuple[float, float, float] = (0.1, 0.1, 0.15)

    # ── Matrix helpers ─────────────────────────────────────────────────────

    def get_view_matrix(self) -> np.ndarray:
        """Compute the view (camera-to-world → world-to-camera) matrix.

        Returns:
            4×4 float64 view matrix.
        """
        return mat4_look_at(self.position, self.target, self.up)

    def get_projection_matrix(self) -> np.ndarray:
        """Compute the projection matrix (perspective or orthographic).

        Returns:
            4×4 float64 projection matrix.
        """
        if self.ortho:
            half = self.ortho_size
            aspect = self.aspect
            from nexus3d.math3d.core import mat4_orthographic

            return mat4_orthographic(
                -half * aspect, half * aspect, -half, half, self.near, self.far
            )
        return mat4_perspective(self.fov, self.aspect, self.near, self.far)

    # ── Mutators ────────────────────────────────────────────────────────────

    def look_at(self, target: np.ndarray, position: Optional[np.ndarray] = None) -> None:
        """Point the camera at *target* from an optional *position*."""
        self.target = np.asarray(target, dtype=np.float64).copy()
        if position is not None:
            self.position = np.asarray(position, dtype=np.float64).copy()

    def set_resolution(self, width: int, height: int) -> None:
        """Update the pixel resolution and recompute the aspect ratio."""
        self.resolution = (int(width), int(height))
        self.aspect = width / max(height, 1)

    # ── Serialisation ───────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-data dictionary."""
        return {
            "name": self.name,
            "position": self.position.tolist(),
            "target": self.target.tolist(),
            "up": self.up.tolist(),
            "fov": self.fov,
            "aspect": self.aspect,
            "near": self.near,
            "far": self.far,
            "ortho": self.ortho,
            "ortho_size": self.ortho_size,
            "resolution": list(self.resolution),
            "background_color": list(self.background_color),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Camera":
        """Reconstruct a :class:`Camera` from a dictionary."""
        cam = cls(name=data.get("name", "Camera"))
        if "position" in data:
            cam.position = np.array(data["position"], dtype=np.float64)
        if "target" in data:
            cam.target = np.array(data["target"], dtype=np.float64)
        if "up" in data:
            cam.up = np.array(data["up"], dtype=np.float64)
        cam.fov = float(data.get("fov", 50.0))
        cam.aspect = float(data.get("aspect", 16.0 / 9.0))
        cam.near = float(data.get("near", 0.1))
        cam.far = float(data.get("far", 1000.0))
        cam.ortho = bool(data.get("ortho", False))
        cam.ortho_size = float(data.get("ortho_size", 10.0))
        if "resolution" in data:
            cam.resolution = tuple(data["resolution"])  # type: ignore[assignment]
        if "background_color" in data:
            cam.background_color = tuple(data["background_color"])  # type: ignore[assignment]
        return cam

    def __repr__(self) -> str:
        return (
            f"Camera(name={self.name!r}, fov={self.fov:.1f}, "
            f"resolution={self.resolution})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SoftwareRasterizer  (pure Python / NumPy fallback)
# ═══════════════════════════════════════════════════════════════════════════════

class SoftwareRasterizer:
    """Pure Python / NumPy software rasterizer for headless rendering.

    Provides a minimal but functional rendering pipeline:

    * Triangle rasterisation with z-buffer (depth testing).
    * Flat and Gouraud (per-vertex) shading.
    * Point and directional lights with simple diffuse + ambient + specular.
    * Optional edge wireframe overlay.

    This is deliberately kept dependency-light (only NumPy) so it can serve
    as a fallback on environments without OpenGL / OSMesa.
    """

    def __init__(self, width: int = 800, height: int = 600) -> None:
        self.width: int = width
        self.height: int = height
        self.color_buffer: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
        self.depth_buffer: np.ndarray = np.full((height, width), np.inf, dtype=np.float32)

    # ── Public API ──────────────────────────────────────────────────────────

    def clear(self, background_color: Tuple[float, float, float] = (0.1, 0.1, 0.15)) -> None:
        """Reset colour and depth buffers."""
        r, g, b = background_color
        self.color_buffer[:] = np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)
        self.depth_buffer[:] = np.inf

    def render_scene(
        self,
        meshes: List[Dict[str, Any]],
        camera: Camera,
        lights: Optional[List[Light]] = None,
    ) -> np.ndarray:
        """Render a list of mesh dictionaries into the colour buffer.

        Each *mesh* dict should contain:

        * ``'vertices'`` – (N, 3) float array of world-space positions.
        * ``'faces'`` – (F, 3) int array of triangle indices.
        * ``'normals'`` – (N, 3) float array of vertex normals (optional;
          face normals are computed automatically if missing).
        * ``'material'`` – a :class:`Material` instance or *None*.
        * ``'transform'`` – (4, 4) float64 model matrix (optional).

        Args:
            meshes: List of mesh dictionaries.
            camera: The :class:`Camera` controlling the viewpoint.
            lights: List of :class:`Light` objects.  Falls back to a single
                directional light from the upper-right-front if *None*.

        Returns:
            The rendered colour buffer as an (H, W, 3) uint8 array.
        """
        bg = camera.background_color if camera else (0.1, 0.1, 0.15)
        self.clear(bg)

        if lights is None:
            lights = [
                Light.directional(direction=normalize(vec3(-0.5, -1.0, -0.3)), intensity=1.0)
            ]

        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix()
        vp = proj @ view

        for mesh in meshes:
            self._render_mesh(mesh, vp, view, lights)

        return self.color_buffer.copy()

    def get_image(self) -> np.ndarray:
        """Return the current colour buffer as an (H, W, 3) uint8 array."""
        return self.color_buffer.copy()

    def save(self, filepath: str) -> None:
        """Save the rendered image to *filepath* (requires Pillow)."""
        if not _pil_available:
            raise RuntimeError("Pillow is required to save images from the software rasterizer.")
        img = Image.fromarray(self.color_buffer, mode="RGB")
        img.save(filepath)
        logger.info("Software rasterizer image saved to %s", filepath)

    # ── Internals ───────────────────────────────────────────────────────────

    def _render_mesh(
        self,
        mesh: Dict[str, Any],
        vp: np.ndarray,
        view: np.ndarray,
        lights: List[Light],
    ) -> None:
        """Rasterise a single mesh."""
        vertices = np.asarray(mesh["vertices"], dtype=np.float64)
        faces = np.asarray(mesh["faces"], dtype=np.int64)

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            return
        if faces.ndim != 2 or faces.shape[1] != 3:
            return

        # Apply model transform if present
        transform = mesh.get("transform")
        if transform is not None:
            transform = np.asarray(transform, dtype=np.float64)
            ones = np.ones((vertices.shape[0], 1), dtype=np.float64)
            verts_h = np.hstack([vertices, ones])
            verts_h = (transform @ verts_h.T).T
            vertices = verts_h[:, :3]

        # Normals
        normals = mesh.get("normals")
        if normals is None:
            # Compute flat face normals and assign to vertices
            face_normals = np.zeros_like(vertices)
            for face in faces:
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                n = normalize(cross(v1 - v0, v2 - v0))
                face_normals[face[0]] += n
                face_normals[face[1]] += n
                face_normals[face[2]] += n
            norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            normals = face_normals / norms
        else:
            normals = np.asarray(normals, dtype=np.float64)
            # Transform normals for non-uniform scale
            if transform is not None:
                normal_matrix = np.linalg.inv(transform).T
                ones = np.ones((normals.shape[0], 1), dtype=np.float64)
                n_h = np.hstack([normals, ones])
                n_h = (normal_matrix @ n_h.T).T
                normals = n_h[:, :3]
                norms = np.linalg.norm(normals, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0
                normals /= norms

        # Material
        material: Optional[Material] = mesh.get("material")
        if material is None:
            material = Material()

        # Compute effective light direction (combine directional lights)
        light_dir = np.zeros(3)
        ambient = 0.15
        for light in lights:
            if light.light_type == "directional":
                light_dir += normalize(-light.direction) * light.intensity
                ambient += 0.05 * light.intensity
            elif light.light_type == "point":
                # Approximate: use a constant direction from a point light
                # (true per-fragment shading would be more accurate)
                light_dir += normalize(-light.direction) * light.intensity

        light_dir = normalize(light_dir)

        # Transform vertices to clip space
        ones = np.ones((vertices.shape[0], 1), dtype=np.float64)
        verts_h = np.hstack([vertices, ones])
        clip = (vp @ verts_h.T).T

        # Perspective divide → NDC
        w = clip[:, 3:4]
        w[w == 0] = 1e-6
        ndc = clip[:, :3] / w

        # NDC → screen
        sx = ((ndc[:, 0] + 1.0) * 0.5 * self.width).astype(np.int32)
        sy = ((1.0 - ndc[:, 1]) * 0.5 * self.height).astype(np.int32)
        sz = ndc[:, 2]  # depth for z-buffer

        base_color_rgb = material.base_color[:3]
        emissive_rgb = material.emissive * material.emissive_strength

        for face in faces:
            i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])

            # Screen coords
            p0 = np.array([sx[i0], sy[i0]], dtype=np.float64)
            p1 = np.array([sx[i1], sy[i1]], dtype=np.float64)
            p2 = np.array([sx[i2], sy[i2]], dtype=np.float64)
            z0, z1, z2 = float(sz[i0]), float(sz[i1]), float(sz[i2])

            # Compute lighting per-vertex (Gouraud)
            colors_v = []
            for idx in (i0, i1, i2):
                n = normals[idx]
                # Diffuse
                diff = max(dot(n, light_dir), 0.0)
                # Specular (Blinn-Phong)
                view_dir_approx = normalize(vec3(0.0, 0.0, 1.0))
                half_vec = normalize(light_dir + view_dir_approx)
                spec = max(dot(n, half_vec), 0.0) ** (max((1.0 - material.roughness) * 64.0, 4.0))

                # Fresnel
                n_dot_v = max(dot(n, view_dir_approx), 0.0)
                fresnel = fresnel_schlick(n_dot_v, 0.04 + 0.96 * material.metallic)

                color = base_color_rgb * (ambient + diff * 0.7) + spec * fresnel * 0.3 + emissive_rgb
                colors_v.append(color)

            c0, c1, c2 = colors_v
            self._rasterize_triangle(
                p0, p1, p2, z0, z1, z2, c0, c1, c2,
            )

    def _rasterize_triangle(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        z0: float,
        z1: float,
        z2: float,
        c0: np.ndarray,
        c1: np.ndarray,
        c2: np.ndarray,
    ) -> None:
        """Rasterise a single triangle using a scanline approach with z-buffer.

        Args:
            p0, p1, p2: Screen-space (x, y) coordinates of the triangle.
            z0, z1, z2: Depth values at each vertex.
            c0, c1, c2: RGB colour vectors at each vertex.
        """
        # Bounding box clipped to screen
        min_x = max(int(min(p0[0], p1[0], p2[0])), 0)
        max_x = min(int(max(p0[0], p1[0], p2[0])) + 1, self.width)
        min_y = max(int(min(p0[1], p1[1], p2[1])), 0)
        max_y = min(int(max(p0[1], p1[1], p2[1])) + 1, self.height)

        if min_x >= max_x or min_y >= max_y:
            return

        # Precompute edge function denominator (2× signed area)
        denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(denom) < 1e-10:
            return  # Degenerate triangle

        inv_denom = 1.0 / denom

        # Total area for perspective-correct interpolation
        total_area = abs(denom) * 0.5

        # Vectorised over all pixels in bounding box
        ys, xs = np.mgrid[min_y:max_y, min_x:max_x]
        px = xs.astype(np.float64) + 0.5
        py = ys.astype(np.float64) + 0.5

        # Barycentric coordinates
        w0 = ((p1[1] - p2[1]) * (px - p2[0]) + (p2[0] - p1[0]) * (py - p2[1])) * inv_denom
        w1 = ((p2[1] - p0[1]) * (px - p2[0]) + (p0[0] - p2[0]) * (py - p2[1])) * inv_denom
        w2 = 1.0 - w0 - w1

        # Inside-triangle mask
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not np.any(inside):
            return

        # Interpolate depth
        z_interp = w0 * z0 + w1 * z1 + w2 * z2

        # Z-buffer test
        depth_test = z_interp < self.depth_buffer[min_y:max_y, min_x:max_x]
        mask = inside & depth_test
        if not np.any(mask):
            return

        # Interpolate colour (Gouraud)
        r = w0 * c0[0] + w1 * c1[0] + w2 * c2[0]
        g = w0 * c0[1] + w1 * c1[1] + w2 * c2[1]
        b = w0 * c0[2] + w1 * c1[2] + w2 * c2[2]

        # Clamp and quantise
        r_out = np.clip(r * 255.0, 0, 255).astype(np.uint8)
        g_out = np.clip(g * 255.0, 0, 255).astype(np.uint8)
        b_out = np.clip(b * 255.0, 0, 255).astype(np.uint8)

        # Write to buffers
        self.depth_buffer[min_y:max_y, min_x:max_x][mask] = z_interp[mask]
        self.color_buffer[min_y:max_y, min_x:max_x, 0][mask] = r_out[mask]
        self.color_buffer[min_y:max_y, min_x:max_x, 1][mask] = g_out[mask]
        self.color_buffer[min_y:max_y, min_x:max_x, 2][mask] = b_out[mask]


# ═══════════════════════════════════════════════════════════════════════════════
# Renderer  (headless, dual-backend)
# ═══════════════════════════════════════════════════════════════════════════════

class Renderer:
    """Headless renderer – renders scenes to images without a display.

    Supports two rendering backends:

    1. **PyRender (OSMesa)** – fast, GPU-quality, physically-based rendering
       when PyRender and OSMesa are installed.
    2. **SoftwareRasterizer** – a pure-Python / NumPy fallback that works
       everywhere at the cost of visual fidelity and performance.

    The backend is selected at construction time.  ``backend='auto'``
    (default) tries PyRender first and silently falls back to software
    rendering when OpenGL is not available.

    Args:
        backend: ``'auto'``, ``'pyrender'``, or ``'software'``.
        resolution: ``(width, height)`` in pixels for the default render
            target.
    """

    def __init__(self, backend: str = "auto", resolution: Tuple[int, int] = (1920, 1080)) -> None:
        self.resolution: Tuple[int, int] = resolution
        self._backend: str = "software"
        self._pyrender_renderer: Any = None
        self._scene: Any = None

        if backend == "auto":
            self._try_pyrender()
            if self._backend != "pyrender":
                logger.info("PyRender unavailable – falling back to software rasterizer.")
        elif backend == "pyrender":
            self._try_pyrender()
            if self._backend != "pyrender":
                raise RuntimeError(
                    "PyRender backend requested but OSMesa / PyRender is not available. "
                    "Install pyrender and osmesa, or use backend='software'."
                )
        elif backend == "software":
            self._backend = "software"
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'auto', 'pyrender', or 'software'.")

        logger.info("Nexus3D Renderer initialised – backend=%s, resolution=%s", self._backend, self.resolution)

    # ── Backend initialisation ──────────────────────────────────────────────

    def _try_pyrender(self) -> None:
        """Attempt to initialise the PyRender (OSMesa) backend."""
        if not _pyrender_available:
            return
        try:
            w, h = self.resolution
            self._pyrender_renderer = pyrender.OffscreenRenderer(w, h)
            self._scene = pyrender.Scene(
                bg_color=[0.1, 0.1, 0.15, 1.0],
                ambient_light=[0.15, 0.15, 0.15],
            )
            self._backend = "pyrender"
            logger.info("PyRender backend initialised successfully.")
        except Exception as exc:
            logger.warning("Failed to initialise PyRender: %s", exc)
            self._pyrender_renderer = None
            self._scene = None
            self._backend = "software"

    # ── Public rendering API ────────────────────────────────────────────────

    def render(
        self,
        scene_data: Dict[str, Any],
        camera: Optional[Camera] = None,
        lights: Optional[List[Light]] = None,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """Render a scene to an image array.

        Args:
            scene_data: Dictionary describing the scene.  Expected keys:

                * ``'meshes'`` – list of dicts, each with ``'vertices'``,
                  ``'faces'``, optional ``'normals'``, ``'uvs'``,
                  ``'material'``, and ``'transform'``.
                * ``'armatures'`` – *(optional)* list of armature dicts for
                  debug visualisation.

            camera: A :class:`Camera` instance.  A sensible default camera
                is created if *None*.
            lights: List of :class:`Light` objects.  A standard 3-point
                lighting rig is created if *None*.
            output_path: If given, the image is also written to this file
                path.

        Returns:
            ``(H, W, 3)`` uint8 NumPy array (RGB).
        """
        if camera is None:
            camera = Camera()
        if lights is None:
            lights = self.create_default_lights()

        if self._backend == "pyrender":
            image = self._render_pyrender(scene_data, camera, lights)
        else:
            image = self._render_software(scene_data, camera, lights)

        if output_path is not None:
            self._save_image(image, output_path)

        return image

    def render_to_file(
        self,
        scene_data: Dict[str, Any],
        filepath: str,
        camera: Optional[Camera] = None,
        lights: Optional[List[Light]] = None,
    ) -> np.ndarray:
        """Convenience: render and immediately save to *filepath*."""
        return self.render(scene_data, camera=camera, lights=lights, output_path=filepath)

    def render_animation(
        self,
        scene_data: List[Dict[str, Any]],
        output_dir: str,
        camera: Optional[Camera] = None,
        lights: Optional[List[Light]] = None,
        prefix: str = "frame_",
        fmt: str = "png",
    ) -> List[str]:
        """Render multiple frames for an animation sequence.

        Args:
            scene_data: List of scene dictionaries, one per frame.
            output_dir: Directory where frame images are saved.
            camera: Camera shared across all frames.
            lights: Lights shared across all frames.
            prefix: Filename prefix (e.g. ``'frame_'`` → ``frame_0000.png``).
            fmt: Image format extension (``'png'``, ``'jpg'``, etc.).

        Returns:
            List of written file-paths, one per frame.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths: List[str] = []
        total = len(scene_data)
        for idx, frame in enumerate(scene_data):
            filename = f"{prefix}{idx:06d}.{fmt}"
            filepath = os.path.join(output_dir, filename)
            self.render(frame, camera=camera, lights=lights, output_path=filepath)
            paths.append(filepath)
            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                logger.info("Rendered frame %d / %d → %s", idx + 1, total, filepath)
        return paths

    # ── Backend-specific rendering ──────────────────────────────────────────

    def _render_pyrender(
        self,
        scene_data: Dict[str, Any],
        camera: Camera,
        lights: List[Light],
    ) -> np.ndarray:
        """Render using the PyRender (OSMesa) backend.

        Returns:
            ``(H, W, 3)`` uint8 image.
        """
        import trimesh

        # Rebuild the scene each call (safe for animated meshes)
        pr_scene = pyrender.Scene(
            bg_color=list(camera.background_color) + [1.0],
            ambient_light=[0.15, 0.15, 0.15],
        )

        # ── Meshes ──────────────────────────────────────────────────────────
        for mesh_data in scene_data.get("meshes", []):
            vertices = np.asarray(mesh_data["vertices"], dtype=np.float64)
            faces = np.asarray(mesh_data["faces"], dtype=np.int64)

            try:
                tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

                normals = mesh_data.get("normals")
                if normals is not None:
                    tri_mesh.vertex_normals = np.asarray(normals, dtype=np.float64)

                material = mesh_data.get("material")
                if material is None:
                    material = Material()

                # Build a simple PBR-friendly visual material for trimesh
                visual_mat = trimesh.visual.material.SimpleMaterial(
                    base_color=material.base_color,
                )

                tri_mesh.visual.material = visual_mat

                # Apply transform
                transform = mesh_data.get("transform")
                if transform is not None:
                    transform = np.asarray(transform, dtype=np.float64)
                else:
                    transform = mat4_identity()

                pr_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)
                pr_scene.add(pr_mesh, pose=transform)
            except Exception as exc:
                logger.warning("Failed to add mesh to PyRender scene: %s", exc)

        # ── Lights ──────────────────────────────────────────────────────────
        for light in lights:
            try:
                light_color = [float(light.color[0]), float(light.color[1]), float(light.color[2])]
                if light.light_type == "directional":
                    pr_light = pyrender.DirectionalLight(color=light_color, intensity=light.intensity * 3.0)
                    # Position the light far away along its negative direction
                    pose = mat4_identity()
                    pose[:3, :3] = self._direction_to_rotation(light.direction)
                    pose[:3, 3] = -normalize(light.direction) * 50.0
                    pr_scene.add(pr_light, pose=pose)
                elif light.light_type == "point":
                    pr_light = pyrender.PointLight(color=light_color, intensity=light.intensity * 3.0)
                    pose = mat4_translate(
                        float(light.position[0]),
                        float(light.position[1]),
                        float(light.position[2]),
                    )
                    pr_scene.add(pr_light, pose=pose)
                elif light.light_type == "spot":
                    pr_light = pyrender.SpotLight(
                        color=light_color,
                        intensity=light.intensity * 3.0,
                        innerConeAngle=light.spot_angle * 0.8,
                        outerConeAngle=light.spot_angle,
                    )
                    pose = mat4_translate(
                        float(light.position[0]),
                        float(light.position[1]),
                        float(light.position[2]),
                    )
                    pose[:3, :3] = self._direction_to_rotation(light.direction)
                    pr_scene.add(pr_light, pose=pose)
                else:
                    logger.debug("Unsupported light type '%s' – skipping.", light.light_type)
            except Exception as exc:
                logger.warning("Failed to add light to PyRender scene: %s", exc)

        # ── Camera ──────────────────────────────────────────────────────────
        camera_node = pr_scene.add(
            pyrender.PerspectiveCamera(yfov=math.radians(camera.fov), aspectRatio=camera.aspect),
        )

        # Build camera pose from look-at
        cam_pose = mat4_identity()
        view_matrix = camera.get_view_matrix()
        cam_pose = np.linalg.inv(view_matrix)

        pr_scene.set_pose(camera_node, cam_pose)

        # ── Render ──────────────────────────────────────────────────────────
        w, h = camera.resolution
        # Resize the offscreen renderer if needed
        if (w, h) != self.resolution:
            self._pyrender_renderer = pyrender.OffscreenRenderer(w, h)
            self.resolution = (w, h)

        colour, _depth = self._pyrender_renderer.render(pr_scene)
        return np.asarray(colour, dtype=np.uint8)

    def _render_software(
        self,
        scene_data: Dict[str, Any],
        camera: Camera,
        lights: List[Light],
    ) -> np.ndarray:
        """Render using the :class:`SoftwareRasterizer` fallback.

        Returns:
            ``(H, W, 3)`` uint8 image.
        """
        w, h = camera.resolution
        rasterizer = SoftwareRasterizer(width=w, height=h)
        meshes = scene_data.get("meshes", [])
        return rasterizer.render_scene(meshes, camera, lights)

    # ── Utility methods ─────────────────────────────────────────────────────

    @staticmethod
    def _direction_to_rotation(direction: np.ndarray) -> np.ndarray:
        """Build a 3×3 rotation matrix that aligns -Z with *direction*.

        This is useful for placing directional / spot lights in PyRender
        which expect the light to shine along its local -Z axis.
        """
        fwd = normalize(direction)
        if abs(dot(fwd, vec3(0, 1, 0))) > 0.999:
            up = vec3(1.0, 0.0, 0.0)
        else:
            up = vec3(0.0, 1.0, 0.0)
        right = normalize(cross(fwd, up))
        true_up = cross(right, fwd)
        rot = np.eye(3, dtype=np.float64)
        rot[:, 0] = right
        rot[:, 1] = true_up
        rot[:, 2] = fwd
        return rot

    @staticmethod
    def create_default_lights() -> List[Light]:
        """Create a standard 3-point lighting rig.

        Returns:
            ``[key_light, fill_light, rim_light]``
        """
        key = Light.directional(
            direction=normalize(vec3(-0.5, -1.0, -0.3)),
            color="#fff5e6",
            intensity=1.0,
            name="KeyLight",
        )
        fill = Light.directional(
            direction=normalize(vec3(0.5, -0.5, 0.5)),
            color="#e6f0ff",
            intensity=0.5,
            name="FillLight",
        )
        rim = Light.directional(
            direction=normalize(vec3(0.0, -0.3, 1.0)),
            color="#ffe6f0",
            intensity=0.7,
            name="RimLight",
        )
        return [key, fill, rim]

    @staticmethod
    def frames_to_video(
        image_paths: List[str],
        output_path: str,
        fps: int = 30,
    ) -> str:
        """Combine rendered frames into a video file.

        Requires ``imageio`` (with FFmpeg backend).

        Args:
            image_paths: Ordered list of image file-paths.
            output_path: Destination video file (e.g. ``'output.mp4'``).
            fps: Frames per second.

        Returns:
            The *output_path* that was written.

        Raises:
            RuntimeError: If ``imageio`` is not available.
        """
        if not _imageio_available:
            raise RuntimeError(
                "imageio is required for video export.  Install it with: pip install imageio[ffmpeg]"
            )
        if not image_paths:
            raise ValueError("image_paths must not be empty.")

        writer = imageio.get_writer(output_path, fps=fps)
        for path in image_paths:
            frame = imageio.imread(path)
            writer.append_data(frame)
        writer.close()
        logger.info("Video saved to %s (%d frames, %d fps)", output_path, len(image_paths), fps)
        return output_path

    # ── Image I/O helper ────────────────────────────────────────────────────

    @staticmethod
    def _save_image(image: np.ndarray, filepath: str) -> None:
        """Write an (H, W, 3) uint8 image to disk."""
        if _pil_available:
            img = Image.fromarray(image, mode="RGB")
            img.save(filepath)
        elif _imageio_available:
            imageio.imwrite(filepath, image)
        else:
            raise RuntimeError("Neither Pillow nor imageio is available for image saving.")
        logger.info("Image saved to %s", filepath)

    def __repr__(self) -> str:
        return f"Renderer(backend={self._backend!r}, resolution={self.resolution})"
