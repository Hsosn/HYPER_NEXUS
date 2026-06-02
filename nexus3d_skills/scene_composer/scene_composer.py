#!/usr/bin/env python3
"""
Nexus3D Advanced Skill: Scene Composer
=======================================
Builds complete 3D scenes with objects, materials, lighting rigs, cameras,
and environment settings. Designed for AI agents to compose complex scenes.

Upgraded v0.3.0 — Adds:
  - Procedural scatter (grid, circle, spiral, random, terrain)
  - Object grouping / hierarchies
  - Auto-composition (rule-of-thirds camera, subject framing)
  - Material override system (per-object, per-group, global)
  - More scene templates (product, landscape, interior, abstract)
  - LOD generation for distant objects

Usage:
    python scene_composer.py --help
    python scene_composer.py create --template "studio" --output scene.json
    python scene_composer.py add-object --scene scene.json --type sphere --position "0,1,0"
    python scene_composer.py scatter --scene scene.json --source Tree --pattern terrain --count 50
    python scene_composer.py auto-compose --scene scene.json --subject "MainObject"
    python scene_composer.py apply-material --scene scene.json --material "chrome" --group "all"
    python scene_composer.py render --scene scene.json --camera "main" --output render.png
"""

import sys
import os
import json
import math
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nexus3d.math3d.core import *
from nexus3d.mesh.primitives import *
from nexus3d.mesh.mesh import Mesh
from nexus3d.rendering.renderer import Renderer, Camera, Light, Material
from nexus3d.utils.helpers import save_json, load_json, ensure_output_dir, NumpyEncoder


# ─── Enums / Constants ──────────────────────────────────────────────────────

class GroupOp(Enum):
    UNION = "union"
    MERGE = "merge"
    GROUP = "group"


# ─── Scene Templates ─────────────────────────────────────────────────────────

SCENE_TEMPLATES = {
    "empty": {
        "description": "Empty scene with ground plane and default lighting",
        "objects": [
            {"type": "plane", "name": "Ground", "params": {"width": 20, "height": 20},
             "position": [0, 0, 0], "material": {"type": "pbr", "color": "#444444", "roughness": 0.9}},
        ],
        "lights": [
            {"type": "directional", "name": "Sun", "direction": [-1, -1, -0.5],
             "color": "#ffffff", "intensity": 1.0},
        ],
        "cameras": [
            {"name": "main", "position": [0, 3, 8], "target": [0, 0.5, 0], "fov": 45},
        ],
        "environment": {"background_color": [0.05, 0.05, 0.08], "ambient": 0.2},
    },
    "studio": {
        "description": "Professional studio setup with 3-point lighting",
        "objects": [
            {"type": "plane", "name": "Ground", "params": {"width": 30, "height": 30},
             "position": [0, 0, 0], "material": {"type": "pbr", "color": "#333333", "roughness": 0.95}},
            {"type": "plane", "name": "BackWall", "params": {"width": 20, "height": 10},
             "position": [0, 5, -10], "rotation": [90, 0, 0],
             "material": {"type": "pbr", "color": "#222222", "roughness": 1.0}},
            {"type": "plane", "name": "LeftWall", "params": {"width": 20, "height": 10},
             "position": [-10, 5, 0], "rotation": [0, 0, 90],
             "material": {"type": "pbr", "color": "#1a1a1a", "roughness": 1.0}},
        ],
        "lights": [
            {"type": "directional", "name": "KeyLight", "direction": [-0.5, -1, -0.8],
             "color": "#fff5e6", "intensity": 1.2},
            {"type": "directional", "name": "FillLight", "direction": [0.8, -0.5, -0.5],
             "color": "#e6f0ff", "intensity": 0.6},
            {"type": "directional", "name": "RimLight", "direction": [0, -0.5, 1],
             "color": "#ffe6f0", "intensity": 0.8},
            {"type": "directional", "name": "TopLight", "direction": [0, -1, 0],
             "color": "#ffffff", "intensity": 0.3},
        ],
        "cameras": [
            {"name": "front", "position": [0, 3, 8], "target": [0, 1, 0], "fov": 40},
            {"name": "side", "position": [8, 3, 0], "target": [0, 1, 0], "fov": 40},
            {"name": "top", "position": [0, 10, 0.01], "target": [0, 0, 0], "fov": 50},
            {"name": "closeup", "position": [0, 1.5, 3], "target": [0, 1, 0], "fov": 35},
        ],
        "environment": {"background_color": [0.02, 0.02, 0.03], "ambient": 0.15},
    },
    "product": {
        "description": "Clean product showcase with turntable-ready lighting",
        "objects": [
            {"type": "plane", "name": "Ground", "params": {"width": 10, "height": 10},
             "position": [0, 0, 0], "material": {"type": "pbr", "color": "#f5f5f5", "roughness": 0.5, "metallic": 0.0}},
        ],
        "lights": [
            {"type": "directional", "name": "Key", "direction": [-0.4, -1, -0.6],
             "color": "#ffffff", "intensity": 1.4},
            {"type": "directional", "name": "Fill", "direction": [0.6, -0.3, -0.5],
             "color": "#e8f0ff", "intensity": 0.5},
            {"type": "directional", "name": "Rim", "direction": [0.2, -0.4, 1],
             "color": "#ffffff", "intensity": 0.9},
            {"type": "point", "name": "TopFill", "position": [0, 6, 0],
             "color": "#ffffff", "intensity": 0.3, "range": 15},
        ],
        "cameras": [
            {"name": "front", "position": [0, 2, 5], "target": [0, 0.8, 0], "fov": 35},
            {"name": "three_quarter", "position": [3, 2.5, 4], "target": [0, 0.8, 0], "fov": 35},
        ],
        "environment": {"background_color": [0.95, 0.95, 0.98], "ambient": 0.3},
    },
    "outdoor": {
        "description": "Outdoor scene with sunlight and environment",
        "objects": [
            {"type": "plane", "name": "Ground", "params": {"width": 100, "height": 100},
             "position": [0, 0, 0], "material": {"type": "pbr", "color": "#3a5f3a", "roughness": 0.95}},
        ],
        "lights": [
            {"type": "directional", "name": "Sun", "direction": [-0.3, -1, -0.5],
             "color": "#fff8e1", "intensity": 1.5},
            {"type": "directional", "name": "SkyLight", "direction": [0, -1, 0],
             "color": "#87ceeb", "intensity": 0.4},
        ],
        "cameras": [
            {"name": "main", "position": [5, 3, 10], "target": [0, 1, 0], "fov": 50},
        ],
        "environment": {"background_color": [0.4, 0.6, 0.8], "ambient": 0.35},
    },
    "dramatic": {
        "description": "Dark dramatic lighting with strong contrast",
        "objects": [
            {"type": "plane", "name": "Ground", "params": {"width": 20, "height": 20},
             "position": [0, 0, 0], "material": {"type": "pbr", "color": "#111111", "roughness": 0.8}},
        ],
        "lights": [
            {"type": "spot", "name": "SpotMain", "position": [3, 8, 5],
             "direction": [-0.3, -1, -0.5], "color": "#ff6600", "intensity": 2.0,
             "spot_angle": 35, "range": 20},
            {"type": "point", "name": "RimPoint", "position": [-3, 4, -2],
             "color": "#0066ff", "intensity": 1.5, "range": 15},
        ],
        "cameras": [
            {"name": "main", "position": [0, 2, 6], "target": [0, 1, 0], "fov": 40},
        ],
        "environment": {"background_color": [0.01, 0.01, 0.02], "ambient": 0.05},
    },
    "interior": {
        "description": "Cozy interior room with warm lighting",
        "objects": [
            {"type": "plane", "name": "Floor", "params": {"width": 8, "height": 10},
             "position": [0, 0, 0], "material": {"type": "pbr", "color": "#5c3a1e", "roughness": 0.8}},
            {"type": "plane", "name": "BackWall", "params": {"width": 8, "height": 4},
             "position": [0, 2, -5], "rotation": [90, 0, 0],
             "material": {"type": "pbr", "color": "#f5e6d3", "roughness": 0.9}},
            {"type": "plane", "name": "LeftWall", "params": {"width": 10, "height": 4},
             "position": [-4, 2, 0], "rotation": [0, 0, 90],
             "material": {"type": "pbr", "color": "#e8dcc8", "roughness": 0.9}},
        ],
        "lights": [
            {"type": "point", "name": "CeilingLight", "position": [0, 3.8, 0],
             "color": "#fff5d6", "intensity": 0.8, "range": 8},
            {"type": "point", "name": "Lamp", "position": [2.5, 1.2, 0.5],
             "color": "#ffcc66", "intensity": 0.6, "range": 5},
        ],
        "cameras": [
            {"name": "wide", "position": [0, 1.5, 4.5], "target": [0, 1, -1], "fov": 60},
        ],
        "environment": {"background_color": [0.1, 0.1, 0.12], "ambient": 0.15},
    },
    "landscape": {
        "description": "Natural landscape with terrain",
        "objects": [
            {"type": "plane", "name": "Terrain", "params": {"width": 200, "height": 200},
             "position": [0, 0, 0], "material": {"type": "pbr", "color": "#4a7a3a", "roughness": 1.0}},
        ],
        "lights": [
            {"type": "directional", "name": "Sun", "direction": [-0.3, -1, -0.5],
             "color": "#fffae0", "intensity": 1.8},
            {"type": "directional", "name": "Sky", "direction": [0, -1, 0],
             "color": "#88bbdd", "intensity": 0.5},
        ],
        "cameras": [
            {"name": "overview", "position": [20, 15, 30], "target": [0, 0, 0], "fov": 45},
        ],
        "environment": {"background_color": [0.5, 0.7, 0.9], "ambient": 0.4},
    },
    "abstract": {
        "description": "Abstract art scene with glowing shapes",
        "objects": [
            {"type": "sphere", "name": "Core", "params": {"radius": 0.8, "rings": 48, "segments": 48},
             "position": [0, 0, 0], "material": {"type": "emissive", "color": "#ff0066", "emissive_strength": 3.0}},
            {"type": "torus", "name": "Ring1", "params": {"major_radius": 2.0, "minor_radius": 0.05},
             "position": [0, 0, 0], "rotation": [45, 0, 0],
             "material": {"type": "pbr", "color": "#00ffcc", "metallic": 1.0, "roughness": 0.1}},
            {"type": "torus", "name": "Ring2", "params": {"major_radius": 2.8, "minor_radius": 0.03},
             "position": [0, 0, 0], "rotation": [0, 45, 30],
             "material": {"type": "pbr", "color": "#ff00ff", "metallic": 0.8, "roughness": 0.0}},
        ],
        "lights": [
            {"type": "point", "name": "Glow1", "position": [0, 3, 3],
             "color": "#ff0066", "intensity": 2.0, "range": 10},
            {"type": "point", "name": "Glow2", "position": [3, -1, -2],
             "color": "#00ffcc", "intensity": 2.0, "range": 10},
        ],
        "cameras": [
            {"name": "main", "position": [5, 2, 5], "target": [0, 0, 0], "fov": 35},
        ],
        "environment": {"background_color": [0.01, 0.01, 0.02], "ambient": 0.05},
    },
}


# ─── Material Library ────────────────────────────────────────────────────────

MATERIAL_PRESETS = {
    "default": {"type": "pbr", "color": "#888888", "roughness": 0.5, "metallic": 0.0},
    "chrome": {"type": "pbr", "color": "#cccccc", "metallic": 1.0, "roughness": 0.05},
    "gold": {"type": "pbr", "color": "#ffd700", "metallic": 1.0, "roughness": 0.15},
    "copper": {"type": "pbr", "color": "#b87333", "metallic": 1.0, "roughness": 0.25},
    "plastic_red": {"type": "pbr", "color": "#cc0000", "metallic": 0.0, "roughness": 0.4},
    "plastic_white": {"type": "pbr", "color": "#f0f0f0", "metallic": 0.0, "roughness": 0.35},
    "rubber": {"type": "pbr", "color": "#222222", "metallic": 0.0, "roughness": 0.9},
    "glass": {"type": "glass", "transmission": 0.9, "ior": 1.5, "roughness": 0.0},
    "wood": {"type": "pbr", "color": "#8b6914", "metallic": 0.0, "roughness": 0.7},
    "stone": {"type": "pbr", "color": "#808080", "metallic": 0.0, "roughness": 0.9},
    "emissive_red": {"type": "emissive", "color": "#ff0000", "emissive_strength": 2.0},
    "emissive_blue": {"type": "emissive", "color": "#0044ff", "emissive_strength": 2.0},
}


# ─── Scene Object ────────────────────────────────────────────────────────────

class SceneObject:
    """A single object in the scene. Supports grouping/hierarchies."""

    def __init__(self, name: str, obj_type: str = "cube",
                 position=None, rotation=None, scale=None,
                 params=None, material=None, mesh_data=None,
                 parent=None):
        self.name = name
        self.obj_type = obj_type
        self.position = np.array(position or [0, 0, 0], dtype=np.float64)
        self.rotation_euler = np.array(rotation or [0, 0, 0], dtype=np.float64)  # degrees
        self.scale_val = np.array(scale or [1, 1, 1], dtype=np.float64)
        self.params = params or {}
        self.base_material = material or MATERIAL_PRESETS["default"].copy()
        self._override_material = None
        self._mesh_data = mesh_data
        self._mesh = None
        self.parent = parent
        self.children: List['SceneObject'] = []

    def add_child(self, child: 'SceneObject'):
        child.parent = self
        self.children.append(child)

    def get_world_position(self) -> np.ndarray:
        if self.parent:
            return self.parent.get_world_position() + self.position
        return self.position.copy()

    def get_effective_material(self):
        if self._override_material:
            return self._override_material
        if self.parent and self.parent._override_material:
            return self.parent._override_material
        return self.base_material

    def get_effective_scale(self) -> np.ndarray:
        if self.parent:
            return self.parent.get_effective_scale() * self.scale_val
        return self.scale_val.copy()

    def apply_material_override(self, material: dict):
        self._override_material = material

    def clear_material_override(self):
        self._override_material = None

    @property
    def mesh(self) -> Mesh:
        """Lazily generate mesh from primitive."""
        if self._mesh is None:
            if self._mesh_data:
                self._mesh = Mesh.from_primitive(self.name, self._mesh_data)
            else:
                prim_map = {
                    "cube": lambda: create_cube(size=self.params.get("size", 1.0)),
                    "sphere": lambda: create_sphere(
                        radius=self.params.get("radius", 1.0),
                        rings=self.params.get("rings", 32),
                        segments=self.params.get("segments", 32)),
                    "cylinder": lambda: create_cylinder(
                        radius=self.params.get("radius", 0.5),
                        height=self.params.get("height", 2.0),
                        segments=self.params.get("segments", 32)),
                    "cone": lambda: create_cone(
                        radius=self.params.get("radius", 0.5),
                        height=self.params.get("height", 2.0),
                        segments=self.params.get("segments", 32)),
                    "torus": lambda: create_torus(
                        major_radius=self.params.get("major_radius", 1.0),
                        minor_radius=self.params.get("minor_radius", 0.3)),
                    "plane": lambda: create_plane(
                        width=self.params.get("width", 10.0),
                        height=self.params.get("height", 10.0),
                        width_segments=self.params.get("width_segments", 1),
                        height_segments=self.params.get("height_segments", 1)),
                    "arrow": lambda: create_arrow(length=self.params.get("length", 1.0)),
                }
                if self.obj_type in prim_map:
                    data = prim_map[self.obj_type]()
                    self._mesh = Mesh.from_primitive(self.name, data)
                else:
                    raise ValueError(f"Unknown object type: {self.obj_type}")

            # Apply transforms
            from nexus3d.math3d.core import euler_to_quat, compose_matrix
            t = self.position
            r = euler_to_quat(np.radians(self.rotation_euler))
            s = self.get_effective_scale()
            self._mesh.apply_transform(compose_matrix(t, r, s))
            self._mesh.compute_normals()

        return self._mesh

    def generate_lod(self, level: int = 1, reduction: float = 0.5) -> Optional[Mesh]:
        """Generate a lower-detail version of this object's mesh."""
        if level == 0 or self._mesh is None:
            return self._mesh
        # Simple LOD: decimate by reducing face count
        m = self._mesh
        n_verts = len(m.vertices)
        n_faces = len(m.faces)
        if n_faces < 12:
            return m
        target_faces = max(int(n_faces * (reduction ** level)), 12)
        step = max(1, n_faces // target_faces)
        lod_faces = m.faces[::step][:target_faces]
        from nexus3d.mesh.mesh import Mesh
        return Mesh(name=f"{self.name}_LOD{level}", vertices=m.vertices, faces=lod_faces)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.obj_type,
            "position": self.position.tolist(),
            "rotation": self.rotation_euler.tolist(),
            "scale": self.scale_val.tolist(),
            "params": self.params,
            "material": self.base_material,
            "override_material": self._override_material,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SceneObject':
        obj = cls(
            name=data["name"],
            obj_type=data.get("type", "cube"),
            position=data.get("position"),
            rotation=data.get("rotation"),
            scale=data.get("scale"),
            params=data.get("params"),
            material=data.get("material"),
        )
        if data.get("override_material"):
            obj._override_material = data["override_material"]
        for child_data in data.get("children", []):
            child = cls.from_dict(child_data)
            obj.add_child(child)
        return obj


# ─── Scatter Engine ──────────────────────────────────────────────────────────

class ScatterEngine:
    """Procedural object scattering on surfaces and volumes."""

    @staticmethod
    def scatter_on_terrain(source: SceneObject, count: int, width: float, depth: float,
                           height_range: tuple = (0, 0), avoid_radius: float = 0.5,
                           seed: int = 42) -> List[SceneObject]:
        """Scatter objects randomly on a terrain plane with avoidance."""
        rng = random.Random(seed)
        results = []
        placed = []
        h_min, h_max = height_range

        for i in range(count):
            for attempt in range(50):
                x = rng.uniform(-width / 2, width / 2)
                z = rng.uniform(-depth / 2, depth / 2)

                # Avoidance check
                ok = True
                for px, pz in placed:
                    if math.hypot(x - px, z - pz) < avoid_radius:
                        ok = False
                        break
                if not ok:
                    continue

                y = rng.uniform(h_min, h_max)
                placed.append((x, z))

                obj = SceneObject(
                    name=f"{source.name}_scatter_{i}",
                    obj_type=source.obj_type,
                    position=[x, y, z],
                    rotation=[0, rng.uniform(0, 360), 0],
                    scale=[rng.uniform(0.8, 1.2) * s for s in source.scale_val],
                    params=source.params.copy(),
                    material=source.base_material.copy(),
                )
                results.append(obj)
                break

        return results

    @staticmethod
    def scatter_grid(source: SceneObject, count_x: int, count_z: int, spacing: float,
                     jitter: float = 0.0, seed: int = 42) -> List[SceneObject]:
        """Scatter objects in a grid pattern with optional jitter."""
        rng = random.Random(seed)
        results = []

        for ix in range(count_x):
            for iz in range(count_z):
                jx = rng.uniform(-jitter, jitter) if jitter > 0 else 0
                jz = rng.uniform(-jitter, jitter) if jitter > 0 else 0
                x = (ix - (count_x - 1) / 2) * spacing + jx
                z = (iz - (count_z - 1) / 2) * spacing + jz

                obj = SceneObject(
                    name=f"{source.name}_grid_{ix}_{iz}",
                    obj_type=source.obj_type,
                    position=[x, 0, z],
                    rotation=[0, rng.uniform(0, 360), 0],
                    params=source.params.copy(),
                    material=source.base_material.copy(),
                )
                results.append(obj)

        return results

    @staticmethod
    def scatter_circle(source: SceneObject, count: int, radius: float,
                       seed: int = 42) -> List[SceneObject]:
        """Scatter objects in a circle."""
        rng = random.Random(seed)
        results = []

        for i in range(count):
            angle = 2 * math.pi * i / count + rng.uniform(-0.05, 0.05)
            r = radius * rng.uniform(0.9, 1.1)
            x = math.cos(angle) * r
            z = math.sin(angle) * r

            obj = SceneObject(
                name=f"{source.name}_circle_{i}",
                obj_type=source.obj_type,
                position=[x, 0, z],
                rotation=[0, math.degrees(angle) + 90, 0],
                params=source.params.copy(),
                material=source.base_material.copy(),
            )
            results.append(obj)

        return results


# ─── Scene Composer ──────────────────────────────────────────────────────────

class SceneComposer:
    """Compose and manage 3D scenes."""

    def __init__(self, name: str = "Scene"):
        self.name = name
        self.objects: Dict[str, SceneObject] = {}
        self.groups: Dict[str, List[str]] = {}
        self.lights: List[dict] = []
        self.cameras: Dict[str, dict] = {}
        self.environment = {"background_color": [0.05, 0.05, 0.08], "ambient": 0.2}
        self.global_material_override: Optional[dict] = None

    # ── Scene creation ──────────────────────────────────────────────────────

    @classmethod
    def from_template(cls, template_name: str, name: str = "Scene") -> 'SceneComposer':
        if template_name not in SCENE_TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(SCENE_TEMPLATES.keys())}")

        tmpl = SCENE_TEMPLATES[template_name]
        scene = cls(name=name)

        for obj_data in tmpl.get("objects", []):
            obj = SceneObject(
                name=obj_data["name"],
                obj_type=obj_data["type"],
                position=obj_data.get("position"),
                rotation=obj_data.get("rotation"),
                params=obj_data.get("params"),
                material=obj_data.get("material"),
            )
            scene.objects[obj.name] = obj

        scene.lights = tmpl.get("lights", [])
        for cam_data in tmpl.get("cameras", []):
            scene.cameras[cam_data["name"]] = cam_data
        scene.environment = tmpl.get("environment", scene.environment)

        return scene

    @classmethod
    def from_file(cls, filepath: str) -> 'SceneComposer':
        data = load_json(filepath)
        scene = cls(name=data.get("name", "Scene"))

        for obj_data in data.get("objects", []):
            obj = SceneObject.from_dict(obj_data)
            scene.objects[obj.name] = obj

        scene.lights = data.get("lights", [])
        scene.cameras = data.get("cameras", {})
        scene.environment = data.get("environment", scene.environment)
        scene.groups = data.get("groups", {})

        return scene

    # ── Object management ───────────────────────────────────────────────────

    def add_object(self, name: str, obj_type: str = "cube",
                   position=None, rotation=None, scale=None,
                   params=None, material=None,
                   parent_name: str = None) -> SceneObject:
        obj = SceneObject(name, obj_type, position, rotation, scale, params, material)
        if parent_name and parent_name in self.objects:
            self.objects[parent_name].add_child(obj)
        self.objects[name] = obj
        return obj

    def remove_object(self, name: str):
        if name in self.objects:
            obj = self.objects[name]
            if obj.parent:
                obj.parent.children = [c for c in obj.parent.children if c.name != name]
            del self.objects[name]

    def get_by_group(self, group_name: str) -> List[SceneObject]:
        names = self.groups.get(group_name, [])
        return [self.objects[n] for n in names if n in self.objects]

    # ── Lighting ────────────────────────────────────────────────────────────

    def add_light(self, light_data: dict):
        self.lights.append(light_data)

    def add_camera(self, name: str, position=None, target=None, fov=45.0):
        self.cameras[name] = {
            "name": name,
            "position": position or [0, 3, 8],
            "target": target or [0, 0, 0],
            "fov": fov,
        }

    def set_environment(self, background_color=None, ambient=None):
        if background_color:
            self.environment["background_color"] = background_color
        if ambient is not None:
            self.environment["ambient"] = ambient

    # ── Duplication ────────────────────────────────────────────────────────

    def duplicate_object(self, source_name: str, new_name: str,
                         position_offset=None, scale_factor=None) -> SceneObject:
        if source_name not in self.objects:
            raise ValueError(f"Object '{source_name}' not found")

        src = self.objects[source_name]
        new_pos = src.position + np.array(position_offset or [0, 0, 0])
        new_scale = src.scale_val * (scale_factor or 1.0)

        obj = SceneObject(
            new_name, src.obj_type,
            position=new_pos.tolist(),
            rotation=src.rotation_euler.tolist(),
            scale=new_scale.tolist(),
            params=src.params.copy(),
            material=src.base_material.copy(),
        )
        self.objects[new_name] = obj
        return obj

    # ── Array patterns ──────────────────────────────────────────────────────

    def create_array(self, source_name: str, pattern: str = "grid",
                     count_x=3, count_y=1, count_z=3, spacing=2.0) -> List[str]:
        new_names = []

        if pattern == "grid":
            for ix in range(count_x):
                for iy in range(count_y):
                    for iz in range(count_z):
                        offset = [
                            (ix - (count_x - 1) / 2) * spacing,
                            iy * spacing,
                            (iz - (count_z - 1) / 2) * spacing,
                        ]
                        n = f"{source_name}_{ix}_{iy}_{iz}"
                        self.duplicate_object(source_name, n, offset)
                        new_names.append(n)

        elif pattern == "circle":
            for i in range(count_x):
                angle = 2 * math.pi * i / count_x
                offset = [math.cos(angle) * spacing, 0, math.sin(angle) * spacing]
                n = f"{source_name}_ring_{i}"
                self.duplicate_object(source_name, n, offset)
                new_names.append(n)

        elif pattern == "spiral":
            for i in range(count_x):
                t = i / count_x
                angle = t * 4 * math.pi
                radius = spacing * (1 - t) + spacing * 0.2 * t
                offset = [math.cos(angle) * radius, t * count_y * spacing,
                          math.sin(angle) * radius]
                n = f"{source_name}_spiral_{i}"
                self.duplicate_object(source_name, n, offset)
                new_names.append(n)

        elif pattern == "line":
            for i in range(count_x):
                offset = [(i - (count_x - 1) / 2) * spacing, 0, 0]
                n = f"{source_name}_line_{i}"
                self.duplicate_object(source_name, n, offset)
                new_names.append(n)

        return new_names

    # ── Scatter ─────────────────────────────────────────────────────────────

    def scatter_objects(self, source_name: str, pattern: str = "random",
                        count: int = 20, width: float = 10.0, depth: float = 10.0,
                        spacing: float = 2.0, jitter: float = 0.0,
                        avoid_radius: float = 0.5, height_range: str = "0,0",
                        seed: int = 42) -> List[str]:
        if source_name not in self.objects:
            raise ValueError(f"Object '{source_name}' not found")

        source = self.objects[source_name]
        h_min, h_max = [float(x) for x in height_range.split(",")]

        if pattern == "random" or pattern == "terrain":
            scattered = ScatterEngine.scatter_on_terrain(
                source, count, width, depth,
                height_range=(h_min, h_max),
                avoid_radius=avoid_radius, seed=seed)
        elif pattern == "grid":
            scattered = ScatterEngine.scatter_grid(
                source, int(math.sqrt(count)), int(math.sqrt(count)),
                spacing, jitter, seed)
        elif pattern == "circle":
            scattered = ScatterEngine.scatter_circle(
                source, count, spacing, seed)
        else:
            raise ValueError(f"Unknown scatter pattern: {pattern}")

        names = []
        for obj in scattered:
            self.objects[obj.name] = obj
            names.append(obj.name)

        return names

    # ── Groups ──────────────────────────────────────────────────────────────

    def create_group(self, group_name: str, object_names: List[str]):
        self.groups[group_name] = [n for n in object_names if n in self.objects]

    def add_to_group(self, group_name: str, object_name: str):
        if object_name not in self.objects:
            raise ValueError(f"Object '{object_name}' not found")
        if group_name not in self.groups:
            self.groups[group_name] = []
        if object_name not in self.groups[group_name]:
            self.groups[group_name].append(object_name)

    # ── Material overrides ──────────────────────────────────────────────────

    def apply_material_override(self, material_name: str,
                                target: str = "all", group_name: str = None):
        """Apply material override to objects.

        Args:
            material_name: Name of material preset or dict
            target: "all", "selected", or "group"
            group_name: Group name if target="group"
        """
        mat = MATERIAL_PRESETS.get(material_name, material_name)
        if isinstance(mat, str):
            mat = {"type": "pbr", "color": mat}

        if target == "all":
            for obj in self.objects.values():
                obj.apply_material_override(mat)
        elif target == "group" and group_name:
            for obj in self.get_by_group(group_name):
                obj.apply_material_override(mat)
        elif target == "global":
            self.global_material_override = mat

    def clear_material_overrides(self, target: str = "all"):
        if target == "all":
            for obj in self.objects.values():
                obj.clear_material_override()
            self.global_material_override = None

    # ── Auto-composition ────────────────────────────────────────────────────

    def auto_compose(self, subject_name: str = None, camera_name: str = "auto",
                     rule_of_thirds: bool = True):
        """Automatically position camera for best composition.

        Uses rule-of-thirds framing, subject centering, and
        optimal distance calculation based on scene bounds.
        """
        if not self.objects:
            return

        # Find subject or use first object
        if subject_name and subject_name in self.objects:
            subject = self.objects[subject_name]
            target_pos = subject.get_world_position()
        else:
            # Use scene center
            all_pos = np.array([o.get_world_position() for o in self.objects.values()])
            target_pos = all_pos.mean(axis=0)

        # Calculate bounding sphere radius
        if subject_name and subject_name in self.objects:
            bbox_min, bbox_max = subject.mesh.get_bounding_box()
            size = max(magnitude(bbox_max - bbox_min), 1.0)
        else:
            all_pos = np.array([o.get_world_position() for o in self.objects.values()])
            if len(all_pos) > 0:
                center = all_pos.mean(axis=0)
                max_dist = max(magnitude(p - center) for p in all_pos)
                size = max(max_dist * 2, 1.0)
            else:
                size = 1.0

        # Optimal camera distance (fill ~2/3 of frame)
        fov_rad = math.radians(40)
        cam_dist = max(size / (2 * math.tan(fov_rad / 2)), 1.0) * 1.5

        # Place camera on a 45-degree angle, slightly above
        angle = math.radians(45)
        elevation = math.radians(30)

        if rule_of_thirds:
            # Offset camera slightly for rule of thirds
            offset_x = 0.15 * cam_dist
            offset_y = 0.1 * cam_dist
        else:
            offset_x = offset_y = 0

        cx = target_pos[0] + cam_dist * math.cos(angle) * math.cos(elevation) + offset_x
        cy = target_pos[1] + cam_dist * math.sin(elevation) + offset_y
        cz = target_pos[2] + cam_dist * math.sin(angle) * math.cos(elevation)

        self.cameras[camera_name] = {
            "name": camera_name,
            "position": [float(cx), float(cy), float(cz)],
            "target": target_pos.tolist(),
            "fov": 40.0,
        }

    # ── Rendering ───────────────────────────────────────────────────────────

    def get_render_data(self, camera_name: str = None) -> dict:
        meshes = []
        for obj in self.objects.values():
            m = obj.mesh
            mat = obj.get_effective_material()
            if self.global_material_override:
                mat = {**mat, **self.global_material_override}
            mesh_data = {
                "vertices": m.vertices.tolist(),
                "faces": m.faces.tolist(),
            }
            if m.normals is not None:
                mesh_data["normals"] = m.normals.tolist()
            if m.uvs is not None:
                mesh_data["uvs"] = m.uvs.tolist()
            if mat:
                mesh_data["material"] = mat
            meshes.append(mesh_data)

        cam_data = self.cameras.get(camera_name,
                                     list(self.cameras.values())[0] if self.cameras else None)
        return {"meshes": meshes, "camera": cam_data, "lights": self.lights,
                "environment": self.environment}

    def render(self, camera_name: str = None, output_path: str = None,
               width: int = 1920, height: int = 1080) -> np.ndarray:
        renderer = Renderer(resolution=(width, height))
        render_data = self.get_render_data(camera_name)

        cam_data = render_data.get("camera")
        camera = Camera()
        if cam_data:
            camera.position = np.array(cam_data["position"], dtype=np.float64)
            camera.target = np.array(cam_data["target"], dtype=np.float64)
            camera.fov = cam_data.get("fov", 45.0)
        camera.set_resolution(width, height)
        camera.background_color = tuple(self.environment.get("background_color", [0.05, 0.05, 0.08]))

        lights = []
        for ld in self.lights:
            lt = ld.get("type", "directional")
            if lt == "directional":
                lights.append(Light.directional(
                    direction=ld.get("direction", [-1, -1, -1]),
                    color=ld.get("color", "#ffffff"),
                    intensity=ld.get("intensity", 1.0)))
            elif lt == "point":
                lights.append(Light.point(
                    position=ld.get("position", [0, 3, 0]),
                    color=ld.get("color", "#ffffff"),
                    intensity=ld.get("intensity", 1.0),
                    range=ld.get("range", 10.0)))
            elif lt == "spot":
                lights.append(Light.spot(
                    position=ld.get("position", [0, 5, 0]),
                    direction=ld.get("direction", [0, -1, 0]),
                    color=ld.get("color", "#ffffff"),
                    intensity=ld.get("intensity", 1.0),
                    spot_angle=ld.get("spot_angle", 45.0),
                    range=ld.get("range", 10.0)))

        scene_data = {"meshes": render_data["meshes"]}

        if output_path:
            ensure_output_dir(Path(output_path).parent)
            renderer.render_to_file(scene_data, output_path, camera=camera, lights=lights)
            return np.array([])

        return renderer.render(scene_data, camera=camera, lights=lights)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "objects": [obj.to_dict() for obj in self.objects.values()],
            "groups": self.groups,
            "lights": self.lights,
            "cameras": self.cameras,
            "environment": self.environment,
        }

    def save(self, filepath: str):
        save_json(self.to_dict(), filepath)

    def get_stats(self) -> dict:
        total_verts = sum(obj.mesh.vertex_count() for obj in self.objects.values())
        total_faces = sum(obj.mesh.face_count() for obj in self.objects.values())
        return {
            "name": self.name,
            "objects": len(self.objects),
            "groups": len(self.groups),
            "lights": len(self.lights),
            "cameras": len(self.cameras),
            "total_vertices": total_verts,
            "total_faces": total_faces,
            "templates": list(SCENE_TEMPLATES.keys()),
            "materials": list(MATERIAL_PRESETS.keys()),
            "object_names": list(self.objects.keys()),
        }


# ─── Main CLI ────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Nexus3D Scene Composer v0.3.0")
    sub = parser.add_subparsers(dest="command")

    # Create
    create_p = sub.add_parser("create", help="Create scene from template")
    create_p.add_argument("--template", "-t", default="studio",
                         choices=list(SCENE_TEMPLATES.keys()))
    create_p.add_argument("--name", default="Scene")
    create_p.add_argument("--output", "-o", required=True)

    # Add object
    add_p = sub.add_parser("add-object", help="Add object to scene")
    add_p.add_argument("--scene", "-s", required=True)
    add_p.add_argument("--name", required=True)
    add_p.add_argument("--type", default="cube",
                      choices=["cube", "sphere", "cylinder", "cone", "torus", "plane", "arrow"])
    add_p.add_argument("--position", default="0,1,0")
    add_p.add_argument("--rotation", default="0,0,0")
    add_p.add_argument("--scale", default="1,1,1")
    add_p.add_argument("--material", default="default", help="Material preset")
    add_p.add_argument("--parent", default=None, help="Parent object name")

    # Scatter
    scatter_p = sub.add_parser("scatter", help="Scatter objects procedurally")
    scatter_p.add_argument("--scene", "-s", required=True)
    scatter_p.add_argument("--source", required=True)
    scatter_p.add_argument("--pattern", default="random",
                          choices=["random", "terrain", "grid", "circle"])
    scatter_p.add_argument("--count", type=int, default=20)
    scatter_p.add_argument("--width", type=float, default=10.0)
    scatter_p.add_argument("--depth", type=float, default=10.0)
    scatter_p.add_argument("--spacing", type=float, default=2.0)
    scatter_p.add_argument("--height-range", default="0,0")
    scatter_p.add_argument("--seed", type=int, default=42)

    # Array
    arr_p = sub.add_parser("array", help="Create object array")
    arr_p.add_argument("--scene", "-s", required=True)
    arr_p.add_argument("--object", required=True)
    arr_p.add_argument("--pattern", default="grid",
                      choices=["grid", "circle", "line", "spiral"])
    arr_p.add_argument("--count", type=int, default=5)
    arr_p.add_argument("--spacing", type=float, default=2.0)

    # Group
    group_p = sub.add_parser("group", help="Create object group")
    group_p.add_argument("--scene", "-s", required=True)
    group_p.add_argument("--name", required=True)
    group_p.add_argument("--objects", required=True, help="Comma-separated object names")

    # Material override
    mat_p = sub.add_parser("apply-material", help="Apply material override")
    mat_p.add_argument("--scene", "-s", required=True)
    mat_p.add_argument("--material", required=True, help="Material preset name")
    mat_p.add_argument("--target", default="all",
                      choices=["all", "group", "global"])
    mat_p.add_argument("--group", default=None, help="Group name if target=group")

    # Auto-compose
    comp_p = sub.add_parser("auto-compose", help="Auto-compose camera position")
    comp_p.add_argument("--scene", "-s", required=True)
    comp_p.add_argument("--subject", default=None, help="Subject object name")
    comp_p.add_argument("--camera", default="auto")
    comp_p.add_argument("--no-thirds", action="store_true", help="Disable rule of thirds")

    # Render
    render_p = sub.add_parser("render", help="Render scene")
    render_p.add_argument("--scene", "-s", required=True)
    render_p.add_argument("--camera", default=None)
    render_p.add_argument("--width", type=int, default=1920)
    render_p.add_argument("--height", type=int, default=1080)
    render_p.add_argument("--output", "-o", required=True)

    # Stats
    stats_p = sub.add_parser("stats", help="Show scene statistics")
    stats_p.add_argument("--scene", "-s", required=True)

    # List templates
    sub.add_parser("list-templates", help="List available templates")

    # List materials
    sub.add_parser("list-materials", help="List available materials")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "create":
        scene = SceneComposer.from_template(args.template, args.name)
        scene.save(args.output)
        print(json.dumps({"template": args.template, "file": args.output,
                          "stats": scene.get_stats()}, indent=2))

    elif args.command == "add-object":
        scene = SceneComposer.from_file(args.scene)
        pos = [float(x) for x in args.position.split(",")]
        rot = [float(x) for x in args.rotation.split(",")]
        scl = [float(x) for x in args.scale.split(",")]
        mat = MATERIAL_PRESETS.get(args.material, MATERIAL_PRESETS["default"])
        scene.add_object(args.name, args.type, pos, rot, scl, material=mat,
                        parent_name=args.parent)
        scene.save(args.scene)
        print(json.dumps({"added": args.name, "total_objects": len(scene.objects)}, indent=2))

    elif args.command == "scatter":
        scene = SceneComposer.from_file(args.scene)
        names = scene.scatter_objects(args.source, args.pattern, args.count,
                                     args.width, args.depth, args.spacing,
                                     height_range=args.height_range,
                                     seed=args.seed)
        scene.save(args.scene)
        print(json.dumps({"pattern": args.pattern, "created": len(names)}, indent=2))

    elif args.command == "array":
        scene = SceneComposer.from_file(args.scene)
        names = scene.create_array(args.object, args.pattern,
                                  count_x=args.count, spacing=args.spacing)
        scene.save(args.scene)
        print(json.dumps({"pattern": args.pattern, "created": len(names)}, indent=2))

    elif args.command == "group":
        scene = SceneComposer.from_file(args.scene)
        names = [n.strip() for n in args.objects.split(",")]
        scene.create_group(args.name, names)
        scene.save(args.scene)
        print(json.dumps({"group": args.name, "members": len(names)}, indent=2))

    elif args.command == "apply-material":
        scene = SceneComposer.from_file(args.scene)
        scene.apply_material_override(args.material, args.target, args.group)
        scene.save(args.scene)
        print(json.dumps({"applied": args.material, "target": args.target}, indent=2))

    elif args.command == "auto-compose":
        scene = SceneComposer.from_file(args.scene)
        scene.auto_compose(args.subject, args.camera, not args.no_thirds)
        scene.save(args.scene)
        cam = scene.cameras.get(args.camera)
        print(json.dumps({"camera": args.camera, "position": cam["position"] if cam else None}, indent=2))

    elif args.command == "render":
        scene = SceneComposer.from_file(args.scene)
        scene.render(args.camera, args.output, args.width, args.height)
        print(json.dumps({"rendered": args.output, "camera": args.camera,
                          "resolution": [args.width, args.height]}, indent=2))

    elif args.command == "stats":
        scene = SceneComposer.from_file(args.scene)
        print(json.dumps(scene.get_stats(), indent=2))

    elif args.command == "list-templates":
        print(json.dumps(list(SCENE_TEMPLATES.keys()), indent=2))

    elif args.command == "list-materials":
        print(json.dumps(list(MATERIAL_PRESETS.keys()), indent=2))


if __name__ == "__main__":
    main()
