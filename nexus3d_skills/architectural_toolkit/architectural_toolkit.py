"""
Architectural Toolkit — Parametric Building & Urban Generation
===============================================================
Generates architectural structures, building massing, facade
systems, floor plans, and urban layouts using parametric rules.
Integrates with the Nexus3D scene system for visualization.
"""

from __future__ import annotations

import json
import math
import random
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class BuildingType(Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    OFFICE = "office"
    INDUSTRIAL = "industrial"
    MIXED_USE = "mixed_use"
    TOWER = "tower"
    SKYSCRAPER = "skyscraper"
    PAVILION = "pavilion"
    CUSTOM = "custom"


class FacadeStyle(Enum):
    CURTAIN_WALL = "curtain_wall"
    PANELIZED = "panelized"
    BRICK = "brick"
    STONE = "stone"
    MODULAR = "modular"
    PARAMETRIC = "parametric"
    GREEN = "green"
    HISTORIC = "historic"
    MODERNIST = "modernist"


class RoofType(Enum):
    FLAT = "flat"
    GABLE = "gable"
    HIPPED = "hipped"
    MANSARD = "mansard"
    SHED = "shed"
    BUTTERFLY = "butterfly"
    SAWTOOTH = "sawtooth"
    DOME = "dome"
    GREEN = "green"


@dataclass
class FloorPlan:
    """A single floor's plan geometry."""
    level: int = 0
    height: float = 3.0
    area: float = 100.0
    polygon: List[Tuple[float, float]] = field(default_factory=list)
    rooms: List[Dict[str, Any]] = field(default_factory=list)
    openings: List[Dict[str, Any]] = field(default_factory=list)  # doors, windows
    circulation: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class BuildingMass:
    """Core massing for a building."""
    width: float = 20.0
    depth: float = 15.0
    height: float = 30.0
    num_floors: int = 10
    floor_height: float = 3.0
    setbacks: List[float] = field(default_factory=list)  # per floor
    rotation: float = 0.0
    podium_width: float = 0.0
    podium_height: float = 0.0
    taper: float = 0.0           # % reduction per floor
    split: float = 0.0           # tower offset ratio

    @property
    def total_height(self) -> float:
        return self.num_floors * self.floor_height

    def floor_plan_at(self, level: int) -> Tuple[float, float]:
        """Return (width, depth) at a given floor level."""
        taper_factor = 1.0 - (level / max(self.num_floors, 1)) * self.taper
        setback = self.setbacks[level] if level < len(self.setbacks) else 0.0
        w = max(2.0, self.width * taper_factor - setback * 2)
        d = max(2.0, self.depth * taper_factor - setback * 2)
        return (w, d)


# ---------------------------------------------------------------------------
# Parametric Building Generator
# ---------------------------------------------------------------------------

class BuildingGenerator:
    """Generate complete building definitions from parametric rules."""

    STYLE_PRESETS = {
        BuildingType.TOWER: {
            "width": (12, 25), "depth": (12, 25), "height": (50, 200),
            "num_floors": (15, 60), "floor_height": 3.0,
            "taper": (0.0, 0.02), "facade": FacadeStyle.CURTAIN_WALL,
            "roof": RoofType.FLAT,
        },
        BuildingType.OFFICE: {
            "width": (20, 40), "depth": (15, 30), "height": (20, 60),
            "num_floors": (5, 20), "floor_height": 3.5,
            "taper": (0.0, 0.005), "facade": FacadeStyle.PANELIZED,
            "roof": RoofType.FLAT,
        },
        BuildingType.RESIDENTIAL: {
            "width": (10, 20), "depth": (8, 15), "height": (10, 30),
            "num_floors": (3, 10), "floor_height": 2.8,
            "taper": (0.0, 0.01), "facade": FacadeStyle.BRICK,
            "roof": RoofType.GABLE,
        },
        BuildingType.INDUSTRIAL: {
            "width": (30, 60), "depth": (20, 40), "height": (8, 15),
            "num_floors": (1, 3), "floor_height": 5.0,
            "taper": 0.0, "facade": FacadeStyle.PANELIZED,
            "roof": RoofType.SHED,
        },
        BuildingType.SKYSCRAPER: {
            "width": (15, 30), "depth": (15, 30), "height": (200, 400),
            "num_floors": (60, 120), "floor_height": 3.2,
            "taper": (0.005, 0.015), "facade": FacadeStyle.CURTAIN_WALL,
            "roof": RoofType.FLAT,
        },
        BuildingType.PAVILION: {
            "width": (15, 40), "depth": (15, 40), "height": (4, 10),
            "num_floors": (1, 2), "floor_height": 3.5,
            "taper": 0.0, "facade": FacadeStyle.MODERNIST,
            "roof": RoofType.BUTTERFLY,
        },
    }

    @classmethod
    def generate(cls, building_type: BuildingType = BuildingType.OFFICE,
                 seed: int = 0, **overrides) -> Dict[str, Any]:
        """Generate a building from parametric presets."""
        rng = random.Random(seed)
        preset = dict(cls.STYLE_PRESETS.get(building_type, cls.STYLE_PRESETS[BuildingType.OFFICE]))

        # Resolve ranges
        mass = BuildingMass()
        for field_name in ("width", "depth", "height", "num_floors", "taper"):
            val = preset.get(field_name)
            if isinstance(val, tuple):
                setattr(mass, field_name, rng.uniform(val[0], val[1]) if isinstance(val[0], float)
                        else rng.randint(val[0], val[1]))
            elif val is not None:
                setattr(mass, field_name, val)

        if "floor_height" in preset:
            mass.floor_height = preset["floor_height"]

        # Apply overrides
        for k, v in overrides.items():
            if hasattr(mass, k):
                setattr(mass, k, v)

        mass.num_floors = max(1, int(mass.num_floors))
        mass.height = mass.num_floors * mass.floor_height

        # Setbacks (subtle random per floor)
        mass.setbacks = [rng.uniform(0, 0.5) for _ in range(mass.num_floors)]

        facade = overrides.get("facade", preset.get("facade", FacadeStyle.CURTAIN_WALL))
        roof = overrides.get("roof", preset.get("roof", RoofType.FLAT))

        # Generate floors
        floors = []
        for i in range(mass.num_floors):
            w, d = mass.floor_plan_at(i)
            z = i * mass.floor_height
            floors.append({
                "level": i,
                "z": z,
                "width": w,
                "depth": d,
                "area": w * d,
                "height": mass.floor_height,
            })

        return {
            "uid": uuid.uuid4().hex[:12],
            "type": building_type.value,
            "mass": asdict(mass),
            "floors": floors,
            "facade": facade.value if isinstance(facade, FacadeStyle) else str(facade),
            "roof": roof.value if isinstance(roof, RoofType) else str(roof),
            "seed": seed,
        }

    @classmethod
    def list_types(cls) -> List[str]:
        return [b.value for b in BuildingType]


# ---------------------------------------------------------------------------
# Facade Generator
# ---------------------------------------------------------------------------

class FacadeGenerator:
    """Generate facade patterns and window layouts."""

    @staticmethod
    def curtain_wall(width: float, height: float,
                     panel_width: float = 1.5,
                     panel_height: float = 3.0,
                     mullion_width: float = 0.05) -> Dict[str, Any]:
        """Generate curtain wall grid."""
        num_x = max(1, int(width / (panel_width + mullion_width)))
        num_y = max(1, int(height / (panel_height + mullion_width)))
        return {
            "style": "curtain_wall",
            "panels_x": num_x,
            "panels_y": num_y,
            "panel_width": panel_width,
            "panel_height": panel_height,
            "mullion_width": mullion_width,
            "total_width": num_x * panel_width + (num_x - 1) * mullion_width,
            "total_height": num_y * panel_height + (num_y - 1) * mullion_width,
        }

    @staticmethod
    def panelized(width: float, height: float,
                  panel_width: float = 3.0,
                  panel_height: float = 3.0,
                  gap: float = 0.02) -> Dict[str, Any]:
        """Generate panelized facade."""
        num_x = max(1, int(width / (panel_width + gap)))
        num_y = max(1, int(height / (panel_height + gap)))
        return {
            "style": "panelized",
            "panels_x": num_x,
            "panels_y": num_y,
            "panel_width": panel_width,
            "panel_height": panel_height,
            "gap": gap,
        }

    @staticmethod
    def parametric(width: float, height: float,
                   density: float = 0.6,
                   variation: float = 0.3,
                   seed: int = 0) -> Dict[str, Any]:
        """Generate parametric facade with varying panel sizes."""
        rng = random.Random(seed)
        panels = []
        y = 0.0
        while y < height:
            x = 0.0
            row = []
            while x < width:
                pw = rng.uniform(0.8, 2.0) * (1.0 + variation * (rng.random() - 0.5))
                ph = rng.uniform(1.0, 2.5) * (1.0 + variation * (rng.random() - 0.5))
                show = rng.random() < density
                row.append({"x": x, "y": y, "w": pw, "h": ph, "visible": show})
                x += pw
            panels.append(row)
            y += max(rng.uniform(1.5, 3.0), 0.5)
        return {
            "style": "parametric",
            "panels": panels,
            "density": density,
            "variation": variation,
        }

    @staticmethod
    def brick(width: float, height: float,
              brick_length: float = 0.23,
              brick_height: float = 0.076,
              mortar: float = 0.01) -> Dict[str, Any]:
        """Generate brick pattern facade."""
        num_x = max(1, int(width / (brick_length + mortar)))
        num_y = max(1, int(height / (brick_height + mortar)))
        return {
            "style": "brick",
            "bricks_x": num_x,
            "bricks_y": num_y,
            "brick_length": brick_length,
            "brick_height": brick_height,
            "mortar": mortar,
            "offset_every_row": True,
        }

    STYLES = {
        "curtain_wall": curtain_wall,
        "panelized": panelized,
        "parametric": parametric,
        "brick": brick,
    }

    @classmethod
    def generate(cls, style: str, width: float, height: float,
                 **kwargs) -> Dict[str, Any]:
        generator = cls.STYLES.get(style)
        if generator:
            return generator(width, height, **kwargs)
        return cls.curtain_wall(width, height)

    @classmethod
    def list_styles(cls) -> List[str]:
        return list(cls.STYLES.keys())


# ---------------------------------------------------------------------------
# Roof Generator
# ---------------------------------------------------------------------------

class RoofGenerator:
    """Generate roof geometry from parametric rules."""

    @staticmethod
    def flat(width: float, depth: float,
             parapet_height: float = 0.3) -> Dict[str, Any]:
        return {"type": "flat", "width": width, "depth": depth,
                "parapet_height": parapet_height}

    @staticmethod
    def gable(width: float, depth: float,
              pitch: float = 0.5,
              overhang: float = 0.3) -> Dict[str, Any]:
        """Gable roof with specified pitch."""
        ridge_height = (width / 2 + overhang) * pitch
        return {
            "type": "gable",
            "width": width,
            "depth": depth,
            "pitch": pitch,
            "ridge_height": ridge_height,
            "overhang": overhang,
        }

    @staticmethod
    def hipped(width: float, depth: float,
               pitch: float = 0.4,
               overhang: float = 0.3) -> Dict[str, Any]:
        ridge_length = max(0, width - 2 * depth * pitch)
        ridge_height = depth * pitch
        return {
            "type": "hipped",
            "width": width,
            "depth": depth,
            "pitch": pitch,
            "ridge_length": ridge_length,
            "ridge_height": ridge_height,
            "overhang": overhang,
        }

    @staticmethod
    def butterfly(width: float, depth: float,
                  pitch: float = 0.3,
                  valley_depth: float = 1.0) -> Dict[str, Any]:
        """Butterfly (inverted gable) roof."""
        return {
            "type": "butterfly",
            "width": width,
            "depth": depth,
            "pitch": pitch,
            "valley_depth": valley_depth,
        }

    @staticmethod
    def dome(diameter: float, height: float,
             segments: int = 16) -> Dict[str, Any]:
        return {
            "type": "dome",
            "diameter": diameter,
            "height": height,
            "segments": segments,
        }

    TYPES = {
        "flat": flat,
        "gable": gable,
        "hipped": hipped,
        "butterfly": butterfly,
        "dome": dome,
    }

    @classmethod
    def generate(cls, roof_type: str, width: float, depth: float,
                 **kwargs) -> Dict[str, Any]:
        generator = cls.TYPES.get(roof_type)
        if generator:
            return generator(width, depth, **kwargs)
        return cls.flat(width, depth)

    @classmethod
    def list_types(cls) -> List[str]:
        return list(cls.TYPES.keys())


# ---------------------------------------------------------------------------
# Urban Layout Generator
# ---------------------------------------------------------------------------

class UrbanPlanner:
    """Generate urban block layouts and city grids."""

    @staticmethod
    def generate_grid(num_blocks_x: int = 5, num_blocks_y: int = 5,
                      block_width: float = 60.0,
                      block_depth: float = 60.0,
                      street_width: float = 12.0,
                      sidewalk_width: float = 2.0) -> Dict[str, Any]:
        """Generate a regular city grid."""
        blocks = []
        for y in range(num_blocks_y):
            for x in range(num_blocks_x):
                blocks.append({
                    "uid": uuid.uuid4().hex[:12],
                    "position": (
                        x * (block_width + street_width),
                        0.0,
                        y * (block_depth + street_width),
                    ),
                    "width": block_width,
                    "depth": block_depth,
                    "grid_x": x,
                    "grid_y": y,
                })

        total_w = num_blocks_x * (block_width + street_width) - street_width
        total_d = num_blocks_y * (block_depth + street_width) - street_width

        return {
            "type": "grid",
            "blocks": blocks,
            "streets": {
                "width": street_width,
                "sidewalk": sidewalk_width,
                "num_horizontal": num_blocks_y - 1,
                "num_vertical": num_blocks_x - 1,
            },
            "dimensions": {"width": total_w, "depth": total_d},
            "num_blocks": num_blocks_x * num_blocks_y,
        }

    @staticmethod
    def generate_radial(num_rings: int = 4,
                        segments: int = 8,
                        block_depth: float = 50.0,
                        radial_width: float = 15.0) -> Dict[str, Any]:
        """Generate a radial/circular city layout."""
        blocks = []
        for ring in range(num_rings):
            radius = (ring + 1) * (block_depth + radial_width)
            for seg in range(segments):
                angle = (seg / segments) * 2 * math.pi
                px = radius * math.cos(angle)
                pz = radius * math.sin(angle)
                bw = block_depth
                bd = block_depth
                blocks.append({
                    "uid": uuid.uuid4().hex[:12],
                    "position": (px, 0.0, pz),
                    "width": bw,
                    "depth": bd,
                    "ring": ring,
                    "segment": seg,
                    "angle": angle,
                })

        return {
            "type": "radial",
            "blocks": blocks,
            "num_rings": num_rings,
            "segments": segments,
            "road_width": radial_width,
        }

    @staticmethod
    def generate_random(bbox: Tuple[float, float, float, float],
                        num_blocks: int = 20,
                        min_size: float = 20.0,
                        max_size: float = 80.0,
                        seed: int = 0) -> Dict[str, Any]:
        """Generate random block layout within a bounding box."""
        rng = random.Random(seed)
        min_x, min_z, max_x, max_z = bbox
        blocks = []
        for _ in range(num_blocks):
            w = rng.uniform(min_size, max_size)
            d = rng.uniform(min_size, max_size)
            x = rng.uniform(min_x, max_x - w)
            z = rng.uniform(min_z, max_z - d)
            blocks.append({
                "uid": uuid.uuid4().hex[:12],
                "position": (x, 0.0, z),
                "width": w,
                "depth": d,
            })

        return {
            "type": "random",
            "blocks": blocks,
            "bbox": bbox,
            "seed": seed,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_building(building_type: str = "office",
                             seed: int = 0,
                             **params) -> Dict[str, Any]:
    """Generate a parametric building."""
    try:
        bt = BuildingType(building_type)
    except ValueError:
        bt = BuildingType.OFFICE
    return BuildingGenerator.generate(bt, seed=seed, **params)


async def list_building_types() -> List[str]:
    """List available building types."""
    return BuildingGenerator.list_types()


async def generate_facade(style: str = "curtain_wall",
                           width: float = 20.0,
                           height: float = 30.0,
                           **params) -> Dict[str, Any]:
    """Generate a facade pattern."""
    return FacadeGenerator.generate(style, width, height, **params)


async def list_facade_styles() -> List[str]:
    """List available facade styles."""
    return FacadeGenerator.list_styles()


async def generate_roof(roof_type: str = "flat",
                         width: float = 20.0,
                         depth: float = 15.0,
                         **params) -> Dict[str, Any]:
    """Generate a roof structure."""
    return RoofGenerator.generate(roof_type, width, depth, **params)


async def list_roof_types() -> List[str]:
    """List available roof types."""
    return RoofGenerator.list_types()


async def generate_urban_layout(layout_type: str = "grid",
                                 seed: int = 0,
                                 **params) -> Dict[str, Any]:
    """Generate an urban layout."""
    planner = UrbanPlanner()
    if layout_type == "grid":
        result = planner.generate_grid(**params)
    elif layout_type == "radial":
        result = planner.generate_radial(**params)
    elif layout_type == "random":
        result = planner.generate_random(
            params.get("bbox", (-200, -200, 200, 200)),
            params.get("num_blocks", 20),
            params.get("min_size", 20.0),
            params.get("max_size", 80.0),
            seed,
        )
    else:
        result = planner.generate_grid(**params)
    return result


async def calculate_building_area(mass_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate area metrics for a building mass."""
    mass = BuildingMass(**{k: v for k, v in mass_data.items()
                           if k in BuildingMass.__dataclass_fields__})
    total_area = 0.0
    floor_areas = []
    for i in range(mass.num_floors):
        w, d = mass.floor_plan_at(i)
        area = w * d
        floor_areas.append(area)
        total_area += area
    return {
        "total_floor_area": total_area,
        "floor_areas": floor_areas,
        "footprint": mass.width * mass.depth,
        "num_floors": mass.num_floors,
        "height": mass.total_height,
        "avg_floor_area": total_area / max(mass.num_floors, 1),
        "farr": total_area / (mass.width * mass.depth),  # floor area ratio
    }
