# CSG Architect

**Category:** 3D / Procedural Modeling  
**Skill Path:** `nexus3d_skills/csg_architect/`

## Overview

Architectural boolean carving for windows/doors, wall generation, room carving.

```
Nexus3D Advanced Skill: CSG Architect
======================================
Professional architectural modeling using real Constructive Solid Geometry (CSG)
with BSP-tree boolean operations for architecturally accurate models.

Upgraded v0.3.0 — Adds:
  - Multi-story building with floor slabs and interior walls
  - Parametric staircase (straight, L-shaped, U-shaped, spiral)
  - Column types (Doric, Ionic, Corinthian, modern)
  - Window and door templates (casement, sliding, arch, french)
  - Roof types (flat, gable, hip, mansard, shed)
  - IFC-style metadata export
  - Opening cutout system for true wall openings

Usage:
    python csg_architect.py --output-dir ./output
    python csg_architect.py --demo room
    python csg_architect.py --demo staircase --params '{"steps":15,"type":"spiral"}'
    python csg_architect.py --demo building --params '{"floors":3,"roof":"gable"}'
```

## Capabilities

- Architectural boolean carving for windows/doors, wall generation, room carving.

## Usage

```bash
# Run the skill directly
python nexus3d_skills/csg_architect/csg_architect.py --help
```

The skill can also be invoked through the agent via `ml_*` tool bridge by importing its module.

## Outputs

- Procedural 3D assets (mesh, armature, animation, material)
- JSON-serialized scene data
- Optional rendered frames (PNG) or animation exports (BVH/glTF/FBX)

## See also

- `nexus3d/` — the underlying 3D engine primitives
- `nexus/tools/builtin/nexus3d_tools.py` — `nexus3d_*` tools that wrap this engine
