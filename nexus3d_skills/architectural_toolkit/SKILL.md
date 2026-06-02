# Architectural Toolkit

**Category:** 3D / Procedural Modeling  
**Skill Path:** `nexus3d_skills/architectural_toolkit/`

## Overview

Building generation with rooms, floors, windows, doors, staircases, roofs, facades.

```
Architectural Toolkit — Parametric Building & Urban Generation
===============================================================
Generates architectural structures, building massing, facade
systems, floor plans, and urban layouts using parametric rules.
Integrates with the Nexus3D scene system for visualization.
```

## Capabilities

- Building generation with rooms, floors, windows, doors, staircases, roofs, facades.

## Usage

```bash
# Run the skill directly
python nexus3d_skills/architectural_toolkit/architectural_toolkit.py --help
```

The skill can also be invoked through the agent via `ml_*` tool bridge by importing its module.

## Outputs

- Procedural 3D assets (mesh, armature, animation, material)
- JSON-serialized scene data
- Optional rendered frames (PNG) or animation exports (BVH/glTF/FBX)

## See also

- `nexus3d/` — the underlying 3D engine primitives
- `nexus/tools/builtin/nexus3d_tools.py` — `nexus3d_*` tools that wrap this engine
