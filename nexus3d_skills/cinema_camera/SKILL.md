# Cinema Camera

**Category:** 3D / Procedural Modeling  
**Skill Path:** `nexus3d_skills/cinema_camera/`

## Overview

Cinematic camera rigging — dolly, crane, tracking, orbit, rule-of-thirds composition.

```
Nexus3D Advanced Skill: Cinema Camera
======================================
Professional cinematic camera tools with depth of field tables, camera animation
paths, exposure calculation, lens profiles, camera shake, and multi-camera setups.

Upgraded v0.3.0 — Adds:
  - Multi-camera sequence (cuts between cameras on a timeline)
  - Camera shake / handheld simulation with configurable intensity
  - Exposure auto-calc (given EV100, get aperture/shutter/ISO combos)
  - Lens distortion profile application (barrel, pincushion, mustache)
  - Camera path editor (insert/delete keyframes, set interpolation)

Usage:
    python cinema_camera.py dof-table --lens 50mm --aperture 2.8
    python cinema_camera.py generate-path orbit --duration 10 --output path.json
    python cinema_camera.py camera-shake --intensity 0.3 --frequency 5.0 --duration 4.0
    python cinema_camera.py exposure-auto --ev 12 --iso 800
    python cinema_camera.py multi-cam --config cameras.json --duration 12.0
    python cinema_camera.py all-demos
```

## Capabilities

- Cinematic camera rigging — dolly, crane, tracking, orbit, rule-of-thirds composition.

## Usage

```bash
# Run the skill directly
python nexus3d_skills/cinema_camera/cinema_camera.py --help
```

The skill can also be invoked through the agent via `ml_*` tool bridge by importing its module.

## Outputs

- Procedural 3D assets (mesh, armature, animation, material)
- JSON-serialized scene data
- Optional rendered frames (PNG) or animation exports (BVH/glTF/FBX)

## See also

- `nexus3d/` — the underlying 3D engine primitives
- `nexus/tools/builtin/nexus3d_tools.py` — `nexus3d_*` tools that wrap this engine
