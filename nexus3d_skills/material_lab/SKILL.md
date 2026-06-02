# Material Lab

**Category:** 3D / Procedural Modeling  
**Skill Path:** `nexus3d_skills/material_lab/`

## Overview

PBR parameter tuning, procedural texture generation, material presets.

```
Material Lab — Advanced Physically-Based Material System
========================================================
Generates, composites, and optimises PBR material graphs with
displacement, subsurface scattering, anisotropy, clear-coat,
and procedural texture generation. Integrates with the
Nexus3D material pipeline and ML‑driven texture enhancement.
```

## Capabilities

- PBR parameter tuning, procedural texture generation, material presets.

## Usage

```bash
# Run the skill directly
python nexus3d_skills/material_lab/material_lab.py --help
```

The skill can also be invoked through the agent via `ml_*` tool bridge by importing its module.

## Outputs

- Procedural 3D assets (mesh, armature, animation, material)
- JSON-serialized scene data
- Optional rendered frames (PNG) or animation exports (BVH/glTF/FBX)

## See also

- `nexus3d/` — the underlying 3D engine primitives
- `nexus/tools/builtin/nexus3d_tools.py` — `nexus3d_*` tools that wrap this engine
