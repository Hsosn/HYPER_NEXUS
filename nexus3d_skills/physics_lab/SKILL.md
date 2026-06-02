# Physics Lab

**Category:** 3D / Procedural Modeling  
**Skill Path:** `nexus3d_skills/physics_lab/`

## Overview

Rigid bodies, constraints, collisions, gravity, simulation stepping.

```
Physics Lab — Advanced Physics Simulation Toolkit
==================================================
Provides rigid body dynamics, collision detection (AABB, sphere,
mesh), spring-mass systems, soft bodies, and simulation stepping
for the Nexus3D environment.
```

## Capabilities

- Rigid bodies, constraints, collisions, gravity, simulation stepping.

## Usage

```bash
# Run the skill directly
python nexus3d_skills/physics_lab/physics_lab.py --help
```

The skill can also be invoked through the agent via `ml_*` tool bridge by importing its module.

## Outputs

- Procedural 3D assets (mesh, armature, animation, material)
- JSON-serialized scene data
- Optional rendered frames (PNG) or animation exports (BVH/glTF/FBX)

## See also

- `nexus3d/` — the underlying 3D engine primitives
- `nexus/tools/builtin/nexus3d_tools.py` — `nexus3d_*` tools that wrap this engine
