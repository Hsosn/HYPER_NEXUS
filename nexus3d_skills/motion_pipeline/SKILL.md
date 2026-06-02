# Motion Pipeline

**Category:** 3D / Procedural Modeling  
**Skill Path:** `nexus3d_skills/motion_pipeline/`

## Overview

Motion capture processing — BVH import, retargeting, blending, looping.

```
Motion Pipeline — Advanced Motion Synthesis & Retargeting
==========================================================
Provides a complete motion data pipeline: loading, blending,
retargeting, motion graph traversal, and procedural footstep
IK. Designed for the Nexus3D animation system.
```

## Capabilities

- Motion capture processing — BVH import, retargeting, blending, looping.

## Usage

```bash
# Run the skill directly
python nexus3d_skills/motion_pipeline/motion_pipeline.py --help
```

The skill can also be invoked through the agent via `ml_*` tool bridge by importing its module.

## Outputs

- Procedural 3D assets (mesh, armature, animation, material)
- JSON-serialized scene data
- Optional rendered frames (PNG) or animation exports (BVH/glTF/FBX)

## See also

- `nexus3d/` — the underlying 3D engine primitives
- `nexus/tools/builtin/nexus3d_tools.py` — `nexus3d_*` tools that wrap this engine
