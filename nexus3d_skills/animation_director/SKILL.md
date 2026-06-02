# Animation Director

**Category:** 3D / Procedural Modeling  
**Skill Path:** `nexus3d_skills/animation_director/`

## Overview

Advanced animation creation, layering, blending, and export pipeline.

```
Nexus3D Advanced Skill: Animation Director
===========================================
Advanced animation creation, layering, blending, and export pipeline.
Creates complex multi-track animations with procedural motion, keyframe editing,
motion blending, and export to industry-standard formats (BVH, glTF, FBX).

Upgraded v0.3.0 — Adds:
  - Foot IK locking for walk/run/jump (prevents foot sliding)
  - Animation layering (upper body + lower body independent layers)
  - Motion graph construction (blend-tree transitions)
  - More animation types (dance, punch, kick, wave, crouch)
  - Keyframe editor with bezier interpolation
  - Motion analysis metrics

Usage:
    python animation_director.py --help
    python animation_director.py create --type "walk_to_run" --duration 4.0 --output anim.json
    python animation_director.py blend --anim-a walk.json --anim-b run.json --weight 0.5 --output blend.json
    python animation_director.py layer --base idle.json --overlay wave.json --bones "LeftArm,LeftForeArm"
    python animation_director.py export --animation anim.json --armature arm.json --format bvh --output motion.bvh
```

## Capabilities

- Advanced animation creation, layering, blending, and export pipeline.

## Usage

```bash
# Run the skill directly
python nexus3d_skills/animation_director/animation_director.py --help
```

The skill can also be invoked through the agent via `ml_*` tool bridge by importing its module.

## Outputs

- Procedural 3D assets (mesh, armature, animation, material)
- JSON-serialized scene data
- Optional rendered frames (PNG) or animation exports (BVH/glTF/FBX)

## See also

- `nexus3d/` — the underlying 3D engine primitives
- `nexus/tools/builtin/nexus3d_tools.py` — `nexus3d_*` tools that wrap this engine
