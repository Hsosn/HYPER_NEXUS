# Character Creator

**Category:** 3D / Procedural Modeling  
**Skill Path:** `nexus3d_skills/character_creator/`

## Overview

Humanoid/creature character generation with adjustable proportions.

```
Nexus3D Advanced Skill: Character Creator
==========================================
Creates fully rigged 3D humanoid characters with customizable proportions,
automatic mesh generation, skinning, pose libraries, and blend shapes.

Upgraded v0.3.0 — Adds:
  - Facial blend shapes (smile, frown, surprise, blink, eyebrow raise)
  - Clothing generation (shirt, pants, skirt, hat) as separate mesh layers
  - LOD (Level of Detail) generation for performance optimization
  - More body presets (child, elder, athletic, cartoon)
  - Asymmetric body proportions (individual limb control)
  - Export to glTF format support
  - Character metadata / annotations

Usage:
    python character_creator.py --help
    python character_creator.py create --name "Hero" --height 1.85 --style muscular
    python character_creator.py facial-expression --name "Hero" --expression "smile" --weight 0.8
    python character_creator.py clothe --name "Hero" --outfit "casual" --output clothed.obj
    python character_creator.py lod --name "Hero" --level 1 --output hero_lod1.obj
    python character_creator.py export --name "Hero" --format obj --output hero.obj
```

## Capabilities

- Humanoid/creature character generation with adjustable proportions.

## Usage

```bash
# Run the skill directly
python nexus3d_skills/character_creator/character_creator.py --help
```

The skill can also be invoked through the agent via `ml_*` tool bridge by importing its module.

## Outputs

- Procedural 3D assets (mesh, armature, animation, material)
- JSON-serialized scene data
- Optional rendered frames (PNG) or animation exports (BVH/glTF/FBX)

## See also

- `nexus3d/` — the underlying 3D engine primitives
- `nexus/tools/builtin/nexus3d_tools.py` — `nexus3d_*` tools that wrap this engine
