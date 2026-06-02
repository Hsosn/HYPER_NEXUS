# Scene Composer

**Category:** 3D / Procedural Modeling  
**Skill Path:** `nexus3d_skills/scene_composer/`

## Overview

Scene layout — object placement, grouping, lighting setup, environment staging.

```
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
```

## Capabilities

- Scene layout — object placement, grouping, lighting setup, environment staging.

## Usage

```bash
# Run the skill directly
python nexus3d_skills/scene_composer/scene_composer.py --help
```

The skill can also be invoked through the agent via `ml_*` tool bridge by importing its module.

## Outputs

- Procedural 3D assets (mesh, armature, animation, material)
- JSON-serialized scene data
- Optional rendered frames (PNG) or animation exports (BVH/glTF/FBX)

## See also

- `nexus3d/` — the underlying 3D engine primitives
- `nexus/tools/builtin/nexus3d_tools.py` — `nexus3d_*` tools that wrap this engine
