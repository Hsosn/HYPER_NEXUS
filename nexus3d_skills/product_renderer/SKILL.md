# Product Renderer

**Category:** 3D / Procedural Modeling  
**Skill Path:** `nexus3d_skills/product_renderer/`

## Overview

Product visualization — turntable animation, studio lighting, reflections.

```
Product Renderer — Professional Product Visualization Studio
=============================================================
Provides automated studio lighting (3-point + HDR), camera
framing, turntable animation, material assignment, and render
preset management for photorealistic product renders.
```

## Capabilities

- Product visualization — turntable animation, studio lighting, reflections.

## Usage

```bash
# Run the skill directly
python nexus3d_skills/product_renderer/product_renderer.py --help
```

The skill can also be invoked through the agent via `ml_*` tool bridge by importing its module.

## Outputs

- Procedural 3D assets (mesh, armature, animation, material)
- JSON-serialized scene data
- Optional rendered frames (PNG) or animation exports (BVH/glTF/FBX)

## See also

- `nexus3d/` — the underlying 3D engine primitives
- `nexus/tools/builtin/nexus3d_tools.py` — `nexus3d_*` tools that wrap this engine
