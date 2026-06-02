# Studio Renderer

**Category:** 3D / Procedural Modeling  
**Skill Path:** `nexus3d_skills/studio_renderer/`

## Overview

Multi-pass rendering, denoising, compositing, output formatting.

```
Studio Renderer — Batch Rendering & Compositing Pipeline
=========================================================
Manages render jobs, batch processing, AOV output, compositing
layers, and post-processing effects for the Nexus3D rendering
pipeline. Designed to work alongside the Product Renderer.
```

## Capabilities

- Multi-pass rendering, denoising, compositing, output formatting.

## Usage

```bash
# Run the skill directly
python nexus3d_skills/studio_renderer/studio_renderer.py --help
```

The skill can also be invoked through the agent via `ml_*` tool bridge by importing its module.

## Outputs

- Procedural 3D assets (mesh, armature, animation, material)
- JSON-serialized scene data
- Optional rendered frames (PNG) or animation exports (BVH/glTF/FBX)

## See also

- `nexus3d/` — the underlying 3D engine primitives
- `nexus/tools/builtin/nexus3d_tools.py` — `nexus3d_*` tools that wrap this engine
