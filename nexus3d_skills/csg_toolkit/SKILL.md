# CSG Toolkit

**Category:** 3D / Procedural Modeling  
**Skill Path:** `nexus3d_skills/csg_toolkit/`

## Overview

General CSG modeling — complex boolean operations, shape carving, model merging.

```
Nexus3D Advanced Skill: CSG Toolkit
====================================
Boolean operations and compound shapes using real CSG union, subtract, and
intersect operations with volume conservation verification, mesh quality
analysis, CSG slicing, and multi-material region extraction.

Upgraded v0.3.0 — Adds:
  - CSG slicing (slice a mesh along a plane, keeping either or both halves)
  - Multi-material region extraction (label faces by original shape membership)
  - Mesh quality analysis (manifold check, non-manifold edges, zero-area faces)
  - Expanded gallery with 16 shapes
  - Volume conservation verification suite

Usage:
    python csg_toolkit.py operation --op union --shapes cube,sphere
    python csg_toolkit.py complex --shape mold
    python csg_toolkit.py verify
    python csg_toolkit.py slice --mesh input.obj --plane-y 0.5
    python csg_toolkit.py analyze --mesh input.obj
    python csg_toolkit.py gallery --output-dir ./csg_gallery
```

## Capabilities

- General CSG modeling — complex boolean operations, shape carving, model merging.

## Usage

```bash
# Run the skill directly
python nexus3d_skills/csg_toolkit/csg_toolkit.py --help
```

The skill can also be invoked through the agent via `ml_*` tool bridge by importing its module.

## Outputs

- Procedural 3D assets (mesh, armature, animation, material)
- JSON-serialized scene data
- Optional rendered frames (PNG) or animation exports (BVH/glTF/FBX)

## See also

- `nexus3d/` — the underlying 3D engine primitives
- `nexus/tools/builtin/nexus3d_tools.py` — `nexus3d_*` tools that wrap this engine
