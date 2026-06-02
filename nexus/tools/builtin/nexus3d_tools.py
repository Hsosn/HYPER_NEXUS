"""
Nexus3D Engine Tools — bridges the Nexus3D headless 3D engine into AgentNexus.

Tools registered:
- nexus3d_create_mesh     : Create 3D mesh primitives (cube, sphere, cylinder, etc.)
- nexus3d_transform_mesh  : Apply 4x4 transform to mesh vertices
- nexus3d_csg_boolean     : CSG boolean operations (union, subtract, intersect)
- nexus3d_create_armature : Create humanoid armature/skeleton
- nexus3d_solve_ik        : Solve inverse kinematics
- nexus3d_animate_procedural : Generate procedural animations (walk, breathe, idle)
- nexus3d_render_scene    : Render a 3D scene to PNG image
- nexus3d_path_trace      : Path-trace a scene for production-quality output
- nexus3d_material_library : Query PBR material library
- nexus3d_cinematic_dof   : Calculate depth-of-field parameters
- nexus3d_physics_simulate: Run physics simulation
- nexus3d_raycast         : Ray intersection test

The Nexus3D engine runs as a sub-process with its own FastAPI server on port 8420,
or alternatively, calls are made directly via the Python API (preferred for speed).
"""
from __future__ import annotations

import asyncio
import json
import sys
import traceback
from pathlib import Path
from typing import Any

# Ensure nexus3d is importable
_NEXUS3D_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "nexus3d"
if str(_NEXUS3D_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEXUS3D_ROOT))

from ..registry import tool

# ── Import-time detection ─────────────────────────────────────────────────────

_NEXUS3D_AVAILABLE = False
try:
    import numpy as _np
    from nexus3d.math3d.core import (
        compose_matrix, decompose_matrix, euler_to_quat, solve_2bone_ik,
        solve_fabrik, solve_ccd, PhysicsBody, vec3, normalize,
        ray_sphere_intersect, ray_plane_intersect, ray_triangle_intersect,
    )
    from nexus3d.mesh.primitives import (
        create_cube, create_sphere, create_cylinder, create_cone,
        create_torus, create_plane, create_arrow, create_circle, create_grid,
    )
    from nexus3d.mesh.mesh import Mesh
    from nexus3d.rigging.armature import create_humanoid_armature, Armature
    from nexus3d.animation.anim import ProceduralAnimation
    from nexus3d.materials.pipeline import MaterialLibrary
    from nexus3d.cinematic.camera import DepthOfField
    _NEXUS3D_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERROR = str(e)


def _check_available() -> str | None:
    """Return error message if Nexus3D is not available."""
    if not _NEXUS3D_AVAILABLE:
        return f"Nexus3D engine not available: {_IMPORT_ERROR}. Install with: pip install numpy trimesh scipy Pillow"
    return None


def _numpy_safe(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types."""
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return obj


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: Create Mesh
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="nexus3d_create_mesh",
    description=(
        "Create a 3D mesh primitive and save to OBJ file in workspace. "
        "Types: cube, sphere, cylinder, cone, torus, plane. "
        "Output: OBJ file for download + JSON with vertex/face counts."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "mesh_type": {
                "type": "string",
                "enum": ["cube", "sphere", "cylinder", "cone", "torus", "plane", "arrow", "circle", "grid"],
                "description": "Type of primitive to create (alias: shape)",
            },
            "shape": {
                "type": "string",
                "description": "Alias for mesh_type (e.g. shape='sphere')",
            },
            "name": {"type": "string", "description": "Name for the mesh"},
            "size": {"type": "number", "description": "Size parameter (default 1.0)"},
            "radius": {"type": "number", "description": "Radius parameter (default 1.0)"},
            "height": {"type": "number", "description": "Height parameter (default 2.0)"},
            "segments": {"type": "integer", "description": "Segment count (default 32)"},
        },
        "required": [],
    },
    risk="low",
    category="3d_engine",
)
async def nexus3d_create_mesh(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        mesh_type = params.get("mesh_type") or params.get("shape", "cube")
        name = params.get("name", f"Mesh_{mesh_type}")
        size = float(params.get("size", 1.0))
        radius = float(params.get("radius", 1.0))
        height = float(params.get("height", 2.0))
        segments = int(params.get("segments", 32))

        primitive_map = {
            "cube": lambda: create_cube(size=size),
            "sphere": lambda: create_sphere(radius=radius, rings=segments, segments=segments),
            "cylinder": lambda: create_cylinder(radius=radius, height=height, segments=segments),
            "cone": lambda: create_cone(radius=radius, height=height, segments=segments),
            "torus": lambda: create_torus(major_radius=radius, minor_radius=size * 0.3,
                                          major_segments=segments, minor_segments=segments // 2),
            "plane": lambda: create_plane(width=size, height=size),
            "arrow": lambda: create_arrow(length=size),
            "circle": lambda: create_circle(radius=radius, segments=segments),
            "grid": lambda: create_grid(size=size * 10, divisions=int(segments)),
        }

        if mesh_type not in primitive_map:
            return f"Error: Unknown mesh type '{mesh_type}'. Available: {list(primitive_map.keys())}"

        mesh_data = primitive_map[mesh_type]()
        
        # Always save to workspace for user download
        from ...config import BASE_DIR
        workspace = Path(BASE_DIR) / "data" / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        mesh = Mesh.from_primitive(name, mesh_data)
        filepath = workspace / f"{name}.obj"
        mesh.save_obj(str(filepath))
        
        result = {
            "name": name,
            "type": mesh_type,
            "vertex_count": len(mesh_data["vertices"]),
            "face_count": len(mesh_data["faces"]),
            "file": f"{name}.obj",
            "workspace_path": str(filepath),
            "download": "✅ File saved to workspace. See below for download.",
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"[Nexus3D Error] {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: Transform Mesh
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="nexus3d_transform_mesh",
    description=(
        "Apply a 4x4 transform matrix to mesh vertices. Provide vertices, faces, "
        "and a 16-element row-major matrix. Returns transformed vertices."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "vertices": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
                "description": "Vertex positions [[x,y,z], ...]",
            },
            "faces": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
                "description": "Face indices (optional)",
            },
            "translation": {"type": "array", "items": {"type": "number"}, "description": "[x, y, z]"},
            "rotation_degrees": {"type": "array", "items": {"type": "number"}, "description": "[rx, ry, rz] in degrees"},
            "scale": {"type": "array", "items": {"type": "number"}, "description": "[sx, sy, sz]"},
        },
        "required": ["vertices"],
    },
    risk="low",
    category="3d_engine",
)
async def nexus3d_transform_mesh(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        vertices = _np.array(params["vertices"], dtype=_np.float64)
        faces = _np.array(params["faces"], dtype=_np.int64) if params.get("faces") else None

        t = _np.array(params.get("translation", [0, 0, 0]), dtype=_np.float64)
        r = euler_to_quat(_np.radians(_np.array(params.get("rotation_degrees", [0, 0, 0]), dtype=_np.float64)))
        s = _np.array(params.get("scale", [1, 1, 1]), dtype=_np.float64)
        matrix = compose_matrix(t, r, s)

        # Apply: v' = M * [v, 1]^T
        ones = _np.ones((vertices.shape[0], 1), dtype=_np.float64)
        verts_h = _np.hstack([vertices, ones])
        transformed = (matrix @ verts_h.T).T[:, :3]

        result = {
            "vertex_count": len(transformed),
            "vertices": transformed.tolist()[:500],  # Cap preview
            "face_count": len(faces) if faces is not None else 0,
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"[Nexus3D Error] {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: CSG Boolean
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="nexus3d_csg_boolean",
    description=(
        "Perform CSG boolean operations (union/subtract/intersect) on 3D shapes. "
        "Creates hollow objects, cutouts, boolean combinations. "
        "Output: OBJ file saved to workspace for download."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["union", "subtract", "intersect"],
                "description": "Boolean operation: union, subtract, intersect",
            },
            "shape_a": {"type": "string", "description": "First shape (cube/sphere/cylinder/cone/torus)"},
            "shape_b": {"type": "string", "description": "Second shape"},
            "offset_b": {"type": "array", "items": {"type": "number"}, "description": "Offset for shape B [x,y,z]"},
        },
        "required": ["operation"],
    },
    risk="low",
    category="3d_engine",
)
async def nexus3d_csg_boolean(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        from nexus3d.csg.engine import CSGMesh, MeshCleaner

        operation = params.get("operation", "union")
        shape_map = {
            'cube': lambda: CSGMesh.from_cube(1.0),
            'sphere': lambda: CSGMesh.from_sphere(0.7, segments=16),
            'cylinder': lambda: CSGMesh.from_cylinder(0.5, 1.5, segments=16),
            'cone': lambda: CSGMesh.from_cone(0.5, 1.5, segments=16),
            'torus': lambda: CSGMesh.from_torus(0.8, 0.25),
        }

        a_type = params.get("shape_a", "cube")
        b_type = params.get("shape_b", "sphere")
        a = shape_map.get(a_type, CSGMesh.from_cube)()
        b = shape_map.get(b_type, CSGMesh.from_sphere)()

        off = _np.array(params.get("offset_b", [0.5, 0, 0]), dtype=_np.float64)
        for i in range(len(b.polygons)):
            for j in range(len(b.polygons[i].vertices)):
                b.polygons[i].vertices[j] = b.polygons[i].vertices[j] + off

        ops = {
            'union': lambda: a.union(b),
            'subtract': lambda: a.subtract(b),
            'intersect': lambda: a.intersect(b),
        }
        if operation not in ops:
            return f"Unknown CSG operation: {operation}"

        result_mesh = ops[operation]()
        vertices, faces = MeshCleaner.clean(*result_mesh.to_vertices_faces())

        # Always save to workspace
        from ...config import BASE_DIR
        workspace = Path(BASE_DIR) / "data" / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        mesh = Mesh(name=f"csg_{operation}", vertices=vertices, faces=faces)
        filepath = workspace / f"csg_{operation}.obj"
        mesh.save_obj(str(filepath))

        result = {
            "operation": operation,
            "shape_a": a_type,
            "shape_b": b_type,
            "vertices": len(vertices),
            "faces": len(faces),
            "volume": result_mesh.volume(),
            "file": f"csg_{operation}.obj",
            "workspace_path": str(filepath),
            "download": "✅ CSG result saved to workspace for download.",
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"[Nexus3D Error] {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: Create Armature
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="nexus3d_create_armature",
    description=(
        "Create a humanoid skeleton/rig for character animation. "
        "Returns bone hierarchy (Hips, Spine, Chest, Head, Arms, Legs). "
        "Output: JSON bone data for animation."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Armature name"},
            "height": {"type": "number", "description": "Character height in meters (default 1.8)"},
        },
    },
    risk="low",
    category="3d_engine",
)
async def nexus3d_create_armature(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        name = params.get("name", "Armature")
        height = float(params.get("height", 1.8))

        armature = create_humanoid_armature(height=height, name=name)
        armature.update_transforms()

        bone_info = {}
        for bone_name, bone in armature.bones.items():
            bone_info[bone_name] = {
                "head": _numpy_safe(bone.world_head),
                "tail": _numpy_safe(bone.world_tail),
            }

        result = {
            "name": armature.name,
            "bone_count": len(armature.bones),
            "bones": bone_info,
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"[Nexus3D Error] {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: Solve IK
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="nexus3d_solve_ik",
    description=(
        "Solve inverse kinematics for a joint chain. Solvers: 2bone (2-joint analytical), "
        "fabrik (Forward And Backward Reaching Inverse Kinematics), ccd (Cyclic Coordinate Descent). "
        "Provide joint positions and a target position."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "solver": {
                "type": "string",
                "enum": ["2bone", "fabrik", "ccd"],
                "description": "IK solver algorithm",
            },
            "joints": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
                "description": "Joint positions [[x,y,z], ...]",
            },
            "target": {"type": "array", "items": {"type": "number"}, "description": "Target [x,y,z]"},
        },
        "required": ["solver", "joints", "target"],
    },
    risk="low",
    category="3d_engine",
)
async def nexus3d_solve_ik(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        solver = params.get("solver", "fabrik")
        joints_np = [_np.array(j, dtype=_np.float64) for j in params["joints"]]
        target_np = _np.array(params["target"], dtype=_np.float64)

        if solver == "2bone" and len(joints_np) >= 2:
            la = float(_np.linalg.norm(joints_np[1] - joints_np[0]))
            lb = float(_np.linalg.norm(joints_np[-1] - joints_np[-2]))
            elbow, wrist = solve_2bone_ik(joints_np[0], target_np, la, lb)
            solved = [joints_np[0].tolist(), elbow.tolist(), wrist.tolist()]
        elif solver == "fabrik":
            solved = [j.tolist() for j in solve_fabrik(joints_np, target_np)]
        elif solver == "ccd":
            solved = [j.tolist() for j in solve_ccd(joints_np, target_np)]
        else:
            return f"Unknown solver: {solver}"

        return json.dumps({
            "solver": solver,
            "original_joints": [j.tolist() for j in joints_np],
            "solved_joints": solved,
            "target": params["target"],
        }, indent=2)
    except Exception as e:
        return f"[Nexus3D Error] {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: Animate Procedural
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="nexus3d_animate_procedural",
    description=(
        "Generate procedural animation keyframes. Types: walk (full walk cycle), "
        "breathe (chest breathing), idle_sway (hip idle), sine_wave (parametric). "
        "Returns per-frame pose data with bone rotations."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "animation_type": {
                "type": "string",
                "enum": ["walk", "breathe", "idle_sway", "sine_wave"],
                "description": "Type of procedural animation",
            },
            "duration": {"type": "number", "description": "Duration in seconds (default 2.0)"},
            "fps": {"type": "integer", "description": "Frames per second (default 30)"},
            "save_to_workspace": {"type": "boolean", "description": "Save to workspace as JSON"},
        },
        "required": ["animation_type"],
    },
    risk="low",
    category="3d_engine",
)
async def nexus3d_animate_procedural(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        anim_type = params.get("animation_type", "walk")
        duration = float(params.get("duration", 2.0))
        fps = int(params.get("fps", 30))
        num_frames = int(duration * fps)

        frames = []
        for i in range(num_frames):
            t = i / fps
            frame = {"frame": i, "time": round(t, 4)}

            if anim_type == "walk":
                pose = ProceduralAnimation.walk_cycle(t)
                frame["pose"] = {}
                for bone, (pos_off, rot) in pose.items():
                    frame["pose"][bone] = {
                        "position_offset": pos_off.tolist(),
                        "rotation": rot.tolist(),
                    }
            elif anim_type == "breathe":
                val = ProceduralAnimation.breathing(t)
                frame["pose"] = {"Chest": {"position_offset": [0, float(val), 0],
                                             "rotation": [1, 0, 0, 0]}}
            elif anim_type == "idle_sway":
                val = ProceduralAnimation.idle_sway(t)
                frame["pose"] = {"Hips": {"position_offset": val.tolist(),
                                           "rotation": [1, 0, 0, 0]}}
            elif anim_type == "sine_wave":
                frame["value"] = float(ProceduralAnimation.sine_wave(t))

            frames.append(frame)

        result = {
            "type": anim_type,
            "duration": duration,
            "fps": fps,
            "total_frames": num_frames,
        }

        # Always save to workspace
        from ...config import BASE_DIR
        workspace = Path(BASE_DIR) / "data" / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        filepath = workspace / f"anim_{anim_type}.json"
        filepath.write_text(json.dumps({"frames": frames, **result}, indent=2))
        result["file"] = f"anim_{anim_type}.json"
        result["workspace_path"] = str(filepath)
        result["download"] = "✅ Animation saved to workspace."

        # Return summary + first few frames
        result["frames_preview"] = frames[:3]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"[Nexus3D Error] {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: Material Library
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="nexus3d_material_library",
    description=(
        "Query the PBR material library. List all available materials or get a "
        "specific material by name. Materials include albedo, metallic, roughness, "
        "normal strength, and emission properties."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "material_name": {"type": "string", "description": "Material name to look up (leave empty to list all)"},
        },
    },
    risk="low",
    category="3d_engine",
)
async def nexus3d_material_library(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        name = params.get("material_name", "")
        if name:
            mat = MaterialLibrary.get(name)
            return json.dumps(mat, indent=2)
        else:
            available = MaterialLibrary.list_materials()
            return json.dumps({
                "available_materials": available,
                "count": len(available),
            }, indent=2)
    except KeyError:
        return f"Material '{name}' not found. Available: {MaterialLibrary.list_materials()}"
    except Exception as e:
        return f"[Nexus3D Error] {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: Cinematic DOF
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="nexus3d_cinematic_dof",
    description=(
        "Calculate cinematic depth-of-field parameters. Given focal length, aperture (f-stop), "
        "and focus distance, computes near plane, far plane, hyperfocal distance, and total DOF."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "focal_length_mm": {"type": "number", "description": "Focal length in mm (default 50)"},
            "aperture_fstop": {"type": "number", "description": "F-stop value (default 2.8)"},
            "focus_distance_m": {"type": "number", "description": "Focus distance in meters (default 10)"},
        },
    },
    risk="low",
    category="3d_engine",
)
async def nexus3d_cinematic_dof(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        dof = DepthOfField(
            focal_length=float(params.get("focal_length_mm", 50)),
            aperture=float(params.get("aperture_fstop", 2.8)),
            focus_distance=float(params.get("focus_distance_m", 10)),
        )
        far = dof.far_plane()
        return json.dumps({
            "focal_length_mm": params.get("focal_length_mm", 50),
            "aperture_fstop": params.get("aperture_fstop", 2.8),
            "focus_distance_m": params.get("focus_distance_m", 10),
            "near_plane_m": round(dof.near_plane(), 3),
            "far_plane_m": round(far, 3) if far < float('inf') else "infinity",
            "hyperfocal_m": round(dof.hyperfocal_distance(), 3),
        }, indent=2)
    except Exception as e:
        return f"[Nexus3D Error] {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: Physics Simulate
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="nexus3d_physics_simulate",
    description=(
        "Run a simple rigid-body physics simulation with gravity. Configure mass, "
        "position, velocity, gravity, and duration. Returns trajectory data."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "mass": {"type": "number", "description": "Body mass in kg (default 1.0)"},
            "position": {"type": "array", "items": {"type": "number"}, "description": "Start position [x,y,z]"},
            "velocity": {"type": "array", "items": {"type": "number"}, "description": "Start velocity [x,y,z]"},
            "gravity": {"type": "number", "description": "Gravity m/s^2 (default 9.81)"},
            "duration": {"type": "number", "description": "Simulation duration in seconds (default 2.0)"},
            "dt": {"type": "number", "description": "Time step (default 0.016)"},
        },
    },
    risk="low",
    category="3d_engine",
)
async def nexus3d_physics_simulate(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        body = PhysicsBody(
            mass=float(params.get("mass", 1.0)),
            position=_np.array(params.get("position", [0, 10, 0]), dtype=_np.float64),
            velocity=_np.array(params.get("velocity", [5, 0, 0]), dtype=_np.float64),
        )

        gravity = float(params.get("gravity", 9.80665))
        duration = float(params.get("duration", 2.0))
        dt = float(params.get("dt", 0.016))

        trajectory = []
        steps = int(duration / dt)
        for step in range(steps + 1):
            trajectory.append({
                "step": step,
                "time": round(step * dt, 4),
                "position": body.position.tolist(),
                "velocity": body.velocity.tolist(),
            })
            body.apply_force(vec3(0, -gravity * body.mass, 0))
            body.step(dt)

        # Return summary + key frames
        result = {
            "mass": params.get("mass", 1.0),
            "gravity": gravity,
            "duration": duration,
            "steps": steps,
            "final_position": trajectory[-1]["position"],
            "trajectory_preview": trajectory[:5],
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"[Nexus3D Error] {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: Raycast
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="nexus3d_raycast",
    description=(
        "Ray intersection test. Cast a ray from origin in a direction and test "
        "against geometry (sphere, plane, triangle). Returns hit point and distance."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "origin": {"type": "array", "items": {"type": "number"}, "description": "Ray origin [x,y,z]"},
            "direction": {"type": "array", "items": {"type": "number"}, "description": "Ray direction [x,y,z]"},
            "geometry_type": {"type": "string", "enum": ["sphere", "plane", "triangle"]},
            "center": {"type": "array", "items": {"type": "number"}, "description": "Geometry center (sphere)"},
            "radius": {"type": "number", "description": "Sphere radius"},
        },
        "required": ["origin", "direction", "geometry_type"],
    },
    risk="low",
    category="3d_engine",
)
async def nexus3d_raycast(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        ray_o = _np.array(params["origin"], dtype=_np.float64)
        ray_d = normalize(_np.array(params["direction"], dtype=_np.float64))
        geo_type = params.get("geometry_type", "sphere")

        result = {"hit": False, "geometry_type": geo_type}

        if geo_type == "sphere":
            center = _np.array(params.get("center", [0, 0, 0]), dtype=_np.float64)
            radius = float(params.get("radius", 1.0))
            hit = ray_sphere_intersect(ray_o, ray_d, center, radius)
            if hit:
                result = {
                    "hit": True,
                    "t_near": hit[0], "t_far": hit[1],
                    "point_near": (ray_o + ray_d * hit[0]).tolist(),
                    "point_far": (ray_o + ray_d * hit[1]).tolist(),
                }
        elif geo_type == "plane":
            point = _np.array(params.get("center", [0, 0, 0]), dtype=_np.float64)
            normal = _np.array(params.get("direction", [0, 1, 0]), dtype=_np.float64)
            t = ray_plane_intersect(ray_o, ray_d, point, normal)
            if t is not None:
                result = {"hit": True, "t": t, "point": (ray_o + ray_d * t).tolist()}

        return json.dumps(result, indent=2)
    except Exception as e:
        return f"[Nexus3D Error] {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: Nexus3D Info
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="nexus3d_info",
    description=(
        "Get Nexus3D engine information: version, available capabilities, "
        "supported features (mesh types, IK solvers, animation, rendering, CSG, materials)."
    ),
    parameters_schema={"type": "object", "properties": {}},
    risk="low",
    category="3d_engine",
)
async def nexus3d_info(_params: dict) -> str:
    if not _NEXUS3D_AVAILABLE:
        return json.dumps({"available": False, "error": _IMPORT_ERROR}, indent=2)

    try:
        import nexus3d
        version = getattr(nexus3d, "__version__", "unknown")
    except Exception:
        version = "0.2.0"

    return json.dumps({
        "available": True,
        "version": version,
        "engine": "Nexus3D",
        "capabilities": {
            "mesh_primitives": ["cube", "sphere", "cylinder", "cone", "torus", "plane", "arrow", "circle", "grid"],
            "csg_operations": ["union", "subtract", "intersect"],
            "csg_shapes": ["cube", "sphere", "cylinder", "cone", "torus"],
            "ik_solvers": ["2bone", "fabrik", "ccd"],
            "animation_types": ["walk", "breathe", "idle_sway", "sine_wave"],
            "rigging": ["humanoid_armature", "auto_skin_weights"],
            "materials": "PBR library available",
            "cinematic": ["depth_of_field", "exposure"],
            "physics": ["rigid_body", "gravity"],
            "math": ["transforms", "quaternions", "raycast"],
        },
        "skills": [
            "architectural_toolkit", "character_creator", "scene_composer",
            "animation_director", "motion_pipeline", "product_renderer",
            "physics_lab", "csg_architect", "studio_renderer",
            "cinema_camera", "csg_toolkit", "material_lab",
        ],
    }, indent=2)
