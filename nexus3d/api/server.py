"""Nexus3D REST API - HTTP interface for AI agents.

Start with: nexus3d serve --host 0.0.0.0 --port 8420
API docs at: http://localhost:8420/docs
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import json
import os
import tempfile
import shutil

# ─── Custom JSON Encoder ─────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def json_response(data: Any, status_code: int = 200):
    return JSONResponse(
        content=json.loads(json.dumps(data, cls=NumpyEncoder)),
        status_code=status_code
    )


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class Vec3Input(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class IKRequest(BaseModel):
    type: str = Field(..., description="IK solver: '2bone', 'fabrik', or 'ccd'")
    joints: List[List[float]] = Field(..., description="List of joint positions [[x,y,z], ...]")
    target: List[float] = Field(..., description="Target position [x,y,z]")
    pole: Optional[List[float]] = None
    tolerance: float = 0.01
    max_iterations: int = 100
    bend_factor: float = 0.5


class TransformRequest(BaseModel):
    vertices: List[List[float]] = Field(..., description="Vertex positions")
    faces: Optional[List[List[int]]] = None
    matrix: List[float] = Field(..., description="4x4 transform matrix (16 floats, row-major)")


class MeshCreateRequest(BaseModel):
    type: str = Field(..., description="Primitive type: cube, sphere, cylinder, cone, torus, plane, arrow")
    name: str = "Mesh"
    params: Dict[str, Any] = {}


class ArmatureCreateRequest(BaseModel):
    type: str = "humanoid"
    name: str = "Armature"
    height: float = 1.8


class ArmatureIKRequest(BaseModel):
    armature: Dict[str, Any] = Field(..., description="Armature data (from create)")
    end_bone: str = Field(..., description="Name of the end-effector bone")
    target: List[float] = Field(..., description="Target world position")
    solver: str = "fabrik"
    pole: Optional[List[float]] = None


class ProceduralAnimRequest(BaseModel):
    type: str = Field(..., description="walk, breathe, idle_sway, sine_wave")
    params: Dict[str, Any] = {}
    duration: float = 2.0
    fps: int = 30


class AnimEvaluateRequest(BaseModel):
    animation: Dict[str, Any] = Field(..., description="Animation clip data")
    time: float = 0.0


class BVHExportRequest(BaseModel):
    armature: Dict[str, Any]
    animation: Dict[str, Any]


class RenderRequest(BaseModel):
    meshes: List[Dict[str, Any]] = Field(..., description="List of mesh data dicts")
    camera_pos: List[float] = [0, 2, 5]
    camera_target: List[float] = [0, 0, 0]
    camera_fov: float = 60.0
    width: int = 1920
    height: int = 1080
    lighting: str = "3point"


class PhysicsSimRequest(BaseModel):
    bodies: List[Dict[str, Any]] = []
    gravity: float = 9.80665
    duration: float = 2.0
    dt: float = 1.0 / 60.0


# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Nexus3D API",
    description="Headless 3D engine for AI agents — create, rig, animate, and render 3D content via HTTP",
    version="0.2.0",
    contact={"name": "Nexus AI Lab"},
    license={"name": "MIT"},
)


# ─── Health & Info ───────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.2.0", "engine": "Nexus3D"}


@app.get("/capabilities")
async def get_capabilities():
    """List available engine capabilities."""
    try:
        import pyrender
        render_backend = "pyrender (OSMesa)"
    except ImportError:
        render_backend = "software_rasterizer"

    try:
        import trimesh
        mesh_backend = "trimesh"
    except ImportError:
        mesh_backend = "numpy"

    return {
        "math": ["vectors", "matrices", "quaternions", "ray_intersection", "geometry"],
        "physics": ["rigid_body", "gravity", "collision"],
        "ik_solvers": ["2bone", "fabrik", "ccd"],
        "mesh_primitives": ["cube", "sphere", "cylinder", "cone", "torus", "plane", "arrow", "circle", "grid"],
        "mesh_operations": ["transform", "subdivide", "merge", "extrude", "inset"],
        "rigging": ["armature", "humanoid_skeleton", "ik", "skin_weights"],
        "animation": ["keyframe", "procedural", "mixing", "blend_shapes"],
        "animation_export": ["bvh", "gltf", "fbx_keyframes"],
        "rendering": {
            "backend": render_backend,
            "features": ["pbr_materials", "pbr_lighting", "shadows"],
            "output_formats": ["png", "jpg", "mp4"]
        },
        "mesh_backend": mesh_backend,
        "formats": ["obj", "json", "bvh", "gltf", "png", "mp4"],
        "path_tracing": {
            "available": True,
            "features": ["global_illumination", "pbr_brdf", "cook_torrance", "ggx_ndf",
                        "smith_geometry", "schlick_fresnel", "soft_shadows",
                        "area_lights", "ambient_occlusion", "sss_approximation",
                        "bvh_acceleration", "russian_roulette", "progressive_rendering",
                        "hdr_tonemapping", "depth_of_field", "importance_sampling"],
            "tonemap_options": ["aces", "reinhard", "linear"],
        },
        "csg": {
            "available": True,
            "operations": ["union", "subtract", "intersect"],
            "primitives": ["cube", "sphere", "cylinder", "cone", "torus"],
            "architecture": ["wall", "door", "window", "room", "stairs", "pillar", "arch", "floor_plan"],
            "mesh_cleanup": ["weld_vertices", "remove_degenerates", "fix_normals"],
        },
        "materials": {
            "procedural_textures": ["noise", "checker", "brick", "wood", "marble", "gradient", "voronoi", "image"],
            "uv_projections": ["planar", "box", "spherical", "cylindrical"],
            "material_library": True,
            "layered_materials": True,
            "ibl": True,
        },
        "cinematic": {
            "camera_animation": True,
            "depth_of_field": True,
            "motion_blur": True,
            "camera_shake_presets": ["handheld", "earthquake", "breathing", "wind", "vehicle"],
            "camera_presets": ["dolly_zoom", "orbit", "tracking_shot", "crane_shot", "reveal", "push_in"],
            "exposure_system": True,
            "lens_profiles": ["14mm_ultrawide", "24mm_wide", "35mm_standard", "50mm_normal", "85mm_portrait", "135mm_telephoto"],
        },
    }


# ─── Math Endpoints ──────────────────────────────────────────────────────────

@app.post("/api/v1/math/ik")
async def solve_ik(req: IKRequest):
    """Solve inverse kinematics for a joint chain."""
    try:
        from nexus3d.math3d.core import solve_2bone_ik, solve_fabrik, solve_ccd, vec3
    except ImportError:
        raise HTTPException(500, "Math engine not available")

    joints_np = [np.array(j, dtype=np.float64) for j in req.joints]
    target_np = np.array(req.target, dtype=np.float64)

    try:
        if req.type == "2bone" and len(joints_np) >= 2:
            pole = np.array(req.pole, dtype=np.float64) if req.pole else None
            la = np.linalg.norm(joints_np[1] - joints_np[0])
            lb = np.linalg.norm(joints_np[-1] - joints_np[-2])
            elbow, wrist = solve_2bone_ik(joints_np[0], target_np, la, lb, pole, req.bend_factor)
            result_joints = [joints_np[0].tolist(), elbow.tolist(), wrist.tolist()]
        elif req.type == "fabrik":
            result_joints = solve_fabrik(joints_np, target_np, req.tolerance, req.max_iterations)
            result_joints = [j.tolist() for j in result_joints]
        elif req.type == "ccd":
            result_joints = solve_ccd(joints_np, target_np, req.tolerance, req.max_iterations)
            result_joints = [j.tolist() for j in result_joints]
        else:
            raise HTTPException(400, f"Unknown IK solver: {req.type}")

        return json_response({
            "solver": req.type,
            "original_joints": [j.tolist() for j in joints_np],
            "solved_joints": result_joints,
            "target": req.target
        })
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/math/transform/compose")
async def compose_transform(
    translation: List[float] = [0, 0, 0],
    rotation: List[float] = [1, 0, 0, 0],  # quaternion [w, x, y, z]
    scale: List[float] = [1, 1, 1]
):
    """Compose a 4x4 transform from translation, rotation (quaternion), and scale."""
    try:
        from nexus3d.math3d.core import compose_matrix
        t = np.array(translation, dtype=np.float64)
        r = np.array(rotation, dtype=np.float64)
        s = np.array(scale, dtype=np.float64)
        matrix = compose_matrix(t, r, s)
        return json_response({"matrix": matrix.tolist()})
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/math/transform/decompose")
async def decompose_transform(matrix: List[float]):
    """Decompose a 4x4 matrix into translation, rotation, scale."""
    try:
        from nexus3d.math3d.core import decompose_matrix
        m = np.array(matrix, dtype=np.float64).reshape(4, 4)
        t, r, s = decompose_matrix(m)
        return json_response({
            "translation": t.tolist(),
            "rotation": r.tolist(),
            "scale": s.tolist()
        })
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/math/raycast")
async def raycast(
    origin: List[float] = [0, 0, 5],
    direction: List[float] = [0, 0, -1],
    geometry_type: str = "sphere",
    geometry_params: Dict[str, Any] = {}
):
    """Ray intersection test against geometry."""
    try:
        from nexus3d.math3d.core import (
            ray_sphere_intersect, ray_plane_intersect,
            ray_triangle_intersect, vec3, normalize
        )
    except ImportError:
        raise HTTPException(500, "Math engine not available")

    ray_o = np.array(origin, dtype=np.float64)
    ray_d = normalize(np.array(direction, dtype=np.float64))

    result = {"hit": False}

    if geometry_type == "sphere":
        center = np.array(geometry_params.get("center", [0, 0, 0]), dtype=np.float64)
        radius = geometry_params.get("radius", 1.0)
        hit = ray_sphere_intersect(ray_o, ray_d, center, radius)
        if hit:
            result = {"hit": True, "t_near": hit[0], "t_far": hit[1],
                       "point_near": (ray_o + ray_d * hit[0]).tolist(),
                       "point_far": (ray_o + ray_d * hit[1]).tolist()}
    elif geometry_type == "plane":
        point = np.array(geometry_params.get("point", [0, 0, 0]), dtype=np.float64)
        normal = np.array(geometry_params.get("normal", [0, 1, 0]), dtype=np.float64)
        t = ray_plane_intersect(ray_o, ray_d, point, normal)
        if t is not None:
            result = {"hit": True, "t": t, "point": (ray_o + ray_d * t).tolist()}
    elif geometry_type == "triangle":
        v0 = np.array(geometry_params["v0"], dtype=np.float64)
        v1 = np.array(geometry_params["v1"], dtype=np.float64)
        v2 = np.array(geometry_params["v2"], dtype=np.float64)
        t = ray_triangle_intersect(ray_o, ray_d, v0, v1, v2)
        if t is not None:
            result = {"hit": True, "t": t, "point": (ray_o + ray_d * t).tolist()}

    return json_response(result)


@app.post("/api/v1/math/physics/simulate")
async def physics_simulate(req: PhysicsSimRequest):
    """Run a simple physics simulation."""
    try:
        from nexus3d.math3d.core import PhysicsBody, vec3
    except ImportError:
        raise HTTPException(500, "Math engine not available")

    bodies = []
    for bd in req.bodies:
        body = PhysicsBody(
            mass=bd.get("mass", 1.0),
            position=np.array(bd.get("position", [0, 0, 0]), dtype=np.float64),
            velocity=np.array(bd.get("velocity", [0, 0, 0]), dtype=np.float64),
            is_static=bd.get("static", False),
        )
        if "force" in bd:
            body.apply_force(np.array(bd["force"], dtype=np.float64))
        bodies.append(body)

    trajectory = []
    steps = int(req.duration / req.dt)
    for step in range(steps + 1):
        frame = {}
        for i, body in enumerate(bodies):
            frame[f"body_{i}"] = {
                "position": body.position.tolist(),
                "velocity": body.velocity.tolist(),
            }
            # Apply gravity
            body.apply_force(vec3(0, -req.gravity * body.mass, 0))
            body.step(req.dt)
        trajectory.append({"step": step, "time": round(step * req.dt, 6), "bodies": frame})

    return json_response({"duration": req.duration, "steps": steps, "trajectory": trajectory})


# ─── Mesh Endpoints ──────────────────────────────────────────────────────────

@app.post("/api/v1/mesh/create")
async def create_mesh(req: MeshCreateRequest):
    """Create a mesh primitive."""
    try:
        from nexus3d.mesh.primitives import (
            create_cube, create_sphere, create_cylinder, create_cone,
            create_torus, create_plane, create_arrow
        )
    except ImportError:
        raise HTTPException(500, "Mesh engine not available")

    primitive_map = {
        "cube": create_cube,
        "sphere": create_sphere,
        "cylinder": create_cylinder,
        "cone": create_cone,
        "torus": create_torus,
        "plane": create_plane,
        "arrow": create_arrow,
    }

    if req.type not in primitive_map:
        raise HTTPException(400, f"Unknown primitive: {req.type}. Available: {list(primitive_map.keys())}")

    try:
        func = primitive_map[req.type]
        params = {k: v for k, v in req.params.items() if k in func.__code__.co_varnames}
        mesh_data = func(**params)
        return json_response({
            "name": req.name,
            "type": req.type,
            "vertex_count": len(mesh_data["vertices"]),
            "face_count": len(mesh_data["faces"]),
            "vertices": mesh_data["vertices"].tolist(),
            "faces": mesh_data["faces"].tolist(),
            "normals": mesh_data["normals"].tolist() if mesh_data.get("normals") is not None else None,
            "uvs": mesh_data["uvs"].tolist() if mesh_data.get("uvs") is not None else None,
        })
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/mesh/transform")
async def transform_mesh(req: TransformRequest):
    """Apply a 4x4 transform to mesh vertices."""
    try:
        from nexus3d.math3d.core import vec3
    except ImportError:
        raise HTTPException(500, "Math engine not available")

    vertices = np.array(req.vertices, dtype=np.float64)
    matrix = np.array(req.matrix, dtype=np.float64).reshape(4, 4)

    # Apply transform: v' = M * [v, 1]^T
    ones = np.ones((vertices.shape[0], 1), dtype=np.float64)
    verts_h = np.hstack([vertices, ones])
    transformed = (matrix @ verts_h.T).T[:, :3]

    return json_response({
        "vertices": transformed.tolist(),
        "faces": req.faces,
        "vertex_count": len(transformed),
    })


@app.post("/api/v1/mesh/import")
async def import_mesh(file: UploadFile = File(...)):
    """Import a mesh from OBJ file."""
    try:
        from nexus3d.mesh.mesh import Mesh
    except ImportError:
        raise HTTPException(500, "Mesh engine not available")

    suffix = Path(file.filename).suffix.lower() if file.filename else ".obj"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        mesh = Mesh.from_file(tmp_path)
        return json_response({
            "filename": file.filename,
            "name": mesh.name,
            "vertex_count": mesh.vertex_count(),
            "face_count": mesh.face_count(),
            "vertices": mesh.vertices.tolist(),
            "faces": mesh.faces.tolist(),
            "normals": mesh.normals.tolist() if mesh.normals is not None else None,
            "uvs": mesh.uvs.tolist() if mesh.uvs is not None else None,
        })
    except Exception as e:
        raise HTTPException(500, f"Import failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/api/v1/mesh/export")
async def export_mesh(
    vertices: List[List[float]],
    faces: List[List[int]],
    normals: Optional[List[List[float]]] = None,
    uvs: Optional[List[List[float]]] = None,
    name: str = "mesh",
    format: str = "obj"
):
    """Export mesh data to a downloadable file."""
    try:
        from nexus3d.mesh.mesh import Mesh
        from nexus3d.utils.helpers import ensure_output_dir
    except ImportError:
        raise HTTPException(500, "Mesh engine not available")

    try:
        vertices_np = np.array(vertices, dtype=np.float64)
        faces_np = np.array(faces, dtype=np.int64)
        normals_np = np.array(normals, dtype=np.float64) if normals else None
        uvs_np = np.array(uvs, dtype=np.float64) if uvs else None

        mesh = Mesh(name=name, vertices=vertices_np, faces=faces_np,
                     normals=normals_np, uvs=uvs_np)

        out_dir = ensure_output_dir("/tmp/nexus3d_exports")
        if format == "obj":
            filepath = out_dir / f"{name}.obj"
            mesh.save_obj(str(filepath))
        else:
            filepath = out_dir / f"{name}.json"
            with open(filepath, 'w') as f:
                json.dump(mesh.to_dict(), f, cls=NumpyEncoder, indent=2)

        return FileResponse(str(filepath), filename=f"{name}.{format}",
                           media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(500, str(e))


# ─── Rigging Endpoints ───────────────────────────────────────────────────────

@app.post("/api/v1/rig/create")
async def create_armature(req: ArmatureCreateRequest):
    """Create an armature/skeleton."""
    try:
        from nexus3d.rigging.armature import create_humanoid_armature
    except ImportError:
        raise HTTPException(500, "Rigging engine not available")

    try:
        if req.type == "humanoid":
            armature = create_humanoid_armature(height=req.height, name=req.name)
        else:
            raise HTTPException(400, f"Unknown armature type: {req.type}")

        armature.update_transforms()
        return json_response({
            "name": armature.name,
            "bone_count": len(armature.bones),
            "bone_names": list(armature.bones.keys()),
            "data": armature.to_dict()
        })
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/rig/ik")
async def solve_armature_ik(req: ArmatureIKRequest):
    """Solve IK on an armature."""
    try:
        from nexus3d.rigging.armature import Armature
    except ImportError:
        raise HTTPException(500, "Rigging engine not available")

    try:
        armature = Armature.from_dict(req.armature)
        target = np.array(req.target, dtype=np.float64)
        pole = np.array(req.pole, dtype=np.float64) if req.pole else None

        armature.solve_ik(req.end_bone, target, pole=pole, solver=req.solver)
        armature.update_transforms()

        # Get solved bone positions
        bone_positions = {}
        for name, bone in armature.bones.items():
            bone_positions[name] = {
                "head": bone.world_head.tolist(),
                "tail": bone.world_tail.tolist(),
            }

        return json_response({
            "solved": True,
            "solver": req.solver,
            "target": req.target,
            "bone_positions": bone_positions,
            "armature": armature.to_dict()
        })
    except Exception as e:
        raise HTTPException(500, str(e))


# ─── Animation Endpoints ─────────────────────────────────────────────────────

@app.post("/api/v1/anim/procedural")
async def generate_procedural_animation(req: ProceduralAnimRequest):
    """Generate a procedural animation."""
    try:
        from nexus3d.animation.anim import ProceduralAnimation, AnimationClip, AnimTrack, AnimCurve
    except ImportError:
        raise HTTPException(500, "Animation engine not available")

    try:
        frames = []
        num_frames = int(req.duration * req.fps)

        for i in range(num_frames):
            t = i / req.fps
            frame_data = {"frame": i, "time": round(t, 4)}

            if req.type == "walk":
                pose = ProceduralAnimation.walk_cycle(t, **req.params)
                frame_data["pose"] = {}
                for bone_name, (pos_off, rot) in pose.items():
                    frame_data["pose"][bone_name] = {
                        "position_offset": pos_off.tolist(),
                        "rotation": rot.tolist()
                    }
            elif req.type == "breathe":
                pose = ProceduralAnimation.breathing(t, **req.params)
                frame_data["pose"] = {"Chest": {"position_offset": [0, pose, 0], "rotation": [1, 0, 0, 0]}}
            elif req.type == "idle_sway":
                pose = ProceduralAnimation.idle_sway(t, **req.params)
                frame_data["pose"] = {"Hips": {"position_offset": pose.tolist(), "rotation": [1, 0, 0, 0]}}
            elif req.type == "sine_wave":
                val = ProceduralAnimation.sine_wave(t, **req.params)
                frame_data["value"] = val
            else:
                raise HTTPException(400, f"Unknown procedural type: {req.type}")

            frames.append(frame_data)

        return json_response({
            "type": req.type,
            "duration": req.duration,
            "fps": req.fps,
            "total_frames": num_frames,
            "frames": frames
        })
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/anim/export/bvh")
async def export_bvh(req: BVHExportRequest):
    """Export animation as BVH motion capture file."""
    try:
        from nexus3d.rigging.armature import Armature
        from nexus3d.animation.anim import AnimationClip
    except ImportError:
        raise HTTPException(500, "Animation engine not available")

    try:
        armature = Armature.from_dict(req.armature)
        clip = AnimationClip.from_dict(req.animation)

        out_dir = Path("/tmp/nexus3d_exports")
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / "animation.bvh"

        from nexus3d.animation.anim import AnimationExporter
        AnimationExporter.to_bvh(armature, clip, str(filepath))

        return FileResponse(str(filepath), filename="animation.bvh",
                           media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(500, str(e))


# ─── Render Endpoints ────────────────────────────────────────────────────────

@app.post("/api/v1/render")
async def render_scene(req: RenderRequest):
    """Render a 3D scene to an image."""
    try:
        from nexus3d.rendering.renderer import Renderer, Camera, Light
        from nexus3d.rendering.renderer import Material
    except ImportError:
        raise HTTPException(500, "Rendering engine not available")

    try:
        camera = Camera()
        camera.position = np.array(req.camera_pos, dtype=np.float64)
        camera.target = np.array(req.camera_target, dtype=np.float64)
        camera.set_resolution(req.width, req.height)
        camera.fov = req.camera_fov

        if req.lighting == "3point":
            lights = [
                Light.directional(direction=[-1, -1, -1], color="#ffffff", intensity=1.0),
                Light.directional(direction=[1, 0, -1], color="#8888ff", intensity=0.5),
                Light.directional(direction=[0, 1, 0], color="#ffff88", intensity=0.3),
            ]
        else:
            lights = Light.create_default_lights() if hasattr(Light, 'create_default_lights') else []

        renderer = Renderer(resolution=(req.width, req.height))

        # Convert mesh data to renderable format
        scene_data = {"meshes": req.meshes}

        out_dir = Path("/tmp/nexus3d_renders")
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / f"render_{hash(str(req.meshes)) % 100000}.png"

        renderer.render_to_file(scene_data, str(filepath), camera=camera, lights=lights)

        return FileResponse(str(filepath), filename="render.png",
                           media_type="image/png")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/render/video")
async def render_video(
    fps: int = 30,
    output_name: str = "animation.mp4",
    file: UploadFile = File(...)
):
    """Upload a zip of frames and get a video back."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise HTTPException(500, "imageio not available")

    import zipfile
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        out_dir = Path("/tmp/nexus3d_video")
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(tmp_path, out_dir)

        # Collect and sort images
        frames = sorted(out_dir.glob("*.png")) + sorted(out_dir.glob("*.jpg"))
        if not frames:
            raise HTTPException(400, "No image files found in zip")

        video_path = out_dir / output_name
        writer = imageio.get_writer(str(video_path), fps=fps)
        for f in frames:
            img = imageio.imread(str(f))
            writer.append_data(img)
        writer.close()

        return FileResponse(str(video_path), filename=output_name,
                           media_type="video/mp4")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)
        shutil.rmtree(out_dir, ignore_errors=True)


# ─── Path Tracer Endpoints ──────────────────────────────────────────────────

class PathTraceRequest(BaseModel):
    width: int = 400
    height: int = 300
    spp: int = 16
    max_bounces: int = 8
    camera_pos: List[float] = [0, 2, 6]
    camera_target: List[float] = [0, 0.5, 0]
    camera_fov: float = 60.0
    aperture: float = 0.0
    focus_distance: float = 10.0
    tonemap: str = "aces"
    background_color: List[float] = [0.4, 0.6, 1.0]
    objects: List[Dict[str, Any]] = []


@app.post("/api/v1/render/pathtrace")
async def render_pathtrace(req: PathTraceRequest):
    """Render using Monte Carlo path tracing for production-quality output."""
    try:
        from nexus3d.rendering.pathtracer import PathTracer, PTMaterial
    except ImportError:
        raise HTTPException(500, "Path tracer not available")

    try:
        import numpy as np
        pt = PathTracer(width=req.width, height=req.height)
        pt.set_tonemap(req.tonemap)
        pt.set_camera(
            np.array(req.camera_pos, dtype=np.float64),
            np.array(req.camera_target, dtype=np.float64),
            fov=req.camera_fov,
            aperture=req.aperture,
            focal_distance=req.focus_distance,
        )
        pt.set_environment(np.array(req.background_color, dtype=np.float64))

        # Default scene if no objects specified
        if not req.objects:
            pt.add_plane(np.array([0,1,0]), np.array([0,0,0]),
                        PTMaterial(albedo=np.array([0.4,0.4,0.4]), roughness=0.9))
            pt.add_sphere(np.array([0,1,0]), 1.0,
                        PTMaterial(albedo=np.array([0.8,0.1,0.1]), metallic=0.3, roughness=0.3))
        else:
            for obj in req.objects:
                otype = obj.get("type", "sphere")
                mat_data = obj.get("material", {})
                mat = PTMaterial(
                    albedo=np.array(mat_data.get("albedo", [0.8,0.8,0.8])),
                    metallic=mat_data.get("metallic", 0.0),
                    roughness=mat_data.get("roughness", 0.5),
                    emission=np.array(mat_data.get("emission", [0,0,0])),
                    transmission=mat_data.get("transmission", 0.0),
                    ior=mat_data.get("ior", 1.5),
                )
                pos = np.array(obj.get("position", [0,0,0]))
                if otype == "sphere":
                    pt.add_sphere(pos, obj.get("radius", 1.0), mat)
                elif otype == "plane":
                    normal = np.array(obj.get("normal", [0,1,0]))
                    pt.add_plane(normal, pos, mat)

        image = pt.render(samples_per_pixel=req.spp, max_bounces=req.max_bounces)

        out_dir = Path("/tmp/nexus3d_pathtrace")
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / f"pt_{hash(str(req.objects)) % 100000}.png"
        pt.save(str(filepath))

        return FileResponse(str(filepath), filename="pathtrace.png",
                           media_type="image/png")
    except Exception as e:
        raise HTTPException(500, str(e))


# ─── CSG Endpoints ──────────────────────────────────────────────────────────

class CSGRequest(BaseModel):
    operation: str = Field(..., description="union, subtract, or intersect")
    shape_a: Dict[str, Any] = Field(..., description="First shape: {type, params}")
    shape_b: Optional[Dict[str, Any]] = None


@app.post("/api/v1/csg/boolean")
async def csg_boolean(req: CSGRequest):
    """Perform real boolean CSG operations on meshes."""
    try:
        from nexus3d.csg.engine import CSGMesh, MeshCleaner
    except ImportError:
        raise HTTPException(500, "CSG engine not available")

    try:
        import numpy as np
        shape_map = {
            'cube': lambda: CSGMesh.from_cube(1.0),
            'sphere': lambda: CSGMesh.from_sphere(0.7, segments=16),
            'cylinder': lambda: CSGMesh.from_cylinder(0.5, 1.5, segments=16),
            'cone': lambda: CSGMesh.from_cone(0.5, 1.5, segments=16),
            'torus': lambda: CSGMesh.from_torus(0.8, 0.25),
        }

        a_type = req.shape_a.get("type", "cube")
        a = shape_map.get(a_type, CSGMesh.from_cube)()

        if req.shape_b is not None:
            b_type = req.shape_b.get("type", "sphere")
            b = shape_map.get(b_type, CSGMesh.from_sphere)()
            offset = req.shape_b.get("offset", [0, 0, 0])
            off = np.array(offset, dtype=np.float64)
            for i in range(len(b._polygons)):
                for j in range(len(b._polygons[i].vertices)):
                    b._polygons[i].vertices[j] = b._polygons[i].vertices[j] + off
        else:
            b = CSGMesh.from_sphere(0.7, segments=16)

        ops = {
            'union': lambda: a.union(b),
            'subtract': lambda: a.subtract(b),
            'intersect': lambda: a.intersect(b),
        }
        if req.operation not in ops:
            raise HTTPException(400, f"Unknown operation: {req.operation}")

        result = ops[req.operation]()
        vertices, faces = MeshCleaner.clean(*result.to_vertices_faces())

        return json_response({
            "operation": req.operation,
            "vertices": len(vertices),
            "faces": len(faces),
            "volume": result.volume(),
            "vertices_data": vertices.tolist(),
            "faces_data": faces.tolist(),
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ─── Material Endpoints ─────────────────────────────────────────────────────

@app.get("/api/v1/materials/library")
async def list_materials():
    """List all available PBR materials."""
    try:
        from nexus3d.materials.pipeline import MaterialLibrary
    except ImportError:
        raise HTTPException(500, "Material library not available")
    return json_response({"materials": MaterialLibrary.list()})


@app.get("/api/v1/materials/{name}")
async def get_material(name: str):
    """Get a specific PBR material."""
    try:
        from nexus3d.materials.pipeline import MaterialLibrary
    except ImportError:
        raise HTTPException(500, "Material library not available")
    try:
        mat = MaterialLibrary.get(name)
        return json_response(mat)
    except KeyError:
        raise HTTPException(404, f"Material '{name}' not found")


# ─── Cinematic Endpoints ───────────────────────────────────────────────────

class DOFRequest(BaseModel):
    focal_length: float = 50.0
    aperture: float = 2.8
    focus_distance: float = 10.0

@app.post("/api/v1/cinematic/dof")
async def calculate_dof(req: DOFRequest):
    """Calculate depth of field parameters."""
    try:
        from nexus3d.cinematic.camera import DepthOfField
    except ImportError:
        raise HTTPException(500, "Cinematic module not available")
    dof = DepthOfField(focal_length=req.focal_length, aperture=req.aperture,
                       focus_distance=req.focus_distance)
    far = dof.far_plane()
    return json_response({
        "focal_length_mm": req.focal_length,
        "aperture_fstop": req.aperture,
        "focus_distance_m": req.focus_distance,
        "near_plane_m": round(dof.near_plane(), 3),
        "far_plane_m": round(far, 3) if far < float('inf') else None,
        "hyperfocal_m": round(dof.hyperfocal_distance(), 3),
    })


class ExposureRequest(BaseModel):
    aperture: float = 2.8
    shutter_time: float = 1.0/48.0
    iso: float = 800.0

@app.post("/api/v1/cinematic/exposure")
async def calculate_exposure(req: ExposureRequest):
    """Calculate exposure parameters."""
    try:
        from nexus3d.cinematic.camera import Exposure
    except ImportError:
        raise HTTPException(500, "Cinematic module not available")
    exp = Exposure(aperture=req.aperture, shutter_time=req.shutter_time, iso=req.iso)
    return json_response({
        "ev100": round(exp.get_ev(), 2),
        "aperture": req.aperture,
        "shutter_time": req.shutter_time,
        "iso": req.iso,
    })


# ─── Run with uvicorn ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8420)
