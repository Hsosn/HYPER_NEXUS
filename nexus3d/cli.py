#!/usr/bin/env python3
"""Nexus3D CLI - Headless 3D engine for AI agents.

Usage:
    nexus3d create mesh MyCube --type sphere
    nexus3d render scene model.obj --output render.png
    nexus3d animate procedural --type walk --duration 3.0
    nexus3d math ik --type fabrik --joints "0,0,0;0,1,0;0,2,0" --target "1,1,1"
    nexus3d serve --port 8420
"""

import sys
import os
import json
import click
import numpy as np
from pathlib import Path

# Add project root to path for development
try:
    # When installed as package, imports work directly
    from nexus3d.utils.helpers import (
        parse_vec3, parse_rotation, ensure_output_dir,
        save_json, load_json, NumpyEncoder
    )
except ImportError:
    # For development: add parent dir to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from nexus3d.utils.helpers import (
        parse_vec3, parse_rotation, ensure_output_dir,
        save_json, load_json, NumpyEncoder
    )


# ─── Output helper ───────────────────────────────────────────────────────────

def output_json(data, verbose=False):
    """Print data as JSON."""
    click.echo(json.dumps(data, cls=NumpyEncoder, indent=2 if verbose else None))


# ─── Main CLI ────────────────────────────────────────────────────────────────

@click.group()
@click.option('--output-dir', '-o', default='.', help='Output directory')
@click.option('--format', '-f', default='obj', help='Output format')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, output_dir, format, verbose):
    """Nexus3D - Headless 3D Engine for AI Agents."""
    ctx.ensure_object(dict)
    ctx.obj['output_dir'] = Path(output_dir)
    ctx.obj['format'] = format
    ctx.obj['verbose'] = verbose


# ─── CREATE commands ─────────────────────────────────────────────────────────

@cli.group()
@click.pass_context
def create(ctx):
    """Create 3D objects."""
    pass


@create.command('mesh')
@click.argument('name')
@click.option('--type', 'mesh_type', required=True,
              type=click.Choice(['cube', 'sphere', 'cylinder', 'cone', 'torus', 'plane', 'arrow', 'circle', 'grid']))
@click.option('--size', default=1.0, help='Size parameter')
@click.option('--radius', default=1.0, help='Radius parameter')
@click.option('--height', default=2.0, help='Height parameter')
@click.option('--segments', default=32, help='Segment count')
@click.option('--save', is_flag=True, help='Save to file')
@click.pass_context
def create_mesh(ctx, name, mesh_type, size, radius, height, segments, save):
    """Create a 3D mesh primitive."""
    try:
        from nexus3d.mesh.primitives import (
            create_cube, create_sphere, create_cylinder, create_cone,
            create_torus, create_plane, create_arrow, create_circle, create_grid
        )
    except ImportError as e:
        click.echo(json.dumps({"error": f"Mesh engine not available: {e}"}))
        sys.exit(1)

    primitive_map = {
        'cube': lambda: create_cube(size=size),
        'sphere': lambda: create_sphere(radius=radius, rings=segments, segments=segments),
        'cylinder': lambda: create_cylinder(radius=radius, height=height, segments=segments),
        'cone': lambda: create_cone(radius=radius, height=height, segments=segments),
        'torus': lambda: create_torus(major_radius=radius, minor_radius=size * 0.3,
                                       major_segments=segments, minor_segments=segments // 2),
        'plane': lambda: create_plane(width=size, height=size),
        'arrow': lambda: create_arrow(length=size),
        'circle': lambda: create_circle(radius=radius, segments=segments),
        'grid': lambda: create_grid(size=size * 10, divisions=int(segments)),
    }

    try:
        mesh_data = primitive_map[mesh_type]()
        result = {
            "name": name,
            "type": mesh_type,
            "vertex_count": len(mesh_data["vertices"]),
            "face_count": len(mesh_data["faces"]),
        }

        out_dir = ensure_output_dir(ctx.obj['output_dir'])

        if save:
            from nexus3d.mesh.mesh import Mesh
            mesh = Mesh.from_primitive(name, mesh_data)
            filepath = out_dir / f"{name}.obj"
            mesh.save_obj(str(filepath))
            result["file"] = str(filepath)
            click.echo(f"Saved: {filepath}")

        output_json(result, ctx.obj['verbose'])
    except Exception as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)


@create.command('armature')
@click.argument('name')
@click.option('--type', 'arm_type', default='humanoid', type=click.Choice(['humanoid']))
@click.option('--height', default=1.8, help='Height for humanoid')
@click.option('--save', is_flag=True, help='Save to JSON file')
@click.pass_context
def create_armature(ctx, name, arm_type, height, save):
    """Create an armature/skeleton."""
    try:
        from nexus3d.rigging.armature import create_humanoid_armature
    except ImportError as e:
        click.echo(json.dumps({"error": f"Rigging engine not available: {e}"}))
        sys.exit(1)

    try:
        armature = create_humanoid_armature(height=height, name=name)
        armature.update_transforms()

        result = {
            "name": armature.name,
            "type": arm_type,
            "bone_count": len(armature.bones),
            "bone_names": list(armature.bones.keys()),
        }

        if save:
            out_dir = ensure_output_dir(ctx.obj['output_dir'])
            filepath = out_dir / f"{name}.json"
            save_json(armature.to_dict(), str(filepath))
            result["file"] = str(filepath)
            click.echo(f"Saved: {filepath}")

        output_json(result, ctx.obj['verbose'])
    except Exception as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)


# ─── MODIFY commands ─────────────────────────────────────────────────────────

@cli.group()
@click.pass_context
def modify(ctx):
    """Modify 3D objects."""
    pass


@modify.command('transform')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--translate', default='0,0,0', help='Translation X,Y,Z')
@click.option('--rotate', default='0,0,0', help='Rotation X,Y,Z (degrees)')
@click.option('--scale', default='1,1,1', help='Scale X,Y,Z')
@click.option('--output', '-o', help='Output file path')
@click.pass_context
def modify_transform(ctx, input_file, translate, rotate, scale_vec, output):
    """Apply transform to a mesh file."""
    try:
        from nexus3d.mesh.mesh import Mesh
        from nexus3d.math3d.core import compose_matrix, euler_to_quat
    except ImportError as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)

    try:
        mesh = Mesh.from_file(input_file)
        t = parse_vec3(translate)
        r = euler_to_quat(np.radians(parse_vec3(rotate)))
        s = parse_vec3(scale_vec)
        matrix = compose_matrix(t, r, s)
        mesh.apply_transform(matrix)

        out_dir = ctx.obj['output_dir']
        if output:
            filepath = Path(output)
        else:
            out_dir = ensure_output_dir(out_dir)
            filepath = out_dir / f"transformed_{Path(input_file).name}"

        if filepath.suffix == '.obj':
            mesh.save_obj(str(filepath))
        else:
            save_json(mesh.to_dict(), str(filepath))

        output_json({"input": input_file, "output": str(filepath),
                      "vertices": mesh.vertex_count(), "faces": mesh.face_count()},
                     ctx.obj['verbose'])
    except Exception as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)


# ─── ANIMATE commands ────────────────────────────────────────────────────────

@cli.group()
@click.pass_context
def animate(ctx):
    """Animation operations."""
    pass


@animate.command('procedural')
@click.option('--type', 'anim_type', required=True,
              type=click.Choice(['walk', 'breathe', 'idle_sway', 'sine_wave']))
@click.option('--duration', default=2.0, help='Duration in seconds')
@click.option('--fps', default=30, help='Frames per second')
@click.option('--output', '-o', help='Output file path')
@click.pass_context
def animate_procedural(ctx, anim_type, duration, fps, output):
    """Generate a procedural animation."""
    try:
        from nexus3d.animation.anim import ProceduralAnimation
    except ImportError as e:
        click.echo(json.dumps({"error": f"Animation engine not available: {e}"}))
        sys.exit(1)

    try:
        frames = []
        num_frames = int(duration * fps)

        for i in range(num_frames):
            t = i / fps
            frame = {"frame": i, "time": round(t, 4)}

            if anim_type == "walk":
                pose = ProceduralAnimation.walk_cycle(t)
                frame["pose"] = {}
                for bone, (pos_off, rot) in pose.items():
                    frame["pose"][bone] = {
                        "position_offset": pos_off.tolist(),
                        "rotation": rot.tolist()
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

        if output:
            out_dir = ensure_output_dir(ctx.obj['output_dir'])
            filepath = Path(output) if Path(output).is_absolute() else out_dir / output
            save_json({"frames": frames, **result}, str(filepath))
            result["file"] = str(filepath)
            click.echo(f"Saved: {filepath}")
        else:
            result["frames"] = frames

        output_json(result, ctx.obj['verbose'])
    except Exception as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)


@animate.command('export')
@click.option('--armature', 'arm_file', required=True, type=click.Path(exists=True),
              help='Armature JSON file')
@click.option('--animation', 'anim_file', required=True, type=click.Path(exists=True),
              help='Animation JSON file')
@click.option('--format', 'fmt', default='bvh', type=click.Choice(['bvh', 'gltf']))
@click.option('--output', '-o', help='Output file path')
@click.pass_context
def animate_export(ctx, arm_file, anim_file, fmt, output):
    """Export animation to BVH or glTF format."""
    try:
        from nexus3d.rigging.armature import Armature
        from nexus3d.animation.anim import AnimationClip, AnimationExporter
    except ImportError as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)

    try:
        armature = Armature.from_dict(load_json(arm_file))
        animation = AnimationClip.from_dict(load_json(anim_file))

        out_dir = ensure_output_dir(ctx.obj['output_dir'])
        if output:
            filepath = Path(output) if Path(output).is_absolute() else out_dir / output
        else:
            filepath = out_dir / f"animation.{fmt}"

        if fmt == 'bvh':
            AnimationExporter.to_bvh(armature, animation, str(filepath))
        elif fmt == 'gltf':
            AnimationExporter.to_gltf(animation, str(filepath))

        output_json({"exported": str(filepath), "format": fmt}, ctx.obj['verbose'])
    except Exception as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)


# ─── RENDER commands ─────────────────────────────────────────────────────────

@cli.group()
@click.pass_context
def render(ctx):
    """Rendering operations."""
    pass


@render.command('scene')
@click.argument('mesh_file', type=click.Path(exists=True))
@click.option('--camera-pos', default='0,2,5', help='Camera position X,Y,Z')
@click.option('--look-at', default='0,0,0', help='Camera look-at target')
@click.option('--fov', default=60.0, help='Camera FOV in degrees')
@click.option('--width', default=1920, help='Image width')
@click.option('--height', default=1080, help='Image height')
@click.option('--output', '-o', help='Output image path')
@click.pass_context
def render_scene(ctx, mesh_file, camera_pos, look_at, fov, width, height, output):
    """Render a mesh to an image."""
    try:
        from nexus3d.rendering.renderer import Renderer, Camera, Light
        from nexus3d.mesh.mesh import Mesh
    except ImportError as e:
        click.echo(json.dumps({"error": f"Rendering engine not available: {e}"}))
        sys.exit(1)

    try:
        mesh = Mesh.from_file(mesh_file)

        camera = Camera()
        camera.position = parse_vec3(camera_pos)
        camera.target = parse_vec3(look_at)
        camera.set_resolution(int(width), int(height))
        camera.fov = float(fov)

        lights = [
            Light.directional(direction=[-1, -1, -1], intensity=1.0),
            Light.directional(direction=[1, 0, -1], intensity=0.5),
            Light.directional(direction=[0, 1, 0], intensity=0.3),
        ]

        renderer = Renderer(resolution=(int(width), int(height)))

        scene_data = {"meshes": [{
            "vertices": mesh.vertices.tolist(),
            "faces": mesh.faces.tolist(),
            "normals": mesh.normals.tolist() if mesh.normals is not None else None,
        }]}

        out_dir = ensure_output_dir(ctx.obj['output_dir'])
        if output:
            filepath = Path(output) if Path(output).is_absolute() else out_dir / output
        else:
            filepath = out_dir / f"render_{Path(mesh_file).stem}.png"

        renderer.render_to_file(scene_data, str(filepath), camera=camera, lights=lights)

        output_json({"rendered": str(filepath), "resolution": [int(width), int(height)]},
                     ctx.obj['verbose'])
    except Exception as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)


@render.command('video')
@click.argument('frames_dir', type=click.Path(exists=True))
@click.option('--fps', default=30, help='Frames per second')
@click.option('--output', '-o', help='Output video path')
@click.pass_context
def render_video(ctx, frames_dir, fps, output):
    """Combine rendered frames into a video."""
    try:
        import imageio.v2 as imageio
    except ImportError as e:
        click.echo(json.dumps({"error": f"imageio not available: {e}"}))
        sys.exit(1)

    try:
        frames_dir = Path(frames_dir)
        images = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
        if not images:
            click.echo(json.dumps({"error": "No images found in directory"}))
            sys.exit(1)

        out_dir = ensure_output_dir(ctx.obj['output_dir'])
        if output:
            filepath = Path(output) if Path(output).is_absolute() else out_dir / output
        else:
            filepath = out_dir / "animation.mp4"

        writer = imageio.get_writer(str(filepath), fps=fps)
        for img_path in images:
            img = imageio.imread(str(img_path))
            writer.append_data(img)
        writer.close()

        output_json({"video": str(filepath), "frames": len(images), "fps": fps},
                     ctx.obj['verbose'])
    except Exception as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)


# ─── MATH commands ───────────────────────────────────────────────────────────

@cli.group()
@click.pass_context
def math_cmds(ctx):
    """Math calculations."""
    pass


@math_cmds.command('ik')
@click.option('--type', 'ik_type', required=True,
              type=click.Choice(['2bone', 'fabrik', 'ccd']))
@click.option('--joints', required=True, help='Joint positions: "x,y,z;x,y,z;..."')
@click.option('--target', required=True, help='Target position: "x,y,z"')
@click.option('--pole', default=None, help='Pole position for bend direction')
@click.pass_context
def math_ik(ctx, ik_type, joints, target, pole):
    """Solve inverse kinematics."""
    try:
        from nexus3d.math3d.core import solve_2bone_ik, solve_fabrik, solve_ccd, vec3
    except ImportError as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)

    try:
        joints_list = [parse_vec3(j.strip()) for j in joints.split(';')]
        target_vec = parse_vec3(target)
        pole_vec = parse_vec3(pole) if pole else None

        if ik_type == "2bone" and len(joints_list) >= 2:
            la = float(np.linalg.norm(joints_list[1] - joints_list[0]))
            lb = float(np.linalg.norm(joints_list[-1] - joints_list[-2]))
            elbow, wrist = solve_2bone_ik(joints_list[0], target_vec, la, lb, pole_vec)
            solved = [joints_list[0].tolist(), elbow.tolist(), wrist.tolist()]
        elif ik_type == "fabrik":
            solved = [j.tolist() for j in solve_fabrik(joints_list, target_vec)]
        elif ik_type == "ccd":
            solved = [j.tolist() for j in solve_ccd(joints_list, target_vec)]
        else:
            click.echo(json.dumps({"error": f"Unknown IK type: {ik_type}"}))
            sys.exit(1)

        output_json({
            "solver": ik_type,
            "original_joints": [j.tolist() for j in joints_list],
            "solved_joints": solved,
            "target": target_vec.tolist(),
        }, ctx.obj['verbose'])
    except Exception as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)


@math_cmds.command('physics')
@click.option('--mass', default=1.0, help='Body mass')
@click.option('--position', default='0,10,0', help='Starting position')
@click.option('--velocity', default='5,0,0', help='Starting velocity')
@click.option('--gravity', default=9.80665, help='Gravity (m/s^2)')
@click.option('--duration', default=2.0, help='Simulation duration')
@click.option('--dt', default=0.016, help='Time step')
@click.pass_context
def math_physics(ctx, mass, position, velocity, gravity, duration, dt):
    """Run a physics simulation."""
    try:
        from nexus3d.math3d.core import PhysicsBody, vec3
    except ImportError as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)

    try:
        body = PhysicsBody(
            mass=mass,
            position=parse_vec3(position),
            velocity=parse_vec3(velocity),
        )

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

        output_json({
            "mass": mass,
            "gravity": gravity,
            "duration": duration,
            "steps": steps,
            "trajectory": trajectory,
        }, ctx.obj['verbose'])
    except Exception as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)


@math_cmds.command('transform')
@click.option('--position', default='0,0,0', help='Translation X,Y,Z')
@click.option('--rotation', default='0,0,0', help='Euler rotation X,Y,Z (degrees)')
@click.option('--scale', default='1,1,1', help='Scale X,Y,Z')
@click.pass_context
def math_transform(ctx, position, rotation, scale_vec):
    """Compose a transform matrix from TRS components."""
    try:
        from nexus3d.math3d.core import compose_matrix, decompose_matrix, euler_to_quat
    except ImportError as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)

    try:
        t = parse_vec3(position)
        r = euler_to_quat(np.radians(parse_vec3(rotation)))
        s = parse_vec3(scale_vec)
        matrix = compose_matrix(t, r, s)

        # Verify by decomposing back
        t2, r2, s2 = decompose_matrix(matrix)

        output_json({
            "matrix": matrix.tolist(),
            "translation": t.tolist(),
            "rotation_quat": r.tolist(),
            "scale": s.tolist(),
            "verified": {
                "translation": t2.tolist(),
                "rotation_quat": r2.tolist(),
                "scale": s2.tolist(),
            }
        }, ctx.obj['verbose'])
    except Exception as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)


# ─── PATH TRACE commands ──────────────────────────────────────────────────────

@cli.group()
@click.pass_context
def trace(ctx):
    """Path tracing commands."""
    pass


@trace.command('render')
@click.option('--output', '-o', required=True, help='Output image path')
@click.option('--width', default=800, help='Image width (path tracing is slow at high res)')
@click.option('--height', default=600, help='Image height')
@click.option('--spp', default=32, help='Samples per pixel')
@click.option('--max-bounces', default=8, help='Maximum ray bounces')
@click.option('--aperture', default=0.0, help='Camera aperture (0 = pinhole)')
@click.option('--focus-dist', default=10.0, help='Focus distance for DoF')
@click.option('--fov', default=60.0, help='Camera FOV')
@click.option('--tonemap', default='aces', type=click.Choice(['aces', 'reinhard', 'linear']))
@click.option('--scene', default='default', type=click.Choice(['default', 'glass', 'mirrors', 'cornell']))
@click.pass_context
def trace_render(ctx, output, width, height, spp, max_bounces, aperture, focus_dist, fov, tonemap, scene):
    """Render a scene using Monte Carlo path tracing."""
    try:
        from nexus3d.rendering.pathtracer import PathTracer, PTMaterial
    except ImportError as e:
        click.echo(json.dumps({"error": f"Path tracer not available: {e}"}))
        sys.exit(1)

    try:
        import numpy as np
        pt = PathTracer(width=int(width), height=int(height))
        pt.set_tonemap(tonemap)

        # Set camera
        cam_pos = np.array([0.0, 2.0, 6.0])
        cam_target = np.array([0.0, 0.5, 0.0])
        pt.set_camera(cam_pos, cam_target, fov=float(fov),
                       aperture=float(aperture), focal_distance=float(focus_dist))

        # Scene definitions
        if scene == 'default':
            # Ground + spheres
            pt.add_plane(np.array([0,1,0]), np.array([0,0,0]),
                        PTMaterial(albedo=np.array([0.4,0.4,0.4]), roughness=0.9))
            pt.add_sphere(np.array([0,1,0]), 1.0, PTMaterial(albedo=np.array([0.8,0.1,0.1]), metallic=0.3, roughness=0.3))
            pt.add_sphere(np.array([-2,0.5,0.5]), 0.5, PTMaterial(albedo=np.array([0.1,0.1,0.8]), metallic=0.5, roughness=0.2))
            pt.add_sphere(np.array([2,0.7,-0.5]), 0.7, PTMaterial(albedo=np.array([0.1,0.8,0.1]), roughness=0.8))
        elif scene == 'glass':
            pt.add_plane(np.array([0,1,0]), np.array([0,0,0]),
                        PTMaterial(albedo=np.array([0.3,0.3,0.3]), roughness=0.8))
            pt.add_sphere(np.array([0,1,0]), 1.0, PTMaterial(albedo=np.array([1,1,1]), transmission=0.95, ior=1.5, roughness=0.0))
            pt.add_sphere(np.array([-1.8,0.6,1.0]), 0.6, PTMaterial(albedo=np.array([0.8,0.2,0.1]), metallic=1.0, roughness=0.1))
            pt.add_sphere(np.array([1.5,0.5,0.5]), 0.5, PTMaterial(albedo=np.array([0.1,0.8,0.8]), roughness=0.9))
        elif scene == 'mirrors':
            pt.add_plane(np.array([0,1,0]), np.array([0,0,0]),
                        PTMaterial(albedo=np.array([0.2,0.2,0.2]), roughness=0.9))
            pt.add_sphere(np.array([0,1,0]), 1.0, PTMaterial(metallic=1.0, roughness=0.02, albedo=np.array([0.95,0.93,0.88])))
            pt.add_sphere(np.array([-2,0.6,1.0]), 0.6, PTMaterial(metallic=1.0, roughness=0.05, albedo=np.array([0.95,0.63,0.54])))
            pt.add_sphere(np.array([2,0.7,-0.5]), 0.7, PTMaterial(albedo=np.array([0.9,0.9,0.95]), roughness=0.9))
        elif scene == 'cornell':
            pt.add_plane(np.array([0,1,0]), np.array([0,0,0]),
                        PTMaterial(albedo=np.array([0.73,0.73,0.73]), roughness=0.9))
            pt.add_sphere(np.array([-0.8,0.5,0.3]), 0.5, PTMaterial(albedo=np.array([1.0,0.0,0.0]), roughness=1.0, emission=np.array([5.0,0.0,0.0])))
            pt.add_sphere(np.array([0.8,0.5,-0.3]), 0.5, PTMaterial(albedo=np.array([0.0,0.0,1.0]), roughness=1.0, emission=np.array([0.0,0.0,5.0])))

        pt.set_environment(np.array([0.4, 0.6, 1.0]))

        image = pt.render(samples_per_pixel=int(spp), max_bounces=int(max_bounces))
        pt.save(str(output))

        result = {
            "renderer": "path_tracer",
            "scene": scene,
            "resolution": [int(width), int(height)],
            "spp": int(spp),
            "max_bounces": int(max_bounces),
            "tonemap": tonemap,
            "output": str(output),
        }
        output_json(result, ctx.obj['verbose'])
    except Exception as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)


# ─── CSG commands ────────────────────────────────────────────────────────────

@cli.group()
@click.pass_context
def csg(ctx):
    """Constructive Solid Geometry operations."""
    pass


@csg.command('wall')
@click.option('--width', default=4.0, help='Wall width')
@click.option('--height', default=3.0, help='Wall height')
@click.option('--thickness', default=0.15, help='Wall thickness')
@click.option('--add-door', is_flag=True, help='Add a door opening')
@click.option('--add-window', is_flag=True, help='Add a window opening')
@click.option('--save', is_flag=True, help='Save to OBJ file')
@click.pass_context
def csg_wall(ctx, width, height, thickness, add_door, add_window, save):
    """Create a wall with CSG-cut door/window openings."""
    try:
        from nexus3d.csg.engine import CSGMesh, ArchitectureKit, MeshCleaner
        from nexus3d.mesh.mesh import Mesh
    except ImportError as e:
        click.echo(json.dumps({"error": f"CSG engine not available: {e}"}))
        sys.exit(1)

    try:
        wall = ArchitectureKit.wall(float(width), float(height), float(thickness))

        if add_door:
            wall = ArchitectureKit.add_door(wall, door_width=1.0, door_height=2.1, position=(0, 0, 0))

        if add_window:
            wall = ArchitectureKit.add_window(wall, window_width=1.2, window_height=1.0,
                                                window_bottom=0.9, position=(0, 0, 0))

        vertices, faces = MeshCleaner.clean(*wall.to_vertices_faces())

        result = {
            "type": "csg_wall",
            "width": width, "height": height, "thickness": thickness,
            "door": add_door, "window": add_window,
            "vertices": len(vertices), "faces": len(faces),
            "volume": wall.volume(),
        }

        if save:
            from nexus3d.utils.helpers import ensure_output_dir
            out_dir = ensure_output_dir(ctx.obj['output_dir'])
            mesh = Mesh(name="csg_wall", vertices=vertices, faces=faces)
            filepath = out_dir / "csg_wall.obj"
            mesh.save_obj(str(filepath))
            result["file"] = str(filepath)

        output_json(result, ctx.obj['verbose'])
    except Exception as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)


@csg.command('boolean')
@click.argument('operation', type=click.Choice(['union', 'subtract', 'intersect']))
@click.option('--shape-a', default='cube', type=click.Choice(['cube', 'sphere', 'cylinder']))
@click.option('--shape-b', default='sphere', type=click.Choice(['cube', 'sphere', 'cylinder']))
@click.option('--offset', default='0.5,0,0', help='Offset for shape B')
@click.option('--save', is_flag=True, help='Save result')
@click.pass_context
def csg_boolean(ctx, operation, shape_a, shape_b, offset, save):
    """Perform boolean operation on two shapes."""
    try:
        from nexus3d.csg.engine import CSGMesh, MeshCleaner
        from nexus3d.mesh.mesh import Mesh
    except ImportError as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)

    try:
        import numpy as np
        shape_map = {
            'cube': lambda: CSGMesh.from_cube(1.0),
            'sphere': lambda: CSGMesh.from_sphere(0.7, segments=16),
            'cylinder': lambda: CSGMesh.from_cylinder(0.5, 1.5, segments=16),
        }

        a = shape_map[shape_a]()
        b = shape_map[shape_b]()

        # Apply offset
        off = np.array([float(x) for x in offset.split(',')])
        for i in range(len(b._polygons)):
            for j in range(len(b._polygons[i].vertices)):
                b._polygons[i].vertices[j] = b._polygons[i].vertices[j] + off

        ops = {
            'union': lambda: a.union(b),
            'subtract': lambda: a.subtract(b),
            'intersect': lambda: a.intersect(b),
        }

        result_mesh = ops[operation]()
        vertices, faces = MeshCleaner.clean(*result_mesh.to_vertices_faces())

        result = {
            "operation": operation,
            "shape_a": shape_a, "shape_b": shape_b,
            "offset": offset,
            "vertices": len(vertices), "faces": len(faces),
            "volume": result_mesh.volume(),
        }

        if save:
            from nexus3d.utils.helpers import ensure_output_dir
            out_dir = ensure_output_dir(ctx.obj['output_dir'])
            mesh = Mesh(name=f"csg_{operation}", vertices=vertices, faces=faces)
            filepath = out_dir / f"csg_{operation}.obj"
            mesh.save_obj(str(filepath))
            result["file"] = str(filepath)

        output_json(result, ctx.obj['verbose'])
    except Exception as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)


# ─── MATERIAL commands ──────────────────────────────────────────────────────

@cli.group()
@click.pass_context
def material(ctx):
    """Material and texture operations."""
    pass


@material.command('library')
@click.argument('name')
@click.pass_context
def material_library(ctx, name):
    """Get a material from the PBR library."""
    try:
        from nexus3d.materials.pipeline import MaterialLibrary
    except ImportError as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)

    try:
        mat = MaterialLibrary.get(name)
        output_json(mat, ctx.obj['verbose'])
    except KeyError:
        click.echo(json.dumps({"error": f"Material '{name}' not found. Available: {MaterialLibrary.list()}"}))
        sys.exit(1)


@material.command('generate-texture')
@click.option('--type', 'tex_type', required=True,
              type=click.Choice(['noise', 'checker', 'brick', 'wood', 'marble', 'gradient', 'voronoi']))
@click.option('--output', '-o', required=True, help='Output image path')
@click.option('--width', default=512, help='Image width')
@click.option('--height', default=512, help='Image height')
@click.option('--save', is_flag=True, help='Save to file')
@click.pass_context
def material_generate_texture(ctx, tex_type, output, width, height, save):
    """Generate a procedural texture."""
    try:
        from nexus3d.materials.pipeline import (
            NoiseTexture, CheckerTexture, BrickTexture, WoodTexture,
            MarbleTexture, GradientTexture, VoronoiTexture
        )
    except ImportError as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)

    try:
        tex_map = {
            'noise': lambda: NoiseTexture(scale=4.0, octaves=6),
            'checker': lambda: CheckerTexture(scale=8.0),
            'brick': lambda: BrickTexture(),
            'wood': lambda: WoodTexture(rings=12.0),
            'marble': lambda: MarbleTexture(turbulence=5.0),
            'gradient': lambda: GradientTexture(direction='radial'),
            'voronoi': lambda: VoronoiTexture(scale=8.0),
        }

        tex = tex_map[tex_type]()

        result = {"type": tex_type, "width": int(width), "height": int(height)}

        if save:
            tex.save(str(output), width=int(width), height=int(height))
            result["file"] = str(output)

        output_json(result, ctx.obj['verbose'])
    except Exception as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)


# ─── CINEMATIC commands ─────────────────────────────────────────────────────

@cli.group()
@click.pass_context
def cinematic(ctx):
    """Cinematic camera operations."""
    pass


@cinematic.command('dof')
@click.option('--focal-length', default=50.0, help='Focal length in mm')
@click.option('--aperture', default=2.8, help='F-stop')
@click.option('--focus-distance', default=10.0, help='Focus distance in meters')
@click.pass_context
def cinematic_dof(ctx, focal_length, aperture, focus_distance):
    """Calculate depth of field parameters."""
    try:
        from nexus3d.cinematic.camera import DepthOfField
    except ImportError as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)

    try:
        dof = DepthOfField(focal_length=float(focal_length),
                           aperture=float(aperture),
                           focus_distance=float(focus_distance))

        result = {
            "focal_length_mm": focal_length,
            "aperture_fstop": aperture,
            "focus_distance_m": focus_distance,
            "near_plane_m": round(dof.near_plane(), 3),
            "far_plane_m": round(dof.far_plane(), 3) if dof.far_plane() < float('inf') else "infinity",
            "hyperfocal_m": round(dof.hyperfocal_distance(), 3),
            "dof_total_m": round(dof.far_plane() - dof.near_plane(), 3) if dof.far_plane() < float('inf') else "infinity",
        }

        output_json(result, ctx.obj['verbose'])
    except Exception as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)


@cinematic.command('exposure')
@click.option('--aperture', default=2.8, help='F-stop')
@click.option('--shutter', default=0.0208, help='Shutter time in seconds (1/48 for 24fps 180deg)')
@click.option('--iso', default=800, help='ISO sensitivity')
@click.pass_context
def cinematic_exposure(ctx, aperture, shutter, iso):
    """Calculate exposure parameters."""
    try:
        from nexus3d.cinematic.camera import Exposure
    except ImportError as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)

    try:
        exp = Exposure(aperture=float(aperture), shutter_time=float(shutter), iso=float(iso))

        result = {
            "aperture": aperture,
            "shutter_time_s": shutter,
            "iso": iso,
            "ev100": round(exp.get_ev(), 2),
            "exposure_value": round(exp.get_ev(), 2),
        }

        output_json(result, ctx.obj['verbose'])
    except Exception as e:
        click.echo(json.dumps({"error": str(e)}))
        sys.exit(1)


# ─── SERVE command ───────────────────────────────────────────────────────────

@cli.command()
@click.option('--host', default='0.0.0.0', help='Server host')
@click.option('--port', default=8420, help='Server port')
@click.option('--workers', default=1, help='Number of workers')
@click.pass_context
def serve(ctx, host, port, workers):
    """Start the Nexus3D API server."""
    try:
        import uvicorn
    except ImportError:
        click.echo(json.dumps({"error": "uvicorn not installed. Run: pip install uvicorn"}))
        sys.exit(1)

    click.echo(f"Starting Nexus3D API server on {host}:{port}", err=True)
    click.echo(f"API docs: http://{host}:{port}/docs", err=True)
    uvicorn.run("nexus3d.api.server:app", host=host, port=int(port), workers=int(workers))


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    cli(obj={})


if __name__ == '__main__':
    main()
