"""Nexus3D Math Engine package."""

from nexus3d.math3d.core import (
    # Vectors
    vec3, vec4, normalize, magnitude, dot, cross, lerp, distance,
    reflect, refract, angle_between, project_onto, perpendicular,
    # Matrices
    mat4_identity, mat4_translate, mat4_scale,
    mat4_rotate_x, mat4_rotate_y, mat4_rotate_z, mat4_rotate_axis,
    mat4_look_at, mat4_perspective, mat4_orthographic,
    decompose_matrix, compose_matrix,
    # Quaternions
    quat, quat_identity, normalize_quat, quat_multiply, quat_conjugate, quat_inverse,
    quat_to_rotation_matrix, rotation_matrix_to_quat,
    quat_from_axis_angle, quat_to_axis_angle,
    quat_slerp, quat_to_euler, euler_to_quat,
    # Geometry
    point_in_triangle, ray_plane_intersect, ray_triangle_intersect,
    ray_sphere_intersect, ray_aabb_intersect, closest_point_on_segment,
    triangle_normal, triangle_area, triangle_centroid,
    compute_bounding_box, compute_centroid, compute_volume, compute_surface_area,
    # Physics
    PhysicsBody, Gravity, apply_gravity, resolve_collision,
    # IK
    solve_2bone_ik, solve_fabrik, solve_ccd,
    # Color
    rgb_to_hsv, hsv_to_rgb, hex_to_rgb, fresnel_schlick, ggx_distribution, smith_geometry,
    # Splines
    bezier_cubic, bezier_cubic_derivative, catmull_rom_spline,
)

__all__ = [
    "vec3", "vec4", "normalize", "magnitude", "dot", "cross", "lerp", "distance",
    "reflect", "refract", "angle_between", "project_onto", "perpendicular",
    "mat4_identity", "mat4_translate", "mat4_scale",
    "mat4_rotate_x", "mat4_rotate_y", "mat4_rotate_z", "mat4_rotate_axis",
    "mat4_look_at", "mat4_perspective", "mat4_orthographic",
    "decompose_matrix", "compose_matrix",
    "quat", "quat_identity", "normalize_quat", "quat_multiply", "quat_conjugate", "quat_inverse",
    "quat_to_rotation_matrix", "rotation_matrix_to_quat",
    "quat_from_axis_angle", "quat_to_axis_angle",
    "quat_slerp", "quat_to_euler", "euler_to_quat",
    "point_in_triangle", "ray_plane_intersect", "ray_triangle_intersect",
    "ray_sphere_intersect", "ray_aabb_intersect", "closest_point_on_segment",
    "triangle_normal", "triangle_area", "triangle_centroid",
    "compute_bounding_box", "compute_centroid", "compute_volume", "compute_surface_area",
    "PhysicsBody", "Gravity", "apply_gravity", "resolve_collision",
    "solve_2bone_ik", "solve_fabrik", "solve_ccd",
    "rgb_to_hsv", "hsv_to_rgb", "hex_to_rgb", "fresnel_schlick", "ggx_distribution", "smith_geometry",
    "bezier_cubic", "bezier_cubic_derivative", "catmull_rom_spline",
]
