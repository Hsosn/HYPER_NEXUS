"""Nexus3D CSG Engine — Constructive Solid Geometry for 3D meshes.

This sub-package provides real boolean operations (union, subtract,
intersect) on closed triangle meshes using BSP trees, as well as
architectural primitives (walls, rooms, windows, doors, stairs,
arches) and post-CSG mesh cleanup utilities.

Quick start::

    from nexus3d.csg import CSGMesh, ArchitectureKit

    # Primitive shapes
    sphere = CSGMesh.from_sphere(1.0, segments=16)
    cube   = CSGMesh.from_cube(1.0)

    # Boolean operations
    result = sphere.subtract(cube)
    verts, faces = result.to_vertices_faces()

    # Architecture
    wall = ArchitectureKit.wall(4.0, 3.0, 0.15)
    wall = ArchitectureKit.add_door(wall, door_width=1.0, door_height=2.1)
    wall = ArchitectureKit.add_window(wall, window_width=1.2)
"""

from nexus3d.csg.engine import (
    # Core classes
    Plane,
    BSPPolygon,
    BSPNode,
    CSGMesh,
    # Cleanup
    MeshCleaner,
    # Architecture
    ArchitectureKit,
    # Module-level helpers
    _mesh_height,
    wall_thickness,
)

__all__ = [
    "Plane",
    "BSPPolygon",
    "BSPNode",
    "CSGMesh",
    "MeshCleaner",
    "ArchitectureKit",
    "_mesh_height",
    "wall_thickness",
]
