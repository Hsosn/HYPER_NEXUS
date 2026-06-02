"""Nexus3D Mesh module — procedural geometry and the core Mesh container.

Public API
----------
:class:`~nexus3d.mesh.mesh.Mesh`           – geometry wrapper with I/O, queries,
                                               transforms, and topology edits.

Primitive generators (all return ``dict`` with *vertices*, *faces*, *normals*, *uvs*):
    :func:`~nexus3d.mesh.primitives.create_cube`
    :func:`~nexus3d.mesh.primitives.create_sphere`
    :func:`~nexus3d.mesh.primitives.create_cylinder`
    :func:`~nexus3d.mesh.primitives.create_cone`
    :func:`~nexus3d.mesh.primitives.create_torus`
    :func:`~nexus3d.mesh.primitives.create_plane`
    :func:`~nexus3d.mesh.primitives.create_grid`
    :func:`~nexus3d.mesh.primitives.create_circle`
    :func:`~nexus3d.mesh.primitives.create_arrow`
"""

from nexus3d.mesh.mesh import Mesh
from nexus3d.mesh.primitives import (
    create_arrow,
    create_circle,
    create_cone,
    create_cube,
    create_cylinder,
    create_grid,
    create_plane,
    create_sphere,
    create_torus,
)

__all__ = [
    "Mesh",
    "create_arrow",
    "create_circle",
    "create_cone",
    "create_cube",
    "create_cylinder",
    "create_grid",
    "create_plane",
    "create_sphere",
    "create_torus",
]
