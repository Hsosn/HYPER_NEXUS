"""Utility functions for Nexus3D."""

import json
import numpy as np
from pathlib import Path
from typing import Any


def parse_vec3(s: str) -> np.ndarray:
    """Parse a string like '1,2,3' or '[1,2,3]' to a 3D numpy array."""
    s = s.strip().strip('[]()')
    parts = [float(x) for x in s.replace(' ', ',').split(',') if x.strip()]
    result = np.zeros(3, dtype=np.float64)
    for i in range(min(3, len(parts))):
        result[i] = parts[i]
    return result


def parse_rotation(s: str) -> np.ndarray:
    """Parse rotation string 'x,y,z' in degrees to quaternion [w, x, y, z]."""
    from nexus3d.math3d.core import euler_to_quat
    angles = parse_vec3(s)
    return euler_to_quat(np.radians(angles))


def parse_matrix(s: str) -> np.ndarray:
    """Parse a string of 16 floats to a 4x4 matrix."""
    s = s.strip().strip('[]()')
    parts = [float(x) for x in s.replace(' ', ',').split(',') if x.strip()]
    if len(parts) == 16:
        return np.array(parts, dtype=np.float64).reshape(4, 4)
    raise ValueError(f"Expected 16 values for matrix, got {len(parts)}")


def ensure_output_dir(path: str) -> Path:
    """Create output directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy types to Python native types."""

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


def save_json(data: Any, filepath: str):
    """Save data to JSON file with numpy support."""
    filepath = Path(filepath)
    ensure_output_dir(filepath.parent)
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)


def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def mesh_to_renderable(mesh) -> dict:
    """Convert a Mesh object to a renderable dict for the renderer."""
    result = {
        'vertices': mesh.vertices.tolist() if hasattr(mesh, 'vertices') and mesh.vertices is not None else [],
        'faces': mesh.faces.tolist() if hasattr(mesh, 'faces') and mesh.faces is not None else [],
    }
    if hasattr(mesh, 'normals') and mesh.normals is not None:
        result['normals'] = mesh.normals.tolist()
    if hasattr(mesh, 'uvs') and mesh.uvs is not None:
        result['uvs'] = mesh.uvs.tolist()
    return result
