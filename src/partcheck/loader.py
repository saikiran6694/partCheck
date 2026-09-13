"""Load an STL file, normalize it to millimetres, and prepare it for checks."""

from __future__ import annotations

import logging

import trimesh

logger = logging.getLogger(__name__)

# STL carries no unit information, so the caller states what unit the file
# was authored in; we scale to millimetres from there.
_SCALE_TO_MM = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": 25.4,
}


def load_part(path: str, units: str = "mm") -> trimesh.Trimesh:
    """Load an STL file and prepare it for checks.

    Steps: load, scale to millimetres, merge duplicate vertices, fix face
    winding/normals, and rest the part on z = 0 (its lowest point touches
    the build plate / bed).
    """
    if units not in _SCALE_TO_MM:
        raise ValueError(f"Unknown units {units!r}; expected one of {sorted(_SCALE_TO_MM)}")

    loaded = trimesh.load(path, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"{path} did not load as a single triangle mesh (got {type(loaded)})")
    mesh = loaded

    scale = _SCALE_TO_MM[units]
    if scale != 1.0:
        mesh.apply_scale(scale)

    mesh.merge_vertices()
    mesh.fix_normals()

    if not mesh.is_watertight:
        logger.warning(
            "%s is not watertight; some checks (e.g. thin wall) may be unreliable", path
        )

    mesh.apply_translation([0.0, 0.0, -mesh.bounds[0, 2]])

    return mesh
