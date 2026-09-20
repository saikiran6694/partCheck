"""Colour flagged faces on a mesh and export it as a highlighted GLB."""

from __future__ import annotations

import numpy as np
import trimesh

from partcheck.models import Finding, Severity

BASE_COLOR = (200, 205, 215, 255)
WARNING_COLOR = (240, 150, 30, 255)
ERROR_COLOR = (220, 45, 45, 255)


def export_highlighted(mesh: trimesh.Trimesh, findings: list[Finding], path: str) -> None:
    """Export a copy of `mesh` as a GLB with flagged faces coloured red (error)
    or orange (warning). A face flagged by both severities is coloured red."""
    colored = mesh.copy()
    # GLB stores per-vertex colour and interpolates across a triangle, so a
    # shared vertex between a flagged and unflagged face would blend into a
    # muddy gradient instead of a flat colour. Unwelding gives every face its
    # own unshared vertices, keeping each face's colour flat and distinct.
    colored.unmerge_vertices()
    colors = np.tile(np.array(BASE_COLOR, dtype=np.uint8), (len(colored.faces), 1))

    # Warnings first, errors last, so a face flagged by both ends up red.
    for finding in sorted(findings, key=lambda f: f.severity == Severity.ERROR):
        color = ERROR_COLOR if finding.severity == Severity.ERROR else WARNING_COLOR
        colors[finding.face_ids] = color

    colored.visual.face_colors = colors
    colored.export(path)
