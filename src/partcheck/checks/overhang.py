"""Overhang check: surfaces tilted too far from vertical need print support."""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from partcheck.checks.base import Check
from partcheck.models import Finding, Severity

_ON_PLATE_TOLERANCE_MM = 0.01


class OverhangCheck(Check):
    name = "overhang"

    def run(self, mesh: trimesh.Trimesh, config: dict[str, Any]) -> list[Finding]:
        max_angle = float(config.get("max_angle_deg", 45.0))
        min_area = float(config.get("min_area_mm2", 1.0))

        nz = mesh.face_normals[:, 2]
        on_plate = mesh.triangles_center[:, 2] < mesh.bounds[0, 2] + _ON_PLATE_TOLERANCE_MM
        flagged = np.where((nz < -np.sin(np.radians(max_angle))) & ~on_plate)[0]

        if flagged.size == 0:
            return []

        adjacency = mesh.face_adjacency
        flagged_set = set(flagged.tolist())
        adj_mask = np.array(
            [a in flagged_set and b in flagged_set for a, b in adjacency], dtype=bool
        )
        edges = adjacency[adj_mask] if adjacency.size else adjacency
        components = trimesh.graph.connected_components(edges, nodes=flagged)

        findings: list[Finding] = []
        for component in components:
            area = float(mesh.area_faces[component].sum())
            if area < min_area:
                continue
            tilt_deg = float(np.degrees(np.arcsin(np.clip(-nz[component], -1.0, 1.0))).max())
            findings.append(
                Finding(
                    check=self.name,
                    severity=Severity.WARNING,
                    face_ids=[int(i) for i in component],
                    value=round(tilt_deg, 2),
                    threshold=max_angle,
                    message=(
                        f"Overhang region of {area:.2f} mm^2 tilted {tilt_deg:.1f} deg "
                        f"from vertical (limit {max_angle:.1f} deg); needs support material."
                    ),
                )
            )
        return findings
