"""Sharp internal corner check: unfilleted concave corners that a round CNC cutter can't cut."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import trimesh

from partcheck.checks.base import Check
from partcheck.models import Finding, Severity


class SharpCornerCheck(Check):
    name = "sharp_corner"

    def run(self, mesh: trimesh.Trimesh, config: dict[str, Any]) -> list[Finding]:
        max_angle = float(config.get("max_concave_angle_deg", 60.0))
        min_length = float(config.get("min_edge_length_mm", 2.0))

        angles = mesh.face_adjacency_angles
        convex = mesh.face_adjacency_convex
        concave_mask = (~convex) & (angles > np.radians(max_angle))

        if not np.any(concave_mask):
            return []

        edge_vertices = mesh.face_adjacency_edges[concave_mask]
        edge_faces = mesh.face_adjacency[concave_mask]
        edge_angles = angles[concave_mask]

        graph = nx.Graph()
        graph.add_edges_from(edge_vertices.tolist())
        vertex_to_component = {
            v: i for i, comp in enumerate(nx.connected_components(graph)) for v in comp
        }
        edge_component = np.array(
            [vertex_to_component[v0] for v0 in edge_vertices[:, 0]], dtype=np.int64
        )

        findings: list[Finding] = []
        for comp_id in np.unique(edge_component):
            mask = edge_component == comp_id
            vpairs = edge_vertices[mask]
            length = float(
                np.linalg.norm(
                    mesh.vertices[vpairs[:, 0]] - mesh.vertices[vpairs[:, 1]], axis=1
                ).sum()
            )
            if length < min_length:
                continue

            faces = sorted({int(f) for pair in edge_faces[mask] for f in pair})
            worst_angle_deg = float(np.degrees(edge_angles[mask].max()))

            findings.append(
                Finding(
                    check=self.name,
                    severity=Severity.WARNING,
                    face_ids=faces,
                    value=round(length, 2),
                    threshold=min_length,
                    message=(
                        f"Sharp internal corner, {length:.2f} mm long, {worst_angle_deg:.1f} deg "
                        f"(limit {max_angle:.1f} deg); needs a fillet for CNC milling."
                    ),
                )
            )
        return findings
