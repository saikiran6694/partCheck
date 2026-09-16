"""Thin-wall check: flags regions thinner than a minimum manufacturable thickness."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import trimesh

from partcheck.checks.base import Check
from partcheck.models import Finding, Severity

logger = logging.getLogger(__name__)

_SUBDIVIDE_MAX_EDGE_MM = 2.0
_OPPOSITE_NORMAL_DOT_MAX = -0.5


class ThinWallCheck(Check):
    name = "thin_wall"

    def run(self, mesh: trimesh.Trimesh, config: dict[str, Any]) -> list[Finding]:
        min_thickness = float(config.get("min_thickness_mm", 1.0))
        ray_offset = float(config.get("ray_offset_mm", 0.01))

        sampled, face_index = mesh.subdivide_to_size(
            max_edge=_SUBDIVIDE_MAX_EDGE_MM, return_index=True
        )

        origins = sampled.triangles_center - sampled.face_normals * ray_offset
        directions = -sampled.face_normals
        locs, ray_idx, tri_idx = sampled.ray.intersects_location(
            origins, directions, multiple_hits=False
        )

        missed = len(origins) - len(set(ray_idx.tolist()))
        if missed:
            logger.warning(
                "thin_wall: %d/%d sample rays had no hit (mesh may not be watertight); "
                "those areas were skipped",
                missed,
                len(origins),
            )

        if len(ray_idx) == 0:
            return []

        dist = np.linalg.norm(locs - origins[ray_idx], axis=1)
        opposite = (
            np.einsum("ij,ij->i", sampled.face_normals[ray_idx], sampled.face_normals[tri_idx])
            < _OPPOSITE_NORMAL_DOT_MAX
        )
        thin = opposite & (dist < min_thickness)
        if not np.any(thin):
            return []

        orig_faces = face_index[ray_idx[thin]]
        thin_dist = dist[thin]

        # Collapse to the minimum observed thickness per original face.
        min_thickness_by_face: dict[int, float] = {}
        for face, d in zip(orig_faces.tolist(), thin_dist.tolist(), strict=True):
            if face not in min_thickness_by_face or d < min_thickness_by_face[face]:
                min_thickness_by_face[face] = d

        thin_faces = np.array(sorted(min_thickness_by_face), dtype=np.int64)

        adjacency = mesh.face_adjacency
        thin_set = set(thin_faces.tolist())
        if adjacency.size:
            adj_mask = np.array(
                [a in thin_set and b in thin_set for a, b in adjacency], dtype=bool
            )
            edges = adjacency[adj_mask]
        else:
            edges = adjacency
        components = trimesh.graph.connected_components(edges, nodes=thin_faces)

        findings: list[Finding] = []
        for component in components:
            thickness = min(min_thickness_by_face[int(f)] for f in component)
            findings.append(
                Finding(
                    check=self.name,
                    severity=Severity.ERROR,
                    face_ids=[int(i) for i in component],
                    value=round(float(thickness), 3),
                    threshold=min_thickness,
                    message=(
                        f"Thin wall region with minimum thickness {thickness:.2f} mm "
                        f"(limit {min_thickness:.2f} mm)."
                    ),
                )
            )
        return findings
