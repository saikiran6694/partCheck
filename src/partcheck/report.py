"""Build a Report from a loaded mesh and its findings."""

from __future__ import annotations

import trimesh

from partcheck.models import Finding, PartInfo, Report


def build_report(path: str, units: str, mesh: trimesh.Trimesh, findings: list[Finding]) -> Report:
    part = PartInfo(
        path=path,
        units=units,
        watertight=bool(mesh.is_watertight),
        face_count=len(mesh.faces),
        vertex_count=len(mesh.vertices),
        bounds_mm=mesh.bounds.tolist(),
    )

    summary: dict[str, int] = {}
    for finding in findings:
        summary[finding.check] = summary.get(finding.check, 0) + 1

    return Report(part=part, findings=findings, summary=summary)
