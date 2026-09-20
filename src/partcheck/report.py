"""Build a Report from a loaded mesh and its findings, and render it to JSON/Markdown."""

from __future__ import annotations

from pathlib import Path

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


def render_markdown(report: Report) -> str:
    lines = [
        "# PartCheck Report",
        "",
        f"**Part:** {report.part.path}",
        (
            f"**Units:** {report.part.units}  "
            f"**Faces:** {report.part.face_count}  "
            f"**Vertices:** {report.part.vertex_count}  "
            f"**Watertight:** {report.part.watertight}"
        ),
        "",
    ]

    if not report.findings:
        lines.append("No issues found.")
        return "\n".join(lines) + "\n"

    lines += [
        "## Findings",
        "",
        "| Check | Severity | Value | Threshold | Message |",
        "| --- | --- | --- | --- | --- |",
    ]
    for f in report.findings:
        lines.append(
            f"| {f.check} | {f.severity.value} | {f.value:.2f} | {f.threshold:.2f} | {f.message} |"
        )

    lines += ["", "## Summary", "", "| Check | Findings |", "| --- | --- |"]
    for check_name, count in report.summary.items():
        lines.append(f"| {check_name} | {count} |")

    return "\n".join(lines) + "\n"


def write_json(report: Report, path: str | Path) -> None:
    Path(path).write_text(report.model_dump_json(indent=2) + "\n")


def write_markdown(report: Report, path: str | Path) -> None:
    Path(path).write_text(render_markdown(report))
