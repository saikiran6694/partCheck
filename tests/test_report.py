from __future__ import annotations

from pathlib import Path

import trimesh

from partcheck.models import Finding, Report, Severity
from partcheck.report import build_report, render_markdown, write_json, write_markdown


def _finding(**overrides) -> Finding:
    base = {
        "check": "overhang",
        "severity": Severity.WARNING,
        "face_ids": [0, 1],
        "value": 60.0,
        "threshold": 45.0,
        "message": "Overhang region needs support.",
    }
    base.update(overrides)
    return Finding(**base)


def test_build_report_no_findings(cube_mesh: trimesh.Trimesh) -> None:
    report = build_report("cube.stl", "mm", cube_mesh, [])
    assert report.findings == []
    assert report.summary == {}
    assert report.part.face_count == len(cube_mesh.faces)


def test_build_report_summarizes_by_check(cube_mesh: trimesh.Trimesh) -> None:
    findings = [_finding(check="overhang"), _finding(check="overhang"), _finding(check="thin_wall")]
    report = build_report("cube.stl", "mm", cube_mesh, findings)
    assert report.summary == {"overhang": 2, "thin_wall": 1}


def test_render_markdown_no_findings(cube_mesh: trimesh.Trimesh) -> None:
    report = build_report("cube.stl", "mm", cube_mesh, [])
    md = render_markdown(report)
    assert "No issues found." in md


def test_render_markdown_lists_findings(cube_mesh: trimesh.Trimesh) -> None:
    report = build_report("cube.stl", "mm", cube_mesh, [_finding()])
    md = render_markdown(report)
    assert "| Check | Severity | Value | Threshold | Message |" in md
    assert "overhang" in md


def test_write_json_roundtrips(cube_mesh: trimesh.Trimesh, tmp_path: Path) -> None:
    report = build_report("cube.stl", "mm", cube_mesh, [_finding()])
    path = tmp_path / "report.json"
    write_json(report, path)
    reloaded = Report.model_validate_json(path.read_text())
    assert reloaded == report


def test_write_markdown_creates_file(cube_mesh: trimesh.Trimesh, tmp_path: Path) -> None:
    report = build_report("cube.stl", "mm", cube_mesh, [])
    path = tmp_path / "report.md"
    write_markdown(report, path)
    assert path.exists()
    assert "PartCheck Report" in path.read_text()
