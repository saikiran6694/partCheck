from __future__ import annotations

from pathlib import Path

import trimesh

from partcheck.models import Finding, Severity
from partcheck.visualize import BASE_COLOR, ERROR_COLOR, WARNING_COLOR, export_highlighted


def test_export_highlighted_no_findings_is_all_base_color(
    cube_mesh: trimesh.Trimesh, tmp_path: Path
) -> None:
    path = tmp_path / "out.glb"
    export_highlighted(cube_mesh, [], str(path))

    reloaded = trimesh.load(str(path), force="mesh")
    colors = {tuple(c) for c in reloaded.visual.face_colors}
    assert colors == {BASE_COLOR}


def test_export_highlighted_uses_flat_severity_colors(
    cube_mesh: trimesh.Trimesh, tmp_path: Path
) -> None:
    findings = [
        Finding(
            check="thin_wall",
            severity=Severity.ERROR,
            face_ids=[0],
            value=0.4,
            threshold=1.0,
            message="thin",
        ),
        Finding(
            check="overhang",
            severity=Severity.WARNING,
            face_ids=[1],
            value=60.0,
            threshold=45.0,
            message="overhang",
        ),
    ]
    path = tmp_path / "out.glb"
    export_highlighted(cube_mesh, findings, str(path))

    reloaded = trimesh.load(str(path), force="mesh")
    colors = {tuple(c) for c in reloaded.visual.face_colors}
    # Unwelded vertices mean no interpolation, so exactly these 3 flat colors.
    assert colors == {BASE_COLOR, ERROR_COLOR, WARNING_COLOR}


def test_error_wins_over_warning_on_shared_face(cube_mesh: trimesh.Trimesh, tmp_path: Path) -> None:
    findings = [
        Finding(
            check="overhang",
            severity=Severity.WARNING,
            face_ids=[0],
            value=60.0,
            threshold=45.0,
            message="overhang",
        ),
        Finding(
            check="thin_wall",
            severity=Severity.ERROR,
            face_ids=[0],
            value=0.4,
            threshold=1.0,
            message="thin",
        ),
    ]
    path = tmp_path / "out.glb"
    export_highlighted(cube_mesh, findings, str(path))

    reloaded = trimesh.load(str(path), force="mesh")
    colors = {tuple(c) for c in reloaded.visual.face_colors}
    assert WARNING_COLOR not in colors
    assert ERROR_COLOR in colors
