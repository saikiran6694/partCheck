from __future__ import annotations

import trimesh

from partcheck.checks.sharp_corner import SharpCornerCheck

CONFIG = {"max_concave_angle_deg": 60, "min_edge_length_mm": 2.0}


def test_cube_has_no_sharp_corners(cube_mesh: trimesh.Trimesh) -> None:
    findings = SharpCornerCheck().run(cube_mesh, CONFIG)
    assert findings == []


def test_unfilleted_pocket_is_flagged(sharp_pocket_mesh: trimesh.Trimesh) -> None:
    findings = SharpCornerCheck().run(sharp_pocket_mesh, CONFIG)
    assert len(findings) >= 1
    assert all(f.check == "sharp_corner" for f in findings)


def test_filleted_pocket_is_not_flagged(filleted_pocket_mesh: trimesh.Trimesh) -> None:
    findings = SharpCornerCheck().run(filleted_pocket_mesh, CONFIG)
    assert findings == []
