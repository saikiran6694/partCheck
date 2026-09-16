from __future__ import annotations

import trimesh

from partcheck.checks.overhang import OverhangCheck

CONFIG = {"max_angle_deg": 45, "min_area_mm2": 1.0}


def test_cube_has_no_overhangs(cube_mesh: trimesh.Trimesh) -> None:
    findings = OverhangCheck().run(cube_mesh, CONFIG)
    assert findings == []


def test_t_shape_flags_underside_of_crossbar(t_shape_mesh: trimesh.Trimesh) -> None:
    findings = OverhangCheck().run(t_shape_mesh, CONFIG)
    assert len(findings) == 1
    assert findings[0].check == "overhang"
    assert findings[0].value > CONFIG["max_angle_deg"]
