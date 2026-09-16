from __future__ import annotations

import trimesh

from partcheck.checks.thin_wall import ThinWallCheck

CONFIG = {"min_thickness_mm": 1.0, "ray_offset_mm": 0.01}


def test_solid_cube_is_not_thin(cube_mesh: trimesh.Trimesh) -> None:
    findings = ThinWallCheck().run(cube_mesh, CONFIG)
    assert findings == []


def test_thin_shell_is_flagged_near_0_5mm(thin_shell_mesh: trimesh.Trimesh) -> None:
    findings = ThinWallCheck().run(thin_shell_mesh, CONFIG)
    assert len(findings) >= 1
    assert all(f.check == "thin_wall" for f in findings)
    assert all(f.value < CONFIG["min_thickness_mm"] for f in findings)
    thinnest = min(f.value for f in findings)
    assert 0.3 < thinnest < 0.7
