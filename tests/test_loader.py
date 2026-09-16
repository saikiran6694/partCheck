from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from partcheck.loader import load_part


@pytest.fixture
def box_stl(tmp_path: Path) -> Path:
    box = trimesh.creation.box(extents=[10, 10, 10])
    box.apply_translation([0, 0, 100])  # float above the origin
    path = tmp_path / "box.stl"
    box.export(path)
    return path


def test_load_part_rests_on_z_zero(box_stl: Path) -> None:
    mesh = load_part(str(box_stl), units="mm")
    assert np.isclose(mesh.bounds[0, 2], 0.0, atol=1e-6)


def test_load_part_no_scale_for_mm(box_stl: Path) -> None:
    mesh = load_part(str(box_stl), units="mm")
    extents = mesh.bounds[1] - mesh.bounds[0]
    assert np.allclose(extents, [10, 10, 10])


def test_load_part_scales_cm_to_mm(box_stl: Path) -> None:
    mesh = load_part(str(box_stl), units="cm")
    extents = mesh.bounds[1] - mesh.bounds[0]
    assert np.allclose(extents, [100, 100, 100])


def test_load_part_is_watertight(box_stl: Path) -> None:
    mesh = load_part(str(box_stl), units="mm")
    assert mesh.is_watertight


def test_load_part_rejects_unknown_units(box_stl: Path) -> None:
    with pytest.raises(ValueError, match="Unknown units"):
        load_part(str(box_stl), units="furlongs")
