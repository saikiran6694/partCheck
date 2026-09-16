"""Shared pytest fixtures: test shapes built with trimesh primitives (and CadQuery
where a feature, like a fillet, has no trimesh primitive equivalent)."""

from __future__ import annotations

from pathlib import Path

import cadquery as cq
import pytest
import trimesh


def _prepare(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Mimic loader.load_part's mesh prep, for fixtures built in-memory."""
    mesh.merge_vertices()
    mesh.fix_normals(multibody=False)
    mesh.apply_translation([0.0, 0.0, -mesh.bounds[0, 2]])
    return mesh


@pytest.fixture
def cube_mesh() -> trimesh.Trimesh:
    """A plain 20 mm cube: no overhangs, no thin walls, no concave edges at all."""
    return _prepare(trimesh.creation.box(extents=[20, 20, 20]))


@pytest.fixture
def t_shape_mesh() -> trimesh.Trimesh:
    """An upright T: the underside of the crossbar overhangs with no support below it."""
    stem = trimesh.creation.box(extents=[10, 10, 40])
    stem.apply_translation([0, 0, 20])
    bar = trimesh.creation.box(extents=[60, 10, 10])
    bar.apply_translation([0, 0, 45])
    return _prepare(trimesh.util.concatenate([stem, bar]))


@pytest.fixture
def thin_shell_mesh() -> trimesh.Trimesh:
    """A hollow 20 mm box with 0.5 mm walls on every side."""
    outer = trimesh.creation.box(extents=[20, 20, 20])
    inner = trimesh.creation.box(extents=[19, 19, 19])
    shell = outer.difference(inner, engine="manifold")
    return _prepare(shell)


@pytest.fixture
def sharp_pocket_mesh(tmp_path: Path) -> trimesh.Trimesh:
    """A 30x30x10 mm block with an unfilleted 16x16x6 mm blind pocket: sharp internal corners."""
    block = cq.Workplane("XY").box(30, 30, 10)
    pocket = cq.Workplane("XY").rect(16, 16).extrude(6).translate((0, 0, 5 - 6))
    result = block.cut(pocket)

    stl_path = tmp_path / "sharp_pocket.stl"
    cq.exporters.export(result, str(stl_path), tolerance=0.01, angularTolerance=0.1)

    mesh = trimesh.load(str(stl_path), force="mesh")
    return _prepare(mesh)


@pytest.fixture
def filleted_pocket_mesh(tmp_path: Path) -> trimesh.Trimesh:
    """Same pocket as sharp_pocket_mesh, but with 3 mm fillets on every pocket edge."""
    block = cq.Workplane("XY").box(30, 30, 10)
    pocket = cq.Workplane("XY").rect(16, 16).extrude(6)
    pocket = pocket.edges("|Z or <Z").fillet(3).translate((0, 0, 5 - 6))
    result = block.cut(pocket)

    stl_path = tmp_path / "filleted_pocket.stl"
    cq.exporters.export(result, str(stl_path), tolerance=0.01, angularTolerance=0.1)

    mesh = trimesh.load(str(stl_path), force="mesh")
    return _prepare(mesh)
