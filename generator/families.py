"""Four parametric part families, each returning a shape, its parameters, and
ground-truth labels derived directly from those parameters.

Every family targets the manufacturability checks it's designed to exercise;
where a family's geometry has a fixed, non-random answer for a check it
isn't varying (e.g. a plain rectangular pocket is always geometrically sharp
unless filleted), the label reflects that constant truth rather than being
left to a random draw.
"""

from __future__ import annotations

import random
from typing import Any

import cadquery as cq
import numpy as np
import trimesh

STL_TOLERANCE_MM = 0.02
STL_ANGULAR_TOLERANCE_RAD = 0.1  # fine enough that fillet facets stay well under any sane sharp-corner threshold

Params = dict[str, Any]
Labels = dict[str, bool]


def export_shape(shape: cq.Workplane | trimesh.Trimesh, path: str) -> None:
    if isinstance(shape, trimesh.Trimesh):
        shape.export(path)
    else:
        cq.exporters.export(shape, path, tolerance=STL_TOLERANCE_MM, angularTolerance=STL_ANGULAR_TOLERANCE_RAD)


def _sample_thickness(
    rng: random.Random,
    thin_range: tuple[float, float],
    thick_range: tuple[float, float],
    border_range: tuple[float, float] = (0.9, 1.1),
) -> float:
    """~55% clearly thin, ~35% clearly thick, ~10% borderline around 1.0 mm.

    Skewed toward thin because only two of the four families vary
    thin_wall at all; this keeps the overall dataset closer to balanced
    without needing every family to test every check.
    """
    r = rng.random()
    if r < 0.55:
        return rng.uniform(*thin_range)
    if r < 0.90:
        return rng.uniform(*thick_range)
    return rng.uniform(*border_range)


def _sample_arm_angle(rng: random.Random) -> float:
    """~30% clearly under 45deg, ~50% clearly over, ~20% borderline 40-50deg.

    Skewed toward overhang because only two of the four families vary it.
    """
    r = rng.random()
    if r < 0.3:
        return rng.uniform(20, 40)
    if r < 0.8:
        return rng.uniform(50, 90)
    return rng.uniform(40, 50)


def generate_l_bracket(rng: random.Random) -> tuple[cq.Workplane, Params, Labels]:
    """An angle bracket: two flat legs of thickness t meeting at a right angle,
    with an optional fillet on the inner corner."""
    leg_a = rng.uniform(20, 60)
    leg_b = rng.uniform(20, 60)
    width = rng.uniform(12, 25)
    thickness = _sample_thickness(rng, thin_range=(0.4, 0.9), thick_range=(1.1, 4.0))
    fillet_radius = 0.0 if rng.random() < 0.5 else rng.uniform(1.0, 5.0)

    t = thickness
    profile_pts = [(0, 0), (leg_a, 0), (leg_a, t), (t, t), (t, leg_b), (0, leg_b)]
    solid = cq.Workplane("XZ").polyline(profile_pts).close().extrude(width)

    if fillet_radius > 0:
        # extrude() on an "XZ" workplane runs along -Y, so the corner point
        # (verified against the solid's actual edge centers) uses -width/2.
        corner_point = (t, -width / 2, t)
        solid = solid.edges(cq.selectors.NearestToPointSelector(corner_point)).fillet(fillet_radius)

    params: Params = {
        "leg_a_mm": leg_a,
        "leg_b_mm": leg_b,
        "width_mm": width,
        "thickness_mm": thickness,
        "fillet_radius_mm": fillet_radius,
    }
    labels: Labels = {
        "thin_wall": thickness < 1.0,
        "sharp_corner": fillet_radius == 0.0,
        "overhang": False,
    }
    return solid, params, labels


def generate_shelled_box(rng: random.Random) -> tuple[cq.Workplane, Params, Labels]:
    """A hollow box. A closed top always leaves an unsupported ceiling inside;
    an open top does not. The unfilleted cavity always has sharp inner corners."""
    width = rng.uniform(15, 45)
    depth = rng.uniform(15, 45)
    height = rng.uniform(15, 45)
    wall = _sample_thickness(rng, thin_range=(0.4, 0.9), thick_range=(1.1, 3.0))
    closed_top = rng.random() < 0.6  # skewed: only this family and t_bar vary overhang
    orientation_deg = rng.choice([0, 90, 180, 270])

    # A small, wall-scaled fillet on every cavity edge so the inner corners
    # are never sharp -- this family targets thin_wall/overhang only.
    cavity_fillet = max(0.15, min(1.0, wall * 0.5))

    outer = cq.Workplane("XY").box(width, depth, height)
    if closed_top:
        cavity = cq.Workplane("XY").box(width - 2 * wall, depth - 2 * wall, height - 2 * wall)
        cavity = cavity.edges().fillet(cavity_fillet)
    else:
        # Taller than the outer box and shifted up by `wall`, so it clears the
        # bottom cap but cuts all the way through the top; only the bottom
        # and vertical edges need filleting (the top ones are clipped away).
        cavity = cq.Workplane("XY").box(width - 2 * wall, depth - 2 * wall, height)
        cavity = cavity.edges("|Z or <Z").fillet(cavity_fillet)
        cavity = cavity.translate((0, 0, wall))

    result = outer.cut(cavity).rotate((0, 0, 0), (0, 0, 1), orientation_deg)

    params: Params = {
        "width_mm": width,
        "depth_mm": depth,
        "height_mm": height,
        "wall_thickness_mm": wall,
        "closed_top": closed_top,
        "orientation_deg": orientation_deg,
    }
    labels: Labels = {
        "thin_wall": wall < 1.0,
        "sharp_corner": False,
        "overhang": closed_top,
    }
    return result, params, labels


def generate_plate_with_pocket(rng: random.Random) -> tuple[cq.Workplane, Params, Labels]:
    """A plate with a blind pocket, open at the top. Plate thickness and pocket
    depth are kept comfortably clear of the thin-wall threshold on purpose,
    since this family targets sharp_corner only."""
    plate_w = rng.uniform(30, 60)
    plate_d = rng.uniform(30, 60)
    plate_t = rng.uniform(8, 14)
    pocket_w = plate_w * rng.uniform(0.4, 0.6)
    pocket_d = plate_d * rng.uniform(0.4, 0.6)
    pocket_depth = rng.uniform(3.0, plate_t - 3.0)
    corner_radius = 0.0 if rng.random() < 0.5 else rng.uniform(1.0, 4.0)
    corner_radius = min(corner_radius, pocket_w * 0.2, pocket_d * 0.2, pocket_depth * 0.8)

    block = cq.Workplane("XY").box(plate_w, plate_d, plate_t)
    pocket = cq.Workplane("XY").rect(pocket_w, pocket_d).extrude(pocket_depth)
    if corner_radius > 0:
        pocket = pocket.edges("|Z or <Z").fillet(corner_radius)
    pocket = pocket.translate((0, 0, plate_t / 2 - pocket_depth))
    result = block.cut(pocket)

    params: Params = {
        "plate_w_mm": plate_w,
        "plate_d_mm": plate_d,
        "plate_t_mm": plate_t,
        "pocket_w_mm": pocket_w,
        "pocket_d_mm": pocket_d,
        "pocket_depth_mm": pocket_depth,
        "corner_radius_mm": corner_radius,
    }
    labels: Labels = {
        "thin_wall": False,
        "sharp_corner": corner_radius == 0.0,
        "overhang": False,
    }
    return result, params, labels


def generate_t_bar(rng: random.Random) -> tuple[trimesh.Trimesh, Params, Labels]:
    """A vertical stem with an arm cantilevered off it at `arm_angle_deg` from
    vertical. At 0deg the arm points straight up (no overhang); at 90deg it's
    a horizontal crossbar (full overhang), matching the check's own tilt
    measurement so the label threshold lines up exactly with the check's."""
    stem_w = rng.uniform(8, 14)
    stem_h = rng.uniform(40, 60)
    arm_len = rng.uniform(20, 50)
    # Both cross-section dims capped at stem_w: either one being wider than
    # the stem leaves a small overhanging shelf at the joint, regardless of
    # arm_angle_deg (whichever dimension ends up facing sideways at low
    # angles, or facing down at high angles, still can't exceed the stem).
    arm_height = rng.uniform(0.5, 0.95) * stem_w
    arm_depth = rng.uniform(0.5, 0.95) * stem_w
    angle_deg = _sample_arm_angle(rng)
    attach_z = stem_h * rng.uniform(0.5, 0.8)

    stem = trimesh.creation.box(extents=[stem_w, stem_w, stem_h])
    stem.apply_translation([0, 0, stem_h / 2])

    arm = trimesh.creation.box(extents=[arm_len, arm_depth, arm_height])
    arm.apply_translation([arm_len / 2, 0, 0])  # base (attach point) at the origin
    # angle_deg - 90 about Y: 0deg -> arm points straight up, 90deg -> arm
    # horizontal (verified numerically: this makes the check's own tilt
    # measurement on the arm's underside equal angle_deg exactly).
    arm.apply_transform(trimesh.transformations.rotation_matrix(np.radians(angle_deg - 90), [0, 1, 0]))
    arm.apply_translation([0, 0, attach_z])

    # A real union (not concatenation) so the arm's buried base end-cap is
    # removed rather than left as a stray triangle inside the stem, which
    # would otherwise trip the overhang/thin-wall checks independent of angle.
    mesh = trimesh.boolean.union([stem, arm], engine="manifold")

    params: Params = {
        "stem_width_mm": stem_w,
        "stem_height_mm": stem_h,
        "arm_length_mm": arm_len,
        "arm_height_mm": arm_height,
        "arm_depth_mm": arm_depth,
        "arm_angle_deg": angle_deg,
        "attach_z_mm": attach_z,
    }
    labels: Labels = {
        "thin_wall": False,
        # An oblique strut-to-post union always leaves a sharp reentrant edge
        # at the joint without a fillet, regardless of arm_angle_deg -- this
        # is real geometry, not a byproduct of a specific angle.
        "sharp_corner": True,
        "overhang": angle_deg > 45.0,
    }
    return mesh, params, labels


FAMILIES = {
    "l_bracket": generate_l_bracket,
    "shelled_box": generate_shelled_box,
    "plate_with_pocket": generate_plate_with_pocket,
    "t_bar": generate_t_bar,
}
