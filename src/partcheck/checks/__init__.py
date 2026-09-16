"""Manufacturability checks, registered by name in `CHECKS`."""

from partcheck.checks.base import CHECKS, Check
from partcheck.checks.overhang import OverhangCheck
from partcheck.checks.sharp_corner import SharpCornerCheck
from partcheck.checks.thin_wall import ThinWallCheck

__all__ = ["CHECKS", "Check", "OverhangCheck", "SharpCornerCheck", "ThinWallCheck"]
