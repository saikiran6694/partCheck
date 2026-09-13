"""Manufacturability checks, registered by name in `CHECKS`."""

from partcheck.checks.base import CHECKS, Check
from partcheck.checks.overhang import OverhangCheck

__all__ = ["CHECKS", "Check", "OverhangCheck"]
