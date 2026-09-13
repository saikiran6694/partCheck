"""Shared contract for manufacturability checks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import trimesh

from partcheck.models import Finding

CHECKS: dict[str, type[Check]] = {}


class Check(ABC):
    """Base class for a single detection algorithm.

    Subclasses set `name` to the key used in the config file and in
    `CHECKS`, and implement `run` to return zero or more findings.
    """

    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", None):
            CHECKS[cls.name] = cls

    @abstractmethod
    def run(self, mesh: trimesh.Trimesh, config: dict[str, Any]) -> list[Finding]:
        """Run this check against `mesh` using this check's slice of the config."""
        raise NotImplementedError
