"""Typed data models shared across PartCheck: findings and reports."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class Finding(BaseModel):
    """A single flagged region produced by one check."""

    check: str
    severity: Severity
    face_ids: list[int] = Field(default_factory=list)
    value: float
    threshold: float
    message: str


class PartInfo(BaseModel):
    """Metadata about the loaded part, independent of any check."""

    path: str
    units: str
    watertight: bool
    face_count: int
    vertex_count: int
    bounds_mm: list[list[float]]


class Report(BaseModel):
    """Everything a run produces: part metadata, findings, and a per-check summary."""

    part: PartInfo
    findings: list[Finding] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)

    def flagged_checks(self) -> list[str]:
        return [name for name, count in self.summary.items() if count > 0]
