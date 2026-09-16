"""Typer CLI entry point for PartCheck."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from partcheck.checks import (  # noqa: F401  (imports register each check)
    CHECKS,
    OverhangCheck,
    SharpCornerCheck,
    ThinWallCheck,
)
from partcheck.loader import load_part
from partcheck.report import build_report

app = typer.Typer(add_completion=False, help="Automated manufacturability review for 3D parts.")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@app.callback()
def _main() -> None:
    """Automated manufacturability review for 3D parts."""


def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


@app.command()
def check(
    part: Path = typer.Argument(..., exists=True, help="Path to an STL file."),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="YAML config path."),
) -> None:
    """Run all registered checks against PART and print a report."""
    cfg = _load_config(config)
    units = cfg.get("units", "mm")

    mesh = load_part(str(part), units=units)

    findings = []
    for name, check_cls in CHECKS.items():
        check_cfg = cfg.get("checks", {}).get(name, {})
        findings.extend(check_cls().run(mesh, check_cfg))

    report = build_report(str(part), units, mesh, findings)

    typer.echo(f"Part: {report.part.path}")
    typer.echo(
        f"  faces={report.part.face_count} vertices={report.part.vertex_count} "
        f"watertight={report.part.watertight}"
    )
    if not report.findings:
        typer.echo("No issues found.")
        return

    for f in report.findings:
        typer.echo(f"[{f.severity.value.upper()}] {f.check}: {f.message}")

    typer.echo("")
    typer.echo("Summary:")
    for check_name, count in report.summary.items():
        typer.echo(f"  {check_name}: {count} finding(s)")


if __name__ == "__main__":
    app()
