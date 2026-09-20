"""Typer CLI entry point for PartCheck."""

from __future__ import annotations

from enum import Enum
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
from partcheck.models import Report
from partcheck.report import build_report, render_markdown, write_json, write_markdown
from partcheck.visualize import export_highlighted

app = typer.Typer(add_completion=False, help="Automated manufacturability review for 3D parts.")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    MD = "md"


@app.callback()
def _main() -> None:
    """Automated manufacturability review for 3D parts."""


def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _print_text(report: Report) -> None:
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


@app.command()
def check(
    part: Path = typer.Argument(..., exists=True, help="Path to an STL file."),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="YAML config path."),
    out: Path | None = typer.Option(
        None, "--out", help="Output directory for report/GLB files (default: next to PART)."
    ),
    fmt: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", help="Stdout report format: text, json, or md."
    ),
) -> None:
    """Run all registered checks against PART, print a report, and write a JSON
    report, a Markdown report, and a highlighted GLB (flagged faces coloured
    red for errors, orange for warnings)."""
    cfg = _load_config(config)
    units = cfg.get("units", "mm")

    mesh = load_part(str(part), units=units)

    findings = []
    for name, check_cls in CHECKS.items():
        check_cfg = cfg.get("checks", {}).get(name, {})
        findings.extend(check_cls().run(mesh, check_cfg))

    report = build_report(str(part), units, mesh, findings)

    if fmt is OutputFormat.JSON:
        typer.echo(report.model_dump_json(indent=2))
    elif fmt is OutputFormat.MD:
        typer.echo(render_markdown(report))
    else:
        _print_text(report)

    out_dir = out if out is not None else part.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = part.stem

    write_json(report, out_dir / f"{stem}_report.json")
    write_markdown(report, out_dir / f"{stem}_report.md")
    export_highlighted(mesh, findings, str(out_dir / f"{stem}_highlighted.glb"))

    typer.echo("")
    typer.echo(f"Wrote {out_dir / f'{stem}_report.json'}")
    typer.echo(f"Wrote {out_dir / f'{stem}_report.md'}")
    typer.echo(f"Wrote {out_dir / f'{stem}_highlighted.glb'}")


if __name__ == "__main__":
    app()
