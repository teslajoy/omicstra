"""omicstra CLI.

`init` scaffolds a project root. it is an explicit command and creates nothing
on install - no downloads, no model weights, no implicit directories. a new
cohort lives in its own directory, never inside the omicstra package.
"""
from __future__ import annotations

import json
from pathlib import Path

import click

from .config import ProjectConfig
from .eda import run_gate
from .settings import settings

TEMPLATE = {
    "project_id": "",
    "platform": "",
    "k_neighbors": 6,
    "seed": 42,
    "split": "patient_stratified_85_15",
    "encoders": [
        {"name": "", "dim": 0, "role": "primary", "raw": True},
        {"name": "", "dim": 0, "role": "st", "raw": False},
    ],
    "h2_label": "",
    "supervision": "",
    "headline_view": "z_he",
    "hypothesis_metric": "auc",
    "contrastive_runs": [],
    "classical_runs": [],
    "niches_dir": "",
    "runs_dir": "runs",
}


@click.group()
def main() -> None:
    """omicstra - cross-modal reasoning for spatial biology."""


@main.command()
@click.option("--project-dir", required=True, type=click.Path(path_type=Path),
              help="cohort root. created if absent. never inside the omicstra package.")
@click.option("--project-id", default=None, help="cohort id. defaults to the directory name.")
@click.option("--force", is_flag=True, help="overwrite an existing project.json.")
def init(project_dir: Path, project_id: str | None, force: bool) -> None:
    """scaffold a new cohort project root."""
    project_dir = project_dir.expanduser()
    cfg_path = project_dir / "project.json"
    if cfg_path.exists() and not force:
        raise click.ClickException(f"{cfg_path} exists - pass --force to overwrite")

    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "steps").mkdir(exist_ok=True)
    (project_dir / "steps" / ".gitkeep").touch()

    payload = dict(TEMPLATE, project_id=project_id or project_dir.resolve().name)
    cfg_path.write_text(json.dumps(payload, indent=2) + "\n")

    click.echo(f"scaffolded {payload['project_id']} at {project_dir}")
    for line in ("project.json   fill in platform, encoders, labels",
                 "steps/         generated step bodies land here",
                 "",
                 "nothing was downloaded. point the server at it with:",
                 f"  OMICSTRA_PROJECT_DIR={project_dir} python -m omicstra.mcp.server"):
        click.echo(f"  {line}")


@main.command()
@click.option("--project-dir", default=None, type=click.Path(path_type=Path))
@click.option("--project-id", default=None)
def gate(project_dir: Path | None, project_id: str | None) -> None:
    """run the EDA admissibility gate and print the verdict."""
    if project_dir is not None:
        settings.project_dir = project_dir.expanduser()
    try:
        r = run_gate(project_id)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e)) from None
    click.echo(f"verdict: {r.verdict}   declared: {r.declared_verdict}   "
               f"agrees: {r.agrees_with_declared}")
    for c in r.checks:
        click.echo(f"  [{c.status:<14}] {'req' if c.required else '   '} "
                   f"{c.id:<26} {c.authority}")
    for label, items in (("violations", r.violations), ("cautions", r.cautions),
                         ("advisories", r.advisories)):
        if items:
            click.echo(f"{label} ({len(items)}):")
            for i in items:
                click.echo(f"  - {i}")
    if r.escalations:
        click.echo(f"escalations ({len(r.escalations)}) - no default, the system does not pick:")
        for e in r.escalations:
            click.echo(f"  - {e['check']}  {e['package_default']}")
            click.echo(f"      why not inherited: {e['why_not_inherited']}")
            if e.get("consequence"):
                click.echo(f"      forecloses:        {e['consequence']}")
    raise SystemExit(0 if r.passed else 1)


@main.command()
@click.option("--project-dir", default=None, type=click.Path(path_type=Path))
@click.option("--project-id", default=None)
def describe(project_dir: Path | None, project_id: str | None) -> None:
    """print a cohort's declared data structure."""
    if project_dir is not None:
        settings.project_dir = project_dir.expanduser()
    try:
        cfg = ProjectConfig.load(project_id)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e)) from None
    click.echo(f"{cfg.project_id}  ({cfg.project_dir})")
    click.echo(f"  platform    {cfg.platform}")
    click.echo(f"  unit        niche = centre spot + {cfg.k_neighbors} neighbours")
    click.echo(f"  split       {cfg.split}  seed {cfg.seed}")
    for e in cfg.encoders:
        click.echo(f"  encoder     {e.name:<12} {e.dim:>5}d  {e.role}{'  raw' if e.raw else ''}")
    click.echo(f"  supervision {cfg.supervision}  (never a feature)")


if __name__ == "__main__":
    main()