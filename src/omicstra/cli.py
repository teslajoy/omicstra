"""omicstra CLI.

`init` scaffolds a project root. it is an explicit command and creates nothing
on install - no downloads, no model weights, no implicit directories. a new
cohort lives in its own directory, never inside the omicstra package.
"""
from __future__ import annotations

import json
from pathlib import Path

import click

from omicstra.config import ProjectConfig
from omicstra.eda import run_gate
from omicstra.settings import settings

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


@main.command()
@click.option("--project-dir", default=None, type=click.Path(path_type=Path))
@click.option("--project-id", default=None)
def families(project_dir: Path | None, project_id: str | None) -> None:
    """list the question types this cohort can route."""
    from omicstra.routing import list_task_families
    if project_dir is not None:
        settings.project_dir = project_dir.expanduser()
    out = list_task_families(project_id=project_id)
    if out["scope"]:
        click.echo(click.style(out["scope"], dim=True))
    for f in out["families"]:
        mark = "*" if f["routable_on_this_cohort"] else "-"
        click.echo(f"  {mark} {f['task_id']:<30} {f['hypothesis']:<8} {f['level']}")
        click.echo(f"      {f['asks']}")
    click.echo("\n  * routable on this cohort   - defined, no evidence here")


@main.command()
@click.argument("task_id")
@click.option("--question", default="", help="the question as asked, recorded verbatim.")
@click.option("--proposed-method", default=None, help="a method the asker named.")
@click.option("--override-refusal", is_flag=True,
              help="proceed against a contraindication. recorded as dissent.")
@click.option("--project-dir", default=None, type=click.Path(path_type=Path))
@click.option("--project-id", default=None)
def route(task_id: str, question: str, proposed_method: str | None,
          override_refusal: bool, project_dir: Path | None,
          project_id: str | None) -> None:
    """route one question to the method this cohort's evidence supports."""
    from omicstra.routing import resolve
    if project_dir is not None:
        settings.project_dir = project_dir.expanduser()
    r = resolve(task_id, question=question, proposed_method=proposed_method,
                override=override_refusal, project_id=project_id)
    res = ("OVERRIDE_ACK" if (r.contraindicated and r.actor == "human")
           else "REFUSE" if r.contraindicated
           else "ESCALATE" if r.escalated
           else "TIE" if r.tie else "RECOMMEND")
    for c in r.caveats:
        click.echo(click.style(f"  {c}", dim=True))
    click.echo(f"\n[{res}] {r.chosen or '-'}")
    click.echo(f"  why:         {r.why}")
    click.echo(f"  consequence: {r.consequence}")
    click.echo(click.style(f"  record_id:   {r.record_id}", dim=True))


@main.command(name="decisions")
@click.option("--project-dir", default=None, type=click.Path(path_type=Path))
@click.option("--project-id", default=None)
def decisions(project_dir: Path | None, project_id: str | None) -> None:
    """summarise the decision ledger - the reproducibility claim as a number."""
    from omicstra.routing import decision_record
    if project_dir is not None:
        settings.project_dir = project_dir.expanduser()
    d = decision_record(project_id=project_id)
    click.echo(f"{d['total']} decision(s)   {d['ledger']}")
    click.echo(f"  by outcome  {d['by_outcome']}")
    click.echo(f"  by actor    {d['by_actor']}")
    for r in d["decisions"]:
        click.echo(f"    {r['record_id']}  {r['outcome']:<13} {r['chosen'] or '-'}")
    click.echo(click.style(f"\n  {d['interpretation_boundary']}", dim=True))


@main.command()
@click.option("--project-id", default="tnbc-92")
def selftest(project_id: str) -> None:
    """assert the routing rules hold on this cohort. one line per check."""
    from omicstra.routing import list_task_families, load_routing_evidence, resolve

    ev = load_routing_evidence(project_id)
    fails = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        fails += not ok
        click.echo(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    def r(tid, **kw):
        return resolve(tid, project_id=project_id, log=False, **kw)

    check("evidence loads", bool(ev), f"{len(ev.get('tasks', {}))} tasks")
    check("scope statement present", bool(ev.get("scope_statement")))
    fams = list_task_families(project_id=project_id)["families"]
    check("every family declared", len(fams) == 7, f"{len(fams)} families")

    for tid in sorted(ev.get("tasks", {})):
        d = r(tid)
        check(f"{tid} resolves", bool(d.why), d.chosen or ("tie" if d.tie else "-"))
        check(f"{tid} carries scope", ev["scope_statement"] in d.caveats)

    check("recommend names one", bool(r("cross_modal_retrieval").chosen))
    check("tie declines to pick", r("tissue_state_grouping").chosen is None)
    check("clears its un-aligned reference",
          not r("tissue_state_grouping").why.startswith("raw_he scores"), "raw_he 0.164")
    # the shortfall path fires on no current task - assert the rule on a fixture
    synth = {"project_id": "synthetic", "evidence_version": "t",
             "scope_statement": "synthetic", "method_names": {"M1": "method one"},
             "tasks": {"tissue_state_grouping": {
                 "metric": "toy", "source": "none", "higher_is_better": True,
                 "floor": {"id": "nothing", "value": 0.0},
                 "reference": {"id": "raw_modality", "value": 0.9},
                 "candidates": [{"id": "M1", "value": 0.5}]}}}
    from omicstra.routing import resolve as _res
    check("a leader below the reference says so first",
          _res("tissue_state_grouping", evidence=synth, log=False)
          .why.startswith("raw_modality scores 0.9"))
    check("agreeing axes recommend", bool(r("structural_agreement").chosen))
    check("disagreeing axes escalate", r("pathway_discrimination").tie)
    check("named contraindication refuses",
          r("subject_identity_suppression", proposed_method="B1_v3").contraindicated)
    ov = r("subject_identity_suppression", proposed_method="B1_v3", override=True)
    check("override proceeds as dissent", bool(ov.chosen) and ov.actor == "human")
    check("contraindication is task-scoped",
          not r("pathway_transfer", proposed_method="B1_v3").contraindicated)
    check("record_id stable across wording",
          r("cross_modal_retrieval", question="a").record_id ==
          r("cross_modal_retrieval", question="b").record_id)
    check("unknown task not routed", r("no_such_family").status == "not_run")

    click.echo(f"\n{'all checks passed' if not fails else f'{fails} FAILED'}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()