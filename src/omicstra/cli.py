"""omicstra CLI.

`init` scaffolds a project root. it is an explicit command and creates nothing
on install - no downloads, no model weights, no implicit directories. a new
cohort lives in its own directory, never inside the omicstra package.
"""
from __future__ import annotations

import json
from pathlib import Path

import click

from omicstra.contracts.project import ProjectConfig
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
    """omicstra - cross-modal embedding alignment and evidence-based routing."""


@main.command()
@click.option("--project-dir", required=True, type=click.Path(path_type=Path),
              help="cohort root. created if absent. never inside the omicstra package.")
@click.option("--project-id", default=None, help="cohort id. defaults to the directory name.")
@click.option("--force", is_flag=True, help="overwrite an existing project.json.")
@click.option("--public", is_flag=True,
              help="cohort is published and redistributable. sets data_classification=public and relaxes the project .gitignore. DEFAULT is restricted - forgetting this flag fails closed.")
def init(project_dir: Path, project_id: str | None, force: bool, public: bool) -> None:
    """scaffold a new cohort project root.

    the project home is the shape of the package repo MINUS the package, so a
    cohort is its own git repo from minute one and its trace/commit coupling has
    something of its own to attach to.

    `--public` sets data_classification and, as a CONSEQUENCE, relaxes the
    .gitignore. restricted is the default because the failure modes are
    asymmetric: treating public data as restricted costs convenience, treating
    restricted data as public is unrecoverable once it has left.
    """
    project_dir = project_dir.expanduser()
    cfg_path = project_dir / "project.json"
    if cfg_path.exists() and not force:
        raise click.ClickException(f"{cfg_path} exists - pass --force to overwrite")

    classification = "public" if public else "restricted"

    for d in ("steps", "data/inputs", "data/embeddings", "runs", "notebooks"):
        (project_dir / d).mkdir(parents=True, exist_ok=True)
        (project_dir / d / ".gitkeep").touch()

    payload = dict(TEMPLATE, project_id=project_id or project_dir.resolve().name)
    cfg_path.write_text(json.dumps(payload, indent=2) + "\n")

    # every field a PERSON declares about the cohort goes in one file, beside
    # subject_id_column and the two backend fields. platform.json holds every
    # field a person declares about a PLATFORM. one file each, no third home.
    cohort = {"note": "cohort-level declarations. read at runtime, never guessed.",
              "contract": "configs/data_contract.json#roles",
              "data_classification": classification,
              "subject_id_column": "",
              "compute_backend": "mac",
              # the CONNECTING CLIENT supplies the model; this server ships none and
              # holds no key. null until a deployment names its client's endpoint
              # (anthropic, bedrock, ...), so a scaffold never claims otherwise.
              "client_model_backend": None}
    if public:
        # a redistribution claim with no provenance is not checkable
        cohort["provenance"] = {"source": "", "licence": "", "gated": ""}
    (project_dir / "cohort.json").write_text(json.dumps(cohort, indent=2) + "\n")

    (project_dir / ".gitignore").write_text(_project_gitignore(classification))
    if not (project_dir / "program.md").exists():
        (project_dir / "program.md").write_text(
            f"# {payload['project_id']} - program\n\n"
            "the search space, the constraints, and the stopping criteria.\n"
            "a human writes this file, not code.\n\n"
            "## search space\n\n## constraints\n\n## stopping criteria\n")

    click.echo(f"scaffolded {payload['project_id']} at {project_dir}")
    click.echo(f"  data_classification: {classification}"
               + ("" if public else "  (default - pass --public to relax)"))
    for line in ("project.json   fill in platform, encoders, labels",
                 "cohort.json    classification, subject_id_column, backends",
                 "program.md     search space, constraints, stopping criteria",
                 "data/inputs/   raw cohort data lands here",
                 "steps/         generated step bodies land here",
                 ".gitignore     written from the classification above",
                 "",
                 "nothing was downloaded. point the server at it with:",
                 f"  OMICSTRA_PROJECT_DIR={project_dir} python -m omicstra.mcp.server"):
        click.echo(f"  {line}")
    if public and not cohort["provenance"]["source"]:
        click.echo("\n  public cohort: fill provenance.source / licence / gated in "
                   "cohort.json.\n  an unprovenanced public claim is not checkable.")


def _project_gitignore(classification: str) -> str:
    """the .gitignore is a CONSEQUENCE of the classification, not the feature.

    note what does not change: caches, runs and embeddings stay ignored either
    way. they are large and regenerable, and 'public' says a thing may be
    redistributed - not that git is the right place to put six gigabytes of it.
    """
    common = (
        "# regenerable or large - ignored regardless of classification\n"
        "data/embeddings/\n"
        "runs/\n"
        "__pycache__/\n"
        ".ipynb_checkpoints/\n"
        ".env\n"
        ".env.*\n"
        "!.env.example\n"
        "_scratch/\n"
    )
    if classification == "public":
        return (
            "# data_classification: public\n"
            "# raw inputs MAY be committed - but see the size note below.\n"
            + common +
            "\n# public does not mean small. commit raw inputs deliberately,\n"
            "# per file, or keep them out and record how to re-fetch them.\n"
            "# uncomment to keep them out entirely:\n"
            "# data/inputs/\n"
        )
    return (
        "# data_classification: restricted\n"
        "# raw inputs MUST NOT be committed. this line is the enforcement.\n"
        "data/inputs/\n"
        + common
    )


@main.command()
@click.option("--project-dir", default=None, type=click.Path(path_type=Path))
@click.option("--project-id", default=None)
@click.option("--adata-path", default=None, type=click.Path(path_type=Path),
              help="override the object path. normally produced by the inventory "
                   "protocol's A7 step and passed through state.")
@click.option("--step", "steps", multiple=True,
              help="step id to run. repeatable. a step with no params is skipped, "
                   "never guessed at.")
def eda(project_dir, project_id, adata_path, steps):
    """run the EDA through the level-0 graph, gates and all.

    this is the whole orchestrator from a terminal: discover picks the arm, the
    eda subgraph profiles and gates, and a caution PAUSES here and asks. the
    same graph serves MCP as run/resume over a thread_id - the only difference
    is who answers the question.

    discover picks the arm, which means this command does NOT always gate: a
    cohort that already carries an evidence pack routes to ask, and the eda
    subgraph is never entered. that is the point of the arm, and the command
    says so rather than printing an empty verdict. `omicstra gate` runs the gate
    on any cohort, evidence or not.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
    from omicstra.graph import build_omicstra_graph
    from omicstra.graphs.eda import build_eda_graph

    if project_dir is not None:
        settings.project_dir = project_dir.expanduser()

    app = build_omicstra_graph(checkpointer=InMemorySaver(),
                               eda=build_eda_graph(checkpointer=None))
    cfg = {"configurable": {"thread_id": "cli"}}
    payload = {"question": "", "project_id": project_id,
               "params": {s: {} for s in steps} or {"count_statistics": {}}}
    if adata_path:
        payload["adata_path"] = str(adata_path.expanduser())

    out = app.invoke(payload, cfg)
    click.echo(f"cohort:  {settings.project_root(project_id).name}")
    click.echo(f"arm:     {out.get('arm')}   (evidence present: {out.get('has_evidence')})")

    while "__interrupt__" in out:
        v = out["__interrupt__"][0].value
        click.echo("\n" + "-" * 62)
        click.echo(v.get("question", "the graph is asking."))
        for c in v.get("cautions", []):
            click.echo(f"  caution: {c}")
        for e in v.get("escalations", []):
            click.echo(f"  escalated: {e.get('check')} - {e.get('why_not_inherited','')}")
        if v.get("consequence"):
            click.echo(f"  if you accept: {v['consequence']}")
        click.echo(f"  {v.get('no_default', 'the system does not pick')}")
        click.echo("-" * 62)
        out = app.invoke(Command(resume=click.confirm("accept and proceed?", default=False)), cfg)

    if out.get("verdict") is None:
        # the arm decided correctly and the command said nothing about it, so it
        # read as a broken gate. `verdict: None` is not a null result - it means
        # the eda subgraph was never entered, because this cohort already carries
        # an evidence pack and `discover` routed it to ask. say that, and name
        # the command that does run the gate.
        click.echo("\nthe eda gate did not run: this cohort has an evidence pack, so "
                   "`discover`\nrouted it to the ask arm. that is the correct arm for a "
                   "cohort with\nrecorded evidence - the gate is for one without.")
        click.echo("\n  omicstra gate     runs the admissibility gate directly, on any cohort")
        raise SystemExit(0)

    click.echo(f"\nverdict: {out.get('verdict')}")
    if out.get("halted"):
        click.echo("halted - not accepted.")
    for r in out.get("records", []):
        click.echo(f"  [{r.get('status', r.get('verdict', '?')):<15}] {r.get('step_id')}"
                   f"  {str(r.get('result', ''))[:52]}")
    raise SystemExit(1 if out.get("halted") else 0)


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
    """route one question to the method this cohort's evidence supports.

    this goes through the level-0 graph, not `resolve` directly, because a
    contraindicated method must ASK rather than report: the graph stops at
    `ask_human`, checkpoints, and waits. the MCP tool of the same name answers
    directly and returns the refusal with its options, since over a stateless
    protocol a refusal is a result rather than a pause.

    `--override-refusal` pre-answers the gate, so the run is unattended and the
    dissent is still recorded - the same "a declared answer does not fire the
    gate" rule the compute contract applies to its six.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from omicstra.graph import build_omicstra_graph
    from omicstra.graphs.route import build_route_graph

    if project_dir is not None:
        settings.project_dir = project_dir.expanduser()

    app = build_omicstra_graph(checkpointer=InMemorySaver(),
                               route=build_route_graph(checkpointer=None))
    cfg = {"configurable": {"thread_id": f"route::{task_id}"}}
    out = app.invoke({"question": question, "task_id": task_id,
                      "proposed_method": proposed_method,
                      "override": override_refusal,
                      "project_id": project_id}, cfg)

    if not out.get("has_evidence"):
        click.echo(click.style(
            "  this cohort has no evidence pack, so it is not routable. "
            "another cohort's winner is not inherited.", fg="yellow"))
        return

    # the gate. resuming uses the PARENT's thread_id - the subgraph compiles
    # with checkpointer=None and never had one of its own.
    while "__interrupt__" in out:
        v = out["__interrupt__"][0].value
        click.echo("\n" + "-" * 62)
        click.echo(v.get("question", "the graph is asking."))
        for k in ("why", "consequence", "if_you_proceed"):
            if v.get(k):
                click.echo(click.style(f"  {k.replace('_', ' ')}: {v[k]}", dim=True))
        out = app.invoke(Command(resume=click.confirm("\nproceed anyway?",
                                                      default=False)), cfg)

    d = out.get("decision") or {}
    for m in out.get("modality", []):
        click.echo(click.style(f"  {m['step_id'].split('::')[0]}: {m['decision']}", dim=True))
    click.echo(f"\n[{(out.get('resolution') or '-').upper()}] {d.get('chosen') or '-'}")
    click.echo(f"  why:         {d.get('why', '')}")
    click.echo(f"  consequence: {d.get('consequence', '')}")
    click.echo(click.style(f"  record_id:   {d.get('record_id', '')}", dim=True))


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