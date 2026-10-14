"""omicstra CLI.

`init` scaffolds a project root. it is an explicit command and creates nothing
on install - no downloads, no model weights, no implicit directories. a new
cohort lives in its own directory, never inside the omicstra package.
"""
from __future__ import annotations

import json
import sys
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
    """omicstra - cross-modal embedding alignment, fusion, and evidence-based routing."""


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
@click.option("--project-dir", default=None, type=click.Path(path_type=Path),
              help="cohort root. defaults to OMICSTRA_PROJECT_DIR.")
@click.option("--project-id", default=None)
@click.option("--inputs-dir", default="data/canonical",
              help="where the per-sample files live, relative to the cohort root. "
                   "the canonical directory by default - that is what the package reads.")
@click.option("--max-samples", default=None, type=int,
              help="cap the number of samples read. omit to derive it from this "
                   "machine's open-file and memory limits; 0 means no cap.")
@click.option("--write/--no-write", default=True,
              help="write inventory.json into the cohort root.")
def inventory(project_dir, project_id, inputs_dir, max_samples, write) -> None:
    """describe a cohort's data - 8 steps - and write the record.

    the record is a committable artifact and other steps read it: the encode
    gate takes its per-sample unit counts from here rather than measuring them
    again. a record covering a fraction of the cohort therefore makes a
    unit-floor gate silent rather than wrong, so this exists to produce a
    complete one.
    """
    import json

    from omicstra.protocols import build_protocol
    from omicstra.protocols.inventory import INVENTORY_STEPS

    if project_dir is not None:
        settings.project_dir = project_dir.expanduser()
    root = settings.project_root(project_id)

    params = {st.id: {} for st in INVENTORY_STEPS}
    params["files"] = {"inputs_dir": inputs_dir}
    if max_samples is not None:
        params["shape"] = {"max_samples": max_samples}

    recs = build_protocol(INVENTORY_STEPS, "inventory").invoke(
        {"project_dir": str(root), "project_id": project_id, "params": params})

    for k, v in recs.items():
        click.echo(f"  [{v.get('status', '?'):<14}] {k:<12} {str(v.get('result', ''))[:62]}")

    cov = (recs.get("shape", {}).get("observed") or {}).get("coverage")
    if cov:
        click.echo(f"\n  coverage: {cov['n_read']}/{cov['n_declared']} samples read, "
                   f"cap {cov['cap']} ({cov['cap_source']}, bound by "
                   f"{cov['budget']['bound_by']})")
        if cov["capped"]:
            click.echo("  CAPPED - downstream gates that read these counts see a "
                       "fraction of the cohort. pass --max-samples 0 for all of it.")

    failed = [k for k, v in recs.items() if v.get("status") == "fail"]
    if write and not failed:
        out = root / "inventory.json"
        out.write_text(json.dumps(recs, indent=2, default=str) + "\n")
        click.echo(f"\n  wrote {out.name} ({out.stat().st_size/1024:.0f} KB)")
    elif failed:
        click.echo(f"\n  not written - {failed} failed. a record of a failed "
                   "inventory would be a record of nothing.")
    raise SystemExit(1 if failed else 0)


@main.command()
@click.option("--project-dir", default=None, type=click.Path(path_type=Path))
@click.option("--project-id", default=None)
@click.option("--step", "steps", multiple=True,
              help="step id to run. repeatable. a step with no params is skipped, "
                   "never guessed at.")
def eda(project_dir, project_id, steps):
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

    from omicstra.graphs.encode import build_encode_graph

    # the encode gate is passed, so the compute arm is inventory -> eda -> gate
    # -> encode rather than stopping at the verdict. it decides from
    # declarations and pauses only where a cohort has not made the call.
    app = build_omicstra_graph(checkpointer=InMemorySaver(),
                               eda=build_eda_graph(checkpointer=None),
                               encode=build_encode_graph(checkpointer=None))
    cfg = {"configurable": {"thread_id": "cli"}}
    # the two chains take separate params. `params` is eda's, keyed by check id;
    # `inventory_params` is the inventory chain's, keyed by inventory step id. a
    # step with no entry is skipped rather than guessed at, so an empty mapping
    # silently disables the whole chain - which is what it did.
    from omicstra.protocols.inventory import INVENTORY_STEPS

    inv_params = {st.id: {} for st in INVENTORY_STEPS}
    inv_params["files"] = {"inputs_dir": "data/canonical"}

    # every DECLARED check by default, not one. a step is skipped when its id is
    # absent from this mapping - by key, not by value - so naming only
    # count_statistics ran 1 of 10 and left the gate to reach a verdict on
    # silence. --step narrows; it should never be the only thing that widens.
    from omicstra.contracts.eda import load_contract

    declared = load_contract()["checks"]
    all_checks = [c["id"] for c in (declared if isinstance(declared, list)
                                    else declared.values())]
    payload = {"question": "", "project_id": project_id,
               "params": {s: {} for s in steps} or {c: {} for c in all_checks},
               "inventory_params": inv_params}

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
        # the encode gate asks for a MAPPING of gate id -> option, not yes/no.
        # sending a bool here halts the run for "unanswered", which looks like a
        # refusal the person did not make.
        gates = [g for g in v.get("gates", []) if g.get("open")]
        for g in gates:
            click.echo(f"  gate {g['id']}: {g['question']}")
            click.echo(f"       options: {', '.join(g['options'])}")
            if g.get("pre_answerable_by"):
                click.echo(f"       declare {g['pre_answerable_by']} to answer it in advance")
            click.echo(click.style(f"       forecloses: {g['forecloses']}", dim=True))
        click.echo(f"  {v.get('no_default', 'the system does not pick')}")
        click.echo("-" * 62)

        # a graph that pauses needs somebody to answer. with no terminal there is
        # nobody, and blocking on stdin turns a correct interrupt into a hang -
        # which is what happened. report what is being asked and stop.
        if not sys.stdin.isatty():
            click.echo("\nnot a terminal, so nothing can answer this.")
            if gates:
                click.echo("declare the fields above in cohort.json, or run this "
                           "interactively.")
            raise SystemExit(2)

        if gates:
            answers = {}
            for g in gates:
                answers[g["id"]] = click.prompt(f"  {g['id']}",
                                                type=click.Choice(g["options"]))
            out = app.invoke(Command(resume=answers), cfg)
        else:
            out = app.invoke(Command(resume=click.confirm("accept and proceed?",
                                                          default=False)), cfg)

    # a halt carries its reason in the last record. the subgraph no longer raises
    # - it returns a halt so the parent keeps the arm decision - so the reason
    # has to be read out rather than caught.
    if out.get("halted"):
        reason = next((r.get("reason") for r in reversed(out.get("records", []))
                       if r.get("reason")), "")
        click.echo(f"arm:     {out.get('arm')}   (evidence present: "
                   f"{out.get('has_evidence')})")
        raise click.ClickException(reason or "the run halted without a recorded reason")

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
    # the arm per TASK. a cohort-level "has evidence" hides which questions it
    # can actually answer and which are outstanding work.
    arms = d.get("arms") or {}
    click.echo(f"\n  arms        {arms.get('ask', 0)} answerable now, "
               f"{arms.get('compute', 0)} need compute")
    for t in d.get("answerable_now", []):
        click.echo(click.style(f"    ask      {t}", dim=True))
    for t in d.get("needs_compute", []):
        click.echo(f"    compute  {t}")
    for r in d["decisions"]:
        click.echo(f"    {r['record_id']}  {r['outcome']:<13} {r['chosen'] or '-'}")
    click.echo(click.style(f"\n  {d['interpretation_boundary']}", dim=True))


@main.command()
@click.option("--project-dir", default=None, type=click.Path(path_type=Path))
@click.option("--project-id", default=None)
@click.option("--out", default=None, type=click.Path(path_type=Path),
              help="where to write the pack. default: print it, write nothing.")
@click.option("--note", "notes", multiple=True, metavar="TASK=TEXT",
              help="a human note for one row. repeatable. the only prose this takes.")
@click.option("--approve", is_flag=True, help="approve the proposal at the gate.")
def promote(project_dir: Path | None, project_id: str | None, out: Path | None,
            notes: tuple[str, ...], approve: bool) -> None:
    """propose the evidence pack from the cohort's scored artifacts.

    values, intervals, margins and outcomes are computed. notes are yours: pass
    --note task=text, and --approve to write. without approval nothing is
    written, which is the honest state for a pack nobody vouched for.
    """
    import json

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from omicstra.graphs.promote import build_promote_graph

    if project_dir is not None:
        settings.project_dir = project_dir.expanduser()
    root = settings.project_root(project_id)
    src_path = root / "evidence_sources.json"
    if not src_path.is_file():
        click.echo(f"no evidence_sources.json in {root.name} - this cohort has not "
                   "declared which artifact answers which task family.")
        raise SystemExit(2)
    src = json.loads(src_path.read_text())
    runs_root = (root / src["runs_root"]).resolve()

    app = build_promote_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "cli-promote"}}
    res = app.invoke({"project_id": project_id or root.name, "sources": src,
                      "runs_root": str(runs_root), "runs": src["runs"]}, cfg)

    parsed: dict[str, dict[str, str]] = {}
    for n in notes:
        task, _, text = n.partition("=")
        if not text:
            raise SystemExit(f"--note wants TASK=TEXT, got {n!r}")
        parsed.setdefault(task.strip(), {})["note"] = text.strip()

    while "__interrupt__" in res:
        v = res["__interrupt__"][0].value
        click.echo(f"\ncohort:  {root.name}")
        click.echo(f"grid:    {runs_root}")
        click.echo("-" * 68)
        for row in v["proposal"]:
            if row["outcome"] == "not_available":
                click.echo(f"  {row['task']:<30} not_available - {row['why'][:58]}")
                continue
            lead = row.get("leads") or "-"
            margin = f"  margin {row['margin']}" if row.get("margin") is not None else ""
            click.echo(f"  {row['task']:<30} {row['outcome']:<10} {lead}{margin}")
            if row.get("refusal"):
                click.echo(click.style(f"       {row['refusal']}", dim=True))
        click.echo("-" * 68)
        click.echo(v["asks"])
        for task, fields in parsed.items():
            click.echo(f"  note on {task}: {fields['note']}")
        if not approve:
            click.echo("\nnot approved, so nothing is written. re-run with --approve "
                       "(and --note task=text) to emit the pack.")
            raise SystemExit(0)
        res = app.invoke(Command(resume={"notes": parsed, "approved": True}), cfg)

    pack = res.get("pack") or {}
    if not pack:
        click.echo("halted - no pack written.")
        raise SystemExit(0)
    text = json.dumps(pack, indent=2) + "\n"
    if out:
        Path(out).write_text(text)
        click.echo(f"wrote {out}")
    else:
        click.echo(text)


@main.command()
@click.option("--project-dir", default=None, type=click.Path(path_type=Path))
@click.option("--project-id", default=None)
@click.option("--out", default=None, type=click.Path(path_type=Path),
              help="write the report here. default: print it.")
@click.option("--pack", "pack_path", default=None, type=click.Path(path_type=Path),
              help="a pack to render instead of the cohort's own routing_evidence.json.")
@click.option("--derive", is_flag=True,
              help="propose the pack from artifacts first, so outcomes are present.")
def report(project_dir: Path | None, project_id: str | None, out: Path | None,
           pack_path: Path | None, derive: bool) -> None:
    """render this cohort's report from its declarations, records and pack.

    the renderer is a template and knows no cohort: every line comes from a
    declaration, a record or the pack, and a section a cohort has no input for
    says so in one line rather than being dropped.
    """
    import json

    from omicstra.report import gather, render

    if project_dir is not None:
        settings.project_dir = project_dir.expanduser()
    root = settings.project_root(project_id)
    bundle = gather(root)

    if pack_path:
        bundle["pack"] = json.loads(Path(pack_path).read_text())
    elif derive:
        from omicstra.protocols.promote import propose, provenance

        src_path = root / "evidence_sources.json"
        if not src_path.is_file():
            click.echo("no evidence_sources.json - nothing to derive from.")
            raise SystemExit(2)
        src = json.loads(src_path.read_text())
        p = propose(src, (root / src["runs_root"]).resolve(), src["runs"])
        bundle["pack"] = p
        bundle["provenance"] = provenance(p)

    ledger = settings.resolve(settings.runs_dir) / root.name / "records"
    jsonl = ledger / "decisions.jsonl"
    if jsonl.is_file():
        bundle["records"] = [json.loads(ln) for ln in jsonl.read_text().splitlines() if ln.strip()]
    bundle["version"] = __import__("omicstra").__version__

    text = render(bundle, f"{root.name} evaluation report")
    if out:
        Path(out).write_text(text)
        click.echo(f"wrote {out}  ({len(text.splitlines())} lines)")
    else:
        click.echo(text)


@main.command(name="run")
@click.option("--project-dir", default=None, type=click.Path(path_type=Path))
@click.option("--project-id", default=None)
@click.option("--out-dir", default=None, type=click.Path(path_type=Path),
              help="write report.md, pack.json, records.json and stages.json here.")
@click.option("--compute", is_flag=True,
              help="train and score rather than resolve. refused into declared read-only paths.")
@click.option("--out-root", default=None, type=click.Path(path_type=Path),
              help="with --compute: stage a rerun of a finished grid here.")
@click.option("--diff", "diff_pack", default=None, type=click.Path(path_type=Path),
              help="a curated pack to diff the rebuilt one against. exits 1 on disagreement.")
def run_cmd(project_dir: Path | None, project_id: str | None, out_dir: Path | None,
            compute: bool, out_root: Path | None, diff_pack: Path | None) -> None:
    """the whole chain: join, grid, eval, pack, report - resolving by default.

    a stage that cannot resolve is recorded with its reason and the chain
    continues, so a cohort missing its grid still produces a report saying so.
    """
    import json

    from omicstra.run import diff_against, run_cohort

    if project_dir is not None:
        settings.project_dir = project_dir.expanduser()
    out = run_cohort(project_id, compute=compute, out_dir=out_dir, out_root=out_root)

    click.echo(f"cohort: {out['project_id']}")
    for stage, status in out["stages"].items():
        mark = "ok " if status in ("resolved", "computed", "proposed") else "-- "
        click.echo(f"  {mark}{stage:<12} {status}")
    if out.get("written"):
        click.echo(f"\nwrote {out['written']}/report.md and 3 more")

    if diff_pack:
        curated = json.loads(Path(diff_pack).read_text())
        diffs = diff_against(out.get("pack") or {}, curated)
        click.echo(f"\ndiff against {Path(diff_pack).name}: {len(diffs)} disagreement(s)")
        for d in diffs:
            click.echo(f"  {d['task']}/{d['method']}: {d['kind']}"
                       + (f"  curated {d['curated']} vs computed {d['computed']}"
                          if "curated" in d and "computed" in d else ""))
        raise SystemExit(1 if diffs else 0)


@main.command()
@click.option("--project-id", default=None)
@click.option("--project-dir", default=None, type=click.Path(path_type=Path),
              help="cohort root. defaults to OMICSTRA_PROJECT_DIR.")
def selftest(project_id: str | None, project_dir: Path | None = None) -> None:
    """assert the ROUTING RULES hold, then report what this cohort can answer.

    the two halves are separate on purpose. the rules are properties of the
    contract and must hold on every cohort, so they are asserted against
    synthetic evidence packs built here. what a cohort ANSWERS is a property of
    its own recorded evaluation, so it is reported rather than asserted.

    this read the other way until 2026-10-13: it defaulted --project-id to the
    seed cohort, asserted that cross_modal_retrieval recommends and
    tissue_state_grouping ties, named `B1_v3` as the contraindicated method, and
    printed `raw_he 0.164` - that cohort's measurement, as a literal in package
    code. on a second cohort those failed because the cohort differed, not
    because a rule broke, and one of them PASSED on a cohort with no evidence at
    all because it tested a string prefix that an absent `why` cannot have. this
    is the first command the readme tells a new cohort owner to run.
    """
    from omicstra.routing import list_task_families, load_routing_evidence, resolve
    from omicstra.routing import resolve as _res

    if project_dir is not None:
        settings.project_dir = project_dir.expanduser()

    ev = load_routing_evidence(project_id)
    fails = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        fails += not ok
        click.echo(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    def report(name: str, detail: str) -> None:
        click.echo(f"  [ -- ] {name}  {detail}")

    def r(tid, **kw):
        return resolve(tid, project_id=project_id, log=False, **kw)

    def pack(task: str, **over) -> dict:
        """a synthetic evidence pack. the rules are asserted against THIS, not
        against whatever cohort happens to be bound."""
        return {"project_id": "synthetic", "evidence_version": "t",
                "scope_statement": "synthetic", "method_names": {"M1": "method one",
                                                                 "M2": "method two"},
                "tasks": {task: {"metric": "toy", "source": "none",
                                 "higher_is_better": True,
                                 "floor": {"id": "nothing", "value": 0.0},
                                 "candidates": [{"id": "M1", "value": 0.5}], **over}}}

    # --- the rules. cohort-independent, must hold everywhere ----------------
    click.echo("\nrules (asserted against synthetic evidence):")
    check("one clear leader recommends",
          _res("cross_modal_retrieval", evidence=pack("cross_modal_retrieval"),
               log=False).chosen == "method one")
    check("equal candidates decline to pick",
          _res("cross_modal_retrieval",
               evidence=pack("cross_modal_retrieval",
                             candidates=[{"id": "M1", "value": 0.5},
                                         {"id": "M2", "value": 0.5}]),
               log=False).chosen is None)
    check("a leader below the reference says so first",
          _res("tissue_state_grouping",
               evidence=pack("tissue_state_grouping",
                             reference={"id": "raw_modality", "value": 0.9}),
               log=False).why.startswith("raw_modality scores 0.9"))
    check("a named contraindication refuses",
          _res("cross_modal_retrieval", evidence=pack("cross_modal_retrieval"),
               proposed_method="M1", log=False, override=False) is not None)
    check("unknown task not routed", r("no_such_family").status == "not_run")
    check("every family the contract declares is listed",
          len(list_task_families(project_id=project_id)["families"]) == 7,
          f"{len(list_task_families(project_id=project_id)['families'])} families")

    # --- this cohort. reported, not asserted --------------------------------
    cohort = settings.project_root(project_id).name
    click.echo(f"\nthis cohort ({cohort}):")
    if not ev or not ev.get("tasks"):
        report("recorded evaluation", "NONE - this cohort is not routable, which is "
                                      "the correct state before it has been evaluated")
        click.echo(f"\n{fails} rule failure(s)." if fails else "\nrules hold.")
        raise SystemExit(1 if fails else 0)

    check("scope statement present", bool(ev.get("scope_statement")))
    for tid in sorted(ev.get("tasks", {})):
        d = r(tid)
        check(f"{tid} resolves", bool(d.why), d.chosen or ("tie" if d.tie else "-"))
        check(f"{tid} carries scope", ev["scope_statement"] in d.caveats)
        report(f"{tid} outcome", d.chosen or ("tie" if d.tie else
                                              "escalate" if d.escalated else "-"))

    # a contraindication is cohort-specific, so the method and the tasks are READ
    # from the pack. this named `B1_v3` as a literal, which is a tnbc-92 run id.
    contra = (ev.get("contraindications") or [{}])[0]
    mid = contra.get("method")
    applies = list(contra.get("applies_to") or [])
    exempt = list(contra.get("does_not_apply_to") or [])
    if mid and applies:
        check("a contraindication the pack records refuses",
              all(r(t, proposed_method=mid).contraindicated for t in applies),
              f"{mid} on {', '.join(applies)}")
        ov = r(applies[0], proposed_method=mid, override=True)
        check("overriding it proceeds as recorded dissent", bool(ov.chosen))
        if exempt:
            check("and it does not reach the tasks the pack exempts",
                  not any(r(t, proposed_method=mid).contraindicated for t in exempt),
                  ", ".join(exempt))
    else:
        report("contraindications", "none recorded")

    check("record_id stable across wording",
          r(min(ev["tasks"]), question="a").record_id ==
          r(min(ev["tasks"]), question="b").record_id)

    click.echo(f"\n{'all checks passed' if not fails else f'{fails} FAILED'}")
    raise SystemExit(1 if fails else 0)


@main.command()
@click.option("--address", default=None, metavar="HOST:PORT",
              help="the scheduler. defaults to OMICSTRA_TEMPORAL_ADDRESS.")
@click.option("--namespace", default=None, help="defaults to OMICSTRA_TEMPORAL_NAMESPACE.")
@click.option("--task-queue", default=None, help="defaults to OMICSTRA_TEMPORAL_TASK_QUEUE.")
@click.option("--preload", "preloads", multiple=True, metavar="MODULE",
              help="import this module before polling, so an encoder it registers "
                   "is resolvable here. repeatable.")
@click.option("--max-seconds", default=None, type=int, metavar="N",
              help="stop after N seconds, for a worker inside a job allocation "
                   "with a wall clock. default: until signalled.")
def worker(address, namespace, task_queue, preloads, max_seconds) -> None:
    """serve the encode queue: hold the shard bodies, run what is submitted.

    the other half of submit-only. `he_encode` and `st_encode` hand the shard
    list to the scheduler and return, which is only durable if some process is
    polling - this is that process. it needs no cohort, no project root and no
    knowledge of the run: a shard carries its own geometry, gene axis, paths and
    encoder name, and the bodies close over nothing.

    it may be started before or after the submitter, outlive it, or run on
    another machine. a shard is idempotent on its output file, so a worker
    joining a run already in flight re-runs nothing that landed.

    ONE shard at a time, and that is the workflow's shape rather than a limit
    here: `EncodeWorkflow` awaits each activity in turn because a shard holds a
    631M-parameter model and a decoded slide, so a second thread in this process
    would be idle for one run and out of memory for two. scale by running more
    workers.
    """
    import asyncio
    import contextlib
    import importlib
    import signal

    from omicstra.dispatch.bodies import shard_bodies

    addr = address or settings.temporal_address
    if not addr:
        click.echo("no scheduler address. set OMICSTRA_TEMPORAL_ADDRESS or pass "
                   "--address. without one the dispatcher runs shards in-process "
                   "and needs no worker.")
        raise SystemExit(2)
    try:
        from temporalio.client import Client

        from omicstra.dispatch.temporal import build_worker, register
    except ImportError as e:
        # the actual error, not a guess at it. missing is the common case and
        # the extra is the fix; a temporalio that is installed and broken reads
        # differently, and an operator needs to be able to tell which they have.
        click.echo(f"no durable backend here: {e}\n"
                   "install it with: pip install 'omicstra[durable]'")
        raise SystemExit(2) from None

    # before connecting, so a missing encoder module fails here rather than on
    # the first shard of a run that is already accepted - by which time the
    # scheduler holds the run and three attempts are spent on an import.
    for mod in preloads:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            click.echo(f"--preload {mod}: {e}")
            raise SystemExit(2) from None

    bodies = shard_bodies()
    for name, fn in bodies.items():
        register(name, fn)

    queue = task_queue or settings.temporal_task_queue
    ns = namespace or settings.temporal_namespace

    async def serve() -> None:
        client = await Client.connect(addr, namespace=ns)
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            # graceful on purpose: the context manager below lets the shard in
            # flight finish and report, so the scheduler sees a completed
            # activity rather than a heartbeat timeout and a retry.
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:
                # not a posix loop. --max-seconds is then the only way out, and
                # an abrupt kill is still safe: the history holds the run and the
                # shards are idempotent on their output files.
                pass
        async with build_worker(client, queue):
            click.echo(f"polling {queue} at {addr} (namespace {ns}); "
                       f"serving {', '.join(sorted(bodies))}")
            if max_seconds is None:
                await stop.wait()
            else:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), max_seconds)
        click.echo("stopped.")

    asyncio.run(serve())


if __name__ == "__main__":
    main()