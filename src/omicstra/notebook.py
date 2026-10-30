"""one run as a rerunnable notebook: the records in, an .ipynb out.

a cell per step. each calls the SAME step function that ran, with the
parameters that run recorded, and checks the numbers it gets back against the
numbers that were recorded. a clean rerun is the reproduction proof.

the cells call the STEPS, not the measures under them. a notebook that called
`measures/` directly would need a step-to-measure mapping maintained alongside
the chain, and a mapping that drifted would reproduce the wrong thing
convincingly. the step is already the reproducible unit and it calls into
`measures/` itself, so a reader who wants the arithmetic opens that file.

precision
---------
numbers are compared at `PLACES` decimals, which is one digit beyond the `.3f`
the steps report at. this is not a loosened test - it is the precision the
method supports. the vision encoder's float32 kernels differ by ~1e-6 between
backends on the same input, and `measures/` accumulates closed-form statistics
in float64 while the published artifacts are float32. asserting past that
compares rounding, not science.

failure
-------
a cell whose check fails records `fail` and does NOT stop the notebook; the
cells that depend on it record `not_run` with the reason. that is the chain's
own semantics - `protocols.build_protocol` makes an upstream failure not_run
and runs to the end so the record is complete - and the reason is the same: a
notebook that halts halfway leaves no record, and one that carries on
regardless recomputes on a bad input.
"""
from __future__ import annotations

from typing import Any, NamedTuple

PLACES = 4

# the unit is an ANALYSIS UNIT, not a run. a run id is an execution artifact -
# nobody opens `eval-a3f2.ipynb` - and this cohort's own committed notebooks are
# already named for topics: `h3_gpath2vec_v3`, `proposal_h1h2h3_evaluation`. so
# one file per unit, per cohort, regenerated to the latest state, and the run
# that produced the numbers is provenance INSIDE the header.
#
# a unit's CONTEXT is how a cohort becomes the dict its steps read. two shapes
# exist and they are not interchangeable: an adapter-backed chain is handed
# arrays for one key (`arrays(project, key)`), and a cohort-backed chain is
# handed the cohort's location and reads its own inputs. a unit with NO declared
# context loader is listed but cannot be generated - `build` refuses it by name
# rather than emitting a cell that imports a module that does not exist.
#
# unit -> Unit(module, step list, context, what the unit answers)
class Unit(NamedTuple):
    module: str
    steps: str
    # ("arrays", adapter module) | ("cohort", None) | None
    context: tuple[str, str | None] | None
    asks: str


UNITS = {
    "eda": Unit("omicstra.protocols.eda", "EDA_STEPS", ("cohort", None),
                "is this cohort admissible"),
    "inventory": Unit("omicstra.protocols.inventory", "INVENTORY_STEPS",
                      ("cohort", None), "what is this data"),
    "join": Unit("omicstra.protocols.join", "JOIN_STEPS",
                 ("arrays", "omicstra.adapters.join"),
                 "one row per unit of analysis, with every unit accounted for"),
    "evaluate": Unit("omicstra.protocols.evaluate", "EVALUATE_STEPS",
                     ("arrays", "omicstra.adapters.evaluation"),
                     "what each hypothesis may claim"),
    # the pathway assay is two chains, not one, and they FORK AT AGGREGATION -
    # one collapses a niche's spots into a single profile before doing anything,
    # the other never collapses them because MAD/mean needs them as several
    # observations of one niche. so they are two units: a notebook holding both
    # would imply a single method where there are two.
    "assay_pathway_genesets": Unit("omicstra.protocols.pathway",
                                   "VARIABLE_GENE_STEPS",
                                   ("arrays", "omicstra.adapters.pathway"),
                                   "which genes vary within a niche"),
    "assay_pathway_aucell": Unit("omicstra.protocols.pathway", "AUCELL_STEPS",
                                 ("arrays", "omicstra.adapters.pathway"),
                                 "how enriched a niche is, over the whole universe"),
    "assay_niche_membership": Unit("omicstra.protocols.pathway",
                                   "NICHE_MEMBERSHIP_STEPS",
                                   ("arrays", "omicstra.adapters.pathway"),
                                   "which spots form a niche"),
}

# the two assays with no unit here are the ENCODERS. they are not step chains -
# a shard loads a 631M-parameter model and writes an array - so a notebook for
# them re-runs the encoder, at 206 s and 11.3 GB per section on a laptop. that
# is the `--from-raw` case and it is deliberately not the default.


def _cell_id(kind: str, n: int) -> str:
    """nbformat 4.5 requires an id per cell and will hard-error without one.

    derived from the position rather than random, so regenerating a unit's
    notebook produces the same ids and a diff shows what CHANGED rather than
    every cell.
    """
    return f"{kind}-{n:03d}"


def _md(text: str, n: int = 0) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text,
            "id": _cell_id("md", n)}


def _code(src: str, n: int = 0) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": src,
            "execution_count": None, "outputs": [], "id": _cell_id("code", n)}


def _numbers(observed: dict, places: int = PLACES) -> dict[str, float]:
    """the scalar numbers a record carries, rounded to the reported precision.

    nested blocks are skipped: a per-pathway table is checked by the step that
    built it, and a notebook that walked into it would assert on a shape the
    record does not promise.
    """
    out = {}
    for k, v in (observed or {}).items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        out[k] = round(float(v), places)
    return out


def _producers(steps_list) -> dict[str, list[str]]:
    """step id -> the step ids it waits on.

    resolved here, at generation time, where the Step objects are. the first
    version worked this out inside the emitted cell and the comprehension was
    unreadable - a notebook a reader cannot follow is not a record.
    """
    produced_by = {p: s.id for s in steps_list for p in s.produces}
    return {s.id: sorted({produced_by[r] for r in s.requires if r in produced_by})
            for s in steps_list}


def _context(unit: str, b: dict) -> tuple[str, str]:
    """(the import line, the line that builds `ctx`).

    two shapes, because the chains genuinely differ: an adapter-backed chain is
    handed the arrays for one key, and a cohort-backed chain is handed the
    cohort's location and reads its own declared inputs.
    """
    kind, module = UNITS[unit].context
    if kind == "arrays":
        # the join reads one SAMPLE; the evaluation chain reads one fitted RUN
        key = repr(b.get("sample")) if unit == "join" else repr(b.get("run"))
        return f"from {module} import arrays\n", f"ctx = arrays(PROJECT, {key})\n"
    return "", ("ctx = {'project_dir': PROJECT, 'project_id': "
                f"{str(b.get('project_id'))!r}}}\n")


def header_cells(b: dict) -> list[dict]:
    unit = b.get("unit") or "evaluate"
    mod, steps_name, _, asks = UNITS[unit]
    imports, ctx_line = _context(unit, b)
    return [
        _md(f"# {unit} - {asks}\n\n"
            f"cohort `{b.get('project_id')}`. re-derives every number this unit "
            "recorded: one cell per step, each calling the step that ran with "
            "the parameters it recorded.\n\n"
            f"- produced by run `{b.get('run_id') or 'unrecorded'}`"
            + (f", sample `{b.get('sample')}`" if b.get("sample") else "")
            + (f", fitted run `{b.get('run')}`" if b.get("run") else "") + "\n"
            f"- {len(_steps(b))} steps\n"
            f"- numbers are compared at {PLACES} decimals, the precision the "
            "method supports: float32 kernels differ by ~1e-6 between backends, "
            "so asserting past that compares rounding rather than science\n"
            "- a failed check does not stop the notebook. its dependents record "
            "`not_run`, as the chain does"),
        _code(
            "import json\n"
            f"from {mod} import {steps_name}\n"
            + imports + "\n"
            f"PROJECT = {str(b.get('project_dir') or b.get('project_id'))!r}\n"
            f"PARAMS = json.loads(r\'\'\'{_json(b.get('params') or {})}\'\'\')\n"
            f"RECORDED = json.loads(r\'\'\'{_json(_recorded(b))}\'\'\')\n"
            f"RESULT = json.loads(r\'\'\'{_json(_results(b))}\'\'\')\n"
            f"WAITS_ON = json.loads(r\'\'\'{_json(b.get('waits_on') or {})}\'\'\')\n\n"
            f"STEP = {{s.id: s for s in {steps_name}}}\n"
            "status = {}\n"
            + ctx_line
            # the chain threads every record forward under `_records`, and a step
            # may READ it: `hypothesis_verdict` judges from what the steps above
            # recorded rather than recomputing. without this the verdict saw an
            # empty dict and returned not_run for every hypothesis - a wrong
            # answer the cell then reported as `pass`, because the verdict lives
            # in nested blocks that the scalar comparison skips.
            + "ctx['_records'] = {}\n"
            + "print(f'{len(ctx)} context keys, {len(PARAMS)} declared steps')"),
    ]


def step_cells(b: dict) -> list[dict]:
    cells: list[dict] = []
    for r in _steps(b):
        sid = r.get("step_id")
        cells.append(_md(
            f"## {sid}\n\n"
            + (f"**method** {r.get('method')}\n\n" if r.get("method") else "")
            + (f"**criterion** {r.get('criterion')}\n\n" if r.get("criterion") else "")
            + f"recorded `{r.get('status')}` - {r.get('result') or ''}"))
        cells.append(_code(
            f"sid = {sid!r}\n"
            "waiting = [d for d in WAITS_ON.get(sid, []) if status.get(d) != 'pass']\n"
            "if waiting:\n"
            "    status[sid] = 'not_run'\n"
            "    ctx['_records'][sid] = {'step_id': sid, 'status': 'not_run'}\n"
            "    print('not_run - waiting on', waiting)\n"
            "elif sid not in PARAMS:\n"
            "    status[sid] = 'not_run'\n"
            "    ctx['_records'][sid] = {'step_id': sid, 'status': 'not_run'}\n"
            "    print('not_run - this cohort declared no params for the step')\n"
            "else:\n"
            "    out = STEP[sid].fn({**ctx, 'params': PARAMS[sid]})\n"
            "    rec, upd = out if isinstance(out, tuple) else (out, {})\n"
            "    ctx.update(upd)\n"
            "    ctx['_records'][sid] = rec.model_dump()\n"
            f"    got = {{k: round(float(v), {PLACES}) for k, v in (rec.observed or {{}}).items()\n"
            "           if not isinstance(v, bool) and isinstance(v, (int, float))}\n"
            "    want = RECORDED.get(sid, {})\n"
            "    off = {k: (got[k], want[k]) for k in want if k in got and got[k] != want[k]}\n"
            # the one-line result too. a step whose answer lives in nested blocks
            # - the verdict per hypothesis, a per-pathway table - has no scalar
            # to compare, and its `result` is the summary the trace prints, so a
            # difference there is a difference a reader would see.
            "    said = RESULT.get(sid)\n"
            "    if said is not None and said != (rec.result or ''):\n"
            "        off['result'] = (rec.result, said)\n"
            "    status[sid] = 'fail' if off else rec.status\n"
            "    print(rec.status, '|', rec.result)\n"
            f"    if off:\n        print('DIFFERS at {PLACES} dp:', off)"))
    return cells


def footer_cells(b: dict) -> list[dict]:
    return [
        _md("## what this rerun showed"),
        _code(
            # "no fail" is not "reproduced". the first version printed
            # `reproduced` over six not_run steps, which is the one wrong answer
            # this cell exists to rule out - a notebook that re-derived nothing
            # and said the run reproduced.
            "for sid in [s.id for s in STEP.values()]:\n"
            "    if sid in status:\n"
            "        print(f'{status[sid]:16s} {sid}')\n"
            "bad = [k for k, v in status.items() if v == 'fail']\n"
            "ran = [k for k, v in status.items() if v in ('pass', 'not_applicable')]\n"
            "print()\n"
            "if bad:\n"
            "    print(f'DIFFERS: {bad}')\n"
            "elif not ran:\n"
            "    print('NOTHING RE-DERIVED - no step ran, so this is not a "
            "reproduction. see the status beside each step for why')\n"
            "else:\n"
            f"    print(f'reproduced {{len(ran)}} of {{len(STEP)}} step(s)')"),
        _md("a clean rerun is the reproduction proof: every number this run "
            "recorded came back from the same step, on the same declarations, "
            "at the precision the method supports.\n\n"
            "a `fail` is a real difference at that precision - not a rounding "
            "artifact - and its dependents read `not_run` because they were "
            "never given a valid input. a notebook where NOTHING ran is not a "
            "clean rerun: it says so rather than reporting no failures."),
    ]


def build(b: dict) -> dict:
    """the notebook, as an nbformat v4 dict. no cohort in the generator."""
    import importlib

    unit = b.get("unit") or "evaluate"
    if unit not in UNITS:
        raise ValueError(f"unknown unit {unit!r}. known: {sorted(UNITS)}")
    if UNITS[unit].context is None:
        # declared, not guessed. the alternative was a setup cell importing an
        # adapter module that does not exist, which looks like a result until a
        # reader runs it.
        raise ValueError(
            f"the unit {unit!r} declares no context loader, so a notebook for it "
            f"cannot build the dict {UNITS[unit].steps} reads. generable units: "
            f"{sorted(u for u, v in UNITS.items() if v.context)}")
    mod, steps_name, _, _ = UNITS[unit]
    b = {**b, "waits_on": _producers(getattr(importlib.import_module(mod),
                                             steps_name))}
    return {
        "cells": _numbered(header_cells(b) + step_cells(b) + footer_cells(b)),
        "metadata": {"kernelspec": {"display_name": "Python 3",
                                    "language": "python", "name": "python3"},
                     "language_info": {"name": "python"},
                     "omicstra": {"unit": unit,
                                  "produced_by_run": b.get("run_id"),
                                  "project_id": b.get("project_id"),
                                  "places": PLACES}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def _numbered(cells: list[dict]) -> list[dict]:
    """stable ids by position, so a regenerated notebook diffs cleanly."""
    return [{**c, "id": _cell_id("md" if c["cell_type"] == "markdown" else "code", i)}
            for i, c in enumerate(cells)]


def _steps(b: dict) -> list[dict]:
    from omicstra.trace import _steps as s

    return s(b.get("records"))


def _recorded(b: dict) -> dict[str, dict]:
    return {r["step_id"]: _numbers(r.get("observed")) for r in _steps(b)
            if r.get("step_id")}


def _results(b: dict) -> dict[str, str]:
    """the one-line result each step recorded.

    a step whose answer lives in NESTED blocks - the verdict per hypothesis, a
    per-pathway table - has no scalar for `_numbers` to compare, so a rerun
    could return a different verdict and the cell would still report `pass`.
    `result` is the summary the trace prints, which makes a difference there a
    difference a reader would see.
    """
    return {r["step_id"]: (r.get("result") or "") for r in _steps(b)
            if r.get("step_id")}


def _json(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=1, default=str)


def from_ledger(unit: str, project_id: str | None = None, *,
                run: str | None = None, sample: str | None = None,
                run_id: str | None = None) -> dict:
    """the bundle for one unit, read out of the cohort's own ledger.

    a `Record` carries no run id - the run identity is the graph's `thread_id`,
    which lives in the checkpointer and not in the record - so a unit's current
    state is the LAST record each of its steps wrote. that is the same statement
    the file name makes: one notebook per unit, regenerated to the latest state.
    a caller that knows the run names it; what is NOT available is selecting a
    unit's records by run, and the header says `unrecorded` rather than
    implying a run that was never written down.

    the chain is the filter. a cohort's ledger holds every step of every gate,
    so selecting by `step_id in UNITS[unit].steps` is what makes a unit's
    notebook that unit's and not the whole cohort's.
    """
    import importlib

    from omicstra.contracts.routing import ledger
    from omicstra.settings import settings

    if unit not in UNITS:
        raise ValueError(f"unknown unit {unit!r}. known: {sorted(UNITS)}")
    u = UNITS[unit]
    ids = [st.id for st in getattr(importlib.import_module(u.module), u.steps)]
    latest: dict[str, dict] = {}
    for r in ledger(project_id).all():
        if r.step_id in ids:
            latest[r.step_id] = r.model_dump(mode="json")
    return {"unit": unit, "project_id": project_id,
            "project_dir": str(settings.project_root(project_id)),
            "run": run, "sample": sample, "run_id": run_id,
            "params": {sid: (rec.get("params") or {})
                       for sid, rec in latest.items()},
            # in the chain's order, not the ledger's. a step that was re-run
            # appears once, where the protocol puts it.
            "records": [latest[sid] for sid in ids if sid in latest]}


def from_records(unit: str, records: list[dict], project_id: str | None = None,
                 *, run: str | None = None, sample: str | None = None,
                 run_id: str | None = None) -> dict:
    """the bundle for one unit, built from in-memory records.

    same shape as `from_ledger`, but reads from a list of record dicts (as
    carried in the graph state) rather than from the file-backed ledger.
    the chain command accumulates records across all stages; this filters
    them to the steps that belong to one unit.
    """
    import importlib

    from omicstra.settings import settings

    if unit not in UNITS:
        raise ValueError(f"unknown unit {unit!r}. known: {sorted(UNITS)}")
    u = UNITS[unit]
    ids = [st.id for st in getattr(importlib.import_module(u.module), u.steps)]
    latest: dict[str, dict] = {}
    for r in records:
        sid = r.get("step_id") if isinstance(r, dict) else getattr(r, "step_id", None)
        if sid in ids:
            d = r if isinstance(r, dict) else r.model_dump(mode="json")
            latest[sid] = d
    return {"unit": unit, "project_id": project_id,
            "project_dir": str(settings.project_root(project_id)),
            "run": run, "sample": sample, "run_id": run_id,
            "params": {sid: (rec.get("params") or {})
                       for sid, rec in latest.items()},
            "records": [latest[sid] for sid in ids if sid in latest]}


def results_dir(project_id: str | None = None):
    """where a cohort's generated results go: `work_root/results`.

    NOT the repo's `notebooks/`. that directory is a cohort's hand-written
    decision record, committed by a person; a generated notebook is a RESULT,
    and results live under the project root `omicstra init` creates, beside the
    ledger at `work_root/records` and the figures at `work_root/figures`.

    `work_root` is the one place the server may write for a cohort, and
    `dispatch.assert_work_root_writable` refuses it if a cohort has declared it
    read-only - so a generated notebook cannot land inside the artifacts the
    ports are diffed against.

    KNOWN LIMIT. a cohort that declares no `work_dir` resolves its work root to
    `runs_dir/{project_id}` under the PACKAGE, not under its own project root -
    so a cohort scaffolded at /tmp/mine writes its results beside site-packages
    on an installed server. `settings.work_root` documents this and offers
    `work_dir` as the fix, which makes every new cohort responsible for knowing
    about it.

    changing that default is NOT free: several tests isolate themselves by
    redirecting `runs_dir`, and a declared `work_dir` ignores the redirect - so
    a test touching the seed cohort would write into the ledger a report reads,
    which is a thing that has already happened once. the fix belongs with that
    isolation mechanism, and it is item 41 in docs/open_items.md rather than a
    change made in passing here.
    """
    from omicstra.dispatch import assert_work_root_writable

    return assert_work_root_writable(project_id) / "results"


def filenames(unit: str) -> tuple[str, str]:
    """the pair's names, keyed on the unit so re-running replaces it.

    a run-keyed name would leave fifty files nobody opens. this is the cohort's
    current record of that unit, and the run that produced it is in the header.
    """
    return f"{unit}.ipynb", f"{unit}.trace.md"


def write(b: dict, project_id: str | None = None) -> tuple:
    """the notebook and its trace, written side by side. returns both paths."""
    import json as _j

    from omicstra.trace import render

    unit = b.get("unit") or "evaluate"
    nb_name, tr_name = filenames(unit)
    d = results_dir(project_id)
    d.mkdir(parents=True, exist_ok=True)
    nb, tr = d / nb_name, d / tr_name
    nb.write_text(_j.dumps(build(b), indent=1) + "\n")
    tr.write_text(render({**b, "notebook": nb_name},
                         title=f"{unit} - {UNITS[unit][3]}"))
    return nb, tr


__all__ = ["PLACES", "UNITS", "Unit", "build", "filenames", "from_ledger",
           "from_records", "results_dir", "write"]
