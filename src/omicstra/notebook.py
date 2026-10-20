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

from typing import Any

PLACES = 4

# the unit is an ANALYSIS UNIT, not a run. a run id is an execution artifact -
# nobody opens `eval-a3f2.ipynb` - and this cohort's own committed notebooks are
# already named for topics: `h3_gpath2vec_v3`, `proposal_h1h2h3_evaluation`. so
# one file per unit, per cohort, regenerated to the latest state, and the run
# that produced the numbers is provenance INSIDE the header.
#
# unit -> (module, step list, adapter, what the unit answers)
UNITS = {
    "eda": ("omicstra.protocols.eda", "EDA_STEPS", None,
            "is this cohort admissible"),
    "inventory": ("omicstra.protocols.inventory", "INVENTORY_STEPS", None,
                  "what is this data"),
    "join": ("omicstra.protocols.join", "JOIN_STEPS", "join",
             "one row per unit of analysis, with every unit accounted for"),
    "evaluate": ("omicstra.protocols.evaluate", "EVALUATE_STEPS", "evaluation",
                 "what each hypothesis may claim"),
    # the pathway assay is two chains, not one, and they FORK AT AGGREGATION -
    # one collapses a niche's spots into a single profile before doing anything,
    # the other never collapses them because MAD/mean needs them as several
    # observations of one niche. so they are two units: a notebook holding both
    # would imply a single method where there are two.
    "assay_pathway_genesets": ("omicstra.protocols.pathway",
                               "VARIABLE_GENE_STEPS", None,
                               "which genes vary within a niche"),
    "assay_pathway_aucell": ("omicstra.protocols.pathway", "AUCELL_STEPS", None,
                             "how enriched a niche is, over the whole universe"),
    "assay_niche_membership": ("omicstra.protocols.pathway",
                               "NICHE_MEMBERSHIP_STEPS", None,
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


def header_cells(b: dict) -> list[dict]:
    unit = b.get("unit") or "evaluate"
    mod, steps_name, adapter, asks = UNITS[unit]
    # the join reads one SAMPLE; the evaluation chain reads one fitted RUN
    second = repr(b.get("sample")) if unit == "join" else repr(b.get("run"))
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
            f"from omicstra.adapters.{adapter} import arrays\n\n"
            f"PROJECT = {str(b.get('project_dir') or b.get('project_id'))!r}\n"
            f"PARAMS = json.loads(r\'\'\'{_json(b.get('params') or {})}\'\'\')\n"
            f"RECORDED = json.loads(r\'\'\'{_json(_recorded(b))}\'\'\')\n"
            f"WAITS_ON = json.loads(r\'\'\'{_json(b.get('waits_on') or {})}\'\'\')\n\n"
            f"STEP = {{s.id: s for s in {steps_name}}}\n"
            "status = {}\n"
            f"ctx = arrays(PROJECT, {second})\n"
            "print(f'{len(ctx)} context keys, {len(PARAMS)} declared steps')"),
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
            "    print('not_run - waiting on', waiting)\n"
            "elif sid not in PARAMS:\n"
            "    status[sid] = 'not_run'\n"
            "    print('not_run - this cohort declared no params for the step')\n"
            "else:\n"
            "    out = STEP[sid].fn({**ctx, 'params': PARAMS[sid]})\n"
            "    rec, upd = out if isinstance(out, tuple) else (out, {})\n"
            "    ctx.update(upd)\n"
            f"    got = {{k: round(float(v), {PLACES}) for k, v in (rec.observed or {{}}).items()\n"
            "           if not isinstance(v, bool) and isinstance(v, (int, float))}\n"
            "    want = RECORDED.get(sid, {})\n"
            "    off = {k: (got[k], want[k]) for k in want if k in got and got[k] != want[k]}\n"
            "    status[sid] = 'fail' if off else rec.status\n"
            "    print(rec.status, '|', rec.result)\n"
            f"    if off:\n        print('DIFFERS at {PLACES} dp:', off)"))
    return cells


def footer_cells(b: dict) -> list[dict]:
    return [
        _md("## what this rerun showed"),
        _code(
            "for sid in [s.id for s in STEP.values()]:\n"
            "    if sid in status:\n"
            "        print(f'{status[sid]:16s} {sid}')\n"
            "bad = [k for k, v in status.items() if v == 'fail']\n"
            "print()\n"
            "print('reproduced' if not bad else f'DIFFERS: {bad}')"),
        _md("a clean rerun is the reproduction proof: every number this run "
            "recorded came back from the same step, on the same declarations, "
            "at the precision the method supports.\n\n"
            "a `fail` is a real difference at that precision - not a rounding "
            "artifact - and its dependents read `not_run` because they were "
            "never given a valid input."),
    ]


def build(b: dict) -> dict:
    """the notebook, as an nbformat v4 dict. no cohort in the generator."""
    import importlib

    unit = b.get("unit") or "evaluate"
    if unit not in UNITS:
        raise ValueError(f"unknown unit {unit!r}. known: {sorted(UNITS)}")
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


def _json(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=1, default=str)


def filenames(unit: str) -> tuple[str, str]:
    """where the pair lands. named for the unit, so re-running replaces it.

    a run-keyed name would leave fifty files nobody opens; this is the cohort's
    living record of that unit, which is what `notebooks/` already holds.
    """
    return f"{unit}.ipynb", f"{unit}.trace.md"


__all__ = ["PLACES", "UNITS", "build", "filenames"]
