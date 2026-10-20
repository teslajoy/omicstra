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

CHAINS = {
    "evaluate": ("omicstra.protocols.evaluate", "EVALUATE_STEPS"),
    "join": ("omicstra.protocols.join", "JOIN_STEPS"),
}


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def _code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": src,
            "execution_count": None, "outputs": []}


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
    chain = b.get("chain") or "evaluate"
    mod, steps_name = CHAINS.get(chain, CHAINS["evaluate"])
    adapter = "join" if chain == "join" else "evaluation"
    # the join reads one SAMPLE; the evaluation chain reads one fitted RUN
    second = repr(b.get("sample")) if chain == "join" else repr(b.get("run"))
    return [
        _md(f"# {b.get('run_id') or 'run'}\n\n"
            "re-derives every number this run recorded. one cell per step, each "
            "calling the step that ran with the parameters it recorded.\n\n"
            f"- cohort `{b.get('project_id')}` · chain `{chain}` · "
            f"{len(_steps(b))} steps\n"
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

    chain = b.get("chain") or "evaluate"
    mod, steps_name = CHAINS.get(chain, CHAINS["evaluate"])
    b = {**b, "waits_on": _producers(getattr(importlib.import_module(mod),
                                             steps_name))}
    return {
        "cells": header_cells(b) + step_cells(b) + footer_cells(b),
        "metadata": {"kernelspec": {"display_name": "Python 3",
                                    "language": "python", "name": "python3"},
                     "language_info": {"name": "python"},
                     "omicstra": {"run_id": b.get("run_id"),
                                  "chain": b.get("chain"),
                                  "places": PLACES}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def _steps(b: dict) -> list[dict]:
    from omicstra.trace import _steps as s

    return s(b.get("records"))


def _recorded(b: dict) -> dict[str, dict]:
    return {r["step_id"]: _numbers(r.get("observed")) for r in _steps(b)
            if r.get("step_id")}


def _json(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=1, default=str)


__all__ = ["CHAINS", "PLACES", "build"]
