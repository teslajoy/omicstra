"""level 2 - protocols. a fixed order, transcribed, not designed.

a protocol is a DAG whose valid topological order is the published method:

    raw counts -> MT filter -> normalize -> kNN weights -> Moran's I
                                                              |
                                        permutation null -> BH-FDR -> verdict

`requires`/`produces` is therefore an ASSERTION, never a search space. there is
nothing to discover, only to refuse: a reordered DAG is a different and wrong
method carrying a confident trace. no model composes this, and no user edits it.

what DOES vary per cohort, and each has a declared decider:

    applicability   resolved from measured data, deterministic     -> not_applicable
    parameters      declared with an authority per value           -> params
    scope           how much data, against a declared minimum      -> params

a protocol never holds state and never interrupts. if something must ask a
human, it belongs one level up, in a graph.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Callable, Literal

from langchain_core.runnables import Runnable, RunnableLambda

from omicstra.records import DiagnosticRecord

Authority = Literal["universal", "cohort_calibrated", "advisory"]
Ctx = dict[str, Any]


class ProtocolOrderError(ValueError):
    """raised at BUILD time, not run time.

    an invalid order is a wrong method, so it must fail before any data is
    touched - not after a step has already run and written a record.
    """


@dataclass(frozen=True)
class Step:
    id: str
    fn: Callable[[Ctx], DiagnosticRecord]
    produces: frozenset[str] = frozenset()
    requires: frozenset[str] = frozenset()
    applicable_when: Callable[[Ctx], bool] | None = None
    why_not_applicable: str = "does not apply to this cohort"
    authority: Authority = "universal"
    params_space: tuple[str, ...] = field(default=())


def _assert_order(steps: list[Step], name: str) -> None:
    """the order check, at BUILD time.

    `requires` must be satisfied by a step EARLIER in the list, because the list
    IS the method. a step whose prerequisite comes later is not a scheduling
    problem to solve - it is a protocol written down wrong.
    """
    seen: set[str] = set()
    for s in steps:
        if missing := s.requires - seen:
            raise ProtocolOrderError(
                f"{name}: step {s.id!r} requires {sorted(missing)} which no earlier "
                f"step produces. the order is the method - reorder the protocol, "
                f"do not reorder the run."
            )
        seen |= s.produces


def _stamp(dump: dict, t0: float) -> dict:
    """how long THIS STEP took, if it did not already say.

    `records.timed_node` stamps every record a graph node added with the NODE's
    duration, and that number is not additive: the evaluate node runs a
    six-step chain, so all six records read 145.7s and the three places that sum
    the field - `trace.section_run`, `report.py` and `runs_trace` - reported
    874.3s for a run that took 145.7. a step is where the time is actually
    spent, so this is where it is measured, and `timed_node`'s `or d` fallback
    then only fires for a record no chain produced.

    a step that measures itself keeps its own number: several measures already
    time their own work, and that is the tighter figure.
    """
    import time

    if dump.get("duration_s") is None:
        dump["duration_s"] = round(time.perf_counter() - t0, 3)
    return dump


def _supplied(dump: dict, supplied: object) -> dict:
    """what this step was actually GIVEN, recorded as its params.

    the record's `params` field was left empty by every step in every chain, so
    a ledger said which steps ran and not what they ran with - and a generated
    notebook, whose whole claim is "the same step, with the parameters that run
    recorded", could re-derive nothing: all six steps came back `not_run` with
    "labels is not declared".

    recorded HERE and not by the step, because this is where the filter happens:
    a step sees only the keys in its `params_space`, so only this frame knows
    what it saw. `params` is inside `record_id`, which is correct - a step run
    with different parameters is a different decision.
    """
    if isinstance(supplied, dict) and not dump.get("params"):
        dump["params"] = dict(supplied)
    return dump


def _link(step: Step, producer_of: dict[str, str] | None = None) -> Callable[[Ctx], Ctx]:
    """one step as a ctx -> ctx function, so links compose with `|`.

    ctx threads through and each link appends its record. returning ctx rather
    than a record is what makes composition possible - a chain is only a chain
    if the output of one link is the input of the next.
    """
    def run(ctx: Ctx) -> Ctx:
        import time

        t0 = time.perf_counter()
        recs = dict(ctx.get("_records", {}))
        updates: dict = {}
        supplied: object = None
        pof = producer_of or {}

        # an upstream failure makes this step NOT_RUN, never error. a KeyError
        # here would read as a bug in this step when the real cause is that a
        # prerequisite failed. the chain still runs to the end so the record is
        # complete - only the reason changes.
        # `not_applicable` belongs in this set. a step that did not apply
        # produced no artifact, so a step requiring that artifact has nothing to
        # read - and omitting the status here sent it into the `fn` anyway, where
        # a missing ctx key surfaced as an error attributed to the wrong step.
        # not applicable is a reason, and it propagates as one.
        failed_up = sorted({pof[p] for p in step.requires
                            if p in pof and recs.get(pof[p], {}).get("status")
                            in ("fail", "not_run", "error", "not_applicable")})
        if failed_up:
            rec = DiagnosticRecord(
                step_id=step.id, status="not_run",
                result=f"upstream {failed_up} failed",
                decision="not attempted - a prerequisite did not produce its artifact")
            recs[step.id] = _stamp(rec.model_dump(), t0)
            return {**ctx, "_records": recs}

        if step.applicable_when is not None and not step.applicable_when(ctx):
            rec = DiagnosticRecord(
                step_id=step.id, status="not_applicable",
                method="applicability predicate over measured data",
                result=step.why_not_applicable,
                decision="not judged - this check does not apply here")
        elif step.id not in ctx.get("params", {}):
            rec = DiagnosticRecord(
                step_id=step.id, status="not_run",
                result="cohort supplied no params for this step",
                decision="a step with no params is skipped, never guessed at")
        else:
            try:
                # a step sees only the params it DECLARES. params_space was
                # declared and recorded in the trace but never enforced, so
                # whatever a cohort's file happened to carry was splatted into
                # the measure - provenance fields arriving as keyword arguments
                # and surfacing as a TypeError that reads like a broken measure.
                # an undeclared space means no filter, which is the old behaviour.
                supplied = ctx["params"][step.id]
                if step.params_space and isinstance(supplied, dict):
                    supplied = {k: v for k, v in supplied.items() if k in step.params_space}
                out = step.fn({**ctx, "params": supplied})
                # a step may return a bare record, or (record, ctx_updates).
                # updates are what make `produces` DATA rather than a label -
                # without them a later step cannot read what an earlier one
                # found, and `requires` is decoration.
                rec, updates = out if isinstance(out, tuple) else (out, {})
            except Exception as e:
                rec = DiagnosticRecord(
                    step_id=step.id, status="error",
                    result=f"{type(e).__name__}: {e}",
                    decision="step failed; the gate cannot judge this check")

        recs[step.id] = _supplied(_stamp(rec.model_dump(), t0), supplied)
        return {**ctx, **updates, "_records": recs}
    return run


def build_protocol(steps: list[Step], name: str = "protocol") -> Runnable:
    """compose steps into ONE chain: A1 | A2 | ... | An.

    `|` is what makes this a langchain chain rather than a function that happens
    to use langchain types. each link becomes its own traced span named by step
    id, tagged with its authority and carrying its declared params_space in
    metadata - so a trace can be filtered to every cohort_calibrated decision
    without instrumenting anything.

    it is also the shape that ports to a workflow engine unchanged: one link,
    one process.
    """
    _assert_order(steps, name)

    producer_of = {p: s.id for s in steps for p in s.produces}
    links = [
        RunnableLambda(_link(s, producer_of)).with_config(
            run_name=s.id,
            tags=[s.authority],
            metadata={"produces": sorted(s.produces),
                      "requires": sorted(s.requires),
                      "params_space": list(s.params_space)},
        )
        for s in steps
    ]
    chain = reduce(operator.or_, links)
    # the ctx carried the records through; hand back just the records
    return (chain | RunnableLambda(lambda ctx: ctx.get("_records", {}))
            ).with_config(run_name=name)
