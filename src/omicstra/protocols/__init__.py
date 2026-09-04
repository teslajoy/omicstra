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


def _link(step: Step, producer_of: dict[str, str] | None = None) -> Callable[[Ctx], Ctx]:
    """one step as a ctx -> ctx function, so links compose with `|`.

    ctx threads through and each link appends its record. returning ctx rather
    than a record is what makes composition possible - a chain is only a chain
    if the output of one link is the input of the next.
    """
    def run(ctx: Ctx) -> Ctx:
        recs = dict(ctx.get("_records", {}))
        updates: dict = {}
        pof = producer_of or {}

        # an upstream failure makes this step NOT_RUN, never error. a KeyError
        # here would read as a bug in this step when the real cause is that a
        # prerequisite failed. the chain still runs to the end so the record is
        # complete - only the reason changes.
        failed_up = sorted({pof[p] for p in step.requires
                            if p in pof and recs.get(pof[p], {}).get("status") in ("fail", "not_run", "error")})
        if failed_up:
            rec = DiagnosticRecord(
                step_id=step.id, status="not_run",
                result=f"upstream {failed_up} failed",
                decision="not attempted - a prerequisite did not produce its artifact")
            recs[step.id] = rec.model_dump()
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
                out = step.fn({**ctx, "params": ctx["params"][step.id]})
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

        recs[step.id] = rec.model_dump()
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
