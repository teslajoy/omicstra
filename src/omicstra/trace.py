"""one RUN's trace: records in, markdown out, no cohort in the renderer.

`report.py` answers "what has this cohort decided" from a pack and a ledger.
this answers a different question - "how did THIS run reach its answer" - from
the records one run emitted, and it reuses that file's primitives rather than
growing a second renderer.

why the LEDGER and not LangSmith
--------------------------------
the committed decision ledger is always on and always local. external tracing
is opt-in via env and `tracing.assert_may_trace` REFUSES it on a cohort the
contract says may not leave the host, because node inputs are cohort content -
so a deliverable built on it would be unavailable for exactly the cohorts that
matter most. and a span carries timings and a tree, while a `DiagnosticRecord`
carries the CRITERION the number was judged against, which is what a reader
re-deriving the science needs.

so the ledger is the source and LangSmith is a view: its URL is a line in the
trace when tracing is on and permitted, never a dependency.

the four statuses are the chain's, not this file's
--------------------------------------------------
`protocols.build_protocol` makes an upstream failure `not_run` rather than an
error, and runs the chain to the end so the record is complete. a trace that
collapsed that to pass/fail would throw away the distinction the vocabulary
exists for: `not_applicable` is a true answer, `unguarded` is not a pass, and
`not_run` names what it was waiting on.
"""
from __future__ import annotations

from typing import Any

from omicstra.report import _absent, _field, _h, _kv, _table

STATUS_MEANING = {
    "pass": "ran, and met its criterion",
    "fail": "ran, and did not",
    "not_applicable": "does not apply to this cohort - a true answer, not a gap",
    "not_run": "never reached, or a prerequisite did not produce its artifact",
    "error": "the step broke. this is not a verdict about the data",
}


def _steps(records: Any) -> list[dict]:
    """the records as an ordered list, however they were handed over.

    a chain returns {step_id: record}; a ledger is a list of them; a graph's
    state carries them under `records`. all three arrive here.
    """
    if isinstance(records, dict):
        if "records" in records and not records.get("step_id"):
            return _steps(records["records"])
        return [{**v, "step_id": v.get("step_id") or k}
                for k, v in records.items() if isinstance(v, dict)]
    return [r for r in (records or []) if isinstance(r, dict)]


def section_run(b: dict) -> list[str]:
    """what this run was, and under which package and protocol."""
    steps = _steps(b.get("records"))
    code = next((r.get("code_version") for r in steps if r.get("code_version")), None)
    proto = next((r.get("protocol_version") for r in steps if r.get("protocol_version")),
                 None)
    secs = sum(r.get("duration_s") or 0 for r in steps)
    untimed = [r["step_id"] for r in steps if r.get("duration_s") is None]
    out = [_h(2, "the run"), ""]
    out += _kv([
        ("run_id", b.get("run_id")),
        ("cohort", b.get("project_id")),
        ("arm", b.get("arm")),
        ("steps", len(steps) or None),
        ("wall", f"{secs:.1f}s" if secs else None),
        ("code_version", code),
        ("protocol_version", proto),
        ("external trace", b.get("langsmith_url")),
    ])
    if untimed:
        # an untimed step took an UNKNOWN time, which is not the same statement
        # as took no time. naming them beats averaging a zero into a total.
        out += ["", (f"*{len(untimed)} step(s) carry no duration: "
                     f"{', '.join(sorted(untimed))}. an untimed step took an "
                     "unknown time, not no time.*")]
    if not b.get("langsmith_url"):
        out += ["", ("*no external trace. the ledger below is the record, and "
                     "it is always local - external tracing is opt-in and is "
                     "refused on a cohort the contract says may not leave the "
                     "host.*")]
    return out


def section_steps(b: dict) -> list[str]:
    """every step, in the order it ran, with the criterion it was judged on."""
    steps = _steps(b.get("records"))
    if not steps:
        return [_h(2, "the steps"), "", *_absent("steps", "this run emitted no records")]

    out = [_h(2, "the steps"), ""]
    out += _table(["step", "status", "method", "criterion", "result", "s"],
                  [[r.get("step_id"), r.get("status"),
                    _field(r, "method")[:60], _field(r, "criterion")[:60],
                    _field(r, "result")[:60],
                    "" if r.get("duration_s") is None else f"{r['duration_s']:.1f}"]
                   for r in steps])

    seen = {r.get("status") for r in steps}
    out += ["", _h(3, "what each status means"), ""]
    out += _table(["status", "meaning"],
                  [[s, STATUS_MEANING.get(s, "")] for s in
                   sorted(seen, key=lambda x: list(STATUS_MEANING).index(x)
                          if x in STATUS_MEANING else 99)])

    # the decision is the sentence a reader takes away, so it is given whole
    # rather than truncated into the table above
    decided = [r for r in steps if _field(r, "decision")]
    if decided:
        out += ["", _h(3, "what each step decided"), ""]
        for r in decided:
            out += [f"**{r.get('step_id')}** - {_field(r, 'decision')}", ""]
    return out


def section_funnel(b: dict) -> list[str]:
    """the funnel, preferring the one the chain ASSEMBLED over a derived one.

    a step that builds a funnel records it under `observed.stages` - built by
    `protocols.join.Funnel`, which refuses a drop with no reason. that is the
    authoritative article, and the trace reads it rather than re-deriving one
    from every record's n_in/n_out: those count what each step was GIVEN, and
    different steps are given different things. deriving across them produced
    "1 -> 3 (300.0% retained)".
    """
    steps = _steps(b.get("records"))

    for r in steps:
        stages = (r.get("observed") or {}).get("stages")
        if not stages:
            continue
        out = [_h(2, "the funnel"), "",
               (f"*assembled by `{r.get('step_id')}`, which refuses a drop "
                "with no reason.*"), ""]
        out += _table(["stage", "n_in", "n_out", "dropped", "reason"],
                      [[st.get("stage"), st.get("n_in"), st.get("n_out"),
                        st.get("dropped"), str(st.get("reason") or "")[:52]]
                       for st in stages])
        first, last = stages[0], stages[-1]
        if first.get("n_in"):
            kept = (last.get("n_out") or 0) / first["n_in"]
            out += ["", (f"{first['n_in']:,} -> {last.get('n_out'):,} "
                         f"({kept:.1%} retained). every dropped unit is "
                         "attributable to a named stage.")]
        return out

    rows = []
    for r in steps:
        obs = r.get("observed") or {}
        if "n_in" in obs and "n_out" in obs:
            rows.append([r.get("step_id"), obs["n_in"], obs["n_out"],
                         (obs["n_in"] or 0) - (obs["n_out"] or 0),
                         str(obs.get("reason") or "")[:52]])
    if not rows:
        return [_h(2, "the funnel"), "",
                *_absent("funnel", "no step in this run recorded a unit count")]
    out = [_h(2, "the funnel"), ""]
    out += _table(["step", "n_in", "n_out", "dropped", "reason"], rows)

    out += ["", ("no step assembled a funnel, so these are PER STEP counts and "
                 "do not chain - each counts what it was given, and the steps "
                 "were given different things. a drop is read against its own "
                 "step's n_in, and there is no single retained fraction.")]
    return out


def section_guards(b: dict) -> list[str]:
    """which guards ran, and what a failed one did to the claim."""
    rows = []
    for r in _steps(b.get("records")):
        for g in (r.get("observed") or {}).get("guards", []) or []:
            if isinstance(g, dict):
                rows.append([r.get("step_id"), g.get("guard"), g.get("scope"),
                             g.get("status"), str(g.get("note") or "")[:46]])
    if not rows:
        return []
    out = [_h(2, "the guards"), ""]
    out += _table(["step", "guard", "scope", "status", "note"], rows)
    not_run = [r for r in rows if r[3] == "not_run"]
    if not_run:
        out += ["", (f"*{len(not_run)} guard(s) could not run. a guard that "
                     "did not run is not a guard that passed - the verdict "
                     "below says `unguarded` where nothing checked a number.*")]
    return out


def section_verdict(b: dict) -> list[str]:
    """the four-state verdict per hypothesis, with why."""
    verdicts = {}
    for r in _steps(b.get("records")):
        v = (r.get("observed") or {}).get("verdicts")
        if isinstance(v, dict):
            verdicts.update(v)
    if not verdicts:
        return []
    out = [_h(2, "the verdict"), ""]
    out += _table(["hypothesis", "verdict", "why"],
                  [[h, d.get("verdict"), str(d.get("why") or "")[:88]]
                   for h, d in sorted(verdicts.items())])
    return out


def section_reproduce(b: dict) -> list[str]:
    """how to re-derive this, and what a clean rerun proves."""
    nb = b.get("notebook")
    out = [_h(2, "reproducing this"), ""]
    if nb:
        out += [(f"`{nb}` re-derives every number above from the arrays up. "
                 "each cell calls the same function in `measures/` that the "
                 "step called, with the parameters this run recorded, and "
                 "asserts the recorded value."), "",
                ("a clean rerun IS the reproduction proof. a cell whose "
                 "assertion fails marks itself failed and its dependents "
                 "declare `not_run` with the reason - the notebook finishes, "
                 "because a record that stops halfway is not a record."), ""]
    out += [("every parameter above is the cohort's declaration, not a "
             "default. `measures/` holds no cohort constant, and a step that "
             "found no declaration refused rather than guessing.")]
    return out


SECTIONS = (section_run, section_steps, section_funnel, section_guards,
            section_verdict, section_reproduce)


def render(bundle: dict, title: str | None = None) -> str:
    """the whole trace. one heading, then every section that has an input."""
    rid = bundle.get("run_id") or "run"
    out = [_h(1, title or f"trace - {rid}"), ""]
    for fn in SECTIONS:
        part = fn(bundle)
        if part:
            out += part + [""]
    return "\n".join(out).rstrip() + "\n"


__all__ = ["SECTIONS", "STATUS_MEANING", "render"]
