"""one run's trace: the records in, markdown out.

tier 1. the renderer knows no cohort, so every case here is synthetic.

the thing most worth testing is what the trace REFUSES to say: a retention
figure where the counts do not chain, a status collapsed to pass/fail, a guard
that did not run read as one that passed.
"""
from __future__ import annotations

from omicstra.trace import STATUS_MEANING, render, section_funnel, section_run


def _rec(step, status="pass", **kw):
    return {"step_id": step, "kind": "diagnostic", "status": status, **kw}


# ------------------------------------------------------------ the shapes ----
def test_records_arrive_as_a_chain_dict_a_list_or_a_state():
    """build_protocol returns {step_id: record}, a ledger is a list, a graph's
    state carries them under `records`. all three reach the renderer."""
    r = _rec("a", result="x")
    for form in ({"a": r}, [r], {"records": [r]}, {"records": {"a": r}}):
        out = render({"run_id": "t", "records": form})
        assert "| a | pass |" in out, form


def test_a_run_with_no_records_says_so_rather_than_rendering_an_empty_table():
    out = render({"run_id": "empty", "records": []})
    assert "emitted no records" in out


# --------------------------------------------------- the four statuses ------
def test_every_status_the_chain_can_emit_has_a_stated_meaning():
    """the vocabulary is the chain's. a trace that collapsed it to pass/fail
    would throw away the distinction it exists for."""
    assert set(STATUS_MEANING) == {"pass", "fail", "not_applicable", "not_run",
                                   "error"}
    assert "not a pass" not in STATUS_MEANING["pass"]
    assert "true answer" in STATUS_MEANING["not_applicable"]
    assert "not a verdict about the data" in STATUS_MEANING["error"]


def test_only_the_statuses_present_are_explained():
    out = render({"run_id": "t", "records": [_rec("a"), _rec("b", "not_run")]})
    assert "ran, and met its criterion" in out
    assert "never reached" in out
    assert "does not apply to this cohort" not in out


# ------------------------------------------------- what it refuses to say ---
def test_no_retained_fraction_when_the_counts_do_not_chain():
    """each step counts what it was GIVEN, and different steps are given
    different things. deriving a percentage across them printed
    '1 -> 3 (300.0% retained)'."""
    recs = [_rec("a", observed={"n_in": 1, "n_out": 1}),
            _rec("b", observed={"n_in": 2, "n_out": 2}),
            _rec("c", observed={"n_in": 3, "n_out": 3})]
    out = "\n".join(section_funnel({"records": recs}))
    # the word appears in the sentence that says there is no such figure, so
    # the assertion is on the FIGURE: "x -> y (z% retained)"
    import re
    assert not re.search(r"\(\d+(\.\d+)?% retained\)", out)
    assert "do not chain" in out
    assert "no single retained fraction" in out


def test_the_assembled_funnel_is_preferred_over_a_derived_one():
    """a step that builds a funnel records it under observed.stages, built by
    `Funnel`, which refuses a drop with no reason. that is the article."""
    recs = [_rec("x", observed={"n_in": 9, "n_out": 9}),
            _rec("the_funnel", observed={"stages": [
                {"stage": "enumerate", "n_in": 100, "n_out": 100, "dropped": 0,
                 "reason": ""},
                {"stage": "coverage", "n_in": 100, "n_out": 40, "dropped": 60,
                 "reason": "no anchor vector"}]})]
    out = "\n".join(section_funnel({"records": recs}))
    assert "assembled by `the_funnel`" in out
    assert "100 -> 40 (40.0% retained)" in out
    assert "| coverage | 100 | 40 | 60 | no anchor vector |" in out


def test_a_guard_that_did_not_run_is_not_read_as_one_that_passed():
    recs = [_rec("s", observed={"guards": [
        {"guard": "G5_noise_floor", "scope": "v", "status": "not_run",
         "note": "too few permutations"},
        {"guard": "G1_label_granularity", "scope": "l", "status": "pass"}]})]
    out = render({"run_id": "t", "records": recs})
    assert "1 guard(s) could not run" in out
    assert "not a guard that passed" in out


def test_an_untimed_step_is_named_rather_than_counted_as_zero():
    recs = [_rec("a", duration_s=2.5), _rec("b")]
    out = "\n".join(section_run({"records": recs}))
    assert "1 step(s) carry no duration: b" in out
    assert "unknown time, not no time" in out


# ----------------------------------------------------------- the ledger -----
def test_the_absence_of_an_external_trace_is_stated_with_the_reason():
    """external tracing is opt-in and refused on a cohort that may not leave
    the host, so its absence is the normal case and not a gap."""
    out = render({"run_id": "t", "records": [_rec("a")]})
    assert "no external trace" in out
    assert "always local" in out
    assert "may not leave the host" in out


def test_an_external_trace_is_a_pointer_when_there_is_one():
    out = render({"run_id": "t", "records": [_rec("a")],
                  "langsmith_url": "https://smith.example/r/abc"})
    assert "https://smith.example/r/abc" in out
    assert "no external trace" not in out


# ------------------------------------------------------- what it carries ----
def test_the_criterion_travels_with_every_step():
    """a span carries timings; a DiagnosticRecord carries the criterion the
    number was judged against, which is what a reader re-deriving it needs."""
    out = render({"run_id": "t", "records": [
        _rec("a", method="KMeans ARI", criterion="G5 runs", result="0.23",
             decision="the score ships with its null")]})
    assert "KMeans ARI" in out and "G5 runs" in out
    assert "the score ships with its null" in out


def test_the_decision_is_given_whole_not_truncated_into_the_table():
    long = ("a verdict COMPUTED from the statistics this chain produced, not "
            "read from a pack, because the two answer different questions")
    out = render({"run_id": "t", "records": [_rec("a", decision=long)]})
    assert long in out


def test_the_code_and_protocol_versions_are_reported():
    out = render({"run_id": "t", "records": [
        _rec("a", code_version="deadbeef", protocol_version="2026-07-28")]})
    assert "deadbeef" in out and "2026-07-28" in out


def test_the_verdict_table_carries_the_four_state_vocabulary():
    out = render({"run_id": "t", "records": [_rec("v", observed={"verdicts": {
        "H1": {"verdict": "biological", "why": "2 guards ran and passed"},
        "H3": {"verdict": "not_run", "why": "no step produced a statistic"}}})]})
    assert "| H1 | biological |" in out
    assert "| H3 | not_run |" in out


def test_a_section_with_no_input_is_dropped_not_rendered_empty():
    out = render({"run_id": "t", "records": [_rec("a")]})
    assert "the guards" not in out
    assert "the verdict" not in out


# ------------------------------------------------- against the real chains --
def test_it_renders_the_evaluate_chain_without_quoting_a_fraction():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    import test_evaluate_chain as T

    out = render({"run_id": "e", "project_id": "synthetic",
                  "records": T._run(T._cohort())})
    assert "H1 | biological" in out or "H1 | diagnostic" in out
    assert "do not chain" in out, "the evaluate chain's counts are per step"
    assert "300.0%" not in out


def test_it_renders_the_join_chain_with_its_assembled_funnel():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    import test_join_chain as J

    out = render({"run_id": "j", "project_id": "synthetic",
                  "records": J._run(J._cohort(anchor_units=["u0", "u1", "u2"]))})
    assert "assembled by `join_funnel`" in out
    assert "5 -> 3 (60.0% retained)" in out
    assert "no vector in the anchor assay" in out


def test_the_renderer_knows_no_cohort():
    import inspect

    from omicstra import trace

    src = inspect.getsource(trace)
    for name in ("tnbc", "virchow", "novae", "gpath2vec", "mc_megacluster"):
        assert name not in src.lower(), f"{name} reached the renderer"
