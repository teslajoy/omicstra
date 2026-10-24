"""one record store, and it reaches the report.

there were two. the routing path appended to a file-backed `RecordStore` at
`runs/{id}/records/decisions.jsonl`; every gate arm wrote plain dicts into the
langgraph checkpoint, readable only through `runs_record(run_id)` and only by a
caller holding the handle. so a cohort's ledger said "N routing decisions" while
the gates, the dispatches and the pins lived somewhere else - and `report.py`,
whose first line is "a pack and a ledger in", could reach neither.

the arms' dicts could not even be VALIDATED as Records: the actor vocabulary in
`records.py` was `deterministic | model | human` while the gate nodes wrote
`client` and `system`, `kind` had no `dispatch`, and a selection's `chosen` was a
string while a gate arm answers several gates in one call. three divergences
between a model and the only code writing to it, none of which anything caught,
because nothing read both.
"""
from __future__ import annotations

import json

import pytest

from omicstra.records import _ACTORS, RecordStore, coerce
from omicstra.report import gather, render


@pytest.fixture
def ledger_at(tmp_path, monkeypatch):
    """a ledger under tmp. the arms resolve it through settings, so this does too
    rather than reaching into the repo's own runs/ during a test."""
    from omicstra.settings import settings

    before = settings.runs_dir
    settings.runs_dir = tmp_path / "runs"
    try:
        yield tmp_path / "runs"
    finally:
        settings.runs_dir = before


# --- the model can express what the arms write -----------------------------
def test_the_actor_vocabulary_is_the_one_the_contract_declares():
    """a closed set declared in one file and enforced in another is two sets."""
    from omicstra.protocols.encode import contract

    declared = contract()["gate_schema"]["actor_vocabulary"]
    assert declared["closed_set"] is True
    named = {k for k, v in declared.items() if isinstance(v, str) and not k.startswith("why")}
    named -= {"closed_set"}
    assert named <= set(_ACTORS), f"the contract names an actor no record can carry: {named - set(_ACTORS)}"


@pytest.mark.parametrize("d", [
    {"step_id": "encode_dispatch", "kind": "dispatch", "status": "pass",
     "actor": "system", "executor": "temporal", "n_shards": 4, "out_dir": "/x"},
    {"step_id": "preflight_encode", "kind": "selection", "status": "pass",
     "actor": "client", "confirmed_by": None,
     "chosen": {"encoder_compatibility": "proceed_anyway_recorded_as_dissent",
                "capacity": "continue"}},
    {"step_id": "preflight_evaluate", "kind": "gate", "status": "not_run",
     "actor": "deterministic", "reason": "no fillable task family in scope"},
])
def test_every_dict_an_arm_writes_validates_as_a_record(d):
    r = coerce(d)
    assert r.kind == d["kind"] and r.actor == d["actor"]
    assert r.record_id, "an id is what makes appending twice idempotent"
    # nothing is dropped: a key the model does not declare lands in params
    for k, v in d.items():
        assert k in type(r).model_fields or r.params.get(k) == v, k
    assert r.headline()


def test_the_id_is_over_the_decision_not_the_clock():
    """an id that moved on every call could not answer the question it exists
    for - whether two ledgers differ on an outcome."""
    a = {"step_id": "s", "kind": "gate", "actor": "deterministic",
         "started_at": "2026-01-01T00:00:00+00:00", "duration_s": 1.0}
    b = {**a, "started_at": "2026-06-06T12:00:00+00:00", "duration_s": 9.9}
    assert coerce(a).record_id == coerce(b).record_id
    assert coerce({**a, "status": "fail"}).record_id != coerce(a).record_id


def test_appending_the_same_run_twice_is_one_ledger_entry(tmp_path):
    """the arms carry their whole record list forward in state, so a resume hands
    the store everything the run has ever decided. without the id check a
    three-gate run resumed twice would read as nine decisions, and the actor
    ratio is a published number.
    """
    path = tmp_path / "decisions.jsonl"
    recs = [{"step_id": "a", "kind": "gate", "actor": "deterministic", "status": "pass"},
            {"step_id": "b", "kind": "dispatch", "actor": "system", "status": "pass",
             "executor": "in_process"}]
    store = RecordStore(path)
    assert len(store.append_new(recs)) == 2
    assert len(store.append_new(recs)) == 0, "a resume must not re-record"

    # and a SECOND store over the same file sees them, which is the whole reason
    # the ledger is file-backed: the server answers each call in its own frame
    again = RecordStore(path)
    assert len(again.all()) == 2
    assert again.actor_ratio()["system"] == 1
    assert len(again.append_new([*recs, {"step_id": "c", "kind": "gate",
                                         "actor": "client", "status": "pass"}])) == 1


# --- the arms reach it, through the surface --------------------------------
def test_an_arms_refusal_lands_in_the_cohorts_ledger(bound_synthetic_cohort, ledger_at,
                                                     monkeypatch, tmp_path):
    """end to end through the MCP, on the arm that needs no encoder: a cohort
    with no evidence pack is refused deterministically, and that refusal is a
    decision the ledger has to carry.
    """
    monkeypatch.setenv("OMICSTRA_CHECKPOINT", f"sqlite:{tmp_path / 'ck.sqlite'}")
    import importlib

    import omicstra.mcp.server as S
    S = importlib.reload(S)

    out = S.evaluate_run(project_id=None, hypotheses=["H1"], run_id="led-1")
    assert out["status"] == "refused", out
    assert out["recorded"] == 1, "the refusal is a decision; it must reach the store"

    path = ledger_at / "synthetic" / "records" / "decisions.jsonl"
    assert path.is_file(), f"no ledger at {path}"
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    assert [r["step_id"] for r in rows] == ["preflight_evaluate"]
    assert rows[0]["actor"] == "deterministic" and rows[0]["status"] == "not_run"

    # a re-run of the same refusal is the same decision, not a second one
    out2 = S.evaluate_run(project_id=None, hypotheses=["H1"], run_id="led-2")
    assert out2["recorded"] == 0, "the same decision twice is one ledger entry"


# --- and the report reads it ----------------------------------------------
def test_gather_reads_the_ledger_and_the_report_shows_who_decided(
        bound_synthetic_cohort, ledger_at):
    root = bound_synthetic_cohort
    path = ledger_at / "synthetic" / "records" / "decisions.jsonl"
    RecordStore(path).append_new([
        {"step_id": "preflight_encode", "kind": "gate", "actor": "deterministic",
         "status": "pass"},
        {"step_id": "encode_dispatch", "kind": "dispatch", "actor": "system",
         "status": "pass", "executor": "temporal", "n_shards": 4},
    ])

    b = gather(root)
    assert len(b["records"]) == 2 and b["ledger"] == str(path)
    assert b["actor_ratio"]["system"] == 1 and b["actor_ratio"]["deterministic"] == 1

    md = render(b, "a report")
    assert "8 · decisions" in md
    assert "encode_dispatch" in md and "temporal" in md
    assert "by system | 1" in md


def test_a_cohort_with_no_ledger_says_so_in_one_line(bound_synthetic_cohort, ledger_at):
    """absence is a true statement about that cohort, not a gap in the report."""
    b = gather(bound_synthetic_cohort)
    assert "records" not in b
    md = render(b, "a report")
    assert "8 · decisions" in md
    assert "ledger is empty or absent" in md


# --- R4: what the chain MEASURED, beside what the summary declares ---------
def test_measured_steps_are_classified_from_the_contracts_not_by_shape(
        bound_synthetic_cohort, ledger_at):
    """a record is an inventory step or an EDA check because something DECLARED
    it one, never because its id looked like one."""
    from omicstra.report import _declared_step_ids

    inv, eda = _declared_step_ids()
    assert "shape" in inv and "coords" in inv
    assert "cohort_counts" in eda and "spatial_autocorrelation" in eda
    assert not (inv & eda), "a step id belongs to one chain or the other"


def test_section_four_reports_what_this_run_measured_and_what_it_refused(
        bound_synthetic_cohort, ledger_at):
    """section 2 reports what eda_summary.json DECLARES, which on a curated
    cohort is hand-written and says so. section 4 reports what the chain
    computed, so the two can be compared rather than one standing for both.
    """
    root = bound_synthetic_cohort
    path = ledger_at / "synthetic" / "records" / "decisions.jsonl"
    RecordStore(path).append_new([
        {"step_id": "shape", "kind": "diagnostic", "actor": "deterministic",
         "status": "pass", "result": "280 samples, 286,250 units"},
        {"step_id": "cohort_counts", "kind": "diagnostic", "actor": "deterministic",
         "status": "pass", "result": "median 25,041 features per section"},
        {"step_id": "model_tissue_fit", "kind": "diagnostic", "actor": "deterministic",
         "status": "not_run",
         "reason": "declared in the contract, no implementation registered"},
    ])
    md = render(gather(root), "a report")

    assert "inventory, as this run measured it" in md
    assert "shape" in md and "286,250" in md
    assert "EDA, as this run measured it" in md
    assert "median 25,041" in md
    # a refusal is a RESULT, not an omission - it names what was missing
    assert "refused, with the cause" in md
    assert "no implementation registered" in md


def test_a_declared_verdict_standing_over_a_computed_one_is_surfaced(
        bound_synthetic_cohort, ledger_at):
    """a person writing `proceed` into a committed summary is a recorded
    acceptance of the cautions. it does not make the two verdicts agree, and a
    reader of the summary alone cannot see the difference."""
    root = bound_synthetic_cohort
    (root / "eda_summary.json").write_text(json.dumps({"verdict": "proceed"}))
    RecordStore(ledger_at / "synthetic" / "records" / "decisions.jsonl").append_new([
        {"step_id": "eda_gate", "kind": "diagnostic", "actor": "deterministic",
         "status": "pass", "result": "proceed_with_caution"},
    ])
    md = render(gather(root), "a report")
    assert "declares `proceed` and this run computed `proceed_with_caution`" in md
    assert "travels with every number below" in md


# --- the trace: an order was always there, the cost was not ----------------
def test_a_node_stamps_only_the_records_it_added(monkeypatch):
    """the node returns the whole accumulated list. stamping all of it would
    attribute every earlier step's cost to this one."""
    import time

    from omicstra.records import timed_node

    @timed_node
    def node(state):
        time.sleep(0.02)
        return {"records": [*state.get("records", []),
                            {"step_id": "new", "kind": "gate"}]}

    out = node({"records": [{"step_id": "old", "kind": "gate", "duration_s": 9.9}]})
    assert out["records"][0]["duration_s"] == 9.9, "an earlier step keeps its own"
    assert out["records"][1]["duration_s"] >= 0.02


def test_a_node_that_times_itself_is_not_overwritten():
    """the measuring functions time their own work, which is tighter than the
    node's wall clock. the wrapper fills a gap, it does not take over."""
    from omicstra.records import timed_node

    @timed_node
    def node(state):
        return {"records": [{"step_id": "m", "kind": "diagnostic", "duration_s": 0.5}]}

    assert node({})["records"][0]["duration_s"] == 0.5


def test_a_node_returning_no_records_is_passed_through():
    from omicstra.records import timed_node

    @timed_node
    def node(state):
        return {"report": {"status": "refused"}}

    assert node({}) == {"report": {"status": "refused"}}


def test_every_graph_node_is_timed():
    """a stage that is slow should be visible without anyone having remembered
    to time it, so the wrapper is applied where nodes are REGISTERED."""
    import re
    from pathlib import Path

    g = Path(__file__).resolve().parents[1] / "src" / "omicstra" / "graphs"
    bare = []
    for f in sorted(g.glob("*.py")):
        for m in re.finditer(r'add_node\("([^"]+)", (?!timed_node)([^)]+)\)', f.read_text()):
            bare.append(f"{f.name}:{m.group(1)}")
    assert not bare, f"untimed graph node(s): {bare}"


def test_the_trace_reports_an_untimed_step_as_missing_not_as_zero(
        bound_synthetic_cohort, ledger_at, monkeypatch, tmp_path):
    """a step with no duration took an unknown time, which is a different
    statement from 'took no time' and must not average into a total."""
    monkeypatch.setenv("OMICSTRA_CHECKPOINT", f"sqlite:{tmp_path / 'ck.sqlite'}")
    import importlib

    import omicstra.mcp.server as S
    S = importlib.reload(S)

    out = S.evaluate_run(project_id=None, hypotheses=["H1"], run_id="tr-1")
    assert out["status"] == "refused"
    tr = S.runs_trace("tr-1")
    assert tr["n_steps"] >= 1
    assert tr["seconds_total"] == round(
        sum(s["seconds"] for s in tr["steps"] if s["seconds"] is not None), 3)
    assert all(s["started_at"] for s in tr["steps"]), "an order was always available"


def test_the_trace_refuses_a_handle_from_another_layer(monkeypatch, tmp_path):
    monkeypatch.setenv("OMICSTRA_CHECKPOINT", f"sqlite:{tmp_path / 'ck.sqlite'}")
    import importlib

    import omicstra.mcp.server as S
    S = importlib.reload(S)
    assert S.runs_trace("never-existed")["steps"] == []


# --- the time is the STEP's, because that is the only additive reading -------
def test_each_step_carries_its_own_duration_not_the_nodes():
    """`timed_node` stamps every record a node added with the NODE's duration,
    which is not additive: the evaluate node runs a six-step chain, so all six
    records read 145.7s and the three places that sum the field reported 874.3s
    for a run that took 145.7."""
    import time

    from omicstra.protocols import Step, build_protocol
    from omicstra.records import DiagnosticRecord

    def slow(ctx):
        time.sleep(0.05)
        return DiagnosticRecord(step_id="slow", status="pass")

    def quick(ctx):
        return DiagnosticRecord(step_id="quick", status="pass")

    recs = build_protocol([Step("slow", slow, authority="deterministic"),
                           Step("quick", quick, authority="deterministic")],
                          "t").invoke({"params": {"slow": {}, "quick": {}}})
    assert recs["slow"]["duration_s"] >= 0.05
    assert recs["quick"]["duration_s"] < recs["slow"]["duration_s"], \
        "a fast step must not inherit a slow one's cost"


def test_a_step_that_times_its_own_work_keeps_that_number():
    """several measures already time themselves, and that is the tighter figure."""
    from omicstra.protocols import Step, build_protocol
    from omicstra.records import DiagnosticRecord

    def own(ctx):
        return DiagnosticRecord(step_id="own", status="pass", duration_s=9.5)

    recs = build_protocol([Step("own", own, authority="deterministic")],
                          "t").invoke({"params": {"own": {}}})
    assert recs["own"]["duration_s"] == 9.5


def test_a_step_a_prerequisite_blocked_is_timed_as_well():
    """it took no time, and that is a KNOWN time - different from an unknown
    one, which the trace reports as missing rather than as zero."""
    from omicstra.protocols import Step, build_protocol
    from omicstra.records import DiagnosticRecord

    def boom(ctx):
        raise RuntimeError("no")

    def after(ctx):
        return DiagnosticRecord(step_id="after", status="pass")

    recs = build_protocol(
        [Step("boom", boom, produces={"x"}, authority="deterministic"),
         Step("after", after, requires={"x"}, authority="deterministic")],
        "t").invoke({"params": {"boom": {}, "after": {}}})
    assert recs["after"]["status"] == "not_run"
    assert recs["after"]["duration_s"] is not None


def test_a_known_zero_is_not_replaced_by_the_nodes_duration():
    """a step a prerequisite blocked takes ~0 s, which rounds to 0.0 and is
    FALSY - so `or` replaced three known zeros with the node's 148.1 s and the
    trace summed 588.6 s for a run that took 150. a zero is a measurement; only
    None means nobody measured."""
    import time

    from omicstra.records import timed_node

    @timed_node
    def node(state):
        time.sleep(0.05)
        return {"records": [{"step_id": "blocked", "kind": "diagnostic",
                             "status": "not_run", "duration_s": 0.0},
                            {"step_id": "untimed", "kind": "diagnostic",
                             "status": "pass"}]}

    recs = node({"records": []})["records"]
    assert recs[0]["duration_s"] == 0.0, "a known zero stands"
    assert recs[1]["duration_s"] >= 0.05, "an unmeasured step gets the node's"
