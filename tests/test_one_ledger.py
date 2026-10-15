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
