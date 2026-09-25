"""the encode gate: the only thing here a protocol cannot do.

`protocols/encode` computes the gates and never pauses. this graph is the pause,
so these tests are about the BRANCH and the ANSWER - not about what the gates
say, which is covered next door without a checkpointer.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from omicstra.graphs.encode import build_encode_graph

ROOT = Path(__file__).resolve().parents[1]
BARE: dict = {}
COUNTS = {"big": 1000, "small": 8}


def _run(cohort, answer=None, encoder="novae", counts=None, tid="t", device=None,
         platform=None):
    """drive the graph once, optionally answering the interrupt.

    `platform` is what makes a `ready` assertion honest in a clone. the node
    resolves its shards from the BOUND cohort, and the seed cohort's slides and
    coordinates are git-ignored - so a test that asserted `ready` without owning
    a cohort was asserting "this machine has 151 GB of data", and failed in every
    clone. the tests that need a real shard list bind `bound_synthetic_cohort`.
    """
    app = build_encode_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": tid}}
    out = app.invoke({"cohort": cohort, "encoder": encoder, "device": device,
                      "platform": platform,
                      "unit_counts": counts if counts is not None else COUNTS}, cfg)
    if answer is not None and "__interrupt__" in out:
        out = app.invoke(Command(resume=answer), cfg)
    return out


DECLARED = {"encoder_token_source": "env:HF_TOKEN", "encoder_fallback": "uni2",
            "below_floor_policy": "drop", "scope_policy": "authoritative",
            "output_dir": "data/embeddings"}


# --- the branch ------------------------------------------------------------
def test_a_fully_declared_cohort_never_pauses(bound_synthetic_cohort):
    """the precondition for an end-to-end run. not the gates being bypassed -
    every one is computed and recorded, with its declaration as provenance."""
    out = _run(DECLARED, tid="declared", platform="synthetic")
    assert "__interrupt__" not in out
    assert out["report"]["status"] == "ready"
    assert not any(g["open"] for g in out["gates"])


def test_an_undeclared_cohort_pauses_with_the_evidence():
    out = _run(BARE, tid="bare")
    v = out["__interrupt__"][0].value
    assert v["gates"], "the payload must carry the gates, not just say there are some"
    for g in v["gates"]:
        assert g["options"] and g["forecloses"], \
            "a gate without its options and its consequence cannot be answered"
    assert v["no_default"] == "the system does not pick"


def test_the_payload_names_what_would_pre_answer_each_gate():
    """so the person being interrupted can stop being interrupted next time."""
    v = _run(BARE, tid="hint")["__interrupt__"][0].value
    assert v["pre_answerable"], "the payload must name the declaration that answers each"
    assert all(p.startswith("cohort.json#") for p in v["pre_answerable"].values())


# --- the answer ------------------------------------------------------------
def test_a_valid_answer_proceeds_and_is_recorded_as_human(bound_synthetic_cohort):
    out = _run(BARE, answer={"encoder_compatibility": "uni2",
                             "platform_floor": "drop", "capacity": "d"}, tid="ok",
               platform="synthetic")
    assert out["report"]["status"] == "ready"
    assert out["answers"]["platform_floor"] == "drop_below_floor", "shorthand must normalise"
    assert any(r.get("actor") == "human" for r in out["records"])


def test_a_human_answer_faces_the_same_closed_set_as_a_declaration():
    """THE symmetry. the option set was closed on the interrupt path and open on
    the declaration path until `_resolve_option` landed. closing one is not
    closing it - the ledger must not record a choice nobody was offered.
    """
    out = _run(BARE, answer={"encoder_compatibility": "uni2",
                             "platform_floor": "keep_everything", "capacity": "d"}, tid="bad")
    assert out["halted"] is True
    assert "option set is closed" in out["records"][-1]["reason"]


def test_an_incomplete_answer_halts_and_names_what_is_missing():
    """answering two of three is not answering. proceeding on a partial reply
    would silently default the rest, which is the one thing a gate forbids."""
    out = _run(BARE, answer={"capacity": "d"}, tid="partial")
    assert out["halted"] is True
    r = out["records"][-1]["reason"]
    assert "unanswered" in r and "platform_floor" in r


def test_choosing_halt_halts():
    out = _run(BARE, answer={"encoder_compatibility": "uni2",
                             "platform_floor": "halt", "capacity": "d"}, tid="halt")
    assert out["halted"] is True and out["report"]["status"] == "halted"


def test_refusing_outright_halts_without_running():
    for refusal in (False, "no", "stop"):
        out = _run(BARE, answer=refusal, tid=f"refuse-{refusal}")
        assert out["halted"] is True
        assert out["report"]["status"] == "halted"


def test_an_unknown_gate_id_is_refused():
    out = _run(BARE, answer={"not_a_gate": "x"}, tid="unknown")
    assert out["halted"] is True


# --- compute refuses on its own account ------------------------------------
def test_the_compute_node_refuses_an_open_gate_even_if_routed_there():
    """belt and braces: routing should make this unreachable, and a compute path
    that trusts its own routing is one edit from spending GPU hours on an
    unanswered values call."""
    from omicstra.graphs.encode import encode
    from omicstra.protocols.encode import ComputeRefused

    with pytest.raises(ComputeRefused):
        encode({"cohort": BARE, "encoder": "novae", "unit_counts": COUNTS})


# --- the fixture -----------------------------------------------------------
def test_tnbc92_runs_the_graph_unattended():
    """the SEED cohort's own declarations, which nothing synthetic can stand in
    for - the claim is that this cohort as recorded clears every gate.

    skips where the cohort is absent, which is every clone. that is a real gap in
    what CI covers and is why the four tests above own a cohort instead.
    """
    root = ROOT / "projects" / "tnbc-92"
    rec = root / "data" / "canonical" / "ingest.json"
    if not rec.is_file():
        pytest.skip("no canonical ingest on this machine")
    counts = {s: v["n_spots"] for s, v in json.loads(rec.read_text())["sources"].items()}
    cohort = json.loads((root / "cohort.json").read_text())
    for enc in ("virchow2", "novae"):
        out = _run(cohort, encoder=enc, counts=counts, tid=f"fixture-{enc}")
        assert "__interrupt__" not in out, f"{enc} paused; step 8 cannot run unattended"
        assert out["report"]["status"] == "ready"


# --- what it costs ---------------------------------------------------------
def test_the_person_being_asked_is_told_what_the_run_costs():
    """the gates say what is being decided. the plan says what the decision buys,
    and "run it here or somewhere else" cannot be answered without it."""
    v = _run(BARE, tid="cost", device="cpu")["__interrupt__"][0].value
    assert v["plan"]["estimated"] is True
    assert v["plan"]["hours"] > 0 and v["plan"]["n_units"] == sum(COUNTS.values())


def test_a_run_that_pauses_for_nobody_still_records_what_it_expected_to_cost(
        bound_synthetic_cohort):
    """an unattended run leaves the number the next plan is checked against."""
    out = _run(DECLARED, encoder="virchow2", tid="unattended-cost", device="mps",
               platform="synthetic")
    assert out["report"]["status"] == "ready"
    assert out["report"]["plan"]["hours"] > 0


def test_an_unplannable_run_proceeds_and_says_it_was_never_priced(bound_synthetic_cohort):
    """no device declared is not a reason to stop a cohort - but it must not read
    as a run whose cost was checked and found acceptable."""
    out = _run(DECLARED, encoder="virchow2", tid="unpriced", platform="synthetic")
    assert out["report"]["status"] == "ready"
    assert out["report"]["plan"]["verdict"] == "unknown"
    assert "no device" in out["report"]["plan"]["why_not"]


def test_a_measurement_that_contradicts_the_declaration_reopens_the_gate():
    """a declared output_dir says WHERE to write. it does not say the volume can
    hold it, and when the plan says it cannot, the declaration stops standing."""
    cohort = dict(DECLARED, pool={"id": "tiny", "device": "cpu",
                                  "memory_gb_per_task": 1.0, "free_disk_gb": 0.001})
    out = _run(cohort, encoder="virchow2", counts={"a": 5_000_000}, tid="too-small")
    v = out["__interrupt__"][0].value
    cap = next(g for g in v["gates"] if g["id"] == "capacity")
    assert cap["observed"]["verdict"] == "does_not_fit"
    assert cap["observed"]["declaration_overridden"] == "data/embeddings"
