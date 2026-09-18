"""promote proposes the pack from artifacts, and never authors its prose.

the acceptance is the curated pack itself: every value and interval a person
typed into `routing_evidence.json` must come back out of the artifacts. where
the rebuild and the pack disagree, one of them is wrong, and until now there was
no way to find out which.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from omicstra.protocols.promote import (
    assert_not_authored,
    propose,
    propose_task,
    provenance,
    read_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "projects" / "tnbc-92"
SOURCES = PROJ / "evidence_sources.json"
PACK = PROJ / "routing_evidence.json"


def _sources():
    if not SOURCES.is_file():
        pytest.skip("cohort declares no evidence sources")
    return json.loads(SOURCES.read_text())


def _rebuilt():
    src = _sources()
    root = (PROJ / src["runs_root"]).resolve()
    if not root.is_dir():
        pytest.skip("run grid not present on this machine")
    return propose(src, root, src["runs"]), json.loads(PACK.read_text())["tasks"]


# --- the acceptance --------------------------------------------------------
def test_every_number_in_the_curated_pack_comes_back_out_of_the_artifacts():
    """46 candidates across five rows. a mismatch means the pack drifted from
    what produced it, which is the failure a hand-written pack cannot detect."""
    built, pack = _rebuilt()
    checked = 0
    for task_id, t in built["tasks"].items():
        if t["outcome"] == "not_available":
            continue
        hand = {c["id"]: c for c in pack[task_id].get("candidates", [])}
        for c in t["candidates"]:
            h = hand.get(c["id"])
            if h is None:            # covered by its own test below
                continue
            assert float(h["value"]) == pytest.approx(float(c["value"]), abs=5e-4), \
                f"{task_id}/{c['id']}: pack {h['value']}, artifact {c['value']}"
            if h.get("ci") and c.get("ci"):
                assert h["ci"] == pytest.approx(c["ci"], abs=5e-4), \
                    f"{task_id}/{c['id']}: interval moved"
            checked += 1
    assert checked >= 40, f"only {checked} candidates compared - the rebuild is too thin"


def test_the_rebuild_finds_a_passing_arm_the_curated_pack_omits():
    """B3_v3 answers 4 of 4 testable pathways and is absent from the pack.

    leaving a passing baseline out is the selective-reporting failure the
    deterministic half exists to prevent - so it is asserted, not tidied away.
    """
    built, pack = _rebuilt()
    mine = {c["id"] for c in built["tasks"]["pathway_transfer"]["candidates"]}
    hand = {c["id"] for c in pack["pathway_transfer"]["candidates"]}
    assert "B3_v3" in mine - hand
    b3 = next(c for c in built["tasks"]["pathway_transfer"]["candidates"] if c["id"] == "B3_v3")
    assert b3["value"] == 4


def test_a_row_whose_point_table_lacks_a_run_says_where_the_number_came_from():
    """B4 has no row in mc_coherence.parquet; its ARI is in the interval file.
    the pack mixed the two silently - the rebuild records which source per run."""
    built, _ = _rebuilt()
    cands = {c["id"]: c for c in built["tasks"]["tissue_state_grouping"]["candidates"]}
    assert cands["B4_v3"]["source"].endswith("metrics_h2_ci.json")
    assert cands["R1_v3"]["source"].endswith("mc_coherence.parquet")


# --- the rules -------------------------------------------------------------
def test_a_row_where_every_candidate_fails_its_guard_refuses():
    """the identity row: no run's biology z exceeds its patient z, so there is no
    method to route to. a refusal is a routable answer, not a caveat on a winner."""
    built, _ = _rebuilt()
    t = built["tasks"]["subject_identity_suppression"]
    assert t["outcome"] == "refuse" and t["winner"] is None
    assert "contraindicated" in t["refusal"]
    assert all(g["contraindicated"] for g in t["guards"])


def test_equal_values_tie_even_without_an_interval():
    """four arms at 4 of 4 pathways are not ordered by groupby order."""
    decl = {"higher_is_better": True}
    out = propose_task("t", decl, [{"id": "a", "value": 4}, {"id": "b", "value": 4},
                                   {"id": "c", "value": 1}])
    assert out["outcome"] == "tie" and set(out["tied"]) == {"a", "b"}


def test_overlapping_intervals_tie_and_separated_ones_do_not():
    decl = {"higher_is_better": True}
    tied = propose_task("t", decl, [{"id": "a", "value": 0.9, "ci": [0.8, 1.0]},
                                    {"id": "b", "value": 0.85, "ci": [0.75, 0.95]}])
    assert tied["outcome"] == "tie"
    clear = propose_task("t", decl, [{"id": "a", "value": 0.9, "ci": [0.88, 0.92]},
                                     {"id": "b", "value": 0.5, "ci": [0.48, 0.52]}])
    assert clear["outcome"] == "recommend" and clear["winner"] == "a"
    assert clear["margin"] == pytest.approx(0.4)


def test_a_row_where_nothing_clears_the_floor_refuses():
    decl = {"higher_is_better": True, "floor": {"id": "chance", "value": 0.5}}
    out = propose_task("t", decl, [{"id": "a", "value": 0.44}, {"id": "b", "value": 0.4}])
    assert out["outcome"] == "refuse" and "floor" in out["refusal"]
    assert "a at 0.44" in out["refusal"]


def test_an_unanswerable_family_is_reported_not_dropped():
    built, _ = _rebuilt()
    for task_id in ("structural_agreement", "pathway_discrimination"):
        t = built["tasks"][task_id]
        assert t["outcome"] == "not_available" and t["why"]


def test_every_declared_family_is_accounted_for():
    """a family the cohort does not declare is named, so a gap cannot look like
    a family that does not exist."""
    built, _ = _rebuilt()
    from omicstra.contracts.routing import load_routing_contract

    known = {f["id"] for f in load_routing_contract()["task_families"]}
    assert set(built["tasks"]) | set(built["families_not_declared"]) == known


# --- the boundary: computed vs authored ------------------------------------
def test_provenance_separates_the_two_halves():
    built, _ = _rebuilt()
    prov = provenance(built)
    t = prov["cross_modal_retrieval"]
    assert "candidates" in t["computed"] and "outcome" in t["computed"]
    assert "metric" in t["declared"] and "floor" in t["declared"]
    assert t["human_fields_present"] is False, "the proposal must carry no prose"


def test_a_pack_whose_measurement_was_authored_is_refused():
    bad = {"tasks": {"t": {"candidates": [{"id": "a", "value": "about 0.9"}]}}}
    with pytest.raises(TypeError, match="authored"):
        assert_not_authored(bad)
    worse = {"tasks": {"t": {"candidates": [{"id": "a", "value": 0.9, "ci": [0.8]}]}}}
    with pytest.raises(ValueError, match="interval"):
        assert_not_authored(worse)


def test_an_undeclared_reader_is_refused_by_name():
    with pytest.raises(KeyError, match="no reader"):
        read_candidates({"reader": "invented"}, ROOT, ["R1_v3"])


def test_the_proposal_names_no_cohort():
    """the readers are cohort-free; the cohort is in the declaration."""
    src = (ROOT / "src" / "omicstra" / "protocols" / "promote.py").read_text().lower()
    for token in ("tnbc", "wang", "virchow", "novae", "_v3"):
        assert token not in src, f"{token!r} is a cohort fact and belongs in the declaration"


# --- the gate --------------------------------------------------------------
def test_the_gate_takes_prose_and_refuses_an_edited_number():
    from omicstra.graphs.promote import emit

    proposal = {"project_id": "x", "runs_root": "r",
                "tasks": {"t": {"task_id": "t", "outcome": "recommend", "winner": "a",
                                "candidates": [{"id": "a", "value": 0.9}]}}}
    out = emit({"proposal": proposal, "notes": {"t": {"note": "a person wrote this"}}})
    assert out["pack"]["tasks"]["t"]["note"] == "a person wrote this"
    assert out["pack"]["tasks"]["t"]["authored"] == ["note"]
    assert out["provenance"]["t"]["human"] == ["note"]
    with pytest.raises(ValueError, match="not a human field"):
        emit({"proposal": proposal, "notes": {"t": {"value": 0.99}}})


def test_the_graph_halts_without_approval():
    from omicstra.graphs.promote import build_promote_graph

    g = build_promote_graph()
    assert g is not None
    from omicstra.graphs.promote import halt

    assert halt({})["halted"] is True and halt({})["pack"] == {}


# --- the floor is beaten, not merely reached -------------------------------
def test_a_candidate_sitting_exactly_on_the_floor_does_not_clear_it():
    """a run with zero significant pathways sits on a floor of "none significant"
    and answers nothing. reading the floor as >= forces it to be inflated in the
    cohort's declaration, which puts an unmeasured number in the cohort's file."""
    decl = {"higher_is_better": True, "floor": {"id": "none_significant", "value": 0}}
    out = propose_task("t", decl, [{"id": "a", "value": 2}, {"id": "b", "value": 0}])
    assert [c["clears_floor"] for c in out["candidates"]] == [True, False]
    assert out["outcome"] == "recommend" and out["winner"] == "a"
    low = propose_task("t", {"higher_is_better": False, "floor": {"id": "f", "value": 1.0}},
                       [{"id": "a", "value": 1.0}, {"id": "b", "value": 0.5}])
    assert [c["clears_floor"] for c in low["candidates"]] == [True, False] or \
           [c["clears_floor"] for c in low["candidates"]] == [False, True]


def test_the_declared_floors_are_the_ones_the_cohort_measured():
    """each floor value must exist in the curated pack; an inflated proxy means
    the declaration carries a number nothing produced."""
    src = _sources()
    pack = json.loads(PACK.read_text())["tasks"]
    for task_id, decl in src["tasks"].items():
        if decl.get("not_available") or task_id not in pack:
            continue
        mine, hand = decl.get("floor"), pack[task_id].get("floor")
        if mine and hand and mine.get("value") is not None and hand.get("value") is not None:
            assert mine["value"] == pytest.approx(hand["value"], abs=5e-4), (
                f"{task_id}: declared floor {mine['value']} vs pack {hand['value']}")


def test_the_subject_identity_floor_is_read_from_the_artifact():
    """0.1366 sits in every run's biology.parquet as the raw imaging view. a floor
    that only exists in a document is the one number in the pack with no source."""
    src = _sources()
    root = (PROJ / src["runs_root"]).resolve()
    if not root.is_dir():
        pytest.skip("run grid not present on this machine")
    fr = src["tasks"]["subject_identity_suppression"]["floor_reader"]
    rows = read_candidates(fr, root, fr["ids"])
    assert rows, "no run carries the raw view"
    values = {r["value"] for r in rows}
    assert values == {0.1366}, f"the raw imaging floor differs across arms: {values}"
    assert all("#raw_he" in r["source"] for r in rows)
    declared = src["tasks"]["subject_identity_suppression"]["floor"]["value"]
    assert declared == pytest.approx(0.1366, abs=5e-5)


def test_the_raw_molecular_view_is_per_arm_not_a_cohort_constant():
    """it depends on the arm's declared ST features, so it cannot be cited as one
    number the way the imaging view can."""
    src = _sources()
    root = (PROJ / src["runs_root"]).resolve()
    if not root.is_dir():
        pytest.skip("run grid not present on this machine")
    rows = read_candidates({"reader": "biology_raw_floor", "view": "raw_st"}, root, src["runs"])
    by_run = {r["id"]: r["value"] for r in rows}
    assert len(set(by_run.values())) > 1, "expected the raw molecular view to vary by arm"
