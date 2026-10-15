"""the evaluate gate: what a number is allowed to claim, asked before it is.

the acceptance has two halves and they test different things.

on the SEED cohort the artifacts exist, so the gates fire on real bytes: 15
cited artifacts hashed and none pinned, and 7 of 9 guard checks failing on
H2-C. those numbers are read from the published grid and are skipped where it is
absent.

on a SYNTHETIC cohort nothing exists, and the refusals are what is under test -
no pack, no sources, and a hypothesis whose families the cohort never declared.
a stage that only ever ran on the cohort it was written for is a stage with one
data point.
"""
from __future__ import annotations

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from omicstra.protocols.evaluate import (
    ArtifactDrift,
    EvidencePackMissing,
    cited_paths,
    claims,
    families,
    guard_status,
    hypothesis_root,
    named_without_location,
    preflight_evaluate,
    scope,
    verdicts,
)

SEED = "tnbc-92"


def _seed_or_skip():
    from pathlib import Path

    from omicstra.settings import settings

    root = settings.project_root(SEED)
    if not (root / "routing_evidence.json").is_file():
        pytest.skip("the seed cohort's pack is not on this machine")
    src = json.loads((root / "evidence_sources.json").read_text())
    if not (Path(root / src["runs_root"]) / "eval").is_dir():
        pytest.skip("run grid not present on this machine")
    return root, src


# --- scope, which is read from the contract rather than assumed -------------
def test_the_two_hypothesis_vocabularies_meet_in_one_place():
    """the contract names H2-A, H2-C, H2-B.2 because they are different claims;
    a caller asks about H2 because that is how the proposal states it."""
    assert hypothesis_root("H2-B.2") == "H2"
    assert hypothesis_root("H3-DAG") == "H3"
    assert hypothesis_root("H1") == "H1"
    ids = {f["id"] for f in families(["H2"])}
    assert ids == {"tissue_state_grouping", "subject_identity_suppression",
                   "morphology_decode"}
    assert len(families(None)) == len(families(["H1", "H2", "H3"])), (
        "every declared family belongs to one of the three hypotheses")


def test_every_required_guard_says_whether_it_is_implemented():
    """a required guard silently not run is the failure the guard library exists
    to prevent, so the gap is declared with a reason per guard."""
    g = {d["guard"]: d for d in guard_status(families(None))}
    assert {d["status"] for d in g.values()} <= {"implemented", "not_implemented"}
    for d in g.values():
        assert d["note"], f"{d['guard']} has no reason"
        assert (d["implemented_as"] is None) == (d["status"] == "not_implemented")
    # the two the contract requires and this package does not check
    assert g["raw_floor"]["status"] == "not_implemented"
    assert g["untrained_control"]["status"] == "not_implemented"


# --- the paths a verdict reads, and the ones it cannot locate ---------------
def test_a_bare_filename_in_the_prose_is_reported_not_guessed():
    """`metrics_h2_ci.json` alone does not say where it is, and the two
    candidates on the seed cohort are a shared file and a per-arm one. resolving
    it against a directory that appeared nearby gets one right and one wrong.
    """
    task = {"ci_file": "eval/H2/metrics_h2_ci.json",
            "source": "eval/H2/metrics_h2_ci.json h2c (delta over biology.parquet)"}
    assert cited_paths(task, []) == ["eval/H2/metrics_h2_ci.json"]
    assert named_without_location(task) == ["biology.parquet"]


def test_a_per_run_template_is_expanded_over_the_declared_grid():
    task = {"source": "{run}/metrics_h1_ci.json (bootstrap, 200x)"}
    assert cited_paths(task, ["A", "B"]) == ["A/metrics_h1_ci.json", "B/metrics_h1_ci.json"]
    assert cited_paths(task, []) == [], "a template with no grid locates nothing"


def test_a_name_located_by_a_structured_key_is_not_also_reported_unlocated():
    task = {"table": "eval/H2/mc_coherence.parquet",
            "source": "eval/H2/mc_coherence.parquet (point) + mc_coherence.parquet again"}
    assert named_without_location(task) == []


# --- a guard that did not run is not a guard that passed -------------------
@pytest.mark.parametrize("row,expected", [
    ({"evaluable": False}, "not_run"),
    ({"evaluable": True, "guards_ran": [], "guards_failed": []}, "unguarded"),
    ({"evaluable": True, "guards_ran": ["a"], "guards_failed": []}, "biological"),
    ({"evaluable": True, "guards_ran": ["a"], "guards_failed": ["a"]}, "diagnostic"),
])
def test_claims_distinguishes_unguarded_from_clean(row, expected):
    """"no guard failed" was true of a hypothesis where no guard ran at all, and
    read as a clean bill of health for a number nothing checked."""
    assert claims(row) == expected


# --- the seed cohort: the gates on real bytes ------------------------------
def test_the_seed_cohorts_artifacts_are_hashed_and_unpinned():
    root, _ = _seed_or_skip()
    reqs = {r.id: r for r in preflight_evaluate(project_id=SEED,
                                                hypotheses=["H1", "H2", "H3"])}
    assert set(reqs) == {"hypothesis_scope", "artifact_pinning", "guard_failure"}
    # named by the caller, so the scope gate is closed by the request
    assert not reqs["hypothesis_scope"].open
    assert reqs["hypothesis_scope"].source == "the request"

    pin = reqs["artifact_pinning"].observed
    assert pin["n_artifacts"] >= 10 and pin["absent"] == []
    assert pin["n_declared_pins"] == 0 and len(pin["unpinned"]) == pin["n_artifacts"]
    assert reqs["artifact_pinning"].open, "unpinned artifacts must ASK, not default"
    assert len(pin["sha256"]) == pin["n_artifacts"], "the hashes travel with the question"
    assert set(pin["named_without_location"]) == {"biology.parquet", "perm_nulls.parquet"}


def test_the_guard_gate_fires_on_a_guard_that_ran_and_failed():
    """H2-C's G3 is the bio/patient ratio per arm. no arm on the seed grid
    suppresses patient identity below biology, which is a RESULT - so the gate
    asks whether to report it as diagnostic rather than reporting it as clean.
    """
    _seed_or_skip()
    v = verdicts(SEED, hypotheses=["H2"])
    reqs = {r.id: r for r in preflight_evaluate(project_id=SEED,
                                                hypotheses=["H2"], verdict=v)}
    g = reqs["guard_failure"]
    assert g.open, "a failed guard must ask what it does to the claim"
    assert g.observed["n_failed"] == len(v["H2"]["guards_failed"]) > 0
    assert all("G3_bio_vs_confounder" in f for f in g.observed["failed"])
    assert g.observed["n_not_implemented"] == 2, "raw_floor and untrained_control"


def test_a_hypothesis_whose_guards_all_lack_an_input_is_unguarded_not_clean():
    _seed_or_skip()
    v = verdicts(SEED, hypotheses=["H1", "H2", "H3"])
    assert claims(v["H2"]) == "diagnostic"
    # H3's pathway gate needs each arm's ST features, which nothing declares
    assert v["H3"]["gates"]["pathway_transfer"]["status"] == "not_run"
    assert v["H3"]["gates"]["pathway_transfer"]["needs"] == \
        "evidence_sources.json#run_st_features"
    assert claims(v["H3"]) == "unguarded"


def test_the_rollups_coverage_is_reported_with_the_rollup():
    """each of the three rollups drops a different arm, and nothing noticed until
    they were read side by side."""
    _seed_or_skip()
    v = verdicts(SEED, hypotheses=["H1", "H2", "H3"])
    miss = {h: v[h]["rollup"]["coverage"]["missing"] for h in ("H1", "H2", "H3")}
    assert miss == {"H1": [], "H2": ["B4_v3"], "H3": ["R5_v3"]}


def test_a_declared_pin_that_does_not_match_is_refused_not_asked(monkeypatch):
    """no option at the gate makes a drifted artifact evaluable: the pack's
    numbers were read from the pinned bytes."""
    _seed_or_skip()
    with pytest.raises(ArtifactDrift, match="differ from the pin"):
        preflight_evaluate({"evidence_pins": {"eval/H2/metrics_h2_ci.json": "0" * 64}},
                           project_id=SEED, hypotheses=["H2"])


def test_a_matching_pin_closes_the_gate():
    _seed_or_skip()
    pin = next(r for r in preflight_evaluate(project_id=SEED, hypotheses=["H2"])
               if r.id == "artifact_pinning")
    reqs = {r.id: r for r in preflight_evaluate({"evidence_pins": pin.observed["sha256"]},
                                                project_id=SEED, hypotheses=["H2"])}
    got = reqs["artifact_pinning"]
    assert not got.open and got.answer == "declared_pins"
    assert got.source == "cohort.json#evidence_pins"


# --- the graph, end to end -------------------------------------------------
def test_the_graph_interrupts_then_states_a_verdict_carrying_the_caveat():
    _seed_or_skip()
    from omicstra.graph import _load_cohort
    from omicstra.graphs.evaluate import build_evaluate_graph

    app = build_evaluate_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "eval-test"}}
    out = app.invoke({"project_id": SEED, "arm": "evaluate",
                      "cohort": _load_cohort(SEED),
                      "hypotheses": ["H1", "H2", "H3"]}, cfg)
    asked = {g["id"] for g in out["__interrupt__"][0].value["gates"]}
    assert asked == {"artifact_pinning", "guard_failure"}

    out = app.invoke(Command(resume={"artifact_pinning": "pin_these_bytes",
                                     "guard_failure": "report_as_diagnostic"}), cfg)
    r = out["report"]
    assert r["status"] == "evaluated"
    assert r["hypotheses"]["H2"]["claims"] == "diagnostic"
    assert {h: d["claims"] for h, d in r["hypotheses"].items()} == {
        "H1": "unguarded", "H2": "diagnostic", "H3": "unguarded"}
    # the pin is a PROPOSAL in the ledger; the package does not write a cohort's
    # declaration, because a pin is a committed human act
    pins = [d for d in out["records"] if d.get("step_id") == "evaluate_pins"]
    assert len(pins) == 1 and len(pins[0]["pins"]) == r["artifacts"]["n_artifacts"]
    assert pins[0]["actor"] == "client" and pins[0]["confirmed_by"] is None


def test_a_gate_left_unanswered_halts_and_states_no_verdict():
    _seed_or_skip()
    from omicstra.graph import _load_cohort
    from omicstra.graphs.evaluate import build_evaluate_graph

    app = build_evaluate_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "eval-halt"}}
    app.invoke({"project_id": SEED, "arm": "evaluate", "cohort": _load_cohort(SEED),
                "hypotheses": ["H2"]}, cfg)
    out = app.invoke(Command(resume={"guard_failure": "report_as_diagnostic"}), cfg)
    assert out["report"]["status"] == "halted"
    assert "hypotheses" not in out["report"]


# --- a cohort with none of this -------------------------------------------
def test_a_cohort_with_no_pack_is_refused_with_the_reason(bound_synthetic_cohort):
    with pytest.raises(EvidencePackMissing, match="routing_evidence.json"):
        preflight_evaluate(project_id=None, hypotheses=["H1"])


def test_the_graph_refuses_rather_than_raising_at_the_surface(bound_synthetic_cohort):
    """a client gets an answer. the first version of two read-path tools raised
    through to a ToolError on a cohort whose gate had never run.
    """
    from omicstra.graphs.evaluate import build_evaluate_graph

    app = build_evaluate_graph(checkpointer=InMemorySaver())
    out = app.invoke({"project_id": None, "arm": "evaluate", "cohort": {},
                      "hypotheses": ["H1", "H2", "H3"]},
                     {"configurable": {"thread_id": "eval-synth"}})
    assert out["report"]["status"] == "refused"
    assert out["report"]["refusal"] == "evidence_pack_missing"
    assert "__interrupt__" not in out, "nothing to ask about is not a question"
    assert out["records"][-1]["actor"] == "deterministic"


def test_a_pack_with_no_fillable_family_is_not_run_rather_than_asked(
        bound_synthetic_cohort):
    """the distinction the scope gate exists for: a hypothesis nobody asked
    reads differently from one whose every family this cohort cannot fill.
    """
    root = bound_synthetic_cohort
    (root / "routing_evidence.json").write_text(json.dumps(
        {"project_id": "synthetic", "tasks": {}}))
    (root / "evidence_sources.json").write_text(json.dumps(
        {"project_id": "synthetic", "runs_root": "runs", "runs": [], "tasks": {}}))

    from omicstra.graphs.evaluate import build_evaluate_graph

    app = build_evaluate_graph(checkpointer=InMemorySaver())
    out = app.invoke({"project_id": None, "arm": "evaluate", "cohort": {},
                      "hypotheses": ["H1", "H2", "H3"]},
                     {"configurable": {"thread_id": "eval-empty"}})
    assert out["report"]["status"] == "not_run"
    assert "no hypothesis asked for has a fillable task family" in out["report"]["why"]
    assert set(out["report"]["per_hypothesis"]) == {"H1", "H2", "H3"}
    assert "__interrupt__" not in out


def test_an_undeclared_family_reads_differently_from_one_declared_unavailable():
    pack = {"tasks": {"cross_modal_retrieval": {"candidates": []},
                      "structural_agreement": {"not_available": "no CKA artifact"}}}
    src = {"tasks": {"cross_modal_retrieval": {"source": "x/y.json"}}}
    sc = scope(families(["H1"]), pack, src)
    assert sc["H1"]["fillable"] == ["cross_modal_retrieval"]
    assert sc["H1"]["not_available"] == [{"task": "structural_agreement",
                                          "why": "no CKA artifact"}]
    assert sc["H1"]["evaluable"] is True


# --- a cell is a value AND whether it may be compared ----------------------
PACK = {
    "contraindications": [{
        "method": "M_bad", "applies_to": ["q_one"], "does_not_apply_to": ["q_two"],
        "statement": "amplifies the confounder by construction",
        "evidence": "ratio below the raw floor"}],
    "tasks": {
        "q_one": {"higher_is_better": True, "floor": {"id": "raw", "value": 0.5},
                  "candidates": [{"id": "M_good", "value": 0.9},
                                 {"id": "M_bad", "value": 0.8},
                                 {"id": "M_weak", "value": 0.4}]},
        "q_two": {"higher_is_better": True, "floor": {"id": "raw", "value": 0.1},
                  "candidates": [{"id": "M_bad", "value": 0.8}]},
        "q_none": {"not_available": "no artifact for this family"},
    },
}


def test_a_contraindication_is_task_scoped_and_its_exception_is_honoured():
    """a method contraindicated for one question may be the right answer to
    another - the pack records the exception, and reading the method name alone
    would silently override it."""
    from omicstra.protocols.evaluate import contraindication

    assert contraindication(PACK, "q_one", "M_bad")["statement"].startswith("amplifies")
    assert contraindication(PACK, "q_two", "M_bad") is None, "the exception stands"
    assert contraindication(PACK, "q_one", "M_good") is None


def test_a_contraindicated_cell_is_not_ranked_and_the_reason_stands_in_its_place():
    from omicstra.protocols.evaluate import cell

    c = cell(PACK, "q_one", {"id": "M_bad", "value": 0.8})
    assert c["status"] == "contraindicated" and c["rankable"] is False
    assert "amplifies" in c["why"] and c["evidence"]
    # and it IS rankable on the question the pack exempts
    assert cell(PACK, "q_two", {"id": "M_bad", "value": 0.8})["rankable"] is True


def test_a_method_below_the_floor_has_not_answered_the_question():
    """every hypothesis here is stated comparatively against a baseline, so a
    value that does not beat it is not a weak answer - it is not an answer."""
    from omicstra.protocols.evaluate import cell

    c = cell(PACK, "q_one", {"id": "M_weak", "value": 0.4})
    assert c["status"] == "below_floor" and c["rankable"] is False
    assert "comparative" in c["why"]
    assert cell(PACK, "q_one", {"id": "M_good", "value": 0.9})["status"] == "admissible"


def test_lower_is_better_flips_the_floor_comparison():
    pack = {"tasks": {"q": {"higher_is_better": False,
                            "floor": {"id": "raw", "value": 0.5},
                            "candidates": []}}}
    from omicstra.protocols.evaluate import cell

    assert cell(pack, "q", {"id": "m", "value": 0.4})["status"] == "admissible"
    assert cell(pack, "q", {"id": "m", "value": 0.6})["status"] == "below_floor"


def test_the_hypothesis_claim_rides_on_the_question_not_the_cell():
    """whether the hypothesis was guarded caveats every cell of a question
    equally, so repeating it per cell says the same thing n times."""
    from omicstra.protocols.evaluate import matrix

    m = matrix(PACK, claims={"H1": "unguarded"})
    assert all("claim" not in c for c in m["q_one"]["cells"])
    assert "claim" in m["q_one"]


def test_a_question_with_no_artifact_keeps_its_own_reason():
    from omicstra.protocols.evaluate import matrix

    row = matrix(PACK)["q_none"]
    assert row["status"] == "not_available" and row["cells"] == []
    assert row["why"] == "no artifact for this family"


def test_the_seed_cohorts_matrix_marks_what_cannot_be_compared():
    """on the published pack: two retrieval arms below chance, one method
    contraindicated for two questions, and arms scoring zero on transfer."""
    root, _ = _seed_or_skip()
    import json

    from omicstra.protocols.evaluate import matrix

    pack = json.loads((root / "routing_evidence.json").read_text())
    m = matrix(pack)
    bad = {(t, c["method"]): c["status"] for t, r in m.items()
           for c in r.get("cells", []) if not c["rankable"]}
    assert bad[("cross_modal_retrieval", "B3_v3")] == "below_floor"
    assert bad[("cross_modal_retrieval", "B4_v3")] == "below_floor", "below chance"
    assert bad[("tissue_state_grouping", "B1_v3")] == "contraindicated"
    assert bad[("subject_identity_suppression", "B1_v3")] == "contraindicated"
    # and the exception holds: the same method is comparable on pathway transfer
    pt = {c["method"]: c["rankable"] for c in m["pathway_transfer"]["cells"]}
    assert pt["B1_v3"] is True, "a contraindication is task-scoped, not a verdict"
