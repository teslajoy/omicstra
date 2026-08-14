"""routing: the rules, not the numbers.

these assert on the SHAPE a decision takes - that a tie declines to pick, that a
contraindication never enters a route, that disagreeing axes escalate. the cohort
values are fixtures; if tnbc-92's evidence is re-derived these should still pass,
because what is under test is the contract.

log=False throughout - a test must not write to the cohort's decision ledger.
"""
from __future__ import annotations

import pytest

from omicstra.routing import (
    list_task_families,
    load_routing_contract,
    load_routing_evidence,
    resolve,
)

CONTRACT = load_routing_contract()
EVIDENCE = load_routing_evidence("tnbc-92")
FAMILIES = [f["id"] for f in CONTRACT["task_families"]]


def route(task_id, **kw):
    return resolve(task_id, project_id="tnbc-92", log=False, **kw)


# --- the contract is well formed -------------------------------------------
def test_every_family_declares_a_level_and_hypothesis():
    for f in CONTRACT["task_families"]:
        assert f["level"] and f["hypothesis"], f["id"]


def test_scope_statement_present_and_travels_into_every_record():
    scope = EVIDENCE["scope_statement"]
    assert scope
    for tid in EVIDENCE["tasks"]:
        assert scope in route(tid).caveats, f"{tid} dropped the scope statement"


def test_families_without_evidence_are_listed_not_hidden():
    fams = list_task_families(project_id="tnbc-92")["families"]
    assert {f["task_id"] for f in fams} == set(FAMILIES)
    assert all("routable_on_this_cohort" in f for f in fams)


# --- resolution shape ------------------------------------------------------
@pytest.mark.parametrize("task_id", sorted(EVIDENCE["tasks"]))
def test_every_recorded_task_resolves(task_id):
    r = route(task_id)
    assert r.kind == "selection"
    assert r.record_id
    assert r.why, f"{task_id} resolved with no stated reason"


def test_a_tie_declines_to_pick():
    r = route("tissue_state_grouping")
    assert r.tie and r.chosen is None


def test_a_recommendation_names_exactly_one():
    r = route("cross_modal_retrieval")
    assert not r.tie and r.chosen


def test_this_cohort_clears_its_reference_on_tissue_state_grouping():
    """raw H&E alone scores 0.164; the aligned runs clear it. no shortfall here."""
    r = route("tissue_state_grouping")
    assert not r.why.startswith("raw_he scores"), r.why[:120]


def test_a_leader_below_the_reference_says_so_first():
    """the rule, on synthetic evidence - no cohort currently triggers it.

    the shortfall sentence must lead, not trail. a reader who stops after the
    first clause should still know the un-aligned modality did better.
    """
    ev = {
        "project_id": "synthetic", "evidence_version": "test",
        "scope_statement": "synthetic fixture",
        "method_names": {"M1": "method one", "M2": "method two"},
        "tasks": {"tissue_state_grouping": {
            "metric": "toy metric", "source": "none", "higher_is_better": True,
            "floor": {"id": "nothing", "value": 0.0},
            "reference": {"id": "raw_modality", "value": 0.9},
            "candidates": [{"id": "M1", "value": 0.5}, {"id": "M2", "value": 0.2}]}},
    }
    r = resolve("tissue_state_grouping", evidence=ev, log=False)
    assert r.why.startswith("raw_modality scores 0.9"), r.why[:90]
    assert r.chosen == "method one"


# --- contraindication ------------------------------------------------------
def test_naming_a_contraindicated_method_refuses():
    r = route("subject_identity_suppression", proposed_method="B1_v3")
    assert r.contraindicated and r.chosen is None and r.actor == "deterministic"


def test_override_proceeds_but_records_dissent():
    r = route("subject_identity_suppression", proposed_method="B1_v3", override=True)
    assert r.contraindicated and r.chosen and r.actor == "human"
    assert any("OVERRIDE" in c for c in r.caveats)


def test_a_contraindicated_method_never_enters_a_route():
    banned = {c["method"] for c in EVIDENCE["contraindications"]
              if "tissue_state_grouping" in c["applies_to"]}
    names = EVIDENCE["method_names"]
    r = route("tissue_state_grouping")
    assert not ({names[b] for b in banned} & {r.chosen}) - {None}


def test_contraindication_is_task_scoped_not_a_verdict_on_the_method():
    """CCA is refused for cohort biology and passes pathway transfer."""
    assert route("subject_identity_suppression", proposed_method="B1_v3").contraindicated
    assert not route("pathway_transfer", proposed_method="B1_v3").contraindicated


# --- the multi-metric rule -------------------------------------------------
def test_agreeing_axes_recommend():
    r = route("structural_agreement")
    assert not r.tie and r.chosen


def test_disagreeing_axes_escalate_rather_than_picking_one():
    r = route("pathway_discrimination")
    assert r.tie and r.chosen is None
    assert "different leaders" in r.why


# --- ids -------------------------------------------------------------------
def test_record_id_is_stable_across_wording():
    assert route("cross_modal_retrieval", question="one phrasing").record_id == \
           route("cross_modal_retrieval", question="an entirely other phrasing").record_id


def test_record_id_separates_tasks_and_overrides():
    base = route("subject_identity_suppression").record_id
    assert base != route("cross_modal_retrieval").record_id
    assert base != route("subject_identity_suppression",
                         proposed_method="B1_v3", override=True).record_id


# --- refusing to guess -----------------------------------------------------
def test_unknown_task_is_not_routed():
    r = route("no_such_family")
    assert r.status == "not_run" and r.chosen is None


def test_a_cohort_without_evidence_is_not_routable():
    r = resolve("cross_modal_retrieval", evidence={}, log=False)
    assert r.status == "not_run" and r.chosen is None
    assert "not inherited" in r.why or "has not been evaluated" in r.why