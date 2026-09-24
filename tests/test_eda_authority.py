"""the authority field: what a criterion's warrant is, and what that changes.

three properties are asserted rather than described:

  - a check with no declared authority is a load error, not a universal check
  - a cohort_calibrated criterion is not evaluated on a cohort that did not
    derive it - it escalates, with no default substituted
  - adding the field did not re-adjudicate the fixture cohort

the fourth test has nothing to do with authority and everything to do with the
graph staying loadable: `from __future__ import annotations` makes langgraph
resolve state annotations against module globals, so a function-local TypedDict
raises NameError - and only once a conditional edge exists, so it surfaces late.
that is asserted here rather than left as a comment.
"""
from __future__ import annotations

import json
import sys
import typing
from pathlib import Path

import pytest

from omicstra import eda
from omicstra.graphs import eda as eda_graph

AUTHORITIES = {"universal", "cohort_calibrated", "advisory"}


# a cohort-free summary: enough fields for every required check, no cohort named.
SYNTHETIC_SUMMARY = {
    "patients": 12,
    "positive_markers_pass": True,
    "negative_markers_pass": True,
    "spatial_autocorrelation_pass": True,
    "batch_correction_needed": False,
    "encoder_input_decision": "raw counts from the platform loader",
    "cross_modal_registration_pass": True,
    "model_tissue_fit": {"some_encoder": {"fit": "validated"}},
    "segmentation_qc_pass": None,
    "multi_section_alignment_pass": None,
    "verdict": "proceed",
    "risks": [],
}


@pytest.fixture
def contract():
    return eda.load_contract()


# --- the field itself ------------------------------------------------------
def test_every_check_declares_an_authority(contract):
    for spec in contract["checks"] + contract["learned_checks"]["items"]:
        assert spec["authority"] in AUTHORITIES, spec["id"]


def test_missing_authority_fails_loudly_at_load(tmp_path, contract):
    """missing must not mean permissive - an unclassified threshold is exactly
    the one that gets silently inherited."""
    broken = json.loads(json.dumps(contract))
    del broken["checks"][0]["authority"]
    (tmp_path / "eda_contract.json").write_text(json.dumps(broken))

    with pytest.raises(ValueError, match="authority"):
        eda.load_contract(tmp_path)


def test_unknown_authority_value_also_fails(tmp_path, contract):
    broken = json.loads(json.dumps(contract))
    broken["checks"][0]["authority"] = "probably_fine"
    (tmp_path / "eda_contract.json").write_text(json.dumps(broken))

    with pytest.raises(ValueError, match="authority"):
        eda.load_contract(tmp_path)


# --- the branch ------------------------------------------------------------
def test_new_cohort_escalates_rather_than_inheriting(contract):
    """a cohort with no calibration record inherits no calibrated criterion."""
    r = eda.evaluate(SYNTHETIC_SUMMARY, contract, "synthetic-cohort", calibration={})

    escalated = {c.id for c in r.checks if c.status == "escalate"}
    calibrated = {s["id"] for s in contract["checks"]
                  if s["authority"] == "cohort_calibrated"}
    assert escalated == calibrated
    assert "spatial_autocorrelation" in escalated


def test_escalation_substitutes_no_default(contract):
    r = eda.evaluate(SYNTHETIC_SUMMARY, contract, "synthetic-cohort", calibration={})
    esc = next(e for e in r.escalations if e["check"] == "spatial_autocorrelation")

    # the package default is REPORTED so the human can see what was on offer,
    # and is not applied - the check never reached _apply.
    assert esc["package_default"]["morans_i_threshold"] == 0.3
    assert esc["no_default"]
    assert esc["consequence"]
    check = next(c for c in r.checks if c.id == "spatial_autocorrelation")
    assert check.observed is None


def test_escalation_does_not_read_as_pass(contract):
    r = eda.evaluate(SYNTHETIC_SUMMARY, contract, "synthetic-cohort", calibration={})
    assert r.verdict == "proceed_with_caution"
    assert not any(c.status == "pass" and c.authority == "cohort_calibrated"
                   for c in r.checks)


def test_a_calibrated_cohort_evaluates_and_carries_provenance(contract):
    calibration = {"calibrated": {
        s["id"]: {"status": "derived", "provenance": "measured here",
                  "scope": "all samples"}
        for s in contract["checks"] if s["authority"] == "cohort_calibrated"}}
    r = eda.evaluate(SYNTHETIC_SUMMARY, contract, "synthetic-cohort", calibration)

    assert not r.escalations
    check = next(c for c in r.checks if c.id == "spatial_autocorrelation")
    assert check.status == "pass"
    assert "measured here" in check.provenance


def test_an_unaccepted_calibration_entry_still_escalates(contract):
    """recording a value is not adopting it - status decides."""
    calibration = {"calibrated": {"spatial_autocorrelation": {
        "status": "escalate", "provenance": "recorded, not accepted"}}}
    r = eda.evaluate(SYNTHETIC_SUMMARY, contract, "synthetic-cohort", calibration)

    check = next(c for c in r.checks if c.id == "spatial_autocorrelation")
    assert check.status == "escalate"


def test_advisory_never_gates(contract):
    """an advisory check fails and the verdict does not move."""
    advisory = json.loads(json.dumps(contract))
    advisory["checks"] = [dict(advisory["checks"][0], authority="advisory",
                               rule="is_true", field="a_failing_field")]
    r = eda.evaluate({**SYNTHETIC_SUMMARY, "a_failing_field": False},
                     advisory, "synthetic-cohort", calibration={})

    assert r.verdict == "proceed"
    assert not r.violations and not r.cautions
    assert any("cohort_counts" in a for a in r.advisories)


# --- the fixture cohort, unchanged -----------------------------------------
def test_fixture_verdict_is_not_re_adjudicated():
    """adding a field and a branch must not move the committed cohort."""
    r = eda.run_gate()

    assert r.verdict == "proceed_with_caution"
    assert r.declared_verdict == "proceed"
    assert not r.escalations
    assert not r.violations
    assert all(c.authority in AUTHORITIES for c in r.checks)


def test_fixture_calibrated_checks_carry_their_provenance():
    r = eda.run_gate()
    check = next(c for c in r.checks if c.id == "spatial_autocorrelation")

    assert check.status == "pass"
    assert check.provenance, "a calibrated check that passes must say on what basis"


# --- the graph -------------------------------------------------------------
def test_state_schema_is_declared_at_module_scope():
    """a function-local TypedDict raises NameError under PEP 563 - and only once
    a conditional edge is present, so it surfaces late. assert it."""
    state = eda_graph.EDAState

    assert "<locals>" not in state.__qualname__
    assert getattr(sys.modules[state.__module__], state.__name__, None) is state
    assert typing.get_type_hints(state)  # resolves against module globals


def test_graph_registry_hardcodes_no_cohort():
    text = Path(eda_graph.__file__).read_text().lower()
    for cohort_token in ("tnbc", "wang", "cn1/c1", "virchow", "novae"):
        assert cohort_token not in text, f"{cohort_token} leaked into the graph"


def test_graph_gate_escalates_a_calibrated_step(monkeypatch, tmp_path):
    """a step that PASSED, on a cohort that never derived the bar it passed
    against, does not reach the verdict as a pass."""
    monkeypatch.setattr(eda.settings, "project_dir", tmp_path)  # no calibration file
    records = [
        {"step_id": "spatial_autocorrelation", "status": "pass",
         "result": "5 of 5 above threshold", "caveats": []},
        {"step_id": "count_statistics", "status": "pass",
         "result": "integer counts", "caveats": []},
    ]
    out = eda_graph.gate({"project_id": "cohort-b", "records": records})

    assert [e["check"] for e in out["escalations"]] == ["spatial_autocorrelation"]
    assert out["verdict"] == "proceed_with_caution"
    assert eda_graph._route(out) == "escalate"
    # a step the contract declares no check for is still judged on its own status
    assert "count_statistics" not in str(out["escalations"])


def test_graph_escalation_reuses_the_gate_interrupt():
    """one interrupt, not two: escalations ride the existing escalate node."""
    import inspect
    assert "interrupt(" in inspect.getsource(eda_graph.escalate)
    assert "escalations" in inspect.getsource(eda_graph.escalate)
    # the gate routes caution -> escalate, and escalations are cautions
    assert eda_graph._route({"verdict": "proceed_with_caution"}) == "escalate"

# --- unit and coverage ------------------------------------------------------
def test_every_check_declares_what_one_answer_covers():
    """section, platform or cohort - from a closed set, never assumed.

    a cohort may carry several platforms with different answers, so collapsing a
    platform check to one status throws away the thing it measured.
    """
    from omicstra.contracts.eda import load_contract
    d = load_contract()
    allowed = set(d["units"]["values"])
    assert allowed == {"section", "platform", "cohort"}
    for c in d["checks"]:
        assert c.get("unit") in allowed, f"{c['id']} declares no unit"
        assert c.get("unit_basis"), f"{c['id']} declares a unit with no reason"


def test_only_section_checks_declare_an_aggregation():
    """platform and cohort checks produce one answer, so there is nothing to
    aggregate. a rule on them would be a knob with no meaning."""
    from omicstra.contracts.eda import load_contract
    d = load_contract()
    for c in d["checks"]:
        if c["unit"] == "section":
            assert c.get("aggregation"), f"{c['id']} is per section and declares no rule"
            assert c["aggregation"]["rule"] in set(d["aggregation"]["values"])
        else:
            assert "aggregation" not in c, f"{c['id']} is not per section but declares one"


def test_a_fraction_rule_carries_p_and_the_reason_for_it():
    """p is a package decision about sections in general.

    it is NOT fitted to a cohort. choosing p so that a recorded verdict
    reproduces would put a cohort measurement inside a package contract and make
    the acceptance circular - the cohort would match by construction.
    """
    from omicstra.contracts.eda import load_contract
    for c in load_contract()["checks"]:
        agg = c.get("aggregation") or {}
        if agg.get("rule") != "fraction":
            continue
        assert isinstance(agg.get("p"), float) and 0 < agg["p"] <= 1, f"{c['id']}: p unset"
        assert agg.get("p_basis"), f"{c['id']}: p with no warrant"


def test_the_contract_does_not_argue_with_itself_about_p():
    """the fraction clause said p was set by acceptance against a recorded verdict,
    which is the one thing the rest of the block forbids. a contract that states
    both rules teaches the next reader whichever they happen to read first."""
    from omicstra.contracts.eda import load_contract
    agg = load_contract()["aggregation"]
    assert "set by acceptance" not in agg["fraction"]
    assert "refused" in agg["fraction"]
    assert agg.get("p_is_never_fitted"), "the rule must be stated once, at the top"


def test_no_cohort_id_appears_in_an_aggregation_warrant():
    """the lint that keeps p honest. a reason naming a cohort is a fitted p."""
    import json

    from omicstra.contracts.eda import load_contract
    blob = json.dumps(load_contract()).lower()
    for token in ("tnbc", "hest", "wang"):
        assert token not in blob, f"a cohort name reached the eda contract: {token}"


def test_no_protocol_reads_a_single_assembled_object():
    """the shape the design rejected, held out by a lint.

    `adata_path` named one object that no inventory step ever produced, so the
    eda chain could not run through the graph at all. it is gone, and so is the
    --adata-path override: a flag for a rejected shape is how the shape comes
    back. this fails if either returns.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "omicstra"
    offenders = []
    for py in root.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            # a string CONSTANT naming it is prose; a subscript or attribute is use
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                    and node.slice.value == "adata_path":
                offenders.append(f"{py.relative_to(root)}: reads ctx/state['adata_path']")
    assert not offenders, "single-object path is back: " + "; ".join(offenders)


def test_the_cli_offers_no_single_object_override():
    import inspect

    from omicstra import cli
    assert "--adata-path" not in inspect.getsource(cli)
