"""the preflight gates, on declarations rather than on a person.

these exist because the gate logic is the part that must be right before any GPU
time is spent, and it is testable without a checkpointer, a thread_id or a human -
which is the whole reason it sits in protocols/ rather than in the graph.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from omicstra.protocols.encode import (
    ComputeRefused,
    assert_clear,
    contract,
    gate_record,
    preflight,
)

ROOT = Path(__file__).resolve().parents[1]
BARE = {"subject_id_column": "patient_id"}          # a cohort that declares nothing


def test_an_undeclared_cohort_leaves_every_applicable_gate_open():
    reqs = preflight(BARE, "virchow2")
    assert reqs, "virchow2 is gated; at least one gate must apply"
    assert all(r.open for r in reqs)
    assert all(r.answer is None and r.source is None for r in reqs)


def test_a_declaration_answers_its_gate_and_says_where_from():
    """a declared answer is still a HUMAN answer - the provenance travels.

    this is what stops a fully-declared unattended run from reading as though
    the system decided anything.
    """
    cohort = dict(BARE, encoder_token_source="env:HF_TOKEN")
    gw = next(r for r in preflight(cohort, "virchow2") if r.id == "gated_weights")
    assert gw.answer == "env:HF_TOKEN"
    assert gw.source == "cohort.json#encoder_token_source"
    assert not gw.open


def test_a_pointer_to_another_file_is_not_answered_from_cohort_json():
    """the pointer names its own file. a gate answerable elsewhere must not be
    silently answered from here just because a key of that name happens to exist.
    """
    from omicstra.protocols import encode

    assert encode._declared({"anything": "x"}, "platform.json#anything") is None
    assert encode._declared({"anything": "x"}, None) is None
    assert encode._declared({"anything": "x"}, "cohort.json#anything") == "x"


def test_an_ungated_encoder_is_never_asked_about_terms():
    """not applicable, not answered. recording a 'yes' would imply someone was
    asked, and nobody was."""
    ids = {r.id for r in preflight(BARE, "novae")}
    assert "gated_weights" not in ids
    assert "gated_weights" in {r.id for r in preflight(BARE, "virchow2")}


def test_the_floor_gate_is_silent_when_nothing_is_below_the_floor():
    """a gate with nothing to decide must not fire. asking a person to rule on an
    empty set is how a gate stops being read."""
    counts = {"a": 1000, "b": 900}
    assert "platform_floor" not in {r.id for r in preflight(BARE, "novae", unit_counts=counts)}


def test_the_floor_gate_carries_the_units_that_tripped_it():
    """the evidence is the point - a gate that says only 'some units are small'
    cannot be answered."""
    counts = {"big": 1000, "small": 8, "tiny": 3}
    pf = next(r for r in preflight(BARE, "novae", unit_counts=counts)
              if r.id == "platform_floor")
    assert pf.observed["floor"] == 512                  # the ENCODER's floor, not the cohort's
    assert pf.observed["n_below"] == 2
    assert set(pf.observed["below"]) == {"small", "tiny"}


def test_an_encoder_with_no_declared_floor_has_no_floor_gate():
    counts = {"a": 1, "b": 2}
    assert "platform_floor" not in {r.id for r in preflight(BARE, "virchow2", unit_counts=counts)}


def test_scope_fires_only_below_the_declared_minimum():
    counts = {"a": 100, "b": 100}
    assert "scope_below_minimum" not in {r.id for r in preflight(BARE, "virchow2",
                                                                 unit_counts=counts)}
    sc = next(r for r in preflight(BARE, "virchow2", unit_counts=counts, min_scope=500)
              if r.id == "scope_below_minimum")
    assert sc.observed == {"units": 200, "declared_minimum": 500}


def test_an_unknown_encoder_does_not_crash_the_preflight():
    """the gates must still be computable for an encoder this install cannot
    load - that is exactly when a person most needs to see them."""
    reqs = preflight(BARE, "no_such_encoder")
    assert {r.id for r in reqs} >= {"encoder_compatibility", "capacity"}


# --- refusing to compute ----------------------------------------------------
def test_compute_refuses_while_a_gate_is_open_and_names_the_fix():
    with pytest.raises(ComputeRefused) as e:
        assert_clear(preflight(BARE, "virchow2"))
    msg = str(e.value)
    assert "gated_weights" in msg
    assert "cohort.json#encoder_token_source" in msg, "the refusal must name what would answer it"
    assert "forecloses" in msg, "and what accepting gives up"


def test_compute_proceeds_once_every_gate_is_declared():
    cohort = dict(BARE, encoder_token_source="env:HF_TOKEN",
                  encoder_fallback="uni2", output_dir="data/embeddings")
    assert_clear(preflight(cohort, "virchow2"))          # must not raise


def test_the_record_reports_pending_as_not_run_not_as_failure():
    """an unanswered gate is a pending decision, not a violation. `fail` would
    put it in the ledger as a broken check."""
    r = gate_record(preflight(BARE, "virchow2"), "virchow2")
    assert r.status == "not_run"
    assert "no default" in r.decision
    assert gate_record(preflight(
        dict(BARE, encoder_token_source="env:HF_TOKEN", encoder_fallback="uni2",
             output_dir="d"), "virchow2"), "virchow2").status == "pass"


# --- against the contract, and against the cohort ---------------------------
def test_every_gate_this_protocol_emits_exists_in_the_contract():
    declared = {g["id"] for g in contract()["gates"]}
    for enc in ("virchow2", "novae"):
        assert {r.id for r in preflight(BARE, enc)} <= declared


def test_no_mid_run_gate_is_returned_by_preflight():
    mid = {g["id"] for g in contract()["gates"] if g["when"] == "mid_run"}
    assert mid, "the contract declares at least one mid-run gate"
    assert not {r.id for r in preflight(BARE, "virchow2")} & mid


def test_the_options_are_closed_and_copied_from_the_contract():
    by_id = {g["id"]: g for g in contract()["gates"]}
    for r in preflight(BARE, "virchow2"):
        assert list(r.options) == by_id[r.id]["options"]
        assert isinstance(r.options, tuple), "the option set is closed, not appendable"


def test_tnbc92_floor_gate_matches_the_recorded_cohort():
    """the real evidence, when this machine has it: Novae's 512-spot floor
    against the ingested spot counts."""
    rec = ROOT / "projects" / "tnbc-92" / "data" / "canonical" / "ingest.json"
    if not rec.is_file():
        pytest.skip("no canonical ingest on this machine")
    counts = {s: v["n_spots"] for s, v in json.loads(rec.read_text())["sources"].items()}
    cohort = json.loads((ROOT / "projects" / "tnbc-92" / "cohort.json").read_text())
    pf = [r for r in preflight(cohort, "novae", unit_counts=counts) if r.id == "platform_floor"]
    assert pf, "some tnbc-92 subarrays sit below Novae's floor - the gate must fire"
    assert pf[0].observed["floor"] == 512
    assert 0 < pf[0].observed["n_below"] < pf[0].observed["n_samples"]


# --- the fixture runs unattended -------------------------------------------
def test_tnbc92_fires_zero_preflight_gates():
    """the end-to-end fixture must run with no person in the loop.

    not because human gates are a nuisance - they are the point elsewhere - but
    because tnbc-92 is the cohort whose answers are already published. a gate
    that fires here would be asking a question the published grid already
    answered, and step 8's `omicstra run` could never complete unattended.
    """
    root = ROOT / "projects" / "tnbc-92"
    rec = root / "data" / "canonical" / "ingest.json"
    if not rec.is_file():
        pytest.skip("no canonical ingest on this machine")
    counts = {s: v["n_spots"] for s, v in json.loads(rec.read_text())["sources"].items()}
    cohort = json.loads((root / "cohort.json").read_text())

    for enc in ("virchow2", "novae"):
        reqs = preflight(cohort, enc, unit_counts=counts)
        assert reqs, f"{enc}: gates must still be COMPUTED, only pre-answered"
        assert not [r for r in reqs if r.open], \
            f"{enc}: {[r.id for r in reqs if r.open]} unanswered - cannot run unattended"
        assert all(r.source for r in reqs), "a resolved gate must carry its provenance"
        assert_clear(reqs)


def test_the_published_grid_dropped_the_below_floor_subarrays():
    """`below_floor_policy: drop` is AS BUILT, and this is the evidence.

    the novae run manifest recorded `skip_too_small_for_novae` while building the
    cache. the preflight gate recomputes the same count from the fresh canonical
    ingest against the encoder's declared floor. two independent paths, and if
    they ever disagree the declaration has stopped describing what was run.
    """
    manifest = ROOT / "data" / "embeddings" / "novae_niche_full" / "run_manifest.json"
    rec = ROOT / "projects" / "tnbc-92" / "data" / "canonical" / "ingest.json"
    if not (manifest.is_file() and rec.is_file()):
        pytest.skip("cache or canonical ingest absent on this machine")

    m = json.loads(manifest.read_text())
    counts = {s: v["n_spots"] for s, v in json.loads(rec.read_text())["sources"].items()}
    cohort = json.loads((ROOT / "projects" / "tnbc-92" / "cohort.json").read_text())

    assert cohort["below_floor_policy"] == "drop"
    recomputed = sum(1 for n in counts.values() if n < m["novae_min_spots"])
    assert recomputed == m["status_counts"]["skip_too_small_for_novae"], (
        f"the ingest says {recomputed} subarrays are below "
        f"{m['novae_min_spots']} spots; the published run dropped "
        f"{m['status_counts']['skip_too_small_for_novae']}")


def test_mark_is_declared_as_an_alternative_and_marked_not_run():
    """a road not taken is recorded with `run: false`, never silently omitted -
    the same shape as platform.json's 128um tile."""
    cohort = json.loads((ROOT / "projects" / "tnbc-92" / "cohort.json").read_text())
    alt = cohort["declared_alternatives"]["below_floor_policy"]
    assert alt["value"] == "mark" and alt["run"] is False

    # it must resolve onto the gate's closed option set, like any real answer -
    # a substring check would have passed on a value that resolves to nothing.
    from omicstra.protocols.encode import _resolve_option

    options = next(g["options"] for g in contract()["gates"] if g["id"] == "platform_floor")
    assert _resolve_option(alt["value"], tuple(options), "platform_floor") == "run_and_mark"
    assert _resolve_option(cohort["below_floor_policy"], tuple(options),
                           "platform_floor") == "drop_below_floor"


def test_a_declaration_cannot_widen_the_closed_option_set():
    """until this existed the closed set was closed on the interrupt path only.

    a declaration could put any string into `answer` and it would reach the
    ledger as though a person had picked it.
    """
    with pytest.raises(ValueError, match="option set is closed"):
        preflight(dict(BARE, below_floor_policy="keep_them"), "novae",
                  unit_counts={"tiny": 3})


def test_a_shorthand_resolves_to_exactly_one_option_or_raises():
    from omicstra.protocols.encode import _resolve_option

    opts = ("drop_below_floor", "run_and_mark", "halt")
    assert _resolve_option("drop", opts, "platform_floor") == "drop_below_floor"
    assert _resolve_option("halt", opts, "platform_floor") == "halt"
    with pytest.raises(ValueError):
        _resolve_option("run", ("run_and_mark", "run_and_drop"), "x")   # ambiguous


def test_a_value_gate_is_not_forced_onto_its_options():
    """a token source and an output path are SUPPLIED, not selected - their
    options describe the shape of the decision, not the legal answers."""
    cohort = dict(BARE, encoder_token_source="env:HF_TOKEN", output_dir="data/embeddings")
    reqs = {r.id: r for r in preflight(cohort, "virchow2")}
    assert reqs["gated_weights"].answer == "env:HF_TOKEN"
    assert reqs["capacity"].answer == "data/embeddings"


# --- the closed-set rule is declared, not hardcoded --------------------------
def test_every_gate_declares_its_answer_kind():
    """an undeclared value gate looks exactly like a closed one whose validation
    is broken, so the field is required rather than defaulted."""
    for g in contract()["gates"]:
        assert g.get("answer_kind") in {"choice", "value"}, \
            f"{g['id']} does not declare answer_kind"


def test_the_exemption_comes_from_the_contract_not_from_the_resolver():
    """the resolver must not grant itself an exemption when a match fails - that
    is indistinguishable from broken validation.
    """
    from omicstra.protocols import encode

    src = Path(encode.__file__).read_text()
    assert "_FREE_TEXT" not in src, "the exemption list is hardcoded again"

    opts = ("drop_below_floor", "run_and_mark", "halt")
    assert encode._resolve_option("anything", opts, "g", "value") == "anything"
    with pytest.raises(ValueError, match="option set is closed"):
        encode._resolve_option("anything", opts, "g", "choice")


def test_a_choice_gate_flipped_to_value_would_stop_being_validated():
    """states the consequence of the declaration, so changing it is a visible act.

    platform_floor is a choice gate; if someone marks it `value` this test says
    what they have given up rather than letting the refusal quietly disappear.
    """
    pf = next(g for g in contract()["gates"] if g["id"] == "platform_floor")
    assert pf["answer_kind"] == "choice", (
        "platform_floor decides whether 19 subarrays enter the cohort. as a value "
        "gate any string would be accepted and the funnel would record a policy "
        "that was never one of the options.")


def test_the_contract_states_the_rule_a_future_gate_author_needs():
    """the rule lives in the contract's prose, not only in this module - a gate
    author reads the contract, not the resolver."""
    rule = contract()["gate_schema"]["options_are_closed_on_both_paths"]
    assert {"rule", "why", "shorthand", "value_gates", "for_a_new_gate"} <= set(rule)
    assert "_resolve_option" in rule["enforced_by"]
