"""tier 1 for EVALUATE_STEPS: the chain that COMPUTES a verdict.

synthetic only. the point of each case is a state the vocabulary has to be able
to reach - `unguarded` and `not_run` especially, because both were once
indistinguishable from a pass.

the chain's job is to close the four guards the pack-reading gates record as
`not_run`, so several tests below assert that a guard RAN, not only that it
passed. a guard that silently does not run is the failure the guard library
exists to prevent.
"""
from __future__ import annotations

import numpy as np
import pytest

from omicstra.protocols import build_protocol
from omicstra.protocols.evaluate import EVALUATE_STEPS, VERDICTS

N_SUBJECTS, PER_SUBJECT = 6, 40
N = N_SUBJECTS * PER_SUBJECT


def _cohort(*, label_is_subject: bool = False, paired: bool = True,
            aligned: bool = True, alias_scores: bool = False,
            scores: bool = True) -> dict:
    rng = np.random.default_rng(0)
    subject = np.repeat(np.arange(N_SUBJECTS), PER_SUBJECT)
    state = np.tile(np.arange(4), N // 4)          # a label that is NOT the subject

    base = np.eye(4)[state] * 6.0 + rng.normal(0, 0.4, (N, 4))
    he = np.column_stack([base, rng.normal(0, 1, (N, 2))])
    st = (he if aligned else rng.normal(0, 1, he.shape)) + rng.normal(0, 0.2, he.shape)
    he = he / np.linalg.norm(he, axis=1, keepdims=True)
    st = st / np.linalg.norm(st, axis=1, keepdims=True)

    sc: dict[str, np.ndarray] = {}
    if scores:
        sc["signal_a"] = he[:, 0] * 3 + rng.normal(0, 0.3, N)
        # aliased: the second signal is the first plus noise, so the two axes
        # land on the same direction
        sc["signal_b"] = (sc["signal_a"] + rng.normal(0, 0.05, N) if alias_scores
                          else he[:, 1] * 3 + rng.normal(0, 0.3, N))

    ctx: dict = {
        "spaces": {"z_he": he},
        "paired": {"z_he": st} if paired else {},
        "labels": {"state": subject.copy() if label_is_subject else state},
        "confounder": subject,
        "scores": sc,
    }
    return ctx


def _params(**over) -> dict:
    p = {
        "label_granularity": {"labels": ("state",), "tau_label_nmi": 0.5},
        "paired_identifiability": {"ks": (1, 5), "negative_ratio": 5,
                                   "negative_seed": 0},
        "structure_and_its_null": {"k": 4, "n_init": 3, "seed": 42,
                                   "silhouette_sample": 100,
                                   "n_permutations": 25, "probe_max_iter": 200,
                                   "held_out_groups": 2},
        "held_out_comparison": {"ks": (1, 5), "negative_ratio": 5,
                                "negative_seed": 0, "min_units_per_subject": 10},
        "association_and_specificity": {"n_permutations": 40, "alpha": 0.05,
                                        "tau_specificity": 0.5, "tail": "two_sided",
                                        "held_out_groups": 2, "transfer_seed": 42,
                                        "within_cohort_seed": 7,
                                        "min_units_per_side": 20},
        "hypothesis_verdict": {"hypotheses": {
            "H1": ("paired_identifiability", "held_out_comparison"),
            "H2": ("structure_and_its_null",),
            "H3": ("association_and_specificity",)}},
    }
    for k, v in over.items():
        p[k] = {**p[k], **v} if isinstance(v, dict) and k in p else v
    return p


def _run(ctx: dict, params: dict | None = None) -> dict:
    return build_protocol(EVALUATE_STEPS, "evaluate").invoke(
        {**ctx, "params": params or _params()})


def _obs(recs: dict, step: str) -> dict:
    return recs[step]["observed"]


def _verdicts(recs: dict) -> dict:
    return _obs(recs, "hypothesis_verdict")["verdicts"]


# ------------------------------------------------------------ the happy path --
def test_a_well_formed_cohort_reaches_a_verdict_on_every_hypothesis():
    recs = _run(_cohort())
    assert [r["status"] for r in recs.values()] == ["pass"] * len(EVALUATE_STEPS)
    v = _verdicts(recs)
    assert set(v) == {"H1", "H2", "H3"}
    assert all(d["verdict"] in VERDICTS for d in v.values())


def test_every_step_states_both_ends_of_its_funnel():
    recs = _run(_cohort())
    for step in [s.id for s in EVALUATE_STEPS]:
        o = _obs(recs, step)
        assert "n_in" in o and "n_out" in o, f"{step} states only one end"


# ------------------------------------------------------------------ case 1 ----
def test_a_statistic_with_no_guard_on_it_is_unguarded_not_a_pass():
    """'no guard failed' was true of a hypothesis where no guard ran. that state
    now has its own name."""
    recs = _run(_cohort(), _params(hypothesis_verdict={"hypotheses": {
        # identifiability reports AUC and CKA and runs no guard of its own
        "H1": ("paired_identifiability",)}}))
    d = _verdicts(recs)["H1"]
    assert d["verdict"] == "unguarded"
    assert "not a pass" in d["why"]


def test_g5_actually_runs_on_the_clustering_score():
    """the pack-reading gate records G5 as not_run for want of a permutation
    null over ARI. the chain computes it."""
    recs = _run(_cohort())
    pairs = _obs(recs, "structure_and_its_null")["per_pair"]
    assert pairs, "no (view, label) pair was scored"
    for key, block in pairs.items():
        assert block["guard_g5"] in ("pass", "fail"), key
        assert block["null"]["n_null"] == 25
        assert block["null"]["p"] >= block["null"]["p_floor"]


# ------------------------------------------------------------------ case 2 ----
def test_a_label_that_encodes_the_subject_makes_the_score_diagnostic():
    recs = _run(_cohort(label_is_subject=True))
    assert _obs(recs, "label_granularity")["confounded"] == ["state"]
    block = next(iter(_obs(recs, "structure_and_its_null")["per_pair"].values()))
    assert block["claims_biology"] is False
    assert "encodes the confounder" in block["why_diagnostic"]
    assert _verdicts(recs)["H2"]["verdict"] == "diagnostic"


def test_the_score_is_still_reported_when_it_is_diagnostic():
    """a diagnostic number is not a withheld number - it says something else."""
    recs = _run(_cohort(label_is_subject=True))
    block = next(iter(_obs(recs, "structure_and_its_null")["per_pair"].values()))
    assert not np.isnan(block["ari"])
    assert block["probe_accuracy"] is not None


def test_the_label_is_measured_before_anything_is_scored_against_it():
    """order is the method: granularity decides what every later number claims."""
    ids = [s.id for s in EVALUATE_STEPS]
    assert ids.index("label_granularity") < ids.index("structure_and_its_null")
    by_id = {s.id: s for s in EVALUATE_STEPS}
    assert "label_guards" in by_id["structure_and_its_null"].requires


# ------------------------------------------------------------------ case 3 ----
def test_aliased_axes_make_the_cosine_travel_with_every_magnitude():
    """G4's not_run reason named a file holding no vectors. the vectors are the
    CCA's own weights."""
    recs = _run(_cohort(alias_scores=True))
    view = _obs(recs, "association_and_specificity")["per_view"]["z_he"]
    assert view["specificity_status"] == "fail"
    assert view["specificity"]["value"] > 0.5
    assert view["specificity"]["observed"]["worst_pair"] == ["signal_a", "signal_b"]
    assert _verdicts(recs)["H3"]["verdict"] == "diagnostic"


def test_distinct_signals_pass_the_specificity_guard():
    recs = _run(_cohort(alias_scores=False))
    view = _obs(recs, "association_and_specificity")["per_view"]["z_he"]
    assert view["specificity_status"] == "pass"


def test_a_single_signal_cannot_be_checked_for_specificity_and_says_so():
    ctx = _cohort()
    ctx["scores"] = {"signal_a": ctx["scores"]["signal_a"]}
    recs = _run(ctx)
    view = _obs(recs, "association_and_specificity")["per_view"]["z_he"]
    assert view["specificity_status"] == "not_run"
    assert "fewer than two axes" in view["specificity_missing"]


def test_every_association_carries_its_null_and_its_q():
    recs = _run(_cohort())
    rows = _obs(recs, "association_and_specificity")["per_view"]["z_he"]["per_signal"]
    assert len(rows) == 2
    for r in rows:
        assert r["n_null"] == 40 and r["guard_g5"] in ("pass", "fail")
        assert 0.0 < r["q"] <= 1.0 and isinstance(r["significant"], bool)


def test_the_fdr_family_is_named_in_the_record():
    """the published artifact corrected over a family the prose did not state."""
    recs = _run(_cohort())
    o = _obs(recs, "association_and_specificity")
    assert o["per_view"]["z_he"]["fdr_family"] == 2
    assert "family is the signals of ONE view" in o["fdr_family_note"]


# ------------------------------------------------------------------ case 4 ----
def test_g2_compares_a_measured_within_score_against_the_cross_score():
    """the gate records G2 as not_run for want of a within-subject score. a
    split is a declaration; the difference between the two is a measurement."""
    recs = _run(_cohort())
    v = _obs(recs, "held_out_comparison")["per_view"]["z_he"]
    assert v["guard_g2"] in ("pass", "fail")
    assert v["n_subjects_scored"] == N_SUBJECTS
    assert "within_subject_auc" in v and "cross_subject_auc" in v


def test_g2_cannot_run_without_a_cross_subject_score_to_compare():
    """the prerequisite did not apply, so the comparison is not attempted and
    says which step it was waiting on."""
    recs = _run(_cohort(paired=False))
    assert recs["paired_identifiability"]["status"] == "not_applicable"
    assert recs["held_out_comparison"]["status"] == "not_run"
    assert "paired_identifiability" in recs["held_out_comparison"]["result"]
    # and the verdict still lands: H2 and H3 are judged, H1 is not_run
    v = _verdicts(recs)
    assert v["H2"]["verdict"] in VERDICTS and v["H3"]["verdict"] in VERDICTS
    assert v["H1"]["verdict"] == "not_run"


def test_a_direct_caller_with_no_cross_score_gets_the_reason_not_a_crash():
    """the chain reaches this through prerequisite propagation; a notebook
    calling the step alone reaches the step's own branch."""
    from omicstra.protocols.evaluate import held_out_comparison

    ctx = _cohort()
    rec, _ = held_out_comparison({**ctx, "params": _params()["held_out_comparison"]})
    assert rec.status == "not_run"
    assert "nothing to compare" in rec.decision


def test_a_subject_floor_that_no_subject_meets_is_not_run_not_a_pass():
    recs = _run(_cohort(), _params(held_out_comparison={
        "ks": (1, 5), "negative_ratio": 5, "negative_seed": 0,
        "min_units_per_subject": 10_000}))
    assert recs["held_out_comparison"]["status"] == "not_run"
    v = _obs(recs, "held_out_comparison")["per_view"]["z_he"]
    assert v["guard_g2"] == "not_run" and "10000 units" in v["reason"]


# ------------------------------------------------------------------ case 5 ----
@pytest.mark.parametrize("step,field", [
    ("label_granularity", "labels"),
    ("label_granularity", "tau_label_nmi"),
    ("paired_identifiability", "ks"),
    ("paired_identifiability", "negative_ratio"),
    ("paired_identifiability", "negative_seed"),
    ("structure_and_its_null", "k"),
    ("structure_and_its_null", "n_permutations"),
    ("structure_and_its_null", "seed"),
    ("held_out_comparison", "min_units_per_subject"),
    ("association_and_specificity", "n_permutations"),
    ("association_and_specificity", "alpha"),
    ("association_and_specificity", "tau_specificity"),
    ("association_and_specificity", "tail"),
    ("association_and_specificity", "held_out_groups"),
    ("association_and_specificity", "transfer_seed"),
    ("association_and_specificity", "within_cohort_seed"),
    ("association_and_specificity", "min_units_per_side"),
    ("hypothesis_verdict", "hypotheses"),
])
def test_an_absent_declaration_refuses_naming_the_field(step, field):
    p = _params()
    p[step] = {k: v for k, v in p[step].items() if k != field}
    recs = _run(_cohort(), p)
    assert recs[step]["status"] == "not_run"
    assert field in recs[step]["result"]


def test_a_declared_label_with_no_values_is_a_missing_input_not_an_absence():
    recs = _run(_cohort(), _params(label_granularity={
        "labels": ("state", "nosuch"), "tau_label_nmi": 0.5}))
    assert recs["label_granularity"]["status"] == "not_run"
    assert "nosuch" in recs["label_granularity"]["result"]


def test_a_missing_context_key_raises_rather_than_being_invented():
    ctx = _cohort()
    del ctx["confounder"]
    recs = _run(ctx)
    assert recs["label_granularity"]["status"] == "error"
    assert "EvaluationInputMissing" in recs["label_granularity"]["result"]


# ------------------------------------------------------------------ case 6 ----
def test_a_hypothesis_whose_step_never_ran_is_not_run_not_a_silent_pass():
    """distinct from 'no guard failed'. the step that would have produced it is
    named, so the gap is readable."""
    ctx = _cohort()
    ctx["scores"] = {}
    recs = _run(ctx)
    assert recs["association_and_specificity"]["status"] == "not_applicable"
    d = _verdicts(recs)["H3"]
    assert d["verdict"] == "not_run"
    assert d["step_status"]["association_and_specificity"]


def test_the_hypothesis_mapping_is_declared_not_inferred_from_a_step_name():
    """a verdict attached to a statistic that does not test it is worse than
    no verdict."""
    recs = _run(_cohort(), _params(hypothesis_verdict={"hypotheses": {
        "my_own_question": ("structure_and_its_null",)}}))
    v = _verdicts(recs)
    assert set(v) == {"my_own_question"}
    assert v["my_own_question"]["steps"] == ["structure_and_its_null"]


def test_the_tally_covers_the_whole_vocabulary():
    recs = _run(_cohort())
    assert set(_obs(recs, "hypothesis_verdict")["tally"]) == set(VERDICTS)


# ------------------------------------------- the chain's own contract ---------
def test_the_step_that_compares_requires_the_number_it_compares_against():
    """asserted at BUILD time: a chain that put G2 before the cross score would
    raise before touching data."""
    from omicstra.protocols import ProtocolOrderError

    by_id = {s.id: s for s in EVALUATE_STEPS}
    order = ["label_granularity", "held_out_comparison", "paired_identifiability",
             "structure_and_its_null", "association_and_specificity",
             "hypothesis_verdict"]
    with pytest.raises(ProtocolOrderError, match="held_out_comparison"):
        build_protocol([by_id[i] for i in order], "reordered")


def test_the_chain_computes_no_statistic_of_its_own():
    """every number comes from measures/, callable from a notebook."""
    import inspect

    from omicstra.protocols import evaluate as E

    for fn in (E.structure_and_its_null, E.association_and_specificity,
               E.paired_identifiability, E.held_out_comparison):
        src = inspect.getsource(fn)
        assert "from omicstra.measures.evaluation import" in src, fn.__name__
        for hand_rolled in ("KMeans(", "roc_auc_score(", "np.corrcoef",
                            "adjusted_rand_score("):
            assert hand_rolled not in src, f"{fn.__name__} computes its own maths"


def test_the_step_does_not_shadow_the_guard_of_the_same_subject():
    """`GUARDS['G2']` must point at the guard, not at the step - a step that
    shadowed it would leave the registry on a different signature and the first
    symptom would be a guard that appeared to run."""
    import inspect

    from omicstra.protocols.evaluate import GUARDS, held_out_comparison

    assert GUARDS["G2"].__name__ == "held_out_honesty"
    assert list(inspect.signature(GUARDS["G2"]).parameters)[:2] == \
        ["within_score", "cross_score"]
    assert held_out_comparison.__name__ == "held_out_comparison"


def test_a_guard_that_could_not_run_is_named_in_the_verdict():
    """`unguarded` with the reason beside it. a cohort that declared too few
    permutations should be able to read why nothing was guarded."""
    recs = _run(_cohort(), _params(
        association_and_specificity={"n_permutations": 5, "alpha": 0.05,
                                     "tau_specificity": 0.5, "tail": "two_sided",
                                     "held_out_groups": 2, "transfer_seed": 42,
                                     "within_cohort_seed": 7,
                                     "min_units_per_side": 20},
        hypothesis_verdict={"hypotheses": {"H3": ("association_and_specificity",)}}))
    rows = _obs(recs, "association_and_specificity")["per_view"]["z_he"]["per_signal"]
    assert all(r["guard_g5"] == "not_run" for r in rows)
    assert "needs samples" in rows[0]["g5_not_run_because"]
    d = _verdicts(recs)["H3"]
    # G4 still ran over the two axes, so the hypothesis is not unguarded
    assert d["verdict"] in ("biological", "diagnostic")
    assert any("G5_noise_floor" in g for g in d["guards_not_run"])


def test_the_guard_minimum_lives_in_the_guard_not_in_the_step():
    """one source for one bar. the step catches the guard's refusal and reports
    it rather than carrying a copy of the number."""
    import inspect

    from omicstra.protocols import evaluate as E

    for fn in (E.structure_and_its_null, E.association_and_specificity):
        src = inspect.getsource(fn)
        assert "20" not in src, f"{fn.__name__} carries a copy of the guard's floor"
    assert "noise_floor" in inspect.getsource(E._guarded_null)


def test_every_step_that_runs_a_guard_declares_it_in_one_flat_place():
    """the verdict reads `observed["guards"]` and nothing else. a step that
    recorded a guard anywhere else would be a guard the verdict cannot see -
    which is how an aliased cohort came back `unguarded` while G4 had failed."""
    recs = _run(_cohort(alias_scores=True))
    guarded = ("label_granularity", "structure_and_its_null",
               "held_out_comparison", "association_and_specificity")
    for step in guarded:
        gs = _obs(recs, step).get("guards")
        assert gs, f"{step} ran no declared guard"
        for g in gs:
            assert set(g) >= {"guard", "scope", "status"}
            assert g["status"] in ("pass", "fail", "not_run")


def test_the_four_closed_guards_are_the_four_the_gates_could_not_run():
    """the card's done-criterion, as a test: G2, G4 and G5 all RAN."""
    recs = _run(_cohort())
    ran = {g["guard"] for step in (s.id for s in EVALUATE_STEPS)
           for g in _obs(recs, step).get("guards", [])
           if g["status"] in ("pass", "fail")}
    assert {"G2_held_out_honesty", "G4_specificity", "G5_noise_floor"} <= ran


# ------------------------------------- H3 is a TRANSFER test, not a within one --
def test_the_association_is_fitted_on_some_subjects_and_scored_on_others():
    """the split is part of the statistic. the within-cohort CCA was
    significant for every arm of the published grid at once, so it orders
    nothing - which is why the direction has to carry to subjects it was not
    fitted on."""
    recs = _run(_cohort())
    o = _obs(recs, "association_and_specificity")
    assert o["n_train"] + o["n_test"] == N
    assert len(o["held_out_subjects"]) == 2
    assert "projected onto held-out subjects" in o["statistic"]
    row = o["per_view"]["z_he"]["per_signal"][0]
    assert "r_test" in row and "r_train" in row


def test_the_within_cohort_statistic_is_kept_and_labelled_non_discriminating():
    """the published artifact stores both; a reader comparing them should not
    have to re-derive why only one separates the arms."""
    recs = _run(_cohort())
    row = _obs(recs, "association_and_specificity")["per_view"]["z_he"]["per_signal"][0]
    w = row["within_cohort"]
    assert "non-discriminating" in w["note"]
    assert "r" in w and "z" in w and "p" in w


def test_a_signal_that_only_fits_the_training_subjects_does_not_transfer():
    """the case the within-cohort test cannot see: a direction with a high
    train correlation and a held-out correlation at chance."""
    rng = np.random.default_rng(2)
    ctx = _cohort()
    subject = ctx["confounder"]
    held = set(np.random.default_rng(42).choice(np.unique(subject), size=2,
                                                replace=False))
    noise = rng.normal(0, 1, N)
    # the signal exists only on the TRAINING subjects
    ctx["scores"] = {"train_only": np.where([c in held for c in subject],
                                            noise, ctx["spaces"]["z_he"][:, 0] * 5),
                     "other": rng.normal(0, 1, N)}
    recs = _run(ctx)
    rows = {r["signal"]: r
            for r in _obs(recs, "association_and_specificity")["per_view"]["z_he"]["per_signal"]}
    assert abs(rows["train_only"]["r_train"]) > abs(rows["train_only"]["r_test"])


def test_a_cohort_with_too_few_subjects_to_hold_any_out_refuses():
    ctx = _cohort()
    ctx["confounder"] = np.zeros(N, dtype=int)
    recs = _run(ctx)
    assert recs["association_and_specificity"]["status"] == "not_run"
    assert "cannot yield a held-out split" in recs["association_and_specificity"]["result"]
    assert "within-cohort test under another name" in \
        recs["association_and_specificity"]["decision"]


def test_a_split_that_leaves_too_few_units_on_a_side_refuses():
    recs = _run(_cohort(), _params(association_and_specificity={
        "n_permutations": 25, "alpha": 0.05, "tau_specificity": 0.5,
        "tail": "two_sided", "held_out_groups": 2, "transfer_seed": 42,
        "within_cohort_seed": 7, "min_units_per_side": 10_000}))
    assert recs["association_and_specificity"]["status"] == "not_run"
    assert "noise with a p-value" in recs["association_and_specificity"]["decision"]


def test_the_two_nulls_have_separate_declared_seeds():
    """they are different tests, not variants of one, so one seed for both
    would make the within-cohort null a function of the transfer null's."""
    by_id = {s.id: s for s in EVALUATE_STEPS}
    space = by_id["association_and_specificity"].params_space
    assert "transfer_seed" in space and "within_cohort_seed" in space


def test_the_transfer_split_is_the_same_for_every_signal_and_every_view():
    """a split that varied by signal would make the comparison meaningless
    before the statistic was chosen."""
    ctx = _cohort()
    ctx["spaces"] = {"z_he": ctx["spaces"]["z_he"],
                     "z_other": ctx["spaces"]["z_he"][:, ::-1].copy()}
    recs = _run(ctx)
    o = _obs(recs, "association_and_specificity")
    assert set(o["per_view"]) == {"z_he", "z_other"}
    assert o["n_test"] > 0 and len(o["held_out_subjects"]) == 2
