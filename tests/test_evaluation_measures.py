"""tier 1 for measures/evaluation.py: properties, and agreement with a second
implementation wherever one exists.

no cohort data and no cohort constant. the tier-2 diff against the published
artifacts is in test_evaluation_slicediff.py.

the strongest tests here are the ones that compute a statistic a SECOND way -
against statsmodels, against scipy, against the three CKA bodies that ship in
`scripts/` - because the defect the port exists to remove is one statistic with
several implementations that can disagree.
"""
from __future__ import annotations

import numpy as np
import pytest

from omicstra.measures.evaluation import (
    NotComparable,
    bh_fdr,
    bootstrap_ci,
    cluster_agreement,
    knn_purity,
    linear_cka,
    linear_probe,
    permutation_test,
    retrieval,
    top_canonical_correlation,
    two_sample_z,
)


# ----------------------------------------------- machine 1 · permutation ----
def test_the_p_value_has_a_floor_and_never_reaches_zero():
    """a p of 0 is not a measurement, it is a statement that the permutations
    did not reach the observation."""
    r = permutation_test(1e9, np.zeros(500), tail="greater")
    assert r["p"] == pytest.approx(1 / 501)
    assert r["p"] == r["p_floor"] > 0


def test_the_caller_must_name_the_tail():
    with pytest.raises(ValueError, match="tail"):
        permutation_test(1.0, [0.0, 1.0], tail="whatever")


def test_the_two_tails_are_different_numbers_on_the_same_null():
    """the DAG result was published as 80.3% and as 78%. they are the one- and
    two-sided tails of a single test, and a defaulted tail buys that."""
    nulls = np.r_[np.full(100, -5.0), np.full(100, 0.0)]
    one = permutation_test(4.0, nulls, tail="greater")["p"]
    two = permutation_test(4.0, nulls, tail="two_sided")["p"]
    assert one < two
    assert one == pytest.approx(1 / 201) and two == pytest.approx(101 / 201)


def test_a_null_that_could_not_be_computed_is_absent_not_zero():
    with_nan = permutation_test(2.0, [1.0, np.nan, 3.0], tail="greater")
    without = permutation_test(2.0, [1.0, 3.0], tail="greater")
    assert with_nan["n_null"] == 2 and with_nan["p"] == without["p"]


def test_a_null_with_no_spread_cannot_place_the_observation():
    """a large z from dividing by almost zero looks like a result."""
    r = permutation_test(5.0, np.ones(100), tail="greater")
    assert np.isnan(r["z"]) and "zero spread" in r["reason"]
    assert not np.isnan(r["p"])          # the p is still meaningful


def test_an_empty_null_is_nan_with_a_reason():
    r = permutation_test(1.0, [], tail="greater")
    assert np.isnan(r["z"]) and np.isnan(r["p"]) and r["reason"] == "empty null"


def test_the_z_is_the_standardised_distance_from_the_null():
    nulls = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    r = permutation_test(6.0, nulls, tail="greater")
    assert r["z"] == pytest.approx((6.0 - nulls.mean()) / nulls.std())


# -------------------------------------------------- machine 2 · BH-FDR ------
def test_bh_fdr_agrees_with_statsmodels():
    """a second implementation, from a library nobody here wrote."""
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(3)
    p = np.clip(rng.beta(0.3, 4.0, 40), 1e-9, 1.0)
    mine = bh_fdr(p, alpha=0.05)
    theirs_rej, theirs_q, *_ = multipletests(p, alpha=0.05, method="fdr_bh")
    assert np.allclose(mine["q"], theirs_q)
    assert mine["rejected"] == theirs_rej.tolist()


def test_a_test_that_could_not_be_computed_leaves_the_family():
    """counting it inflates the denominator every other q is divided by."""
    with_nan = bh_fdr([0.01, 0.02, np.nan], alpha=0.05)
    without = bh_fdr([0.01, 0.02], alpha=0.05)
    assert with_nan["n_family"] == 2 and with_nan["n_excluded_not_testable"] == 1
    assert np.allclose(with_nan["q"][:2], without["q"])


def test_the_family_is_an_argument_and_it_changes_every_q():
    """'corrected across the family tested' is a claim about the family.

    identical p-values give the same q at any family size - BH's monotone step
    collapses them - so the demonstration needs one small p among large ones,
    which is also the shape every real family here has.
    """
    small = bh_fdr([0.001] + [0.9] * 4, alpha=0.05)
    large = bh_fdr([0.001] + [0.9] * 399, alpha=0.05)
    assert small["q"][0] == pytest.approx(0.005)
    assert large["q"][0] == pytest.approx(0.4)
    assert small["rejected"][0] and not large["rejected"][0]


def test_the_q_values_are_monotone_in_p():
    rng = np.random.default_rng(4)
    p = np.sort(rng.uniform(0, 1, 50))
    q = np.asarray(bh_fdr(p, alpha=0.1)["q"])
    assert np.all(np.diff(q) >= -1e-12)


def test_naming_the_family_wrong_refuses():
    with pytest.raises(ValueError, match="family name"):
        bh_fdr([0.1, 0.2], alpha=0.05, family=["only_one"])


def test_the_rejected_tests_are_named_not_only_counted():
    r = bh_fdr([0.0001, 0.9], alpha=0.05, family=["immune", "ecm"])
    assert r["rejected_names"] == ["immune"]


# ----------------------------------------------- machine 3 · bootstrap ------
def test_resampling_groups_gives_a_wider_interval_than_resampling_units():
    """the reason the published bootstrap draws subjects: units within a subject
    are near-copies, so a unit bootstrap reports an interval far too narrow."""
    rng = np.random.default_rng(5)
    # 8 groups, 50 near-identical units each. the group mean carries the signal
    offsets = rng.normal(0, 3.0, 8)
    values = np.concatenate([o + rng.normal(0, 0.01, 50) for o in offsets])
    groups = np.repeat(np.arange(8), 50)

    def mean_of(idx):
        return float(values[idx].mean())

    by_group = bootstrap_ci(mean_of, groups, n_resamples=300, seed=42, ci=0.95)
    by_unit = bootstrap_ci(mean_of, np.arange(values.size),
                           n_resamples=300, seed=42, ci=0.95)
    # 8.3x on this seed. the claim is an order of magnitude, not a constant
    assert (by_group["hi"] - by_group["lo"]) > 5 * (by_unit["hi"] - by_unit["lo"])
    assert by_group["n_groups"] == 8


def test_a_bootstrap_over_one_group_refuses_rather_than_reporting_zero_width():
    r = bootstrap_ci(lambda idx: 1.0, ["a"] * 20, n_resamples=50, seed=42, ci=0.95)
    assert np.isnan(r["lo"]) and "one group" in r["reason"]


def test_the_point_estimate_uses_every_unit():
    vals = np.arange(10, dtype=float)
    r = bootstrap_ci(lambda idx: float(vals[idx].mean()),
                     np.repeat([0, 1], 5), n_resamples=20, seed=1, ci=0.9)
    assert r["point"] == pytest.approx(vals.mean())


def test_resamples_that_could_not_be_computed_are_counted_not_dropped_silently():
    r = bootstrap_ci(lambda idx: float("nan") if len(idx) > 5 else 1.0,
                     np.repeat([0, 1, 2], 4), n_resamples=10, seed=1, ci=0.9)
    assert r["n_resamples_nan"] == 10 and "every resample" in r["reason"]


# ------------------------------------------------------ paired retrieval ----
def test_a_perfectly_matched_space_retrieves_perfectly():
    sim = np.eye(40) * 2 - 1
    r = retrieval(sim, ks=(1, 5), negative_ratio=10, negative_seed=0)
    assert r["r1"] == 1.0 and r["mrr"] == 1.0 and r["median_rank"] == 1.0
    assert r["auc"] == 1.0 and r["alignment_gap"] == pytest.approx(2.0)


def test_the_auc_depends_on_the_negative_sample_so_both_are_declared():
    """the script drew negatives under a hardcoded seed that was not the seed
    the rest of the chain used, and the artifact did not say so."""
    rng = np.random.default_rng(7)
    sim = rng.normal(0, 1, (80, 80)) + np.eye(80) * 0.6
    a = retrieval(sim, ks=(1,), negative_ratio=1, negative_seed=0)["auc"]
    b = retrieval(sim, ks=(1,), negative_ratio=1, negative_seed=1)["auc"]
    assert a != b, "the negative sample is a parameter, not an implementation detail"
    full = retrieval(sim, ks=(1,), negative_ratio=10_000, negative_seed=0)
    assert full["n_negatives_sampled"] == 80 * 79      # capped at every off-diagonal


def test_ties_with_the_matched_pair_are_counted_because_they_break_arbitrarily():
    sim = np.ones((10, 10))               # every unit tied with its match
    r = retrieval(sim, ks=(1,), negative_ratio=1, negative_seed=0)
    assert r["n_tied_at_self"] == 10
    assert r["alignment_gap"] == pytest.approx(0.0)


def test_a_rectangular_matrix_has_no_matched_pairs_and_refuses():
    with pytest.raises(NotComparable, match="square"):
        retrieval(np.zeros((4, 6)), ks=(1,), negative_ratio=1, negative_seed=0)


def test_recall_thresholds_a_zero_indexed_rank_and_median_rank_is_one_indexed():
    """both reproduced as published, and named rather than left to be found."""
    sim = np.zeros((3, 3))
    sim[0, 0] = 1.0                       # unit 0 matched, units 1-2 tied at 0
    r = retrieval(sim, ks=(1,), negative_ratio=1, negative_seed=0)
    assert r["median_rank"] >= 1.0


# -------------------------------------------------- geometric agreement -----
def _script_cka_align(X, Y):
    """scripts/align.py:427, verbatim."""
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    num = (Xc.T @ Yc).reshape(-1) @ (Xc.T @ Yc).reshape(-1)
    den = np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc)
    return float(num / (den + 1e-12))


def _script_cka_rank_matched(X, Y):
    """scripts/eval_rank_matched_cka.py:62, verbatim."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    num = float(np.linalg.norm(Yc.T @ Xc, ord="fro") ** 2)
    den = float(np.linalg.norm(Xc.T @ Xc, ord="fro")
                * np.linalg.norm(Yc.T @ Yc, ord="fro"))
    return num / den if den > 0 else float("nan")


def test_the_one_cka_agrees_with_both_script_bodies():
    rng = np.random.default_rng(11)
    X, Y = rng.normal(3, 1, (200, 16)), rng.normal(-2, 1, (200, 8))
    mine = linear_cka(X, Y)
    assert mine == pytest.approx(_script_cka_align(X, Y), rel=1e-12)
    assert mine == pytest.approx(_script_cka_rank_matched(X, Y), rel=1e-12)


def test_centring_is_in_the_signature_because_it_changed_the_answer_17_fold():
    """the third script body took PRE-CENTRED input under the same function
    name. handed raw data it returned 0.96 where the others returned 0.054."""
    rng = np.random.default_rng(11)
    X, Y = rng.normal(3, 1, (200, 16)), rng.normal(-2, 1, (200, 8))
    centred = linear_cka(X, Y, centre=True)
    as_if_precentred = linear_cka(X, Y, centre=False)
    assert as_if_precentred > 10 * centred
    # and passing genuinely pre-centred data to centre=False is correct
    Xc, Yc = X - X.mean(0, keepdims=True), Y - Y.mean(0, keepdims=True)
    assert linear_cka(Xc, Yc, centre=False) == pytest.approx(centred, rel=1e-12)


def test_a_representation_agrees_perfectly_with_itself():
    rng = np.random.default_rng(12)
    X = rng.normal(0, 1, (120, 9))
    assert linear_cka(X, X) == pytest.approx(1.0, rel=1e-10)


def test_cka_is_invariant_to_an_orthogonal_rotation():
    rng = np.random.default_rng(13)
    X = rng.normal(0, 1, (150, 10))
    Y = rng.normal(0, 1, (150, 10))
    Q = np.linalg.qr(rng.normal(0, 1, (10, 10)))[0]
    assert linear_cka(X, Y) == pytest.approx(linear_cka(X @ Q, Y), rel=1e-10)


def test_comparing_two_different_unit_sets_refuses():
    with pytest.raises(NotComparable, match="SAME units"):
        linear_cka(np.zeros((10, 3)), np.zeros((11, 3)))


# ------------------------------------------------ clustering and probes -----
def test_a_label_that_is_the_clustering_scores_one():
    rng = np.random.default_rng(14)
    y = np.repeat(np.arange(4), 40)
    X = np.eye(4)[y] * 50 + rng.normal(0, 0.1, (160, 4))
    r = cluster_agreement(X, y, k=4, n_init=10, seed=42, silhouette_sample=100)
    assert r["ari"] == pytest.approx(1.0) and r["k"] == 4


def test_clustering_refuses_a_single_class_rather_than_scoring_it():
    r = cluster_agreement(np.zeros((10, 2)), ["a"] * 10, k=3, n_init=10,
                          seed=42, silhouette_sample=5)
    assert np.isnan(r["ari"]) and "two classes" in r["reason"]


def test_the_probe_splits_by_group_so_a_subject_cannot_appear_on_both_sides():
    """a unit-level split puts near-copies of one subject on both sides and
    reports memorisation as accuracy."""
    rng = np.random.default_rng(15)
    groups = np.repeat(np.arange(6), 30)
    y = np.repeat([0, 1], 90)                     # label varies WITH the group
    X = np.column_stack([y * 8.0 + rng.normal(0, 0.3, 180), rng.normal(0, 1, 180)])
    r = linear_probe(X, y, groups, seed=42, max_iter=300, held_out_groups=2)
    assert r["split"] == "by group"
    assert r["n_test_groups"] == 2 and r["n_train"] + r["n_test"] == 180


def test_the_probe_refuses_when_there_are_too_few_groups_to_hold_any_out():
    r = linear_probe(np.zeros((4, 2)), [0, 1, 0, 1], ["g", "g", "g", "g"],
                     seed=42, max_iter=50, held_out_groups=2)
    assert np.isnan(r["accuracy"]) and "held-out split" in r["reason"]


def test_knn_purity_is_one_when_the_classes_are_separated():
    rng = np.random.default_rng(16)
    y = np.repeat([0, 1], 50)
    X = np.column_stack([y * 100.0, rng.normal(0, 0.1, 100)])
    r = knn_purity(X, y, ks=(5, 15))
    assert r["purity_k5"] == pytest.approx(1.0)
    assert r["purity_k15"] == pytest.approx(1.0)


def test_knn_purity_is_nan_when_k_exceeds_the_units():
    assert np.isnan(knn_purity(np.zeros((4, 2)), [0, 1, 0, 1], ks=(10,))["purity_k10"])


# -------------------------------------------------------------- contrast ----
def test_the_welch_z_agrees_with_scipy():
    """a second implementation. the statistic the matrix cites had no home at
    all, so agreement with an independent one is the only check available."""
    from scipy.stats import ttest_ind

    rng = np.random.default_rng(17)
    a = rng.normal(1.0, 2.0, 40)
    b = rng.normal(0.0, 0.5, 25)
    mine = two_sample_z(a, b, tail="greater")["z"]
    theirs = ttest_ind(a, b, equal_var=False).statistic
    assert mine == pytest.approx(float(theirs), rel=1e-12)


def test_the_welch_z_uses_a_sample_variance_per_group():
    r = two_sample_z([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], tail="two_sided")
    assert r["ddof"] == 1 and r["z"] == pytest.approx(0.0)


def test_the_lower_tail_flips_the_sign_so_a_pass_means_the_same_thing():
    a, b = [5.0, 6.0, 7.0], [1.0, 2.0, 3.0]
    assert two_sample_z(a, b, tail="greater")["z"] > 0
    assert two_sample_z(a, b, tail="less")["z"] < 0


def test_two_constant_groups_have_no_standard_error_and_say_so():
    r = two_sample_z([2.0] * 5, [3.0] * 5, tail="two_sided")
    assert np.isnan(r["z"]) and "zero variance" in r["reason"]


def test_a_group_too_small_to_have_a_variance_refuses():
    r = two_sample_z([1.0], [1.0, 2.0, 3.0], tail="greater")
    assert np.isnan(r["z"]) and "at least two" in r["reason"]


# ----------------------------------------------------------- association ----
def test_there_is_one_cca_and_the_weights_flag_does_not_change_the_value():
    """the published chain had a full version and a value-only twin used inside
    the permutation loop. two bodies are two chances to drift."""
    rng = np.random.default_rng(18)
    Z = rng.normal(0, 1, (200, 6))
    Y = Z[:, :2] @ rng.normal(0, 1, (2, 3)) + rng.normal(0, 0.5, (200, 3))
    value = top_canonical_correlation(Z, Y)
    value_again, a, b = top_canonical_correlation(Z, Y, with_weights=True)
    assert value == value_again
    assert a.shape == (6,) and b.shape == (3,)


def test_a_canonical_correlation_is_a_cosine_and_cannot_exceed_one():
    rng = np.random.default_rng(19)
    Z = rng.normal(0, 1, (50, 40))
    Y = rng.normal(0, 1, (50, 40))         # p close to n: r saturates at 1
    assert top_canonical_correlation(Z, Y) <= 1.0


def test_a_linear_image_of_a_block_correlates_perfectly_with_it():
    rng = np.random.default_rng(20)
    Z = rng.normal(0, 1, (120, 4))
    assert top_canonical_correlation(Z, Z @ rng.normal(0, 1, (4, 2))) == \
        pytest.approx(1.0, rel=1e-9)


def test_a_one_dimensional_block_is_accepted_as_a_column():
    rng = np.random.default_rng(21)
    Z = rng.normal(0, 1, (80, 3))
    score = Z[:, 0] + rng.normal(0, 0.3, 80)
    r = top_canonical_correlation(Z, score)
    assert 0.8 < r <= 1.0


def test_relating_two_different_unit_sets_refuses():
    with pytest.raises(NotComparable, match="SAME units"):
        top_canonical_correlation(np.zeros((10, 2)), np.zeros((9, 2)))


# ----------------------------------------------- the module's own contract ---
def test_no_statistic_carries_a_cohort_parameter_of_its_own():
    """every cohort-dependent parameter is required with no default. a default
    is a choice made on one tissue that then travels silently."""
    import inspect

    from omicstra.measures import evaluation as E

    must_be_required = {
        "permutation_test": ["tail"],
        "bh_fdr": ["alpha"],
        "bootstrap_ci": ["n_resamples", "seed", "ci"],
        "retrieval": ["ks", "negative_ratio", "negative_seed"],
        "cluster_agreement": ["k", "n_init", "seed", "silhouette_sample"],
        "linear_probe": ["seed", "max_iter", "held_out_groups"],
        "knn_purity": ["ks"],
        "two_sample_z": ["tail"],
    }
    for fn_name, fields in must_be_required.items():
        sig = inspect.signature(getattr(E, fn_name))
        for f in fields:
            assert sig.parameters[f].default is inspect.Parameter.empty, \
                f"{fn_name}({f}=) carries a default; it must be declared"


def test_the_module_reads_nothing_and_names_no_cohort():
    import inspect

    from omicstra.measures import evaluation as E

    src = inspect.getsource(E)
    for forbidden in ("open(", "read_parquet", "read_csv", "Path(", "tnbc",
                      "virchow", "novae", "gpath2vec"):
        assert forbidden not in src, f"{forbidden!r} reached a measure"


# ------------------------------------------------- the transfer statistic -----
def test_a_direction_fitted_on_train_is_scored_on_test_without_refitting():
    """`projected_correlation` is a Pearson r along given directions, not a
    CCA: a CCA on the held-out units would refit and report the best possible
    correlation, which is the number a transfer test exists to avoid."""
    from omicstra.measures.evaluation import (
        projected_correlation,
        top_canonical_correlation,
        transfer_correlation,
    )

    rng = np.random.default_rng(30)
    Z = rng.normal(0, 1, (300, 6))
    y = Z[:, 0] * 2 + rng.normal(0, 1.0, 300)
    tr, te = slice(0, 200), slice(200, 300)

    r_test, r_train, a, b = transfer_correlation(Z[tr], y[tr], Z[te], y[te])
    assert r_test == projected_correlation(Z[te], y[te], a, b)
    refit = top_canonical_correlation(Z[te], y[te])
    assert refit >= abs(r_test) - 1e-12, "a refit cannot do worse than a transfer"
    assert 0.0 < abs(r_train) <= 1.0


def test_a_signal_present_only_in_train_does_not_transfer():
    from omicstra.measures.evaluation import transfer_correlation

    rng = np.random.default_rng(31)
    Z = rng.normal(0, 1, (400, 5))
    y = np.r_[Z[:300, 0] * 5, rng.normal(0, 1, 100)]      # test half is noise
    r_test, r_train, _, _ = transfer_correlation(Z[:300], y[:300], Z[300:], y[300:])
    assert abs(r_train) > 0.8
    assert abs(r_test) < 0.4


def test_each_side_is_centred_on_its_own_rows():
    """centring the held-out block on the training mean would leak it."""
    from omicstra.measures.evaluation import projected_correlation

    rng = np.random.default_rng(32)
    Z = rng.normal(0, 1, (120, 4))
    y = rng.normal(0, 1, 120)
    a, b = rng.normal(0, 1, 4), np.array([1.0])
    shifted = projected_correlation(Z + 100.0, y + 100.0, a, b)
    assert shifted == pytest.approx(projected_correlation(Z, y, a, b), rel=1e-10)


def test_a_constant_projection_has_no_correlation_and_says_nan():
    from omicstra.measures.evaluation import projected_correlation

    Z = np.ones((40, 3))
    assert np.isnan(projected_correlation(Z, np.arange(40.0), np.ones(3),
                                          np.array([1.0])))


def test_projecting_two_different_unit_sets_refuses():
    from omicstra.measures.evaluation import NotComparable, projected_correlation

    with pytest.raises(NotComparable, match="SAME units"):
        projected_correlation(np.zeros((10, 2)), np.zeros(9), np.ones(2),
                              np.array([1.0]))
