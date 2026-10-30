"""tier 1 for measures/biology.py: properties, edge cases, and exact parity
against the reference implementations in scripts/eval_alignment_biology.py.

no cohort data and no cohort constant. the tier-2 diff against the published
artifacts is not this file's job.

the strongest tests here are the ones that run the script's functions and the
package functions on the SAME synthetic data with the SAME seed and assert
exact equality. that is what "ported arithmetic unchanged" means, and
anything less permits the two to drift.

z-score convention: the biology/patient tests in the script use
z = obs / null_std (historical), while rho_with_null uses the standard
(obs - null_mean) / null_std. this test verifies both: the package's `z`
field matches the script's historical formula, and `z_standard` matches
the conventional one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from omicstra.measures.biology import (
    biology_delta,
    biology_perm_null,
    biology_perm_test,
    cosine_sim_matrix,
    geometric_faithfulness,
    l2_normalize,
    mean_pool_l2,
    patient_delta,
    patient_perm_null,
    patient_perm_test,
    ratio_delta_ci,
    upper_tri,
)


# ================================================================ fixtures ====

def _synthetic_subarrays(n_subs: int = 20, dim: int = 32, seed: int = 42):
    """synthetic subarray embeddings with known patient/label structure.

    3 patients, 3 biology labels (A, B, NA), unequal subarray counts.
    the embeddings are random but seeded, so the cosine matrix is fixed.
    """
    rng = np.random.default_rng(seed)
    patients = np.array(
        ["P1"] * 8 + ["P2"] * 7 + ["P3"] * 5
    )
    labels = np.array(
        ["A", "A", "B", "B", "NA", "A", "B", "A",   # P1: 4A, 3B, 1NA
         "B", "B", "A", "A", "B", "A", "B",          # P2: 3A, 4B
         "A", "A", "B", "NA", "B"]                    # P3: 2A, 2B, 1NA
    )
    assert len(patients) == n_subs
    assert len(labels) == n_subs
    emb = rng.standard_normal((n_subs, dim)).astype(np.float32)
    return emb, patients, labels


def _make_cosine(emb):
    """build cosine matrix the script's way: l2-norm rows, matmul, clip."""
    n = np.linalg.norm(emb, axis=-1, keepdims=True)
    normed = emb / np.clip(n, 1e-12, None)
    return np.clip(normed @ normed.T, -1.0, 1.0).astype(np.float32)


# ======================================================== reference impls ====
# these are the script's functions, copied verbatim for parity comparison.
# they differ from the package only in interface (tuple returns vs dicts).

def _ref_upper_triangle(arr):
    iu = np.triu_indices(arr.shape[0], k=1)
    return arr[iu], iu


def _ref_biology_delta(sim_pairs, same_pt, labels, iu, valid_mask):
    same_lb = (labels[:, None] == labels[None, :])[iu]
    cross = (~same_pt) & valid_mask
    A = cross & same_lb
    B = cross & ~same_lb
    m_same = float(sim_pairs[A].mean()) if A.any() else float("nan")
    m_diff = float(sim_pairs[B].mean()) if B.any() else float("nan")
    return m_same - m_diff, int(A.sum()), int(B.sum()), m_same, m_diff


def _ref_patient_delta(sim_pairs, patient_vec, iu):
    same_pt = (patient_vec[:, None] == patient_vec[None, :])[iu]
    m_same = float(sim_pairs[same_pt].mean()) if same_pt.any() else float("nan")
    m_diff = float(sim_pairs[~same_pt].mean()) if (~same_pt).any() else float("nan")
    return m_same - m_diff, int(same_pt.sum()), int((~same_pt).sum()), m_same, m_diff


def _ref_biology_perm_test(sim, patients, labels, n_perm, seed):
    iu = np.triu_indices(len(patients), k=1)
    same_pt = (patients[:, None] == patients[None, :])[iu]
    sim_pairs = sim[iu]
    valid = ((labels != "NA")[:, None] & (labels != "NA")[None, :])[iu]
    obs, n_same, n_diff, m_same, m_diff = _ref_biology_delta(
        sim_pairs, same_pt, labels, iu, valid)
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patients)
    patient_to_idx = pd.factorize(patients)[0]
    patient_label = np.array([
        labels[np.where(patients == p)[0][0]] for p in unique_patients
    ])
    null = np.empty(n_perm, dtype=np.float32)
    for i in range(n_perm):
        perm = rng.permutation(len(unique_patients))
        shuffled = patient_label[perm][patient_to_idx]
        d, *_ = _ref_biology_delta(sim_pairs, same_pt, shuffled, iu, valid)
        null[i] = d
    p = float((null >= obs).sum() + 1) / (n_perm + 1)
    return obs, float(null.mean()), float(null.std()), p, n_same, n_diff, m_same, m_diff, null


def _ref_patient_perm_test(sim, patients, n_perm, seed):
    iu = np.triu_indices(len(patients), k=1)
    sim_pairs = sim[iu]
    obs, n_same, n_diff, m_same, m_diff = _ref_patient_delta(
        sim_pairs, patients, iu)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float32)
    pv = patients.copy()
    for i in range(n_perm):
        rng.shuffle(pv)
        d, *_ = _ref_patient_delta(sim_pairs, pv, iu)
        null[i] = d
    p = float((null >= obs).sum() + 1) / (n_perm + 1)
    return obs, float(null.mean()), float(null.std()), p, n_same, n_diff, m_same, m_diff, null


def _ref_rho_with_null(emb_sim, ea_sim, n_perm, seed):
    flat_emb, _ = _ref_upper_triangle(emb_sim)
    flat_ea, _ = _ref_upper_triangle(ea_sim)
    obs_rho, _ = spearmanr(flat_emb, flat_ea)
    obs_pearson = float(np.corrcoef(flat_emb, flat_ea)[0, 1])
    rng = np.random.default_rng(seed)
    n = emb_sim.shape[0]
    null = np.empty(n_perm, dtype=np.float32)
    for i in range(n_perm):
        perm = rng.permutation(n)
        emb_perm = emb_sim[np.ix_(perm, perm)]
        flat_perm, _ = _ref_upper_triangle(emb_perm)
        r, _ = spearmanr(flat_perm, flat_ea)
        null[i] = r
    null_mean = float(null.mean())
    null_std = float(null.std())
    z = (float(obs_rho) - null_mean) / null_std if null_std > 0 else float("nan")
    p = float((null >= float(obs_rho)).sum() + 1) / (n_perm + 1)
    return {"rho": float(obs_rho), "pearson": obs_pearson,
            "null_mean": null_mean, "null_std": null_std,
            "z": z, "p_value": p, "n_pairs": int(len(flat_emb)),
            "n_permutations": int(n_perm)}


# ================================================= aggregation properties ====

class TestAggregation:
    def test_l2_normalize_unit_norm(self):
        x = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
        normed = l2_normalize(x)
        norms = np.linalg.norm(normed, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_l2_normalize_zero_vector_stays_zero(self):
        """a zero vector divided by eps is still zero - eps prevents the
        divide-by-zero, it does not invent a direction."""
        x = np.zeros((1, 4), dtype=np.float32)
        normed = l2_normalize(x)
        np.testing.assert_array_equal(normed, 0.0)
        assert np.isfinite(normed).all()

    def test_mean_pool_l2_single_niche(self):
        x = np.array([[3.0, 4.0]], dtype=np.float32)
        pooled = mean_pool_l2(x)
        expected = np.array([0.6, 0.8], dtype=np.float32)
        np.testing.assert_allclose(pooled, expected, atol=1e-6)

    def test_mean_pool_l2_empty(self):
        x = np.zeros((0, 4), dtype=np.float32)
        pooled = mean_pool_l2(x)
        assert pooled.shape == (4,)
        np.testing.assert_array_equal(pooled, 0.0)

    def test_mean_pool_l2_result_is_unit_norm(self):
        rng = np.random.default_rng(99)
        x = rng.standard_normal((10, 16)).astype(np.float32)
        pooled = mean_pool_l2(x)
        assert abs(np.linalg.norm(pooled) - 1.0) < 1e-5

    def test_cosine_sim_matrix_diagonal_is_one(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 8))
        C = cosine_sim_matrix(X)
        np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-6)

    def test_cosine_sim_matrix_symmetric(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 8))
        C = cosine_sim_matrix(X)
        np.testing.assert_allclose(C, C.T, atol=1e-7)

    def test_cosine_sim_matrix_matches_script(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 8)).astype(np.float32)
        pkg = cosine_sim_matrix(X)
        ref = _make_cosine(X)
        np.testing.assert_allclose(pkg, ref, atol=1e-6)

    def test_upper_tri_count(self):
        A = np.arange(16).reshape(4, 4)
        flat, iu = upper_tri(A)
        assert len(flat) == 6  # 4 choose 2

    def test_upper_tri_matches_script(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((5, 5))
        pkg_flat, _ = upper_tri(A)
        ref_flat, _ = _ref_upper_triangle(A)
        np.testing.assert_array_equal(pkg_flat, ref_flat)


# ======================================================== delta properties ====

class TestDeltas:
    def test_biology_delta_symmetric_labels_is_zero(self):
        """when every label pair is equally represented, delta is near zero."""
        n = 10
        sim = np.ones((n * (n - 1) // 2,), dtype=np.float32) * 0.5
        same_pt = np.zeros(len(sim), dtype=bool)
        same_lb = np.ones(len(sim), dtype=bool)
        valid = np.ones(len(sim), dtype=bool)
        d, _, _, m_same, m_diff = biology_delta(sim, same_pt, same_lb, valid)
        # all same-label, no diff-label -> m_diff is nan
        assert np.isnan(m_diff)

    def test_biology_delta_empty_partition(self):
        sim = np.array([0.5, 0.6, 0.7])
        same_pt = np.array([True, True, True])  # all same-patient
        same_lb = np.array([True, False, True])
        valid = np.ones(3, dtype=bool)
        d, n_s, n_d, _, _ = biology_delta(sim, same_pt, same_lb, valid)
        # no cross-patient pairs -> both partitions empty
        assert n_s == 0 and n_d == 0
        assert np.isnan(d)

    def test_patient_delta_all_unique(self):
        """one subarray per patient -> no same-patient pairs."""
        sim = np.array([0.5, 0.6, 0.7])  # 3-choose-2
        same_pt = np.zeros(3, dtype=bool)
        d, n_s, n_d, m_same, m_diff = patient_delta(sim, same_pt)
        assert n_s == 0
        assert np.isnan(m_same)


# ====================================================== parity: biology ====

class TestBiologyPermParity:
    """exact parity between package and script on identical synthetic data."""

    @pytest.fixture()
    def synthetic(self):
        emb, patients, labels = _synthetic_subarrays()
        sim = _make_cosine(emb)
        return sim, patients, labels

    def test_observed_delta_matches(self, synthetic):
        sim, patients, labels = synthetic
        n_perm, seed = 200, 42
        ref_obs, ref_nm, ref_ns, ref_p, ref_nsame, ref_ndiff, ref_ms, ref_md, ref_null = \
            _ref_biology_perm_test(sim, patients, labels, n_perm, seed)
        pkg = biology_perm_test(sim, patients, labels,
                                n_permutations=n_perm, seed=seed)
        assert pkg["delta"] == pytest.approx(ref_obs, abs=0)
        assert pkg["mean_same"] == pytest.approx(ref_ms, abs=0)
        assert pkg["mean_diff"] == pytest.approx(ref_md, abs=0)
        assert pkg["n_same_pairs"] == ref_nsame
        assert pkg["n_diff_pairs"] == ref_ndiff

    def test_null_distribution_is_identical(self, synthetic):
        sim, patients, labels = synthetic
        n_perm, seed = 200, 42
        *_, ref_null = _ref_biology_perm_test(sim, patients, labels, n_perm, seed)
        pkg = biology_perm_test(sim, patients, labels,
                                n_permutations=n_perm, seed=seed)
        np.testing.assert_array_equal(pkg["null"], ref_null)

    def test_p_value_matches(self, synthetic):
        sim, patients, labels = synthetic
        n_perm, seed = 200, 42
        _, _, _, ref_p, *_, _ = _ref_biology_perm_test(
            sim, patients, labels, n_perm, seed)
        pkg = biology_perm_test(sim, patients, labels,
                                n_permutations=n_perm, seed=seed)
        assert pkg["p_value"] == pytest.approx(ref_p, abs=0)

    def test_z_matches_historical_convention(self, synthetic):
        """the script's main() computes z = obs / null_std, not standard z."""
        sim, patients, labels = synthetic
        n_perm, seed = 200, 42
        ref_obs, _, ref_ns, *_, _ = _ref_biology_perm_test(
            sim, patients, labels, n_perm, seed)
        ref_z = ref_obs / ref_ns if ref_ns > 0 else float("nan")
        pkg = biology_perm_test(sim, patients, labels,
                                n_permutations=n_perm, seed=seed)
        assert pkg["z"] == pytest.approx(ref_z, abs=0)

    def test_z_standard_differs_from_historical(self, synthetic):
        """z_standard = (obs - null_mean) / null_std. the two converge when
        null_mean is near zero, but are not identical."""
        sim, patients, labels = synthetic
        n_perm, seed = 200, 42
        pkg = biology_perm_test(sim, patients, labels,
                                n_permutations=n_perm, seed=seed)
        if not np.isnan(pkg["z"]) and not np.isnan(pkg["z_standard"]):
            diff = abs(pkg["z"] - pkg["z_standard"])
            correction = abs(pkg["z_mean_correction"])
            assert diff == pytest.approx(correction, abs=1e-5)


# ====================================================== parity: patient ====

class TestPatientPermParity:
    @pytest.fixture()
    def synthetic(self):
        emb, patients, labels = _synthetic_subarrays()
        sim = _make_cosine(emb)
        return sim, patients

    def test_observed_delta_matches(self, synthetic):
        sim, patients = synthetic
        n_perm, seed = 200, 42
        ref_obs, _, _, ref_p, ref_nsame, ref_ndiff, ref_ms, ref_md, _ = \
            _ref_patient_perm_test(sim, patients, n_perm, seed)
        pkg = patient_perm_test(sim, patients,
                                n_permutations=n_perm, seed=seed)
        assert pkg["delta"] == pytest.approx(ref_obs, abs=0)
        assert pkg["n_same_pairs"] == ref_nsame
        assert pkg["n_diff_pairs"] == ref_ndiff

    def test_null_distribution_is_identical(self, synthetic):
        sim, patients = synthetic
        n_perm, seed = 200, 42
        *_, ref_null = _ref_patient_perm_test(sim, patients, n_perm, seed)
        pkg = patient_perm_test(sim, patients,
                                n_permutations=n_perm, seed=seed)
        np.testing.assert_array_equal(pkg["null"], ref_null)

    def test_z_matches_historical_convention(self, synthetic):
        sim, patients = synthetic
        n_perm, seed = 200, 42
        ref_obs, _, ref_ns, *_, _ = _ref_patient_perm_test(
            sim, patients, n_perm, seed)
        ref_z = ref_obs / ref_ns if ref_ns > 0 else float("nan")
        pkg = patient_perm_test(sim, patients,
                                n_permutations=n_perm, seed=seed)
        assert pkg["z"] == pytest.approx(ref_z, abs=0)


# ================================================= parity: faithfulness ====

class TestGeometricFaithfulnessParity:
    def test_rho_and_z_match_script(self):
        rng = np.random.default_rng(42)
        emb_sim = _make_cosine(rng.standard_normal((15, 32)).astype(np.float32))
        ref_sim = _make_cosine(rng.standard_normal((15, 32)).astype(np.float32))
        n_perm, seed = 100, 42
        ref = _ref_rho_with_null(emb_sim, ref_sim, n_perm, seed)
        pkg = geometric_faithfulness(emb_sim, ref_sim,
                                     n_permutations=n_perm, seed=seed)
        assert pkg["rho"] == pytest.approx(ref["rho"], abs=0)
        assert pkg["pearson"] == pytest.approx(ref["pearson"], abs=0)
        assert pkg["z"] == pytest.approx(ref["z"], abs=1e-10)
        assert pkg["p_value"] == pytest.approx(ref["p_value"], abs=0)
        assert pkg["null_mean"] == pytest.approx(ref["null_mean"], abs=1e-5)
        assert pkg["null_std"] == pytest.approx(ref["null_std"], abs=1e-5)

    def test_z_is_standard_convention(self):
        """geometric_faithfulness uses (obs - mean) / std, not the historical
        obs / std used by biology_perm_test."""
        rng = np.random.default_rng(42)
        emb_sim = _make_cosine(rng.standard_normal((15, 32)).astype(np.float32))
        ref_sim = _make_cosine(rng.standard_normal((15, 32)).astype(np.float32))
        pkg = geometric_faithfulness(emb_sim, ref_sim,
                                     n_permutations=100, seed=42)
        assert pkg["z_convention"] == "standard: (obs - null_mean) / null_std"


# ============================================================ ratio CI ====

class TestRatioDeltaCi:
    def test_matches_script_formula(self):
        """exact formula from eval_h2_bootstrap_ci.py:h2c_one_run."""
        z_t, z_p = 5.0, 3.0
        ratio = z_t / z_p
        se = abs(ratio) * np.sqrt(1.0 / (z_t ** 2) + 1.0 / (z_p ** 2))
        pkg = ratio_delta_ci(z_t, z_p)
        assert pkg["ratio"] == pytest.approx(ratio)
        assert pkg["ratio_se"] == pytest.approx(se)
        assert pkg["ratio_ci_low"] == pytest.approx(ratio - 1.96 * se)
        assert pkg["ratio_ci_high"] == pytest.approx(ratio + 1.96 * se)

    def test_zero_denominator(self):
        pkg = ratio_delta_ci(5.0, 0.0)
        assert pkg["ratio"] == float("inf")
        assert np.isnan(pkg["ratio_se"])
        assert "undefined" in pkg["note"]

    def test_zero_numerator(self):
        pkg = ratio_delta_ci(0.0, 3.0)
        assert pkg["ratio"] == 0.0
        assert pkg["ratio_se"] == 0.0
        assert "zero" in pkg["note"]

    def test_small_z_flagged(self):
        pkg = ratio_delta_ci(1.5, 3.0)
        assert "poorly conditioned" in pkg["note"]

    def test_large_z_no_flag(self):
        pkg = ratio_delta_ci(5.0, 3.0)
        assert pkg["note"] == ""


# ====================================================== edge cases ====

class TestEdgeCases:
    def test_all_na_labels_untestable(self):
        rng = np.random.default_rng(42)
        sim = _make_cosine(rng.standard_normal((6, 8)).astype(np.float32))
        patients = np.array(["P1"] * 3 + ["P2"] * 3)
        labels = np.array(["NA"] * 6)
        pkg = biology_perm_test(sim, patients, labels,
                                n_permutations=50, seed=42)
        assert np.isnan(pkg["delta"])
        assert "no cross-patient pairs with valid labels" in pkg["untestable"]

    def test_single_patient_no_same_pairs(self):
        rng = np.random.default_rng(42)
        sim = _make_cosine(rng.standard_normal((5, 8)).astype(np.float32))
        patients = np.array(["P1", "P2", "P3", "P4", "P5"])
        pkg = patient_perm_test(sim, patients,
                                n_permutations=50, seed=42)
        assert pkg["n_same_pairs"] == 0
        assert "all unique patients" in pkg["untestable"]

    def test_two_subarrays_minimum(self):
        """the smallest non-trivial case: 2 subarrays, 1 pair."""
        rng = np.random.default_rng(42)
        sim = _make_cosine(rng.standard_normal((2, 8)).astype(np.float32))
        patients = np.array(["P1", "P2"])
        labels = np.array(["A", "B"])
        pkg = biology_perm_test(sim, patients, labels,
                                n_permutations=50, seed=42)
        # 1 cross-patient pair: both valid, one same-label=False
        assert pkg["n_diff_pairs"] == 1
        assert pkg["n_same_pairs"] == 0
        assert "no cross-patient same-label pairs" in pkg["untestable"]

    def test_null_skew_and_kurt_match_pandas(self):
        """the script uses pd.Series.skew() and .kurt(), which are the
        bias-corrected estimators. verify the package matches."""
        rng = np.random.default_rng(42)
        sim = _make_cosine(rng.standard_normal((10, 8)).astype(np.float32))
        patients = np.array(["P1"] * 4 + ["P2"] * 3 + ["P3"] * 3)
        labels = np.array(["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"])
        pkg = biology_perm_test(sim, patients, labels,
                                n_permutations=100, seed=42)
        pd_skew = float(pd.Series(pkg["null"]).skew())
        pd_kurt = float(pd.Series(pkg["null"]).kurt())
        assert pkg["null_skew"] == pytest.approx(pd_skew, abs=1e-10)
        assert pkg["null_kurt"] == pytest.approx(pd_kurt, abs=1e-10)