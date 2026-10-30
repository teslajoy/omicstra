"""subarray-level biology validation statistics (H2-C scoring path).

does a shared embedding space amplify biological signal (MC subtype, TIME
class) relative to patient identity? ported from
`scripts/eval_alignment_biology.py` and `scripts/eval_h2_bootstrap_ci.py`.

arithmetic unchanged from the scripts that produced the published numbers.
what changes is that each statistic exists ONCE and the parameters the
scripts left implicit are arguments a caller must supply.

resolution
----------
subarrays, not niches. the biology test compares subarray-to-subarray
cosine similarities partitioned by patient-level labels, so the unit of
analysis is one subarray's mean embedding, and the permutation shuffles
labels at the patient level (one patient -> many subarrays).

z-score convention
------------------
TWO conventions live in the same script and in the artifacts it produced.

    rho_with_null (geometric faithfulness):
        z = (observed - null_mean) / null_std       standard z

    biology_perm_test, patient_perm_test (biology.parquet):
        z = observed_delta / null_std               historical z

the historical z omits the null-mean subtraction. under the null the
expected delta is approximately zero (permuted labels produce no
systematic same-vs-different gap), so the two formulas converge - but
they are not identical, and the published biology.parquet stores the
historical z. the H2-C ratio (z_TIME / z_patient) is computed FROM those
stored z values, so changing the formula changes the ratio.

this module preserves the historical formula in the `z` field and adds a
`z_standard` field with the conventional (obs - mean) / std. both are
deterministic given the same null, and the difference `null_mean / null_std`
quantifies how far apart they are: below ~0.1 they are practically the
same; above it something is wrong with the null or the test.

factorize convention
--------------------
the script uses `pd.factorize` for patient-to-index mapping, which
assigns indices in first-appearance order. `np.unique(..., return_inverse)`
would assign in sorted order. the two agree when patients are already
sorted, and the observed delta (which uses the original labels, not the
index) is identical either way. the NULL DISTRIBUTION uses the mapping to
propagate shuffled labels, so the same seed produces different
permutations under the two conventions. this module uses `pd.factorize`
for exact seed-level parity with the published artifacts.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================ aggregation ====

def l2_normalize(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """L2-normalise along the last axis."""
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, eps, None)


def mean_pool_l2(per_niche: np.ndarray) -> np.ndarray:
    """L2-norm each niche, mean, L2-norm the mean.

    matches the gpath2vec convention: the subarray embedding is the normalised
    mean of normalised niche embeddings. empty input -> zero vector.
    """
    if per_niche.shape[0] == 0:
        return np.zeros(per_niche.shape[1], dtype=np.float32)
    return l2_normalize(
        l2_normalize(per_niche).mean(axis=0)
    ).ravel().astype(np.float32)


def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """N x N cosine similarity from N x D embeddings, clipped to [-1, 1]."""
    Xn = l2_normalize(np.asarray(X, dtype=np.float64))
    return np.clip(Xn @ Xn.T, -1.0, 1.0).astype(np.float32)


def upper_tri(A: np.ndarray) -> tuple[np.ndarray, tuple]:
    """upper triangle of a square matrix, flattened, with indices."""
    iu = np.triu_indices(A.shape[0], k=1)
    return A[iu], iu


# ============================================================ pair masks ====

def _pair_masks(patients: np.ndarray) -> tuple[tuple, np.ndarray]:
    """upper-triangle indices and same-patient mask for a vector of patient IDs."""
    iu = np.triu_indices(len(patients), k=1)
    same_pt = (patients[:, None] == patients[None, :])[iu]
    return iu, same_pt


# ============================================================ deltas ====

def biology_delta(sim_pairs: np.ndarray, same_patient: np.ndarray,
                  same_label: np.ndarray, valid: np.ndarray
                  ) -> tuple[float, int, int, float, float]:
    """mean cosine of cross-patient same-label pairs minus cross-patient
    different-label pairs. the biology signal, controlled for patient.

    operates on the FLATTENED upper triangle of a cosine matrix.
    `valid` excludes pairs where either subarray has a missing label.

    returns (delta, n_same_pairs, n_diff_pairs, mean_same, mean_diff).
    when a partition is empty the mean is nan and the delta propagates it.
    """
    cross = (~same_patient) & valid
    A = cross & same_label
    B = cross & (~same_label)
    m_same = float(sim_pairs[A].mean()) if A.any() else float("nan")
    m_diff = float(sim_pairs[B].mean()) if B.any() else float("nan")
    return m_same - m_diff, int(A.sum()), int(B.sum()), m_same, m_diff


def patient_delta(sim_pairs: np.ndarray, same_patient: np.ndarray
                  ) -> tuple[float, int, int, float, float]:
    """mean cosine of same-patient pairs minus different-patient pairs.

    the confounder strength: how much of the cosine structure is patient
    identity rather than biology.

    returns (delta, n_same_pairs, n_diff_pairs, mean_same, mean_diff).
    """
    m_same = float(sim_pairs[same_patient].mean()) if same_patient.any() else float("nan")
    m_diff = float(sim_pairs[~same_patient].mean()) if (~same_patient).any() else float("nan")
    return m_same - m_diff, int(same_patient.sum()), int((~same_patient).sum()), m_same, m_diff


# ============================================================ permutation ====

def biology_perm_null(sim: np.ndarray, patients: np.ndarray,
                      labels: np.ndarray, *, n_permutations: int,
                      seed: int) -> tuple[float, np.ndarray, dict]:
    """observed biology delta and its patient-level permutation null.

    the null permutes the label at the PATIENT level: each permutation
    shuffles the patient-to-label mapping and propagates to subarrays.
    this is the correct null because subarrays of one patient are not
    independent, so shuffling subarray labels directly would produce a
    null that is too tight.

    uses pd.factorize for the patient-to-index mapping to match the
    script's seed-level behaviour exactly. see module docstring.

    returns (observed_delta, null_array(n_permutations,), obs_details).
    obs_details has n_same_pairs, n_diff_pairs, mean_same, mean_diff.
    """
    iu, same_pt = _pair_masks(patients)
    sim_pairs = sim[iu]
    valid = ((labels != "NA")[:, None] & (labels != "NA")[None, :])[iu]
    same_lb = (labels[:, None] == labels[None, :])[iu]
    obs, n_same, n_diff, m_same, m_diff = biology_delta(
        sim_pairs, same_pt, same_lb, valid)

    # the null holds the COMPARED PAIR SET fixed (original valid mask) and
    # shuffles only which label each pair carries. recomputing valid from
    # shuffled labels would change the denominator in each permutation,
    # which is a different and looser test.
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patients)
    patient_to_idx = pd.factorize(patients)[0]
    patient_label = np.array([
        labels[np.where(patients == p)[0][0]] for p in unique_patients
    ])
    null = np.empty(n_permutations, dtype=np.float32)
    for i in range(n_permutations):
        perm = rng.permutation(len(unique_patients))
        shuffled = patient_label[perm][patient_to_idx]
        same_lb_s = (shuffled[:, None] == shuffled[None, :])[iu]
        d, *_ = biology_delta(sim_pairs, same_pt, same_lb_s, valid)
        null[i] = d
    return obs, null, {"n_same_pairs": n_same, "n_diff_pairs": n_diff,
                       "mean_same": m_same, "mean_diff": m_diff}


def patient_perm_null(sim: np.ndarray, patients: np.ndarray, *,
                      n_permutations: int, seed: int
                      ) -> tuple[float, np.ndarray, dict]:
    """observed patient delta and its permutation null.

    the null shuffles patient IDs directly: each permutation reassigns
    which subarray belongs to which patient.

    returns (observed_delta, null_array(n_permutations,), obs_details).
    """
    iu, same_pt = _pair_masks(patients)
    sim_pairs = sim[iu]
    obs, n_same, n_diff, m_same, m_diff = patient_delta(sim_pairs, same_pt)

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=np.float32)
    pv = patients.copy()
    for i in range(n_permutations):
        rng.shuffle(pv)
        same_pt_s = (pv[:, None] == pv[None, :])[iu]
        d, *_ = patient_delta(sim_pairs, same_pt_s)
        null[i] = d
    return obs, null, {"n_same_pairs": n_same, "n_diff_pairs": n_diff,
                       "mean_same": m_same, "mean_diff": m_diff}


# ============================================================ full tests ====

def _null_stats(obs: float, null: np.ndarray) -> dict[str, Any]:
    """z, p, and null diagnostics matching the script's biology.parquet schema.

    z convention: see module docstring. `z` is the HISTORICAL formula
    (obs / null_std), `z_standard` is the conventional (obs - null_mean) /
    null_std. the difference `null_mean / null_std` says how far apart they
    are. both are nan when null_std is zero.
    """
    null_mean = float(null.mean())
    null_std = float(null.std())
    # historical z (biology.parquet): obs / null_std
    z_hist = obs / null_std if null_std > 0 else float("nan")
    # standard z: (obs - null_mean) / null_std
    z_std = (obs - null_mean) / null_std if null_std > 0 else float("nan")
    p = float((null >= obs).sum() + 1) / (len(null) + 1)
    return {
        "z": z_hist,
        "z_standard": z_std,
        "z_convention": "obs / null_std (historical, not mean-subtracted)",
        "z_mean_correction": null_mean / null_std if null_std > 0 else float("nan"),
        "p_value": p,
        "p_floor": 1.0 / (len(null) + 1),
        "null_mean": null_mean,
        "null_std": null_std,
        "null_skew": float(pd.Series(null).skew()),
        "null_kurt": float(pd.Series(null).kurt()),
        "null_q05": float(np.quantile(null, 0.05)),
        "null_q95": float(np.quantile(null, 0.95)),
        "obs_percentile_in_null": float((null < obs).mean()),
        "n_permutations": int(len(null)),
    }


def biology_perm_test(sim: np.ndarray, patients: np.ndarray,
                      labels: np.ndarray, *, n_permutations: int,
                      seed: int) -> dict[str, Any]:
    """full biology permutation test.

    returns all fields the script wrote to biology.parquet, plus `z_standard`
    and `null` (the raw null array, for downstream CI or diagnostics).

    when all labels are 'NA' or a pair group is empty, the delta is nan,
    the z is nan, the p is 1.0, and the untestable reason is recorded.
    """
    obs, null, details = biology_perm_null(
        sim, patients, labels,
        n_permutations=n_permutations, seed=seed)
    stats = _null_stats(obs, null)
    untestable = ""
    if details["n_same_pairs"] == 0 and details["n_diff_pairs"] == 0:
        untestable = "no cross-patient pairs with valid labels"
    elif details["n_same_pairs"] == 0:
        untestable = "no cross-patient same-label pairs"
    return {
        "test_type": "biology",
        "delta": obs,
        **details,
        **stats,
        "null": null,
        "untestable": untestable,
    }


def patient_perm_test(sim: np.ndarray, patients: np.ndarray, *,
                      n_permutations: int, seed: int) -> dict[str, Any]:
    """full patient permutation test.

    returns all fields the script wrote to biology.parquet for patient_id
    rows, plus `z_standard` and `null`.

    when there are no same-patient pairs (every subarray from a unique
    patient), the delta is nan and the test is untestable.
    """
    obs, null, details = patient_perm_null(
        sim, patients, n_permutations=n_permutations, seed=seed)
    stats = _null_stats(obs, null)
    untestable = ""
    if details["n_same_pairs"] == 0:
        untestable = "no same-patient pairs (all unique patients)"
    return {
        "test_type": "patient",
        "delta": obs,
        **details,
        **stats,
        "null": null,
        "untestable": untestable,
    }


# ============================================================ faithfulness ====

def geometric_faithfulness(emb_sim: np.ndarray, ref_sim: np.ndarray, *,
                           n_permutations: int, seed: int
                           ) -> dict[str, Any]:
    """Spearman correlation between two cosine matrices, against a
    shuffled-matrix null.

    the null permutes the ROWS AND COLUMNS of the embedding matrix jointly
    (a random relabelling of which subarray is which), then recomputes the
    upper-triangle Spearman.

    z convention: STANDARD (obs - null_mean) / null_std, matching the
    script's rho_with_null.

    ported from eval_alignment_biology.py:rho_with_null.
    """
    flat_emb, _ = upper_tri(emb_sim)
    flat_ref, _ = upper_tri(ref_sim)
    obs_rho, _ = spearmanr(flat_emb, flat_ref)
    obs_pearson = float(np.corrcoef(flat_emb, flat_ref)[0, 1])
    rng = np.random.default_rng(seed)
    n = emb_sim.shape[0]
    null = np.empty(n_permutations, dtype=np.float32)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        emb_perm = emb_sim[np.ix_(perm, perm)]
        flat_perm, _ = upper_tri(emb_perm)
        r, _ = spearmanr(flat_perm, flat_ref)
        null[i] = r
    null_mean = float(null.mean())
    null_std = float(null.std())
    z = (float(obs_rho) - null_mean) / null_std if null_std > 0 else float("nan")
    p = float((null >= float(obs_rho)).sum() + 1) / (n_permutations + 1)
    return {"rho": float(obs_rho), "pearson": obs_pearson,
            "null_mean": null_mean, "null_std": null_std,
            "z": z, "p_value": p,
            "n_pairs": int(len(flat_emb)),
            "n_permutations": int(n_permutations),
            "z_convention": "standard: (obs - null_mean) / null_std"}


# ============================================================ ratio CI ====

def ratio_delta_ci(z_num: float, z_den: float
                   ) -> dict[str, Any]:
    """delta-method (Fieller-style) 95% CI on z_num / z_den.

    each z is normalised by its own permutation null_std, so under H0 each
    has approximately unit variance. the delta-method SE is:
        SE(ratio) = |ratio| * sqrt(1/z_num^2 + 1/z_den^2)

    undefined when z_den == 0 (infinite ratio). also poorly conditioned
    when either z is near zero: SE explodes because the variance of a ratio
    blows up when the denominator's distribution includes zero. both cases
    are flagged in `note`.

    ported from eval_h2_bootstrap_ci.py:h2c_one_run.
    """
    if z_den == 0:
        return {"ratio": float("inf"), "ratio_ci_low": float("nan"),
                "ratio_ci_high": float("nan"), "ratio_se": float("nan"),
                "z_num": float(z_num), "z_den": float(z_den),
                "note": "z_den is zero - ratio undefined"}
    ratio = z_num / z_den
    if z_num == 0:
        return {"ratio": 0.0, "ratio_ci_low": 0.0, "ratio_ci_high": 0.0,
                "ratio_se": 0.0,
                "z_num": float(z_num), "z_den": float(z_den),
                "note": "z_num is zero - ratio is zero with no uncertainty"}
    se = abs(ratio) * np.sqrt(1.0 / (z_num * z_num) + 1.0 / (z_den * z_den))
    note = ""
    if abs(z_num) < 2.0 or abs(z_den) < 2.0:
        note = ("poorly conditioned: a z near zero means the denominator's "
                "distribution includes zero, so the delta-method SE may "
                "understate the true uncertainty")
    return {"ratio": float(ratio),
            "ratio_ci_low": float(ratio - 1.96 * se),
            "ratio_ci_high": float(ratio + 1.96 * se),
            "ratio_se": float(se),
            "z_num": float(z_num), "z_den": float(z_den),
            "note": note}


__all__ = [
    "l2_normalize", "mean_pool_l2", "cosine_sim_matrix", "upper_tri",
    "biology_delta", "patient_delta",
    "biology_perm_null", "patient_perm_null",
    "biology_perm_test", "patient_perm_test",
    "geometric_faithfulness", "ratio_delta_ci",
]