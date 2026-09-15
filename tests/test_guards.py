"""the six guards, each on the failure that earned it a place.

a guard exists because a result looked good and was not. so every test here
states that failure in miniature and asserts the guard catches it - and, for the
ones with a threshold, asserts the ordinary case passes, because a guard that
fires on everything is not a guard.
"""
from __future__ import annotations

import numpy as np
import pytest

from omicstra.protocols.evaluate import (
    GUARDS,
    GuardFailed,
    GuardResult,
    bio_vs_confounder_ratio,
    circular_supervision,
    held_out_honesty,
    nmi_label_confounder,
    noise_floor,
    specificity_matrix,
)

RNG = np.random.default_rng(0)


def test_all_six_are_implemented():
    """they were NotImplementedError stubs through v1.1.0a1."""
    assert set(GUARDS) == {"G1", "G2", "G3", "G4", "G5", "G6"}
    import inspect

    from omicstra.protocols import evaluate
    assert "NotImplementedError" not in inspect.getsource(evaluate)


# --- G1 label granularity ---------------------------------------------------
def test_g1_a_label_that_is_the_subject_fails():
    """the archetype label scored 0.89 against patient_id: a clustering score
    against it measured which patient a niche came from."""
    pat = np.repeat(np.arange(20), 10)
    r = nmi_label_confounder(pat, pat)
    assert not r.passed and r.value == pytest.approx(1.0)
    assert "demote to diagnostic" in r.note


def test_g1_an_independent_label_passes():
    pat = np.repeat(np.arange(20), 10)
    assert nmi_label_confounder(RNG.integers(0, 4, 200), pat).passed


def test_g1_mismatched_lengths_raise_rather_than_score():
    with pytest.raises(ValueError, match="length"):
        nmi_label_confounder([1, 2, 3], [1, 2])


# --- G2 held-out honesty ----------------------------------------------------
def test_g2_a_score_that_dies_across_subjects_fails():
    """CCA: 0.667 within-subarray, 0.000 cross-subarray. the within number was a
    different quantity, not a stronger one."""
    r = held_out_honesty(0.667, 0.0)
    assert not r.passed
    assert "report the cross-subject value only" in r.note


def test_g2_a_score_that_survives_passes():
    assert held_out_honesty(0.86, 0.84).passed


# --- G3 biology vs confounder -----------------------------------------------
def test_g3_below_one_is_anti_helpful():
    """one classical arm reached 0.071 - a contraindication, not a caution."""
    r = bio_vs_confounder_ratio(0.071, 1.0)
    assert not r.passed and "anti-helpful" in r.note


def test_g3_above_one_but_below_the_raw_floor_still_fails():
    """clearing 1 is not enough if not aligning at all does better."""
    r = bio_vs_confounder_ratio(1.2, 1.0, floor=1.5)
    assert not r.passed and "raw-modality floor" in r.note


def test_g3_clearing_both_passes():
    assert bio_vs_confounder_ratio(3.0, 1.0, floor=1.5).passed


# --- G4 specificity ---------------------------------------------------------
def test_g4_aliased_axes_fail_and_name_the_pair():
    """five pathway axes at 0.97 mutual cosine: one direction, five labels."""
    base = RNG.normal(size=(1, 16))
    V = np.repeat(base, 5, axis=0) + RNG.normal(scale=0.02, size=(5, 16))
    r = specificity_matrix(V, names=list("abcde"))
    assert not r.passed and r.value > 0.9
    assert len(r.observed["worst_pair"]) == 2


def test_g4_orthogonal_axes_pass():
    assert specificity_matrix(np.eye(5)).passed


def test_g4_needs_two_axes():
    with pytest.raises(ValueError):
        specificity_matrix(np.ones((1, 4)))


# --- G5 noise floor ---------------------------------------------------------
def test_g5_uses_the_empirical_p_not_a_normal_approximation():
    """(1 + #{null >= obs}) / (n + 1), ported from rho_with_null. the published
    critique of this class of test is about normal approximations."""
    null = np.arange(99, dtype=float)          # 0..98
    r = noise_floor(95.0, null)
    # 4 values >= 95 (95,96,97,98) -> (4 + 1) / (99 + 1)
    assert r.value == pytest.approx(5 / 100)
    assert "empirical" in r.observed["p_form"]


def test_g5_an_observation_inside_its_null_fails():
    r = noise_floor(0.1, RNG.normal(size=999))
    assert not r.passed and "indistinguishable" in r.note


def test_g5_an_observation_beyond_its_null_passes_with_z():
    r = noise_floor(5.0, RNG.normal(size=999))
    assert r.passed and r.observed["z"] > 3


def test_g5_refuses_a_null_too_small_to_mean_anything():
    with pytest.raises(ValueError, match="needs samples"):
        noise_floor(1.0, [0.1, 0.2])


def test_g5_lower_tail():
    assert noise_floor(-5.0, RNG.normal(size=999), tail="less").passed


# --- G6 circular supervision ------------------------------------------------
def test_g6_supervision_in_the_features_fails():
    r = circular_supervision(["virchow2_niche", "mc_weights_niche"], "mc_weights_niche")
    assert not r.passed and "find itself" in r.note


def test_g6_delegates_to_the_join_rather_than_duplicating_it():
    """one implementation. the join is where features and supervision share a
    table, so that is where the assertion has to live."""
    import inspect

    src = inspect.getsource(circular_supervision)
    assert "assert_no_circular_supervision" in src


def test_g6_clean_features_pass():
    assert circular_supervision(["virchow2_niche", "novae_niche"], "mc_weights_niche").passed


# --- the result type --------------------------------------------------------
def test_a_failed_guard_can_be_made_to_raise():
    """the chain raises where a guard's failure invalidates the number rather
    than caveating it."""
    with pytest.raises(GuardFailed, match="G2"):
        held_out_honesty(0.667, 0.0).raise_if_failed()
    assert isinstance(held_out_honesty(0.8, 0.8).raise_if_failed(), GuardResult)


def test_no_guard_names_a_cohort():
    """cohort-free is the property that makes the library travel."""
    import inspect

    from omicstra.protocols import evaluate
    body = inspect.getsource(evaluate).lower()
    # the docstrings may cite a result; the logic may not branch on a cohort
    for token in ("tnbc", "wang", "hest"):
        assert f'"{token}' not in body and f"'{token}" not in body
