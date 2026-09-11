"""the encode acceptance criterion, and the proof it still bites.

a tolerance that passes everything is not a test. these fix the criterion's
shape on synthetic data - which runs anywhere - and the oracle test at the end
runs the real comparison where the cohort and the 75 GB cache exist.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from omicstra.protocols.encode import (
    ACCEPTANCE,
    accepts,
    encoder_provenance,
    retrieval_invariant,
    slice_diff,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "embeddings" / "virchow2_niche"
PORTED = Path("/private/tmp/claude-1227382601/-Users-sanati-BForePC-omicstra"
              "/a5d1bbd1-2b2c-4f74-b8ea-d3518731d752/scratchpad/ported_cpu")


def _blob(n=200, dim=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)).astype(np.float32) * 5.0


# --- the criterion has the shape it claims ---------------------------------
def test_identical_input_passes_every_condition():
    x = _blob()
    d = {"max_rel_to_row_norm": 0.0}
    inv = retrieval_invariant(x, x)
    # against the DECLARED bar, not a tighter one invented here: normalising
    # identical float32 vectors already costs ~2e-7, so 1 - 1e-7 would fail a
    # perfect input and tell us nothing about a port.
    assert inv["knn_set_agreement"] == 1.0
    assert inv["min_cosine"] >= ACCEPTANCE["min_cosine"]
    ok, _ = accepts(d, inv)
    assert ok


def test_a_real_neighbourhood_change_still_fails():
    """THE test. a tie-aware rule that forgives an actual reordering would pass
    a port that moved a published number, which is the failure it exists to catch.
    """
    x = _blob()
    y = x.copy()
    rng = np.random.default_rng(1)
    # perturb a fifth of the rows far beyond any float32 noise
    hit = rng.choice(len(y), size=len(y) // 5, replace=False)
    y[hit] += rng.normal(size=(len(hit), y.shape[1])).astype(np.float32) * 3.0

    inv = retrieval_invariant(y, x)
    assert inv["knn_set_agreement"] < ACCEPTANCE["knn_set_agreement"], \
        "a genuine neighbourhood change was forgiven as a tie"
    ok, why = accepts({"max_rel_to_row_norm": 0.0}, inv)
    assert not ok and "knn" in why


def test_float32_noise_alone_does_not_fail_it():
    """the other side: noise at the scale two backends actually differ by must
    not fail, or the criterion rejects every correct port."""
    x = _blob()
    y = (x + np.random.default_rng(2).normal(size=x.shape).astype(np.float32) * 1e-5)
    inv = retrieval_invariant(y.astype(np.float32), x)
    assert inv["knn_set_agreement"] >= ACCEPTANCE["knn_set_agreement"]


def test_ties_are_not_counted_as_disagreement():
    """duplicate rows make every ranking among them arbitrary. at 8 spots each
    niche overlaps every other, so the pooled vectors ARE duplicates - counting
    the sort order as a difference measures argsort, not the encoder.
    """
    base = _blob(n=4, dim=16, seed=3)
    x = np.repeat(base, 3, axis=0)               # 12 rows, only 4 distinct
    y = x.copy()
    inv = retrieval_invariant(y, x)
    assert inv["knn_set_agreement"] == 1.0
    assert inv["knn_set_exact"] <= 1.0


def test_a_drift_that_clears_knn_can_still_fail_on_magnitude():
    """the three conditions ask three different questions and one cannot stand
    in for another."""
    x = _blob()
    inv = retrieval_invariant(x, x)
    ok, why = accepts({"max_rel_to_row_norm": 1e-3}, inv)
    assert not ok and "drift" in why


def test_the_criterion_is_declared_with_its_reason():
    assert set(ACCEPTANCE) >= {"max_abs_over_row_norm", "min_cosine",
                               "knn_set_agreement", "reason"}
    assert "retrieval" in ACCEPTANCE["reason"] or "H1" in ACCEPTANCE["reason"]
    assert "cache_device_inferred" in ACCEPTANCE, \
        "the cache's device is inferred from evidence, and that must travel"


def test_provenance_records_what_the_cache_never_did():
    """the whole reason the comparison is labelled cross-device.

    it reports torch and timm versions, so it needs them - both are in the
    `encode` extra, which CI deliberately does not install.
    """
    pytest.importorskip("timm", reason="provenance reports the timm version")
    p = encoder_provenance("cpu")
    assert {"device", "torch", "timm", "model_revision", "dtype"} <= set(p)
    assert p["device"] == "cpu"


# --- the oracle: only where the cohort and the cache exist ------------------
# TNBC51_CN26_D1 is 8 spots - below Novae's 512 floor, one of the 19 that
# `below_floor_policy: drop` removes, so it NEVER enters a real run. it is here
# because it is pathological, not because it is representative: at 8 spots every
# niche overlaps every other, the pooled vectors are duplicates, and it is the
# only place in the cohort where the tie rule can be exercised on real data.
PICKS = ["TNBC1_CN1_C1", "TNBC51_CN26_D1", "TNBC22_CN11_E2",
         "TNBC10_CN5_D2", "TNBC10_CN5_E2"]


@pytest.mark.parametrize("sid", PICKS)
def test_the_ported_encoder_reproduces_the_cache(sid):
    """the real slice-diff, on vectors this machine computed with the port.

    skips where either half is absent, which is every machine but this one.
    """
    ported, cached = PORTED / f"{sid}.npy", CACHE / f"{sid}.npy"
    if not (ported.is_file() and cached.is_file()):
        pytest.skip("ported slice or cache absent on this machine")

    a, b = np.load(ported), np.load(cached)
    d = slice_diff(a, cached)
    inv = retrieval_invariant(a, b)
    ok, why = accepts(d, inv)
    assert ok, f"{sid}: {why}"
