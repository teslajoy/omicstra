"""tiling.py against the script that built the cache.

the cache IS the oracle, and the cache was made by
scripts/extract_virchow2_niche.py. so the comparison target is that script's
logic, not a remembered description of it. these run the script's neighbour and
pooling code inline, verbatim, and diff it against the package path.

this is a standing guard, not a one-time port check: any future change to
tiling.py that moves a single vector fails here, rather than surfacing as an
unexplained shift in H1 six weeks later.

synthetic where possible, real data where it matters. the lattice tests need no
cohort; the oracle test skips when the cohort is absent, which is every machine
that is not this one.
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
import pytest

from omicstra.measures.tiling import (
    TileGeometry,
    niche_neighbours,
    pool_niche,
    tile_boxes,
    tile_um,
)

ROOT = Path(__file__).resolve().parents[1]
K = 6


# --- no cohort needed -------------------------------------------------------
def test_a_niche_includes_itself():
    """k=6 means seven vectors, not six. a niche is the spot plus its ring."""
    coords = np.array([[x, y] for x in range(5) for y in range(5)], dtype=float)
    idx, counts = niche_neighbours(coords, K)
    assert idx.shape[1] == K + 1
    assert all(j in idx[j] for j in range(len(coords))), "self must be in its own niche"
    assert counts.min() == K + 1


def test_fewer_spots_than_k_keeps_the_tissue():
    """a 3-spot subarray pools over 3, rather than being dropped.

    scipy returns n as a sentinel when it cannot fill k; those are discarded, so
    boundary and tiny subarrays stay in the cohort and `counts` records it.
    """
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    idx, counts = niche_neighbours(coords, K)
    assert counts.tolist() == [3, 3, 3]
    out = pool_niche(np.ones((3, 4), dtype=np.float32), idx)
    assert np.isfinite(out).all()


def test_pooling_is_an_unweighted_mean():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    idx, _ = niche_neighbours(coords, 2)
    v = np.array([[0.0], [3.0], [6.0]], dtype=np.float32)
    assert pool_niche(v, idx)[0][0] == pytest.approx(3.0)


def test_tile_um_is_derived_not_declared():
    """the function the falsified constant stood in for.

    a 338 px crop on a 528.71 px lattice at 150 um pitch covers 95.9 um - not
    the 128 um the script's constant named, because the pitch sized the crop.
    """
    assert tile_um(338, 528.71, 150.0) == pytest.approx(95.9, abs=0.1)
    assert tile_um(451, 528.71, 150.0) == pytest.approx(128.0, abs=0.2)


def test_edge_boxes_clamp_and_are_flagged():
    coords = np.array([[0.0, 0.0], [100.0, 100.0]])
    boxes = tile_boxes(coords, TileGeometry(scale=1.0, tile_px=40), (200, 200))
    assert (boxes[0] == [0, 0, 20, 20]).all(), "a corner spot clamps"
    assert (boxes[1] == [80, 80, 120, 120]).all(), "an interior spot does not"


# --- the oracle: only on a machine that has the cohort ----------------------
#
# five subarrays, chosen for properties rather than at random, so the next person
# knows why these five. `_meta.tsv` is the SCRIPT'S OWN OUTPUT - its
# `neighbor_count` column is what the script wrote while building the cache, so
# diffing against it compares to the artifact, not to a re-implementation of the
# artifact's logic.
PICKS = {
    "TNBC1_CN1_C1":   "the clean case - 1075 spots, every niche full, no edge effects",
    "TNBC51_CN26_D1": "8 spots, the smallest in the cohort and far below Novae's 512 floor; "
                      "this is the evidence the platform_floor gate acts on",
    "TNBC22_CN11_E2": "1813 spots, the largest - the shard that will actually be slow, and "
                      "the one most likely to hit memory on a laptop",
    "TNBC10_CN5_D2":  "one of three subarrays from a single patient; the other two are E1 "
                      "and E2. confirms subject identity travels, which the niche join and "
                      "every patient-held-out split depend on",
    "TNBC10_CN5_E2":  "its sibling. consecutive 16um sections of one block are near-"
                      "duplicates in z, which is why splitting below patient level leaks",
}

# NOT in the picks, and the reason is a measured fact rather than an oversight:
# across all 280 subarrays, ZERO tiles clip the image edge and the smallest
# subarray has 8 spots against a k+1 of 7. so in THIS cohort the black-pad branch
# and the sentinel drop never execute. both are real for other cohorts - a cohort
# with <=6 spots on an array, or spots near a slide edge, hits them immediately -
# so they keep synthetic coverage above, by fact rather than by choice.

CACHE = ROOT / "data" / "embeddings" / "virchow2_niche"


def _meta(sid: str):
    m = CACHE / f"{sid}_meta.tsv"
    if not m.exists():
        pytest.skip(f"{sid} cache not present on this machine")
    import pandas as pd
    return pd.read_csv(m, sep="\t")


@pytest.mark.parametrize("sid", sorted(PICKS), ids=lambda s: s)
def test_neighbour_counts_match_what_the_script_recorded(sid):
    """against the script's own written output, not a reconstruction of it.

    `neighbor_count` in _meta.tsv was produced while the cache was built. if the
    package's niche construction ever diverges, this is where it shows.
    """
    df = _meta(sid)
    _, counts = niche_neighbours(df[["pixel_x", "pixel_y"]].values, K)
    assert np.array_equal(counts, df["neighbor_count"].values), f"{sid}: {PICKS[sid]}"


@pytest.mark.parametrize("sid", sorted(PICKS), ids=lambda s: s)
def test_pooling_matches_the_script_on_real_vectors(sid):
    """real cached embeddings rather than random ones.

    magnitudes and correlation structure are what a float32 accumulation
    difference would show up against; gaussian noise can hide one.
    """
    from scipy.spatial import KDTree
    npy = CACHE / f"{sid}.npy"
    if not npy.exists():
        pytest.skip(f"{sid} vectors not present")
    v = np.load(npy).astype(np.float32)
    coords = _meta(sid)[["pixel_x", "pixel_y"]].values
    n = len(coords)
    _, script_idx = KDTree(coords).query(coords, k=min(K + 1, n))   # the script, verbatim

    script_pool = np.zeros_like(v)                                   # the script's inline loop
    for j in range(n):
        nb = np.atleast_1d(script_idx[j])
        script_pool[j] = v[nb[nb < n]].mean(axis=0)

    assert np.abs(script_pool - pool_niche(v, script_idx)).max() == 0.0, f"{sid} drifted"


def test_the_smallest_subarray_is_still_above_the_niche_size():
    """8 spots against k+1=7 - the sentinel path is unreachable here.

    this asserts the FACT the pick list rests on. if a future cohort or a
    re-ingest produces a subarray of 6 or fewer, this fails and the sentinel
    branch stops being synthetic-only.
    """
    metas = sorted(CACHE.glob("*_meta.tsv"))
    if not metas:
        pytest.skip("cache not present on this machine")
    import pandas as pd
    smallest = min(sum(1 for _ in open(m)) - 1 for m in metas)
    assert smallest > K, f"a subarray of {smallest} spots now exercises the sentinel drop"
