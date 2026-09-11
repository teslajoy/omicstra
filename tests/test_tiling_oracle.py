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
def _spots():
    hits = sorted((ROOT / "data" / "inputs" / "byArray").glob("*/*/selection.RData"))
    if not hits:
        pytest.skip("cohort inputs not present on this machine")
    import pandas as pd
    r = subprocess.run(["Rscript", "-e", f'load("{hits[0]}"); write.csv(spots, stdout())'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("Rscript unavailable")
    df = pd.read_csv(io.StringIO(r.stdout), index_col=0)
    return hits[0].parent.name, df[["pixel_x", "pixel_y"]].values


def test_neighbours_match_the_script_that_built_the_cache():
    from scipy.spatial import KDTree
    name, coords = _spots()
    n = len(coords)
    _, script_idx = KDTree(coords).query(coords, k=min(K + 1, n))   # the script, verbatim
    pkg_idx, _ = niche_neighbours(coords, K)
    assert np.array_equal(script_idx, pkg_idx), f"neighbour sets drifted on {name}"


def test_pooling_matches_the_script_that_built_the_cache():
    from scipy.spatial import KDTree
    name, coords = _spots()
    n = len(coords)
    _, script_idx = KDTree(coords).query(coords, k=min(K + 1, n))
    v = np.random.default_rng(0).standard_normal((n, 64)).astype(np.float32)

    script_pool = np.zeros((n, 64), dtype=np.float32)          # the script's inline loop
    for j in range(n):
        nb = np.atleast_1d(script_idx[j])
        script_pool[j] = v[nb[nb < n]].mean(axis=0)

    assert np.abs(script_pool - pool_niche(v, script_idx)).max() == 0.0, f"pooling drifted on {name}"
