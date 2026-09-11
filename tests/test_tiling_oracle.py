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

from pathlib import Path

import numpy as np
import pytest

from omicstra.measures.tiling import (
    TileGeometry,
    cut_tiles,
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


# --- cut_tiles: the branch this cohort never takes --------------------------
#
# zero tiles clip the image edge across all 280 subarrays, so the black-pad path
# cannot be exercised on real data and is synthetic-only by necessity. that makes
# it exactly the code most likely to rot, so it gets the same oracle treatment as
# everything else: the pre-refactor body is reproduced verbatim below and the
# package path is diffed against it pixel for pixel.

def _cut_tiles_as_originally_written(image, coords, geom):
    """scripts/extract_virchow2_niche.py's geometry, before the clamp was shared.

    verbatim, including `int(round(...))` where the package now uses `np.rint`.
    do not tidy this - its whole value is being the thing that was there.
    """
    from PIL import Image

    w, h = image.size
    half = geom.tile_px // 2
    out = []
    for x, y in np.asarray(coords, dtype=float):
        # the redundant int(round(...)) is deliberate: it is what the script
        # wrote, and tidying it here would silently retire the oracle.
        cx, cy = int(round(x * geom.scale)), int(round(y * geom.scale))  # noqa: RUF046
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)
        patch = image.crop((x1, y1, x2, y2))
        if patch.size != (geom.tile_px, geom.tile_px):
            canvas = Image.new("RGB", (geom.tile_px, geom.tile_px), (0, 0, 0))
            canvas.paste(patch, (half - (cx - x1), half - (cy - y1)))
            patch = canvas
        out.append(patch.resize((geom.out_px, geom.out_px), Image.LANCZOS))
    return out


def _noise_image(w, h, seed=0):
    from PIL import Image

    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")


def test_cut_tiles_matches_the_original_geometry_everywhere():
    """every clamped case plus interior ones, against the body that was replaced.

    noise rather than a flat fill on purpose: a constant image hides an offset,
    because a tile shifted by a pixel is identical to one that is not.
    """
    img = _noise_image(200, 160, seed=7)
    geom = TileGeometry(scale=1.0, tile_px=40, out_px=32)
    coords = np.array([
        [100.0, 80.0],    # interior, no pad
        [0.0, 0.0],       # both axes clamped low
        [199.0, 159.0],   # both clamped high
        [5.0, 80.0],      # left only
        [195.0, 80.0],    # right only
        [100.0, 3.0],     # top only
        [100.0, 157.0],   # bottom only
        [10.5, 20.5],     # half-pixel: the rounding mode has to agree too
    ])
    got = cut_tiles(img, coords, geom)
    want = _cut_tiles_as_originally_written(img, coords, geom)
    assert len(got) == len(want) == len(coords)
    for j, (a, b) in enumerate(zip(got, want)):
        assert a.size == (geom.out_px, geom.out_px)
        assert np.array_equal(np.asarray(a), np.asarray(b)), \
            f"tile {j} at {coords[j].tolist()} differs from the original geometry"


def test_a_padded_tile_keeps_the_spot_centred():
    """the pad goes where the clamp took pixels from, not into the corner.

    stated independently of the oracle above, because both could agree while both
    being wrong - this asserts the property rather than the history.
    """
    img = _noise_image(200, 160, seed=3)
    geom = TileGeometry(scale=1.0, tile_px=40, out_px=40)
    tile = np.asarray(cut_tiles(img, np.array([[5.0, 80.0]]), geom)[0])
    assert (tile[:, :15] == 0).all(), "the 15 clamped columns should be black"
    assert tile[:, 15:].any(), "the rest must carry image, not more pad"


def test_no_tile_in_this_cohort_actually_pads():
    """the fact the two tests above rest on: the pad branch is unreachable here.

    if a re-ingest ever puts a spot within half a tile of an image edge, this
    fails and the synthetic-only justification stops being true.
    """
    import json

    record = ROOT / "projects" / "tnbc-92" / "data" / "canonical" / "ingest.json"
    if not record.is_file():
        pytest.skip("no canonical ingest on this machine")
    rec = json.loads(record.read_text())

    import pandas as pd
    plat = json.loads((ROOT / "projects" / "tnbc-92" / "platform.json").read_text())
    st = plat["platforms"]["original_st"]
    half = st["he_tile"]["tile_px_hd"] // 2
    scale = st["image_pyramid"]["scale_to_hd"]

    clipped = []
    for sid in rec["samples"][:40]:            # 40 is enough to catch a systematic shift
        df = pd.read_parquet(record.parent / f"{sid}_spots.parquet")
        cx = np.rint(df["x"].to_numpy() * scale).astype(int)
        cy = np.rint(df["y"].to_numpy() * scale).astype(int)
        if (cx - half < 0).any() or (cy - half < 0).any():
            clipped.append(sid)
    assert not clipped, f"spots within half a tile of the origin in: {clipped[:3]}"


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
    smallest = min(len(m.read_text().splitlines()) - 1 for m in metas)
    assert smallest > K, f"a subarray of {smallest} spots now exercises the sentinel drop"
