"""niche construction and tile geometry - the maths, with no cohort in it.

lifted from scripts/extract_virchow2_niche.py. what stayed behind there: reading
Wang's .RData, globbing their directory layout, and the constants 9523 / 31744 /
128.0, which are this cohort's image pyramid and are now declared in
platform.json. what moved here is the part that is the same for any cohort with
spots, coordinates and an image.

the script remains the ORACLE. every function below is diffed against it on a
real subarray, so a future change to this file that alters a single vector is
caught rather than discovered in a result six weeks later.

faithfulness over improvement
-----------------------------
several things here are not what you would write from scratch: neighbours are
taken by rank rather than by distance, a clipped tile is padded with black rather
than reflected, and the pool is unweighted. each reproduces the published grid.
changing any of them is a declared arm, not a tidy-up, because every arm in that
grid saw identical tiles and that is the only reason the comparison is fair.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TileGeometry:
    """how a spot coordinate becomes a crop. every field is declared, not inferred.

    `tile_px` is the integer that sizes the crop and is what the run identity
    hashes. the micron figure it corresponds to is derived from a measured
    nearest-neighbour distance, so hashing that instead would move the identity
    whenever someone re-measures the lattice.
    """
    scale: float        # coordinate space -> image space (platform.image_pyramid)
    tile_px: int        # crop side in image space (platform.he_tile.tile_px_hd)
    out_px: int = 224   # encoder input side


def niche_neighbours(coords: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """k nearest spots per spot, INCLUDING self, plus how many are real.

    two behaviours that look like bugs and are not:

    - the query asks for ``k + 1`` because a point is its own nearest neighbour,
      and the self index is kept: a niche is the spot plus its neighbourhood, not
      the neighbourhood around a hole.
    - where a subarray has fewer spots than requested, scipy returns ``n`` as a
      sentinel index. those are dropped, so boundary spots pool over however many
      neighbours exist rather than being discarded. keeping them is a choice -
      they are real tissue - and `counts` is what makes it auditable afterwards.

    neighbours are taken by RANK, not by a distance cutoff. on an anisotropic
    lattice that means a k=6 niche is not a symmetric disc; it is whatever the
    six nearest happen to be. the published grid was built this way.
    """
    from scipy.spatial import KDTree

    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n == 0:
        return np.empty((0, 0), dtype=int), np.empty(0, dtype=int)

    k_query = min(k + 1, n)
    _, idx = KDTree(coords).query(coords, k=k_query)
    idx = np.atleast_2d(idx.reshape(n, -1))

    counts = np.array([int((row < n).sum()) for row in idx], dtype=np.int32)
    return idx, counts


def tile_boxes(coords: np.ndarray, geom: TileGeometry,
               image_size: tuple[int, int]) -> np.ndarray:
    """one clamped crop box per coordinate: (x1, y1, x2, y2), image space.

    boxes at the image edge come back SMALLER than `tile_px` - `cut_tiles` pads
    those back to size. this function exists so a caller can see WHICH spots sat
    on the boundary without opening the image or re-deriving it from the vectors.
    """
    coords = np.asarray(coords, dtype=float)
    w, h = image_size
    half = geom.tile_px // 2
    cx = np.rint(coords[:, 0] * geom.scale).astype(int)
    cy = np.rint(coords[:, 1] * geom.scale).astype(int)
    return np.stack([np.maximum(0, cx - half), np.maximum(0, cy - half),
                     np.minimum(w, cx + half), np.minimum(h, cy + half)], axis=1)


def cut_tiles(image, coords: np.ndarray, geom: TileGeometry) -> list:
    """coordinates -> encoder-ready tiles. the whole geometry in one place."""
    from PIL import Image

    w, h = image.size
    half = geom.tile_px // 2
    out = []
    for x, y in np.asarray(coords, dtype=float):
        cx, cy = int(round(x * geom.scale)), int(round(y * geom.scale))
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)
        patch = image.crop((x1, y1, x2, y2))
        if patch.size != (geom.tile_px, geom.tile_px):
            canvas = Image.new("RGB", (geom.tile_px, geom.tile_px), (0, 0, 0))
            canvas.paste(patch, (half - (cx - x1), half - (cy - y1)))
            patch = canvas
        out.append(patch.resize((geom.out_px, geom.out_px), Image.LANCZOS))
    return out


def pool_niche(vectors: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """unweighted mean over each spot's neighbourhood, sentinels dropped.

    unweighted on purpose. distance weighting is defensible and would be a
    different arm: it changes every vector, and the grid's fairness rests on all
    ten arms having seen the same pooling.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    n, dim = vectors.shape
    out = np.zeros((n, dim), dtype=np.float32)
    for j in range(n):
        row = np.atleast_1d(idx[j])
        valid = row[row < n]
        out[j] = vectors[valid].mean(axis=0) if len(valid) else vectors[j]
    return out


def tile_um(tile_px: int, nn_px: int | float, pitch_um: float) -> float:
    """how much tissue a crop actually covers, given the measured lattice.

    this is the function the original constant stood in for. it is the reason a
    tile named 128 um was 95.9: the crop was sized as ``128 * nn / pitch`` with a
    pitch that was later falsified, so the pitch was an INPUT to the geometry
    rather than a label on it. derived, never hashed.
    """
    return float(tile_px) * (float(pitch_um) / float(nn_px))