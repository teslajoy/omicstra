"""the pathway chain against the artifact it was transcribed from.

the gene sets that produced the published embeddings are the oracle, and a port
is verified by diffing against them rather than by re-deriving them. this runs
the real comparison where the cohort and the 1.3 GB artifact exist, and skips
elsewhere - every input it needs is git-ignored.

the criterion is EXACT. these are sets of gene symbols chosen by a rank, not
floating-point vectors across backends, so there is no tolerance to declare: a
port that selects a different gene selected a different gene.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "data/embeddings/biological_signals/niche_madmean_genesets.json"
CANON = ROOT / "projects/tnbc-92/data/canonical"
NICHE_META = ROOT / "data/embeddings/virchow2_niche"
E2S = ROOT / "data/inputs/ensembl_to_symbol.tsv"
SUB, K, MT = "TNBC1_CN1_C1", 6, ("MT-", "MTRNR")

# the parameters the published build used, read out of the notebook that wrote
# the oracle and checked against the artifact. they are the COHORT's, and the
# package holds no default for any of them.
PUBLISHED = {
    "niche_qc": {"min_niche_umi": 1000, "min_spot_umi": 200, "min_detected_spots": 3},
    "normalise": {"target_sum": 1e6},
    "variability": {},
    "background": {"min_background": 100},
    "foreground": {"top_fraction": 0.25, "min_foreground": 20},
    "gene_sets": {},
}


def _present() -> bool:
    return (ORACLE.is_file() and E2S.is_file()
            and (CANON / f"{SUB}.h5ad").is_file()
            and (NICHE_META / f"{SUB}_meta.tsv").is_file())


def _oracle_for(sub: str) -> dict[str, list[str]]:
    """this subarray's entries, streamed out of a 1.3 GB file.

    the buffer is truncated after the LAST COMPLETE match rather than to a
    fixed tail. a fixed tail drops an entry longer than it, which silently
    loses exactly the largest sets - and a verifier that quietly misses rows
    turns a passing diff into a false negative.
    """
    pat = re.compile(rf'"({re.escape(sub)}::[^"]+)":\s*\[(.*?)\]')
    out: dict[str, list[str]] = {}
    buf = ""
    with ORACLE.open() as fh:
        while True:
            chunk = fh.read(1 << 22)
            if not chunk:
                break
            buf += chunk
            last = 0
            for m in pat.finditer(buf):
                out[m.group(1)] = [g.strip().strip('"') for g in m.group(2).split(",")]
                last = m.end()
            buf = buf[last:]
            if out and sub not in buf:
                break
    return out


def _mine_for(sub: str) -> dict[str, list[str]]:
    import anndata as ad
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors

    from omicstra.protocols.pathway import (
        background, foreground, gene_sets, niche_qc, normalise, variability,
    )

    e2s = pd.read_csv(E2S, sep="\t")
    e2s_map = dict(zip(e2s["ensembl_id"], e2s["gene_symbol"]))
    all_hgnc = sorted({s for s in e2s_map.values() if s and not s.startswith(MT)})
    gorder = {s: i for i, s in enumerate(all_hgnc)}

    a = ad.read_h5ad(CANON / f"{sub}.h5ad")
    keep_local, keep_global, seen = [], [], set()
    for i, g in enumerate(a.var_names):
        sym = e2s_map.get(re.sub(r"\.\d+$", "", g))
        if sym is None or sym in seen or sym.startswith(MT) or sym not in gorder:
            continue
        seen.add(sym)
        keep_local.append(i)
        keep_global.append(gorder[sym])

    meta = pd.read_csv(NICHE_META / f"{sub}_meta.tsv", sep="\t", index_col=0)
    meta = meta.loc[meta.index.intersection(a.obs_names)]
    pos = {s: i for i, s in enumerate(a.obs_names)}
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    counts = X[[pos[s] for s in meta.index]][:, np.asarray(keep_local)].astype("float32")

    coords = meta[["pixel_x", "pixel_y"]].values
    _, idx = NearestNeighbors(n_neighbors=K + 1).fit(coords).kneighbors(coords)
    sid = meta.index.to_numpy()
    niches = [(f"{sub}::{sid[c]}", counts[idx[c, :K + 1]])
              for c in np.where(meta["neighbor_count"].values >= 7)[0]]

    ctx: dict = {"niches": niches,
                 "gene_symbols": np.array(all_hgnc)[np.asarray(keep_global)].tolist()}
    for fn, key in ((niche_qc, "niche_qc"), (normalise, "normalise"),
                    (variability, "variability"), (background, "background"),
                    (foreground, "foreground"), (gene_sets, "gene_sets")):
        ctx["params"] = PUBLISHED[key]
        ctx.update(fn(ctx)[1])
    return ctx["gene_sets"]


@pytest.mark.skipif(not _present(), reason="cohort or oracle not on this machine")
def test_the_chain_reproduces_the_published_gene_sets_exactly():
    """EXACT, with no tolerance declared. these are sets of symbols chosen by a
    rank, not float vectors across backends - a port that selects a different
    gene selected a different gene.
    """
    mine, orc = _mine_for(SUB), _oracle_for(SUB)
    assert orc, "the oracle yielded no entries for this subarray"
    assert set(mine) == set(orc), (
        f"niche sets differ: {len(set(mine) ^ set(orc))} symmetric difference")
    bad = [k for k in mine if sorted(mine[k]) != sorted(orc[k])]
    assert not bad, f"{len(bad)} of {len(mine)} gene sets differ, e.g. {bad[:3]}"


@pytest.mark.skipif(not _present(), reason="cohort or oracle not on this machine")
def test_the_oracle_reader_does_not_lose_the_largest_entries():
    """the first version of this reader kept a fixed 4 KB tail between chunks
    and dropped the two largest sets - a verifier that quietly misses rows
    turns a passing diff into a false negative, which is worse than failing.
    """
    orc = _oracle_for(SUB)
    sizes = sorted((len(v) for v in orc.values()), reverse=True)
    assert len(orc) > 1000 and sizes[0] > 1000, (
        f"{len(orc)} entries, largest {sizes[0]} - the big ones are missing")
