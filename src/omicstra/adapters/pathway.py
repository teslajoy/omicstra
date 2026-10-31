"""canonical data -> the context the pathway protocol chains read.

the three pathway chains (NICHE_MEMBERSHIP_STEPS, VARIABLE_GENE_STEPS,
AUCELL_STEPS) all need expression counts grouped by niche, plus gene
identifiers and (for AUCell) pathway gene sets. this adapter reads
from canonical h5ad files and pre-built niche parquets, and nothing in
it decides what a threshold should be - those arrive as params.

    arrays(project_dir, sample)  ->  the ctx a pathway step reads

the shape matches `adapters/join.arrays` and `adapters/evaluation.arrays`:
a function that takes a project dir + a key and returns a dict. it is
named in `notebook.UNITS` so the notebook generator can emit a setup
cell that imports it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omicstra.adapters.canonical import CanonicalMissing


class PathwayDataMissing(FileNotFoundError):
    """a required input for the pathway chain is absent."""


def _gene_map(project_dir: str | Path) -> dict[str, str]:
    """ensembl -> symbol, from the cohort's declared mapping.

    absent means refuse. the gene_mapping declaration in cohort.json names the
    file and the key/value columns so a second cohort can use its own release.
    """
    root = Path(project_dir)
    cohort = json.loads((root / "cohort.json").read_text())
    gm = cohort.get("gene_mapping")
    if not gm:
        raise PathwayDataMissing(
            "cohort.json declares no gene_mapping - the pathway chain needs "
            "gene symbols and the canonical data uses Ensembl IDs")
    import pandas as pd

    from omicstra.contracts.project import ProjectConfig
    cfg = ProjectConfig.from_dir(root)
    path = cfg.path(gm["map"])
    if not path.is_file():
        raise PathwayDataMissing(f"gene mapping file not found: {path}")
    df = pd.read_csv(path, sep="\t")
    return dict(zip(df[gm["map_key"]], df[gm["map_value"]], strict=False))


def _niche_dir(project_dir: str | Path) -> Path:
    """the directory holding per-sample niche parquets.

    reads project.json for the niches dir declaration. tnbc-92 stores niches
    at `data/embeddings/niches_v3/`; a second cohort declares its own path.
    """
    root = Path(project_dir)
    raw = json.loads((root / "project.json").read_text())
    from omicstra.contracts.project import ProjectConfig
    cfg = ProjectConfig.from_dir(root)
    d = raw.get("niches_dir")
    if not d:
        raise PathwayDataMissing(
            "project.json declares no niches_dir - the pathway chain needs "
            "pre-built niche parquets with neighbor_spot_ids")
    return cfg.path(d)


def arrays(project_dir: str | Path, sample: str) -> dict[str, Any]:
    """the ctx a pathway protocol step reads, for ONE sample of one cohort.

    reads the canonical h5ad (expression matrix + gene axis), the niche
    parquet (niche membership via neighbor_spot_ids), and the gene mapping
    (Ensembl -> symbols). builds:

        genes           the gene axis (Ensembl IDs)
        gene_symbols    the gene axis in symbols
        niche_members   {centre_spot_id: [neighbour_spot_ids]}
        niches          [(niche_id, counts_matrix)] - centre + neighbours
        niche_groups    {niche_id: sample_id} for z-scoring by batch
        pathway_sets    not loaded here - passed as params or loaded by the
                        pathway_net step from declared gene-set files
    """
    import anndata as ad
    import numpy as np
    import pandas as pd

    root = Path(project_dir)

    h5_path = root / "data" / "canonical" / f"{sample}.h5ad"
    if not h5_path.is_file():
        raise PathwayDataMissing(f"no canonical h5ad for {sample}: {h5_path}")
    adata = ad.read_h5ad(h5_path)

    ensembl_ids = list(adata.var.index)
    ens_to_sym = _gene_map(project_dir)
    symbols = [ens_to_sym.get(e.split(".")[0], e) for e in ensembl_ids]

    niche_path = _niche_dir(project_dir) / f"{sample}.parquet"
    if not niche_path.is_file():
        raise PathwayDataMissing(
            f"no niche parquet for {sample}: {niche_path}")
    niche_df = pd.read_parquet(niche_path)
    if "neighbor_spot_ids" not in niche_df.columns:
        raise PathwayDataMissing(
            f"{niche_path} has no neighbor_spot_ids column")

    spot_index = {sid: i for i, sid in enumerate(adata.obs.index)}
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype="float32")

    niche_members: dict[str, list[str]] = {}
    niches: list[tuple[str, np.ndarray]] = []
    niche_groups: dict[str, str] = {}

    for centre in niche_df.index:
        if centre not in spot_index:
            continue
        neighbours = niche_df.at[centre, "neighbor_spot_ids"]
        if hasattr(neighbours, "tolist"):
            neighbours = neighbours.tolist()
        elif isinstance(neighbours, str):
            neighbours = json.loads(neighbours)
        members = [centre] + [n for n in neighbours if n in spot_index]
        niche_members[centre] = members
        rows = np.stack([X[spot_index[s]] for s in members])
        niches.append((centre, rows))
        niche_groups[centre] = sample

    return {
        "genes": ensembl_ids,
        "gene_symbols": symbols,
        "niche_members": niche_members,
        "niches": niches,
        "niche_groups": niche_groups,
        "project_dir": str(project_dir),
    }
