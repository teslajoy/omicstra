"""adapter: Wang et al. original-ST cohort -> AnnData.

an adapter's only job is to produce AnnData in the shape the EDA steps expect.
it is the one place that knows a cohort's file layout, so no step ever learns
a file format.

this cohort stores raw integer counts in `selection.RData` as a matrix `cnts`
(observations x genes, versioned Ensembl IDs) plus a `spots` data.frame carrying
pixel coordinates. the sibling `rawCountsMatrices/` directory holds normalised
and batch-corrected floats despite its name - the EDA count_statistics step
tests for this rather than trusting either name.

requires Rscript on PATH. pyreadr cannot read R matrix objects, and rpy2 needs a
matching R version, so a subprocess is the reliable path.
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

_R_READ = 'load("{path}"); write.csv({obj}, stdout())'


def _read_rdata(path: Path, obj: str) -> pd.DataFrame:
    r = subprocess.run(["Rscript", "-e", _R_READ.format(path=path, obj=obj)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Rscript failed reading {obj} from {path}: {r.stderr[:300]}")
    if not r.stdout.strip():
        raise RuntimeError(f"Rscript returned nothing for {obj} in {path}")
    return pd.read_csv(io.StringIO(r.stdout), index_col=0)


def list_samples(by_array: str | Path) -> list[tuple[str, str]]:
    """(slide, position) for every sample carrying selection.RData."""
    root = Path(by_array)
    out = []
    for slide in sorted(p for p in root.iterdir() if p.is_dir()):
        for pos in sorted(p for p in slide.iterdir() if p.is_dir()):
            if (pos / "selection.RData").exists():
                out.append((slide.name, pos.name))
    return out


def load_sample(by_array: str | Path, slide: str, position: str,
                gene_map: dict[str, str] | None = None):
    """one sample -> AnnData with raw counts, spatial coords, and a sample id."""
    import anndata as ad
    from scipy.sparse import csr_matrix

    p = Path(by_array) / slide / position / "selection.RData"
    counts = _read_rdata(p, "cnts")
    coords = _read_rdata(p, "spots")

    common = counts.index.intersection(coords.index)
    counts, coords = counts.loc[common], coords.loc[common]

    var_names = list(counts.columns)
    if gene_map:
        # map versioned Ensembl -> symbol where possible; keep the ID otherwise
        var_names = [gene_map.get(str(c).split(".")[0], str(c)) for c in counts.columns]

    a = ad.AnnData(csr_matrix(counts.values.astype(np.float32)))
    a.var_names = var_names
    a.var_names_make_unique()
    a.var["ensembl_id"] = [str(c).split(".")[0] for c in counts.columns]
    a.obs_names = [str(i) for i in counts.index]
    a.obs["sample"] = f"{slide}_{position}"

    xcol = "pixel_x" if "pixel_x" in coords.columns else coords.columns[0]
    ycol = "pixel_y" if "pixel_y" in coords.columns else coords.columns[1]
    a.obsm["spatial"] = coords[[xcol, ycol]].to_numpy(dtype=float)
    return a


def load_gene_map(path: str | Path) -> dict[str, str]:
    df = pd.read_csv(path, sep="\t")
    return dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))


def load_cohort(by_array: str | Path, samples: list[tuple[str, str]] | None = None,
                gene_map_path: str | Path | None = None, max_samples: int | None = None):
    """concatenate samples into one AnnData on the intersection of their gene sets."""
    import anndata as ad

    gene_map = load_gene_map(gene_map_path) if gene_map_path else None
    picks = samples if samples is not None else list_samples(by_array)
    if max_samples:
        picks = picks[:max_samples]

    parts = []
    for slide, pos in picks:
        try:
            parts.append(load_sample(by_array, slide, pos, gene_map))
        except Exception as e:  # a dead sample must not kill the cohort load
            print(f"  skipped {slide}/{pos}: {type(e).__name__}: {e}")
    if not parts:
        raise RuntimeError("no samples loaded")
    return ad.concat(parts, join="inner", index_unique="-")