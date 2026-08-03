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


_R_SPARSE = '''
suppressMessages(library(Matrix))
load("{src}")
m <- as(as(as(cnts, "dMatrix"), "generalMatrix"), "CsparseMatrix")
writeMM(m, "{d}/counts.mtx")
writeLines(rownames(cnts), "{d}/rows.txt")
writeLines(colnames(cnts), "{d}/cols.txt")
write.csv(spots, "{d}/spots.csv")
'''


def _read_sparse(path: Path):
    """counts as Matrix Market + coords as a small CSV.

    the matrix is ~16% dense, so writing triplets instead of a dense CSV grid is
    about 6x faster end to end (12s -> 2s per sample). falls back to the CSV path
    if R's Matrix package is unavailable.
    """
    import tempfile

    import scipy.io as sio

    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run(["Rscript", "-e", _R_SPARSE.format(src=path, d=d)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"sparse read failed for {path}: {r.stderr[:300]}")
        X = sio.mmread(f"{d}/counts.mtx").tocsr()
        rows = Path(f"{d}/rows.txt").read_text().rstrip("\n").split("\n")
        cols = Path(f"{d}/cols.txt").read_text().rstrip("\n").split("\n")
        coords = pd.read_csv(f"{d}/spots.csv", index_col=0)
    return X, rows, cols, coords


def load_sample(by_array: str | Path, slide: str, position: str,
                gene_map: dict[str, str] | None = None, sparse: bool = True):
    """one sample -> AnnData with raw counts, spatial coords, and a sample id."""
    import anndata as ad
    from scipy.sparse import csr_matrix

    p = Path(by_array) / slide / position / "selection.RData"

    if sparse:
        try:
            X, obs_names, var_ids, coords = _read_sparse(p)
        except Exception:
            sparse = False
    if not sparse:
        counts = _read_rdata(p, "cnts")
        coords = _read_rdata(p, "spots")
        X = csr_matrix(counts.values.astype(np.float32))
        obs_names, var_ids = [str(i) for i in counts.index], list(counts.columns)

    # keep only observations that carry coordinates
    keep = [i for i, o in enumerate(obs_names) if o in set(coords.index)]
    X = X[keep]
    obs_names = [obs_names[i] for i in keep]
    coords = coords.loc[obs_names]

    var_names = ([gene_map.get(str(c).split(".")[0], str(c)) for c in var_ids]
                 if gene_map else [str(c) for c in var_ids])

    a = ad.AnnData(X.astype(np.float32))
    a.var_names = var_names
    a.var_names_make_unique()
    a.var["ensembl_id"] = [str(c).split(".")[0] for c in var_ids]
    a.obs_names = obs_names
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