"""ingest: Wang et al. original-ST cohort -> canonical (.h5ad + _spots.parquet).

run ONCE. after this, nothing in omicstra reads .RData, globs byArray/, or knows
that this cohort arrived from R. that is the whole purpose: the package's input
contract is .h5ad plus a coordinates table, and a cohort in any other format
costs an ingest script rather than a change inside src/omicstra.

    byArray/{slide}/{position}/selection.RData   --->  {sid}.h5ad
                                                        {sid}_spots.parquet
                                                        ingest.json

requires Rscript. that dependency lives HERE, in scripts/, and nowhere else -
pyreadr cannot read R matrix objects and rpy2 needs a matching R version, so a
subprocess is the reliable path, but it is a one-time cost rather than a runtime
one.

coordinates stay in the cohort's own space
------------------------------------------
`x` and `y` are written exactly as `pixel_x` / `pixel_y` appear in `spots` -
HE-small space, not microns and not HD pixels. the scale onto the image is
declared in platform.json#image_pyramid. keeping them raw is why the 2026-08-12
pitch correction cost a declaration edit rather than a re-ingest, and why the
128um tile can become an arm without touching a file on disk.

usage:
    python scripts/ingest_wang.py --limit 2      # a couple, to check the shape
    python scripts/ingest_wang.py                # all 281
    python scripts/ingest_wang.py --coords-only  # skip counts: the H&E path only
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parent.parent
BY_ARRAY = ROOT / "data" / "inputs" / "byArray"
IMAGES_HD = ROOT / "data" / "inputs" / "Images" / "imagesHD"
IMAGES_LARGE = ROOT / "data" / "inputs" / "Images" / "imagesLarge"


def rel(p: Path) -> str:
    """a path as the record stores it: relative to the repo root, always.

    every path written into ingest.json goes through here. an absolute path in
    that file would carry one machine's home directory into a record whose whole
    purpose is to be portable, and it would do it silently - the file looks fine
    until someone else reads it. a source outside ROOT is a real condition, not
    something to paper over with a fallback, so it raises.
    """
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        raise ValueError(
            f"{p} is outside the repo root {ROOT}; the record stores repo-relative "
            "paths only, so a source elsewhere needs a declared base rather than "
            "an absolute path here.") from None


def _sha(p: Path, blocks: int = 64) -> str:
    """sha256 of the first blocks*1MB. full-file hashing 281 RData files is slow
    and the prefix is enough to notice a source swap, which is what this is for."""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for _ in range(blocks):
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def _rscript(path: Path, obj: str):
    import pandas as pd

    r = subprocess.run(["Rscript", "-e", f'load("{path}"); write.csv({obj}, stdout())'],
                       capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"Rscript failed on {path}: {r.stderr[:200]}")
    return pd.read_csv(io.StringIO(r.stdout), index_col=0)


def _patient_map() -> dict[tuple[str, str], str]:
    """(slide, position) -> patient, from image filenames at either resolution.

    byArray/ is keyed by slide and position alone, so the patient has to come
    from an image filename - and it must, because patient is the confounder axis
    every split is held out on. BOTH pyramids are read: identity is in the name
    and does not depend on which resolution happens to be present, and reading
    only imagesHD conflates "unknown patient" with "no HD image". those are
    different facts and the record has to say which.
    """
    by_slide_pos = {}
    for d in (IMAGES_LARGE, IMAGES_HD):          # HD last: it wins on a conflict
        for img in d.glob("*.jpg"):
            parts = img.stem.split("_")
            if len(parts) >= 3:
                by_slide_pos[(parts[1], parts[2])] = parts[0]
    return by_slide_pos


# R writes the counts as MatrixMarket rather than CSV. `cnts` is 1075 x 27567 and
# 84% zero on the smallest sample; as dense CSV that is ~30M numbers through a
# pipe, 280 times over. MatrixMarket carries the nonzeros only and scipy reads it
# directly, so the conversion stays seconds per sample rather than minutes.
#
# the id files are written alongside because MatrixMarket records shape but not
# names, and the names are the point: versioned Ensembl ids in the genes, Wang's
# spot ids in the rows.
_COUNTS_R = """
suppressMessages(library(Matrix))
load("%(rdata)s")
stopifnot(exists("cnts"))
m <- as(Matrix(t(cnts), sparse = TRUE), "CsparseMatrix")
writeMM(m, "%(mtx)s")
writeLines(rownames(cnts), "%(spot_ids)s")
writeLines(colnames(cnts), "%(gene_ids)s")
"""


def _read_counts(rdata, tmp):
    """(csr spots x genes, spot_ids, gene_ids) for one selection.RData.

    R hands over genes x spots because that is what writeMM wants from a column
    matrix; the transpose here puts observations first, which is anndata's
    convention and, for this cohort, means one row per spot.
    """
    from scipy.io import mmread

    mtx, sids, gids = tmp / "c.mtx", tmp / "s.txt", tmp / "g.txt"
    script = _COUNTS_R % {"rdata": rdata, "mtx": mtx, "spot_ids": sids, "gene_ids": gids}
    r = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"Rscript counts failed on {rdata}: {r.stderr[:300]}")
    return (mmread(mtx).T.tocsr(),
            sids.read_text().splitlines(), gids.read_text().splitlines())


def write_h5ad(rdata, spots, dest, patient_id, sample_id):
    """one sample -> one .h5ad, spots x genes, raw integer counts.

    three things are asserted rather than assumed, because each is a gate input:

    - counts stay INTEGER. `% expressing` is a detection rate only on raw counts,
      which is the check eda_contract.json calls universal. R stores them as
      double and they are integerish; this narrows the dtype, it rounds nothing.
    - gene ids stay VERSIONED ENSEMBL and unmapped. symbols are for display, and
      mapping here would bake one annotation release into the artifact.
    - the counts rows are REINDEXED onto the coordinate order, never zipped to
      it. they happen to agree on this cohort. a silent zip that is right by luck
      does not survive a cohort where they disagree, and would be undetectable.
    """
    import tempfile

    import anndata as ad
    import numpy as np
    import pandas as pd

    with tempfile.TemporaryDirectory() as td:
        X, spot_ids, gene_ids = _read_counts(rdata, Path(td))

    if not np.allclose(X.data, np.rint(X.data)):
        raise ValueError(f"{sample_id}: counts are not integer - a detection rate "
                         "computed on these would not be a detection rate")
    X.data = X.data.astype(np.int32)

    want = pd.Index(spots.index.astype(str))
    order = pd.Index(spot_ids).get_indexer(want)
    if (order < 0).any():
        raise ValueError(f"{sample_id}: {int((order < 0).sum())} coordinate spot(s) "
                         "absent from the counts matrix; the two halves disagree")
    X = X[order]

    obs = pd.DataFrame({"spot_id": want, "patient_id": patient_id,
                        "sample_id": sample_id}, index=want)
    var = pd.DataFrame(index=pd.Index(gene_ids, name="ensembl_id"))
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.write_h5ad(dest, compression="gzip")
    return {"n_genes": int(a.n_vars), "n_obs": int(a.n_obs), "nnz": int(X.nnz),
            "dtype": "int32", "gene_id": "ensembl_versioned"}


def discover() -> tuple[list[tuple[str, str, Path]], list[dict]]:
    """(ingestable, skipped). ingestable is (sample_id, patient_id, RData).

    the sample id is TNBC{n}_{slide}_{position}, which is the cache's naming.

    two reasons a sample is skipped, and they are NOT the same reason:

      no_patient    no image at any resolution names it. ingesting it would mean
                    guessing the confounder axis, so it is dropped.
      no_hd_image   the patient is known, the counts are there, and the HD image
                    the morphology path cuts tiles from is not. the ST side could
                    be ingested; the H&E side could not, so the sample would enter
                    the grid as half a sample.

    both are returned rather than dropped in silence, because a count that falls
    from 281 to 280 with no reason attached is the kind of thing that gets
    rediscovered a year later.
    """
    by_slide_pos = _patient_map()

    out, skipped = [], []
    for p in sorted(BY_ARRAY.glob("*/*/selection.RData")):
        slide, pos = p.parent.parent.name, p.parent.name
        patient = by_slide_pos.get((slide, pos))
        if patient is None:
            skipped.append({"slide": slide, "position": pos, "rdata": rel(p),
                            "reason": "no_patient",
                            "detail": "no image at either resolution names this "
                                      "slide/position, so the patient - the axis every "
                                      "split is held out on - would be a guess"})
            continue
        if hd_image_for(slide, pos) is None:
            skipped.append({"slide": slide, "position": pos, "rdata": rel(p),
                            "sample_id": f"{patient}_{slide}_{pos}",
                            "patient_id": patient, "reason": "no_hd_image",
                            "detail": "patient is known and counts exist, but there is no "
                                      "imagesHD/ file to cut tiles from. this is the "
                                      "281 -> 280 drop, and it matches the cached grid"})
            continue
        out.append((f"{patient}_{slide}_{pos}", patient, p))
    return out, skipped


def hd_image_for(slide: str, pos: str) -> Path | None:
    hits = sorted(IMAGES_HD.glob(f"*_{slide}_{pos}*.jpg"))
    return hits[0] if hits else None


@click.command()
@click.option("--limit", default=0, help="ingest only the first N samples.")
@click.option("--coords-only", is_flag=True,
              help="write coordinates and skip the counts matrix. the morphology "
                   "path never reads .h5ad, so this is the fast route to tiles.")
@click.option("--out", default=None, type=click.Path(path_type=Path),
              help="canonical directory. defaults to projects/tnbc-92/data/canonical.")
def main(limit: int, coords_only: bool, out: Path | None) -> None:
    """convert this cohort once into the shape the package reads."""
    import pandas as pd

    out = out or (ROOT / "projects" / "tnbc-92" / "data" / "canonical")
    out.mkdir(parents=True, exist_ok=True)

    samples, skipped = discover()
    if limit:
        samples = samples[:limit]
    if not samples:
        raise SystemExit(f"no selection.RData under {BY_ARRAY}")

    record = {"cohort": "tnbc-92", "source": "byArray/{slide}/{position}/selection.RData",
              "coords_space": "HE-small pixels, as recorded in `spots`. the scale onto the "
                              "HD image is declared in platform.json#image_pyramid.",
              "paths_relative_to": "the omicstra repo root - see `rel()`. tnbc-92 is the "
                                   "one cohort whose data sits outside its own project "
                                   "root, which its project.json declares as `../../data`.",
              "subject_id_column": "patient_id", "samples": [], "sources": {}, "images": {},
              # what was WRITTEN, never what was asked for. this field said
              # `not coords_only` while nothing wrote a .h5ad at all, so a full
              # run would have recorded counts it did not have.
              "counts_written": False, "counts": {}, "skipped": skipped}

    for sid, patient, rdata in samples:
        slide, pos = rdata.parent.parent.name, rdata.parent.name
        spots = _rscript(rdata, "spots")
        cols = {c.lower(): c for c in spots.columns}
        xs, ys = cols.get("pixel_x"), cols.get("pixel_y")
        if not (xs and ys):
            click.echo(f"  {sid}: no pixel_x/pixel_y, skipped", err=True)
            continue

        pd.DataFrame({"spot_id": spots.index.astype(str),
                      "x": spots[xs].astype(float),
                      "y": spots[ys].astype(float)}).to_parquet(out / f"{sid}_spots.parquet",
                                                                index=False)
        record["samples"].append(sid)
        record["sources"][sid] = {"rdata": rel(rdata), "sha256_16": _sha(rdata),
                                  "n_spots": len(spots), "patient_id": patient}
        img = hd_image_for(slide, pos)
        if img:
            record["images"][sid] = rel(img)

        counts = ""
        if not coords_only:
            info = write_h5ad(rdata, spots, out / f"{sid}.h5ad", patient, sid)
            record["counts"][sid] = info
            counts = f", {info['n_genes']} genes, {info['nnz']:,} nonzero"
        click.echo(f"  {sid}: {len(spots)} spots{counts}"
                   f"{'' if img else '  (no HD image found)'}")

        # written every sample, not once at the end. an ingest interrupted at
        # sample 200 otherwise leaves 200 .h5ad files and no record of them,
        # which is worse than leaving nothing.
        record["counts_written"] = bool(record["counts"])
        (out / "ingest.json").write_text(json.dumps(record, indent=2) + "\n")

    (out / "ingest.json").write_text(json.dumps(record, indent=2) + "\n")
    try:
        where = out.resolve().relative_to(ROOT)
    except ValueError:
        where = out                       # --out elsewhere: a console line, not the record
    click.echo(f"\nwrote {len(record['samples'])} samples -> {where}")
    for sk in skipped:
        click.echo(f"  skipped {sk['slide']}/{sk['position']}: {sk['reason']}")
    if coords_only:
        click.echo("coordinates only. the molecular path needs .h5ad - rerun without "
                   "--coords-only when the ST side is ported.")


if __name__ == "__main__":
    sys.exit(main())