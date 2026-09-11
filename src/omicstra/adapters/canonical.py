"""the package's input contract: .h5ad plus a coordinates table.

one reader, N ingest scripts. a cohort's native format is converted ONCE by
`scripts/ingest_<cohort>.py` and never again; nothing in the package imports an
R reader, a vendor SDK, or a proprietary loader. the point is not tidiness - it
is that `omicstra` must install and run for someone who has never heard of the
format the seed cohort happened to arrive in.

    cohort native  --ingest script-->  canonical  --package-->  everything
    .RData, .tif                       .h5ad
    .h5ad, .parquet                    _spots.parquet
    vendor bundle                      ingest.json

why coordinates are a SEPARATE file from the counts
---------------------------------------------------
the H&E path needs `spot_id, x, y` and nothing else. if coordinates only lived in
`adata.obsm["spatial"]`, cutting tiles would mean loading a counts matrix -
27,567 genes x 1,075 spots per subarray - to read two columns. the split is what
lets the morphology side run without touching expression at all.

why this file is thin
---------------------
it deliberately does almost nothing. every line of cohort knowledge that ends up
here is a line that a second cohort will have to work around, so the test of this
module is not that it handles tnbc-92 well - it is that adding hest-breast costs
an ingest script and no edit here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class CanonicalMissing(FileNotFoundError):
    """no canonical form for this cohort, and the message says how to make one.

    a refusal rather than a fallback: silently reading a cohort's native files
    would put format knowledge back in the package, which is the thing this
    module exists to prevent.
    """


@dataclass(frozen=True)
class Sample:
    """one unit of ingest - a subarray, a section, a slide. what it is called is
    the cohort's business; what it must provide is not."""
    sample_id: str
    counts: Path | None          # .h5ad. absent is legal: the H&E path never reads it
    spots: Path | None           # _spots.parquet: spot_id, x, y
    image: Path | None           # the stained image, if this cohort has one

    def has(self, *roles: str) -> bool:
        return all(getattr(self, r) is not None and getattr(self, r).exists() for r in roles)


def canonical_dir(project_root: str | Path) -> Path:
    return Path(project_root) / "data" / "canonical"


def read_ingest(project_root: str | Path) -> dict:
    """the ingest record: what was converted, from what, with which shas.

    this is the provenance that survives the conversion. without it the canonical
    files are anonymous - you cannot tell which .RData produced which .h5ad, and
    a re-ingest that silently changes one becomes undetectable.
    """
    p = canonical_dir(project_root) / "ingest.json"
    if not p.is_file():
        raise CanonicalMissing(
            f"no ingest record at {p}. run scripts/ingest_<cohort>.py once to convert "
            "this cohort's native files into .h5ad + _spots.parquet. the package reads "
            "only the canonical form, so that a cohort in any format costs an ingest "
            "script rather than a change here.")
    return json.loads(p.read_text())


def list_samples(project_root: str | Path) -> list[Sample]:
    """every sample the ingest produced, whether or not each part is present.

    a sample missing `counts` is returned rather than skipped: the H&E path is
    entitled to run on it, and the EDA gate is the thing that decides whether an
    absent role is a caution or a halt. hiding it here would make that decision
    for a layer that has more context than this one.
    """
    d = canonical_dir(project_root)
    rec = read_ingest(project_root)
    out = []
    for sid in rec.get("samples", []):
        h5, sp = d / f"{sid}.h5ad", d / f"{sid}_spots.parquet"
        img = rec.get("images", {}).get(sid)
        out.append(Sample(sample_id=sid,
                          counts=h5 if h5.exists() else None,
                          spots=sp if sp.exists() else None,
                          image=Path(img) if img else None))
    return out


def load_spots(sample: Sample):
    """the coordinates table. columns: spot_id, x, y - in the cohort's own space.

    x and y are NOT microns and NOT image pixels-at-full-resolution. they are
    whatever space the cohort records positions in, and `platform.json` declares
    the scale that maps them onto the image. keeping them raw is what lets the
    pitch be corrected later without re-ingesting anything, which is exactly what
    happened here.
    """
    import pandas as pd

    if not sample.has("spots"):
        raise CanonicalMissing(
            f"{sample.sample_id} has no coordinates table. the H&E path needs "
            "spot_id, x, y; re-run the ingest script for this cohort.")
    df = pd.read_parquet(sample.spots)
    missing = {"spot_id", "x", "y"} - set(df.columns)
    if missing:
        raise CanonicalMissing(
            f"{sample.spots.name} is missing {sorted(missing)}. the canonical "
            "coordinates table is exactly spot_id, x, y plus anything else a cohort "
            "wants to carry.")
    return df


def load_counts(sample: Sample):
    """the AnnData. only the molecular path calls this."""
    import anndata as ad

    if not sample.has("counts"):
        raise CanonicalMissing(
            f"{sample.sample_id} has no .h5ad. the molecular path needs one; the "
            "morphology path does not and can proceed without it.")
    return ad.read_h5ad(sample.counts)