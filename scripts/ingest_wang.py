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
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Rscript failed on {path}: {r.stderr[:200]}")
    return pd.read_csv(io.StringIO(r.stdout), index_col=0)


def discover() -> list[tuple[str, str, Path]]:
    """(sample_id, patient_id, selection.RData).

    the sample id is TNBC{n}_{slide}_{position}, which is both the cache's naming
    and the only place the PATIENT is recorded. byArray/ is keyed by slide and
    position alone, so the patient has to come from the HD image filename - and
    it must, because patient is the confounder axis every split is held out on.
    a sample whose patient cannot be determined is skipped rather than ingested
    with a guess.
    """
    by_slide_pos = {}
    for img in IMAGES_HD.glob("*.jpg"):
        parts = img.stem.split("_")
        if len(parts) >= 3:
            by_slide_pos[(parts[1], parts[2])] = parts[0]

    out = []
    for p in sorted(BY_ARRAY.glob("*/*/selection.RData")):
        slide, pos = p.parent.parent.name, p.parent.name
        patient = by_slide_pos.get((slide, pos))
        if patient is None:
            continue
        out.append((f"{patient}_{slide}_{pos}", patient, p))
    return out


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

    samples = discover()
    if limit:
        samples = samples[:limit]
    if not samples:
        raise SystemExit(f"no selection.RData under {BY_ARRAY}")

    record = {"cohort": "tnbc-92", "source": "byArray/{slide}/{position}/selection.RData",
              "coords_space": "HE-small pixels, as recorded in `spots`. the scale onto the "
                              "HD image is declared in platform.json#image_pyramid.",
              "subject_id_column": "patient_id", "samples": [], "sources": {}, "images": {}, "counts_written": not coords_only}

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
        record["sources"][sid] = {"rdata": str(rdata.relative_to(ROOT)), "sha256_16": _sha(rdata),
                                  "n_spots": int(len(spots)), "patient_id": patient}
        img = hd_image_for(slide, pos)
        if img:
            record["images"][sid] = str(img)
        click.echo(f"  {sid}: {len(spots)} spots{'' if img else '  (no HD image found)'}")

    (out / "ingest.json").write_text(json.dumps(record, indent=2) + "\n")
    click.echo(f"\nwrote {len(record['samples'])} samples -> {out.relative_to(ROOT)}")
    if coords_only:
        click.echo("coordinates only. the molecular path needs .h5ad - rerun without "
                   "--coords-only when the ST side is ported.")


if __name__ == "__main__":
    sys.exit(main())