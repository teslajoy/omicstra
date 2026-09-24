"""ingest: HEST-1k breast cohort -> canonical (.h5ad + _spots.parquet).

the second cohort's converter, and the witness for the four-way split. it was
committed as a stub so that finishing it would test one claim: that adding a
cohort needs a declaration, an adapter-shaped conversion and nothing inside
`src/omicstra`. if a line here had to change the package, that is where
"declarations only" tears.

what the conversion is
----------------------
HEST already ships .h5ad, so this is nearly a no-op: link the object, write the
coordinates table beside it, record what came from where. the canonical
contract is ".h5ad plus a coordinates table", and the split exists because the
morphology arm needs spot_id, x, y and must not load a 20,000-gene matrix to
read two columns.

three things this cohort has that the seed cohort cannot reveal
--------------------------------------------------------------
- THREE platforms in one cohort - original-ST, Visium, Xenium. platform is a
  property of a SAMPLE, so platform.json keys samples individually. this script
  reads that declaration and refuses a sample it does not cover, rather than
  guessing from the object's schema: HEST stores Xenium in a spot schema, so
  sniffing columns reports a cell platform as a spot one, confidently.
- a platform with NO PITCH. Xenium declares neither pitch nor spot diameter.
  nothing here invents them, and the geometry step reports the spacing without
  a pitch rather than deriving one.
- a TARGETED PANEL rather than a whole transcriptome, which is a different claim
  for any gene-set statistic. that is the EDA contract's applicability question,
  not this script's.

the objects are LINKED, not copied
----------------------------------
hard links where the filesystem allows, else a symlink. 7.9 GB copied twice is
7.9 GB of the same bytes, and the source is read-only for this cohort. the sha
in the record is what ties the canonical file to what it came from.

usage
-----
    python scripts/ingest_hest.py --project-dir ~/path/to/hest-breast
    python scripts/ingest_hest.py --project-dir ... --limit 3    # the thin path
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import click


def _sha(p: Path, blocks: int = 8) -> str:
    """sha256 of the first blocks*1MB. the full file is up to 700 MB here and the
    prefix is enough to tie a canonical link to its source; the record says so."""
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for _ in range(blocks):
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def _link(src: Path, dest: Path) -> str:
    """hard link, symlink, or copy - whichever the filesystem allows, recorded."""
    if dest.exists() or dest.is_symlink():
        return "present"
    try:
        os.link(src, dest)
        return "hardlink"
    except OSError:
        pass
    try:
        dest.symlink_to(src.resolve())
        return "symlink"
    except OSError:
        import shutil

        shutil.copy2(src, dest)
        return "copy"


def _spots(h5ad: Path, dest: Path) -> dict:
    """the coordinates table: spot_id, x, y, from the object's own positions."""
    import anndata as ad
    import pandas as pd

    a = ad.read_h5ad(h5ad, backed="r")
    if "spatial" not in a.obsm:
        return {"ok": False, "why": "no obsm['spatial'] - nothing to write"}
    xy = a.obsm["spatial"]
    df = pd.DataFrame({"spot_id": list(a.obs_names),
                       "x": [float(v) for v in xy[:, 0]],
                       "y": [float(v) for v in xy[:, 1]]})
    df.to_parquet(dest, index=False)
    return {"ok": True, "n_spots": len(df),
            "extent": [float(df.x.max() - df.x.min()), float(df.y.max() - df.y.min())]}


@click.command()
@click.option("--project-dir", required=True, type=click.Path(path_type=Path, exists=True),
              help="the cohort's project root. a cohort is a project, not part of the package.")
@click.option("--src", default="data/inputs/st",
              help="where the source .h5ad files are, relative to the project root.")
@click.option("--out", default="data/canonical",
              help="where the canonical pair goes. this is what the package reads.")
@click.option("--limit", default=None, type=int,
              help="convert only the first N samples - the thin path before the cohort.")
@click.option("--samples", default=None,
              help="comma-separated sample ids, for a targeted run.")
def main(project_dir: Path, src: str, out: str, limit: int | None, samples: str | None) -> None:
    """convert HEST .h5ad files into the canonical pair, and record provenance."""
    root = project_dir.expanduser().resolve()
    src_dir, out_dir = root / src, root / out
    if not src_dir.is_dir():
        raise SystemExit(f"no source directory at {src_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    plat_path = root / "platform.json"
    if not plat_path.is_file():
        raise SystemExit(
            f"no platform.json in {root}. the platform is a DECLARATION - this cohort has "
            "three of them and the schema cannot be sniffed, so ingest refuses rather "
            "than guessing.")
    by_sample = json.loads(plat_path.read_text()).get("samples", {})

    found = sorted(src_dir.rglob("*.h5ad"))
    wanted = {s.strip() for s in samples.split(",")} if samples else None
    if wanted:
        found = [p for p in found if p.stem in wanted]
    if limit:
        found = found[:limit]

    # the record describes the DIRECTORY, not this invocation. a --limit or
    # --samples run that overwrote it would leave a record listing three samples
    # beside six canonical files, which is a provenance artifact that lies.
    prior = {}
    rec_path = out_dir / "ingest.json"
    if rec_path.is_file():
        prior = json.loads(rec_path.read_text())

    record = {
        "cohort": root.name,
        "source": f"{src}/{{sample_id}}.h5ad - HEST-1k, already .h5ad",
        "coords_space": ("full-resolution image pixels, as the object records them. the micron "
                         "scale is NOT declared here: inventory#geometry measures it per sample "
                         "from the object's own scalefactors."),
        "paths_relative_to": "the cohort's project root",
        "conversion": "link the object, write spot_id/x/y beside it. no values are changed.",
        # the canonical Sample carries an image, and this script never recorded
        # one - so the morphology arm saw no slide for any sample and silently
        # had nothing to encode. the WSIs arrive separately from the objects, so
        # this is re-read on every run rather than assumed from the first.
        "images": dict(prior.get("images", {})),
        "samples": list(prior.get("samples", [])),
        "sources": dict(prior.get("sources", {})),
        "skipped": [s for s in prior.get("skipped", []) if s["sample"] not in {p.stem for p in found}],
    }

    for p in found:
        sid = p.stem
        platform = by_sample.get(sid)
        if platform is None:
            record["skipped"].append({"sample": sid, "why": "no_platform_declared"})
            continue
        how = _link(p, out_dir / f"{sid}.h5ad")
        spots = _spots(p, out_dir / f"{sid}_spots.parquet")
        if not spots.get("ok"):
            record["skipped"].append({"sample": sid, "why": spots["why"]})
            continue
        if sid not in record["samples"]:
            record["samples"].append(sid)
        record["sources"][sid] = {
            "h5ad": str(p.relative_to(root)), "sha256_16": _sha(p), "linked_as": how,
            "platform": platform, "n_spots": spots["n_spots"], "extent_px": spots["extent"],
        }
        wsi = root / "data" / "inputs" / "wsis" / f"{sid}.tif"
        if wsi.is_file():
            record["images"][sid] = str(wsi)
        click.echo(f"  {sid:<12} {platform:<12} {spots['n_spots']:>6} spots  ({how})"
                   f"{'' if wsi.is_file() else '  no slide'}")

    record["samples"] = sorted(record["samples"])
    rec_path.write_text(json.dumps(record, indent=2) + "\n")
    click.echo(f"\n{len(record['samples'])} sample(s) canonical in {out_dir}")
    if record["skipped"]:
        click.echo(f"{len(record['skipped'])} skipped: "
                   + ", ".join(f"{s['sample']} ({s['why']})" for s in record["skipped"][:4]))
    click.echo("wrote ingest.json")


if __name__ == "__main__":
    main()
