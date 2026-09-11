"""ingest: HEST-1k breast cohort -> canonical.

STUB. written alongside ingest_wang.py so the canonical contract is defined by
two cohorts rather than one. it is not runnable until the cohort's files are on
disk, and it is deliberately committed unfinished: an interface defined by a
single cohort is an interface shaped like that cohort.

what this cohort proves about the contract
------------------------------------------
hest already ships .h5ad, so ingest here is nearly a no-op - write a spots
table from `adata.obsm["spatial"]`, record shas, done. that is the point. if
adding hest needed anything in src/omicstra to change, the four-way split
(adapter / declaration / package method / encoder wrapper) would have failed.

three things hest has that tnbc-92 cannot reveal, and which the contract had to
survive:

- THREE platforms in one cohort: original-ST, Visium, and Xenium. platform is a
  property of a SAMPLE, not of a cohort, which is why platform.json keys samples
  individually rather than declaring one platform for the lot.
- a platform with NO PITCH. Xenium is subcellular; `spot_pitch_um` and
  `spot_diameter_um` are null. any geometry that assumes a pitch exists breaks
  here, which is why TileGeometry takes a scale and an integer crop size rather
  than deriving them from a pitch.
- a TARGETED PANEL rather than whole transcriptome. gene-set enrichment over a
  panel is a different claim from enrichment over a transcriptome, and the EDA
  contract's applicability rules are what stop a check running where it cannot
  mean anything.

the cohort itself lives in its own repo. a cohort is a project, not part of the
package - see commit 9690c42.
"""
from __future__ import annotations

import sys


def main() -> int:
    print(__doc__.strip())
    print()
    print("not runnable yet. to finish it:")
    for line in (
        "1  point it at the HEST-1k breast subset (its own repo, not this one)",
        "2  for each sample: copy/symlink the .h5ad, and write _spots.parquet as",
        "   spot_id, x, y from adata.obsm['spatial'] - that is the whole conversion",
        "3  record source shas in data/canonical/ingest.json, keyed by sample_id",
        "4  declare each sample's platform in platform.json: 108 original-ST,",
        "   8 Visium, 9 Xenium. per SAMPLE, never per cohort",
        "5  Xenium samples declare no pitch and no spot diameter. do not invent them",
    ):
        print(f"   {line}")
    print()
    print("if finishing this needs an edit inside src/omicstra, that is the bug.")
    return 1


if __name__ == "__main__":
    sys.exit(main())