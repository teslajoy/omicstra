"""the package's input contract, and the thing that proves it is generalizable.

the contract is .h5ad plus a coordinates table. a cohort's native format is
converted ONCE by scripts/ingest_<cohort>.py, so nothing in src/omicstra imports
an R reader, a vendor SDK, or any other format-specific loader.

the test of that is not whether tnbc-92 works. it is whether a SECOND cohort
costs an ingest script and no edit inside the package - which is why
scripts/ingest_hest.py is committed unfinished beside the real one.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_the_package_imports_no_format_reader():
    """Rscript, pyreadr and rpy2 belong to ingest scripts, not to omicstra.

    this is the whole of option B in one assertion. if it fails, a cohort that
    did not arrive from R has to work around code written for one that did.

    checks what is IMPORTED, not what is mentioned - a first version substring-
    matched the source, so the docstring explaining why a reader is absent
    tripped the check that keeps it absent. a test that forbids discussing a
    thing is a bad test.
    """
    import ast

    banned = {"pyreadr", "rpy2", "rdata"}
    offenders = []
    for py in (ROOT / "src" / "omicstra").rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in banned:
                        offenders.append(f"{py.relative_to(ROOT)}: import {a.name}")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] in banned:
                    offenders.append(f"{py.relative_to(ROOT)}: from {node.module}")
            # an Rscript subprocess is a format reader wearing a different hat
            elif isinstance(node, ast.Constant) and node.value == "Rscript":
                offenders.append(f"{py.relative_to(ROOT)}: shells to Rscript")
    assert not offenders, "format readers inside the package: " + "; ".join(offenders)


def test_two_ingest_scripts_exist_so_the_contract_has_two_witnesses():
    """an interface defined by one cohort is shaped like that cohort.

    hest is committed as a stub on purpose: it ships .h5ad already, so its ingest
    is nearly a no-op, and if finishing it ever needs a change inside
    src/omicstra then the four-way split has failed.
    """
    assert (ROOT / "scripts" / "ingest_wang.py").exists()
    hest = ROOT / "scripts" / "ingest_hest.py"
    assert hest.exists(), "the second witness is missing"
    text = hest.read_text()
    assert "Xenium" in text and "no pitch" in text.lower(), \
        "the stub must record what the second cohort proves: a platform with no pitch"


def test_canonical_refuses_rather_than_falling_back():
    """absent canonical form is a refusal that says how to make one.

    falling back to a cohort's native files would put format knowledge back in
    the package by the side door.
    """
    from omicstra.adapters.canonical import CanonicalMissing, read_ingest
    with pytest.raises(CanonicalMissing) as e:
        read_ingest("/nonexistent/cohort")
    assert "ingest_" in str(e.value), "the refusal must name the fix"


# --- only where the cohort has been ingested --------------------------------
def _root():
    r = ROOT / "projects" / "tnbc-92"
    if not (r / "data" / "canonical" / "ingest.json").exists():
        pytest.skip("cohort not ingested on this machine")
    return r


def test_ingested_coordinates_match_what_the_script_used():
    """bit-identical, not approximately.

    the cache's _meta.tsv records the coordinates the script cut tiles at. if
    ingest moves them by any amount, every tile moves, and the slice-diff at 3.5
    would fail for a reason that has nothing to do with the encoder.
    """
    import numpy as np
    import pandas as pd

    from omicstra.adapters.canonical import list_samples, load_spots
    cache = ROOT / "data" / "embeddings" / "virchow2_niche"
    compared = 0
    for s in list_samples(_root()):
        m = cache / f"{s.sample_id}_meta.tsv"
        if not m.exists():
            continue
        df, meta = load_spots(s), pd.read_csv(m, sep="\t")
        assert np.abs(df["x"].values - meta["pixel_x"].values).max() == 0.0, s.sample_id
        assert np.abs(df["y"].values - meta["pixel_y"].values).max() == 0.0, s.sample_id
        compared += 1
    assert compared, "no ingested sample had a cache entry to compare against"


def test_patient_travels_through_the_adapter():
    """patient is the confounder axis every split is held out on.

    byArray/ is keyed by slide and position only, so the patient has to be
    recovered from the image filename. an ingest that loses it produces a cohort
    that cannot be split honestly - and the first draft of ingest_wang.py did
    exactly that, which is why this test exists.
    """
    from omicstra.adapters.canonical import list_samples, read_ingest
    root = _root()
    rec = read_ingest(root)
    assert rec.get("subject_id_column") == "patient_id"
    for s in list_samples(root):
        pat = rec["sources"][s.sample_id].get("patient_id")
        assert pat, f"{s.sample_id} has no patient"
        assert s.sample_id.startswith(pat + "_"), f"{s.sample_id} disagrees with {pat}"


def test_the_morphology_path_needs_no_counts():
    """a coords-only ingest is a legal, useful state.

    cutting tiles must not require loading a 27,567-gene matrix to read two
    columns, and a sample without .h5ad is returned rather than hidden so the
    EDA gate - not the adapter - decides what an absent role means.
    """
    from omicstra.adapters.canonical import list_samples, load_spots
    for s in list_samples(_root()):
        assert load_spots(s) is not None
