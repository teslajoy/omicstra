"""shared fixtures. one synthetic cohort, because two would drift.

why a synthetic cohort exists at all: five tests in `test_encode_graph.py`
asserted the encode node reports `ready`, and `ready` became conditional on the
cohort's slides and `_spots.parquet` being on disk the moment the node started
resolving shards. both are git-ignored, so in a clone the node correctly answers
`nothing_to_run` and the assertions failed - CI red on a package that was right.

the fix is not a skip. a cohort the tests OWN means a clone exercises the gate
routing rather than stepping over it, and the seed cohort stays what it is: the
thing the oracle diffs against, not a test fixture.
"""
from __future__ import annotations

import json

import pytest

# four sections, twelve spots each, no counts, a 64x64 slide apiece. small enough
# that building it costs nothing and large enough that k=6 niches are real.
SECTIONS = [f"sec{i}" for i in range(4)]
SYNTH_COHORT = {
    "project_id": "synthetic",
    "platform": "synthetic",
    "k_neighbors": 6,
    "seed": 42,
    "split": "patient",
    "subject_id_column": "patient_id",
    "encoders": [],
    "supervision": [],
}
SYNTH_PLATFORM = {
    "platforms": {
        "synthetic": {
            "image_pyramid": {"scale_to_hd": 1.0, "basis": "synthetic"},
            "he_tile": {"tile_um": 100.0, "tile_px_hd": 8, "resize_to": 8,
                        "basis": "synthetic"},
        }
    }
}


def build_synthetic_cohort(dest, sections=SECTIONS, cohort_extra=None):
    """a canonical cohort on disk: declarations, coordinates, one slide each.

    returns the project root. the shape is the package's input contract - .h5ad
    plus a coordinates table plus an ingest record - with the .h5ad absent,
    because the H&E arm is entitled to run on a section that has no counts.
    """
    import numpy as np
    import pandas as pd
    from PIL import Image

    root = dest / "synthetic"
    canon = root / "data" / "canonical"
    canon.mkdir(parents=True, exist_ok=True)
    imgs = root / "images"
    imgs.mkdir(exist_ok=True)

    (root / "project.json").write_text(json.dumps({**SYNTH_COHORT, **(cohort_extra or {})}))
    (root / "cohort.json").write_text(json.dumps({**SYNTH_COHORT, **(cohort_extra or {})}))

    rng = np.random.default_rng(0)
    for sid in sections:
        xy = rng.integers(8, 56, size=(12, 2))
        pd.DataFrame({"spot_id": [f"{sid}_{i}" for i in range(12)],
                      "x": xy[:, 0].astype(float),
                      "y": xy[:, 1].astype(float)}).to_parquet(canon / f"{sid}_spots.parquet")
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype="uint8")).save(
            imgs / f"{sid}.png")

    plat = json.loads(json.dumps(SYNTH_PLATFORM))
    plat["samples"] = {sid: "synthetic" for sid in sections}
    (root / "platform.json").write_text(json.dumps(plat))
    (canon / "ingest.json").write_text(json.dumps(
        {"samples": list(sections),
         "images": {sid: str(imgs / f"{sid}.png") for sid in sections},
         "sources": {sid: {"n_spots": 12} for sid in sections}}))
    return root


@pytest.fixture
def synthetic_cohort(tmp_path):
    """the cohort on disk, with nothing bound. for a test that passes the root."""
    pytest.importorskip("PIL", reason="the synthetic slide needs pillow")
    return build_synthetic_cohort(tmp_path)


@pytest.fixture
def bound_synthetic_cohort(synthetic_cohort):
    """the cohort BOUND as settings.project_dir, restored afterwards.

    try/finally is not optional here. a leaked `project_dir` turned one new test
    into thirteen unrelated failures once, because every later test in the
    session resolved the wrong cohort root.
    """
    from omicstra.settings import settings

    before = settings.project_dir
    settings.project_dir = synthetic_cohort
    try:
        yield synthetic_cohort
    finally:
        settings.project_dir = before


@pytest.fixture(autouse=True)
def _runs_dir_is_never_the_repos(tmp_path, monkeypatch):
    """no test writes into the repo's `runs/`.

    every arm appends to `runs/{project_id}/records/decisions.jsonl` now, and
    `settings.runs_dir` resolves under the PACKAGE repo root - so a test calling
    a tool on the seed cohort wrote into the record a report reads. 42 rows
    landed in `runs/synthetic/` and 5 in `runs/tnbc-92/` before this existed,
    from tests that were otherwise careful with tmp_path.

    autouse, because the leak is invisible: the files are git-ignored, nothing
    fails, and the next report is simply wrong about who decided what. a test
    that genuinely needs the real tree reads it by path rather than through
    settings, which is what the artifact tests already do.
    """
    from omicstra.settings import settings

    monkeypatch.setattr(settings, "runs_dir", tmp_path / "_runs")
    # a SUBPROCESS inherits none of that, so it is told by environment too -
    # the kill tests run the server in their own process.
    monkeypatch.setenv("OMICSTRA_RUNS_DIR", str(tmp_path / "_runs"))
