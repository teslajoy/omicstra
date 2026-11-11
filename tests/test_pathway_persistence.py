"""WS 6: pathway artifact persistence.

five tests cover the two new code paths:
  save_scores step  -- the per-sample parquet writer in protocols/pathway.py
  _persist_pathway_scores -- the combine-and-update-project-json helper in graph.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# save_scores step (protocols/pathway.py)
# ---------------------------------------------------------------------------

def _make_aucell_z(n_niches=10, n_pathways=5):
    """minimal aucell_z tuple: (names, arr, nids)."""
    names = [f"P{i}" for i in range(n_pathways)]
    arr = np.random.default_rng(42).standard_normal((n_niches, n_pathways))
    nids = [f"s1::{i}" for i in range(n_niches)]
    return names, arr, nids


def test_save_scores_step_writes_parquet(tmp_path):
    """save_scores writes a parquet at _output_path with niche_id index."""
    from omicstra.protocols.pathway import save_scores

    out = tmp_path / "scores.parquet"
    names, arr, nids = _make_aucell_z()
    ctx = {"aucell_z": (names, arr, nids), "_output_path": str(out)}

    rec, extra = save_scores(ctx)

    assert out.is_file(), "parquet not written"
    df = pd.read_parquet(out)
    assert df.index.name == "niche_id"
    assert list(df.columns) == names
    assert df.shape == (len(nids), len(names))
    assert rec.status == "pass"


def test_save_scores_step_noop_without_output_path():
    """save_scores returns not_applicable and writes nothing when no _output_path."""
    from omicstra.protocols.pathway import save_scores

    names, arr, nids = _make_aucell_z()
    ctx = {"aucell_z": (names, arr, nids)}  # no _output_path

    rec, extra = save_scores(ctx)

    assert rec.status == "not_applicable"
    assert extra == {}


# ---------------------------------------------------------------------------
# _persist_pathway_scores helper (graph.py)
# ---------------------------------------------------------------------------

def _write_sample_parquet(scores_dir: Path, sid: str,
                          n_niches=8, n_pathways=3) -> pd.DataFrame:
    names = [f"Q{i}" for i in range(n_pathways)]
    arr = np.ones((n_niches, n_pathways))
    nids = [f"{sid}::{j}" for j in range(n_niches)]
    df = pd.DataFrame(arr, index=nids, columns=names)
    df.index.name = "niche_id"
    safe = sid.replace("/", "__").replace(":", "__")
    p = scores_dir / f"{safe}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    return df


def test_persist_pathway_scores_creates_combined_parquet(tmp_path):
    """_persist_pathway_scores concatenates per-sample parquets into aucell_v1.parquet."""
    from omicstra.graph import _persist_pathway_scores

    scores_dir = tmp_path / "data" / "embeddings" / "pathway_scores"
    sids = ["A1", "A2"]
    for sid in sids:
        _write_sample_parquet(scores_dir, sid)

    # write a minimal project.json
    proj_path = tmp_path / "project.json"
    proj_path.write_text(json.dumps({"name": "test"}))

    class _Cfg:
        def path(self, rel):
            return tmp_path / rel

    _persist_pathway_scores(tmp_path, scores_dir, sids, _Cfg())

    combined = scores_dir / "aucell_v1.parquet"
    assert combined.is_file()
    df = pd.read_parquet(combined)
    assert len(df) == 16   # 8 niches × 2 samples


def test_persist_pathway_scores_updates_project_json(tmp_path):
    """_persist_pathway_scores writes pathway_scores key into project.json."""
    from omicstra.graph import _persist_pathway_scores

    scores_dir = tmp_path / "data" / "embeddings" / "pathway_scores"
    _write_sample_parquet(scores_dir, "S1")

    proj_path = tmp_path / "project.json"
    proj_path.write_text(json.dumps({"name": "test"}))

    class _Cfg:
        def path(self, rel):
            return tmp_path / rel

    _persist_pathway_scores(tmp_path, scores_dir, ["S1"], _Cfg())

    proj = json.loads(proj_path.read_text())
    assert "pathway_scores" in proj
    assert "path" in proj["pathway_scores"]
    assert "columns" in proj["pathway_scores"]
    assert proj["pathway_scores"]["columns"] == ["Q0", "Q1", "Q2"]


def test_reuse_check_finds_persisted_file(tmp_path):
    """after _persist_pathway_scores, the reuse check in _FLAG_TO_SLOT logic finds the file."""
    from omicstra.graph import _persist_pathway_scores

    scores_dir = tmp_path / "data" / "embeddings" / "pathway_scores"
    _write_sample_parquet(scores_dir, "S1")

    proj_path = tmp_path / "project.json"
    proj_path.write_text(json.dumps({"name": "test"}))

    class _Cfg:
        def path(self, rel):
            return tmp_path / rel

    _persist_pathway_scores(tmp_path, scores_dir, ["S1"], _Cfg())

    proj = json.loads(proj_path.read_text())
    decl = proj.get("pathway_scores")
    assert isinstance(decl, dict) and decl.get("path")
    resolved = tmp_path / decl["path"]
    assert resolved.is_file(), "declared path does not point to the written file"
