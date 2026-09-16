"""step 8: the chain composes, and the rebuilt pack still matches what was published.

finishing proves plumbing. the acceptance is the DIFF - every number in the
curated pack coming back out of the artifacts the chain reads today.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from omicstra.run import diff_against, run_cohort

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "projects" / "tnbc-92"


def _run(tmp_path):
    if not (PROJ / "evidence_sources.json").is_file():
        pytest.skip("cohort absent")
    out = run_cohort("tnbc-92", out_dir=tmp_path / "run")
    if out["stages"].get("niche_join", "").startswith("unavailable"):
        pytest.skip("caches not present on this machine")
    return out


def test_every_stage_resolves_on_the_seed_cohort(tmp_path):
    out = _run(tmp_path)
    for stage in ("embed_he", "embed_st", "niche_join", "align", "eval"):
        assert out["stages"][stage] == "resolved", f"{stage}: {out['stages'][stage]}"
    assert out["stages"]["promote"] == "proposed"


def test_the_rebuilt_pack_still_matches_the_published_one(tmp_path):
    """the only disagreement is the arm the curated pack omits."""
    out = _run(tmp_path)
    curated = json.loads((PROJ / "routing_evidence.json").read_text())
    diffs = diff_against(out["pack"], curated)
    moved = [d for d in diffs if d["kind"] in ("value_moved", "interval_moved")]
    assert not moved, f"the chain no longer reproduces: {moved}"
    assert [d["method"] for d in diffs] == ["B3_v3"]
    assert diffs[0]["kind"] == "absent_from_curated"


def test_the_run_writes_a_report_a_pack_and_its_records(tmp_path):
    out = _run(tmp_path)
    d = Path(out["written"])
    for name in ("report.md", "pack.json", "records.json", "stages.json"):
        assert (d / name).is_file(), name
    assert (d / "report.md").read_text().startswith("# ")
    assert len(json.loads((d / "records.json").read_text())) >= 5


def test_a_cohort_with_no_artifacts_still_reports_what_is_missing(tmp_path):
    """a stage that cannot resolve is recorded, not raised - the report is the
    point, and 'nothing is here' is a result."""
    proj = tmp_path / "bare"
    proj.mkdir()
    (proj / "project.json").write_text(json.dumps(
        {"project_id": "bare", "platform": "p", "niches_dir": "missing"}))
    from omicstra.settings import settings

    old = settings.project_dir
    try:
        settings.project_dir = proj
        out = run_cohort(None, out_dir=tmp_path / "out")
    finally:
        settings.project_dir = old
    assert out["stages"]["niche_join"].startswith("unavailable")
    assert out["stages"]["promote"].startswith("unavailable")
    assert "not applicable" in out["report"]


def test_the_diff_names_each_kind_of_disagreement():
    ours = {"tasks": {"t": {"outcome": "recommend", "candidates": [
        {"id": "a", "value": 0.9}, {"id": "b", "value": 0.5}, {"id": "new", "value": 0.4}]}}}
    curated = {"tasks": {"t": {"candidates": [
        {"id": "a", "value": 0.9}, {"id": "b", "value": 0.8}, {"id": "gone", "value": 0.1}]}}}
    kinds = {d["method"]: d["kind"] for d in diff_against(ours, curated)}
    assert kinds == {"b": "value_moved", "new": "absent_from_curated",
                     "gone": "not_reachable_by_rebuild"}
