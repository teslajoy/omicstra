"""the artifacts that produced the published results are never written to.

this is not only a data-loss guard. those directories are what every port is
diffed against, so a write into one makes the next diff compare a port against
its own output - and pass. the failure invalidates the verification rather than
the data, and it is silent, which is why it refuses rather than warning.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from omicstra.dispatch import (
    Shard,
    WouldOverwriteOracle,
    assert_writable,
    read_only_inputs,
)

ROOT = Path(__file__).resolve().parents[1]


def test_the_cohort_declares_its_oracles():
    pj = ROOT / "projects" / "tnbc-92" / "project.json"
    if not pj.is_file():
        pytest.skip("fixture cohort absent")
    decl = json.loads(pj.read_text()).get("read_only_inputs") or {}
    assert decl.get("policy") == "refuse_writes"
    assert decl.get("paths"), "a cohort with published results must name its oracles"


def test_the_declaration_is_a_typed_field_not_a_loose_key():
    """pydantic drops what it does not declare. this sat in project.json being
    silently ignored, which is the same failure as an undeclared key at a graph
    boundary - the guard read an empty list and protected nothing.
    """
    from omicstra.contracts.project import ProjectConfig

    assert "read_only_inputs" in ProjectConfig.model_fields
    cfg = ProjectConfig.load("tnbc-92") if (
        ROOT / "projects" / "tnbc-92" / "project.json").is_file() else None
    if cfg:
        assert cfg.read_only_inputs, "the declaration did not survive parsing"


def test_the_declared_paths_resolve():
    if not (ROOT / "projects" / "tnbc-92" / "project.json").is_file():
        pytest.skip("fixture cohort absent")
    ro = read_only_inputs("tnbc-92")
    assert len(ro) >= 4, "a declared path that cannot resolve protects nothing"


@pytest.mark.parametrize("target", [
    "data/embeddings/virchow2_niche/TNBC1_CN1_C1.npy",
    "data/embeddings/virchow2_cell/TNBC1_CN1_C1.npy",
    "data/embeddings/novae_niche_full/x.parquet",
    "data/embeddings/niches_v3/TNBC1_CN1_C1.parquet",
])
def test_a_shard_aimed_at_an_oracle_is_refused(target):
    """the exact hazard: the plan tool once named the oracle as its output
    directory, and `is_done()` skipping existing files was the only thing
    standing between that and a contaminated cache."""
    if not (ROOT / "projects" / "tnbc-92" / "project.json").is_file():
        pytest.skip("fixture cohort absent")
    with pytest.raises(WouldOverwriteOracle, match="read-only"):
        assert_writable([Shard(id="s", output=ROOT / target)], "tnbc-92")


def test_a_run_scoped_output_is_allowed():
    if not (ROOT / "projects" / "tnbc-92" / "project.json").is_file():
        pytest.skip("fixture cohort absent")
    assert_writable([Shard(id="s", output=ROOT / "runs/tnbc-92/r1/s.npy")], "tnbc-92")


def test_an_undeclared_cohort_is_not_blocked(tmp_path, monkeypatch):
    """a cohort with no published results has no oracle to protect, and must not
    be prevented from writing anywhere."""
    from omicstra.settings import settings

    monkeypatch.setattr(settings, "project_dir", tmp_path)
    assert read_only_inputs(None) == []
    assert_writable([Shard(id="s", output=tmp_path / "out.npy")], None)


def test_the_refusal_happens_before_any_shard_runs():
    """a refusal that arrives after the first shard has written came too late."""
    import inspect

    from omicstra import dispatch

    src = inspect.getsource(dispatch.run_shards)
    # the BODY only. the docstring mentions `temporal_address` before the guard
    # runs, and matching that compares against prose rather than order.
    body = src.split('"""')[-1]
    i_guard = body.index("assert_writable")
    assert i_guard < body.index("run_shards_locally"), "guard after local execution"
    assert i_guard < body.index("settings.temporal_address"), "guard after backend choice"


def test_the_plan_tool_does_not_name_an_oracle_as_output():
    """it did. the oracle is reported separately, as read-only."""
    if not (ROOT / "projects" / "tnbc-92" / "data" / "canonical" / "ingest.json").is_file():
        pytest.skip("canonical ingest absent")
    from omicstra.mcp.server import describe_compute_plan

    p = describe_compute_plan("virchow2")
    out, oracle = Path(p["output_dir"]).resolve(), Path(p["oracle"]["path"]).resolve()
    assert out != oracle
    assert oracle not in out.parents
    for ro in read_only_inputs("tnbc-92"):
        assert ro != out and ro not in out.parents
