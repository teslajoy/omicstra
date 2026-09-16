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


# --- a finished run grid is an oracle, and a rerun goes somewhere new --------
def test_every_path_the_evidence_pack_cites_is_declared_read_only():
    """"done" is declared, not inferred - and this is what stops a finished project
    forgetting to declare. any runs/ or data/ path the pack reads a routed number
    from must sit inside a read-only input."""
    import re

    pack = ROOT / "projects" / "tnbc-92" / "routing_evidence.json"
    if not pack.is_file():
        pytest.skip("evidence pack absent")
    from omicstra.dispatch import protected_by

    ro = read_only_inputs("tnbc-92")
    cited = set()
    for task in json.loads(pack.read_text())["tasks"].values():
        for m in re.finditer(r"\b(?:runs|data)/[\w\-./{}]+", str(task.get("source", ""))):
            cited.add(m.group(0).rstrip(".;,").split("{")[0].rstrip("/"))
    assert cited, "the pack cites no artifact paths - the pattern is stale"
    uncovered = sorted(p for p in cited if protected_by(ROOT / p, ro) is None)
    assert not uncovered, f"cited but not declared read-only: {uncovered}"


def _grid(tmp_path):
    from omicstra.contracts.project import ProjectConfig
    from omicstra.protocols.align import AlignGrid

    runs = tmp_path / "finished"
    for rid in ("R1", "B1"):
        (runs / rid / "eval").mkdir(parents=True)
        (runs / rid / "embeddings_test.parquet").write_bytes(b"emb-" + rid.encode())
        (runs / rid / "run_config.json").write_text("{}")
        (runs / rid / "split.json").write_text("{}")
        (runs / rid / "eval" / "biology.parquet").write_bytes(b"published")
    (runs / "R1" / "checkpoint.pt").write_bytes(b"ckpt")
    cfg = ProjectConfig(project_id="t", platform="p", project_dir=tmp_path,
                        read_only_inputs={"policy": "refuse_writes", "paths": [str(runs)]})
    grid = AlignGrid(runs_root=str(runs), run_refs={r: str(runs / r / "embeddings_test.parquet")
                                                     for r in ("R1", "B1")})
    pack = tmp_path / "routing_evidence.json"
    pack.write_text('{"tasks": {}}')
    return cfg, grid, pack


def test_scoring_into_a_finished_grid_is_refused_before_any_script_runs(tmp_path, monkeypatch):
    from omicstra.protocols import align

    cfg, grid, pack = _grid(tmp_path)
    calls = []
    monkeypatch.setattr(align, "_run_script", lambda s, a: calls.append((s, a)))
    with pytest.raises(WouldOverwriteOracle, match="out_root"):
        align.run_eval(cfg, grid, compute=True, evidence_out=pack)
    assert calls == []


def test_a_rerun_is_staged_into_out_root_and_scores_only_there(tmp_path, monkeypatch):
    from omicstra.protocols import align

    cfg, grid, pack = _grid(tmp_path)
    out = tmp_path / "rerun"
    calls = []
    monkeypatch.setattr(align, "_run_script", lambda s, a: calls.append((s, a)))
    res, rec = align.run_eval(cfg, grid, compute=True, evidence_out=pack, out_root=out)

    (_script, args), = calls
    assert args[args.index("--runs-dir") + 1] == str(out.resolve())
    assert res.metrics_ref.startswith(str(out.resolve()))
    assert rec.params["runs_root"] == str(out.resolve())
    staged = out / "R1"
    assert (staged / "embeddings_test.parquet").is_symlink()
    assert (staged / "checkpoint.pt").is_symlink()
    assert not (staged / "run_config.json").is_symlink()
    assert not (staged / "eval").exists(), "a staged run must carry none of the outputs"
    assert (Path(grid.runs_root) / "R1" / "eval" / "biology.parquet").read_bytes() == b"published"


def test_out_root_inside_a_finished_grid_is_refused(tmp_path, monkeypatch):
    from omicstra.protocols import align

    cfg, grid, pack = _grid(tmp_path)
    monkeypatch.setattr(align, "_run_script", lambda s, a: None)
    with pytest.raises(WouldOverwriteOracle, match="stage a rerun in"):
        align.run_eval(cfg, grid, compute=True, evidence_out=pack,
                       out_root=Path(grid.runs_root) / "rerun")


def test_a_stage_that_writes_through_a_staged_link_is_caught(tmp_path, monkeypatch):
    """no mapped eval script writes its inputs - this holds that to the file."""
    import os
    import time

    from omicstra.protocols import align

    cfg, grid, pack = _grid(tmp_path)
    out = tmp_path / "rerun"

    def rogue(script, args):
        target = out / "R1" / "embeddings_test.parquet"
        time.sleep(0.01)
        target.write_bytes(b"overwritten through the link")
        os.utime(target.resolve())

    monkeypatch.setattr(align, "_run_script", rogue)
    with pytest.raises(WouldOverwriteOracle, match="wrote through a staged link"):
        align.run_eval(cfg, grid, compute=True, evidence_out=pack, out_root=out)


def test_training_into_a_finished_grid_is_refused(tmp_path, monkeypatch):
    from omicstra.protocols import align

    cfg, grid, _ = _grid(tmp_path)
    nj = align.NicheJoin(manifest_ref="m", niches_dir=str(tmp_path))
    monkeypatch.setattr(align, "_run_script", lambda s, a: pytest.fail("a script ran"))
    with pytest.raises(WouldOverwriteOracle, match="train into"):
        align.run_align(cfg, nj, run_ids=["R2_v3"], compute=True, runs_root=Path(grid.runs_root))


# --- the join builds into a directory of its own ---------------------------
def test_building_the_join_without_an_out_dir_is_refused():
    """the cohort's declared join is what the ports are diffed against."""
    from omicstra.contracts.project import ProjectConfig
    from omicstra.protocols.align import ComputeUnavailable, run_niche_join

    cfg = ProjectConfig(project_id="t", platform="p", niches_dir="n")
    with pytest.raises(ComputeUnavailable, match="out_dir"):
        run_niche_join(cfg, compute=True)


def test_building_the_join_into_a_declared_path_is_refused(tmp_path, monkeypatch):
    from omicstra.contracts.project import ProjectConfig
    from omicstra.protocols import align

    oracle = tmp_path / "niches"
    oracle.mkdir()
    cfg = ProjectConfig(project_id="t", platform="p", project_dir=tmp_path,
                        niches_dir=str(oracle),
                        read_only_inputs={"policy": "refuse_writes", "paths": [str(oracle)]})
    monkeypatch.setattr(align, "_run_script", lambda s, a: pytest.fail("a script ran"))
    with pytest.raises(WouldOverwriteOracle, match="build the niche join into"):
        align.run_niche_join(cfg, compute=True, out_dir=oracle)


def test_a_join_built_from_a_different_pathway_build_is_refused(tmp_path):
    """the manifest records which gpath2vec produced the table; the cohort
    declares which one it means. a mismatch carries the published name over
    different numbers."""
    import json as _json

    from omicstra.contracts.project import ProjectConfig
    from omicstra.protocols.align import ComputeUnavailable, run_niche_join

    d = tmp_path / "niches"
    d.mkdir()
    (d / "manifest.json").write_text(_json.dumps(
        {"sources": {"gpath2vec_sha256": "b" * 64}, "n_niches_total_post_intersection": 10}))
    cfg = ProjectConfig(project_id="t", platform="p", project_dir=tmp_path,
                        niches_dir=str(d), gpath2vec_sha256="a" * 64)
    with pytest.raises(ComputeUnavailable, match="declares"):
        run_niche_join(cfg)
