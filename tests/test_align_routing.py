"""entry-point routing: verify CLI, LangGraph graph and MCP all reach the
LCEL chain for contrastive runs when compute=True.

these tests patch _train_contrastive_chain (the LCEL entry) and _run_script
(the subprocess entry) to verify ROUTING, not numerical correctness. the
chain and parity tests cover the maths; these cover the wiring.

no torch dependency: every code path that would import torch is patched.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omicstra.contracts.project import ProjectConfig
from omicstra.protocols.align import (
    AlignGrid,
    NicheJoin,
    run_align,
)

STEP_IDS = ("load_data", "split", "train", "embed_test", "evaluate_and_save")

FAKE_CHAIN_RECORDS = {
    sid: {"status": "pass", "duration_s": 0.1, "params": {}}
    for sid in STEP_IDS
}


def _make_fixture(tmp_path: Path, run_id: str = "R1_v3") -> tuple[ProjectConfig, NicheJoin, Path]:
    """minimal fixture that triggers the compute path for one contrastive run.

    a config file marks it contrastive, a niches dir exists, and the
    embeddings file is absent -- so run_align enters the training branch.
    """
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    configs = tmp_path / "configs" / "v3"
    configs.mkdir(parents=True)
    (configs / f"{run_id}.json").write_text(json.dumps({
        "run_id": run_id, "loss": "infonce", "fusion": "late",
        "seed": 42, "st_features": ["novae_niche", "gpath2vec_niche"],
    }))

    niches = tmp_path / "niches"
    niches.mkdir()

    cfg = MagicMock(spec=ProjectConfig)
    cfg.runs_dir = str(runs_root)
    cfg.contrastive_runs = [run_id]
    cfg.classical_runs = []

    nj = NicheJoin(manifest_ref="test", niches_dir=str(niches))
    return cfg, nj, runs_root


# ---------------------------------------------------------------------------
# Test 1: run_align default routes through the LCEL chain
# ---------------------------------------------------------------------------
class TestRunAlignDefaultRoute:
    def test_contrastive_uses_package_by_default(self, tmp_path):
        cfg, nj, runs_root = _make_fixture(tmp_path)

        with patch("omicstra.protocols.align._train_contrastive_chain",
                   return_value=FAKE_CHAIN_RECORDS) as mock_chain, \
             patch("omicstra.protocols.align._run_script") as mock_script, \
             patch("omicstra.protocols.align._refuse_protected"), \
             patch("omicstra.protocols.align._project_rel", return_value=runs_root):
            grid, rec = run_align(cfg, nj, compute=True, runs_root=runs_root)

        mock_chain.assert_called_once()
        mock_script.assert_not_called()
        assert rec.params["method"] == "package"
        assert "chain_records" in rec.params
        assert set(rec.params["chain_records"]["R1_v3"]) == set(STEP_IDS)

    def test_explicit_use_package_false_uses_script(self, tmp_path):
        cfg, nj, runs_root = _make_fixture(tmp_path)

        with patch("omicstra.protocols.align._train_contrastive_chain") as mock_chain, \
             patch("omicstra.protocols.align._run_script") as mock_script, \
             patch("omicstra.protocols.align._refuse_protected"), \
             patch("omicstra.protocols.align._project_rel", return_value=runs_root):
            grid, rec = run_align(cfg, nj, compute=True,
                                  runs_root=runs_root, use_package=False)

        mock_script.assert_called_once()
        mock_chain.assert_not_called()
        assert rec.params["method"] == "script"

    def test_explicit_use_package_true_also_works(self, tmp_path):
        """both the default and an explicit True select the package chain."""
        cfg, nj, runs_root = _make_fixture(tmp_path)

        with patch("omicstra.protocols.align._train_contrastive_chain",
                   return_value=FAKE_CHAIN_RECORDS) as mock_chain, \
             patch("omicstra.protocols.align._run_script") as mock_script, \
             patch("omicstra.protocols.align._refuse_protected"), \
             patch("omicstra.protocols.align._project_rel", return_value=runs_root):
            grid, rec = run_align(cfg, nj, compute=True,
                                  runs_root=runs_root, use_package=True)

        mock_chain.assert_called_once()
        mock_script.assert_not_called()
        assert rec.params["method"] == "package"


# ---------------------------------------------------------------------------
# Test 2: LangGraph graph entry point
# ---------------------------------------------------------------------------
class TestGraphRouting:
    def test_graph_align_node_uses_package(self, tmp_path):
        """the align node in graphs/align.py calls run_align, which defaults
        to use_package=True. verify compute propagates through the graph."""
        from langgraph.checkpoint.memory import InMemorySaver

        from omicstra.graphs.align import build_align_graph

        cfg, nj, runs_root = _make_fixture(tmp_path)
        fake_grid = AlignGrid(
            runs_root=str(runs_root),
            run_refs={"R1_v3": str(runs_root / "R1_v3" / "embeddings_test.parquet")},
            computed=["R1_v3"],
        )
        fake_rec = MagicMock()
        fake_rec.model_dump.return_value = {
            "step_id": "align", "status": "pass", "kind": "transform",
            "params": {"method": "package", "chain_records": {"R1_v3": FAKE_CHAIN_RECORDS}},
        }

        with patch("omicstra.graphs.align.preflight_align", return_value=[]), \
             patch("omicstra.protocols.align.run_niche_join", return_value=(nj, MagicMock())), \
             patch("omicstra.protocols.align._train_contrastive_chain",
                   return_value=FAKE_CHAIN_RECORDS) as mock_chain, \
             patch("omicstra.protocols.align._refuse_protected"), \
             patch("omicstra.contracts.project.ProjectConfig.load", return_value=cfg):

            app = build_align_graph(checkpointer=InMemorySaver())
            out = app.invoke(
                {"project_id": "test", "arm": "align", "compute": True},
                {"configurable": {"thread_id": "routing-test"}},
            )

        mock_chain.assert_called_once()
        rec = out.get("records", [])[-1]
        assert rec["params"]["method"] == "package"


# ---------------------------------------------------------------------------
# Test 3: run_cohort (the `omicstra run` backend)
# ---------------------------------------------------------------------------
class TestRunCohortRouting:
    def test_run_cohort_propagates_compute(self, tmp_path):
        """run_cohort calls run_align(compute=compute) and inherits the
        default use_package=True."""
        from omicstra.run import run_cohort

        cfg, nj, runs_root = _make_fixture(tmp_path)
        fake_grid = AlignGrid(
            runs_root=str(runs_root),
            run_refs={"R1_v3": str(runs_root / "R1_v3" / "embeddings_test.parquet")},
            computed=["R1_v3"],
        )
        fake_rec = MagicMock()
        fake_rec.params = {"method": "package", "n_runs": 1, "computed": ["R1_v3"]}
        fake_rec.model_dump.return_value = {
            "step_id": "align", "status": "pass", "kind": "transform",
            "params": fake_rec.params,
        }

        with patch("omicstra.run.ProjectConfig") as mock_cfg_cls, \
             patch("omicstra.run.run_embed_he", return_value=(None, MagicMock(params={}))), \
             patch("omicstra.run.run_embed_st", return_value=(None, MagicMock(params={}))), \
             patch("omicstra.run.run_niche_join", return_value=(nj, MagicMock(params={}))), \
             patch("omicstra.run.run_align", return_value=(fake_grid, fake_rec)) as mock_ra, \
             patch("omicstra.run.run_eval", return_value=(None, MagicMock(params={}))), \
             patch("omicstra.run.settings") as mock_settings:
            mock_cfg_cls.load.return_value = cfg
            cfg.project_dir = tmp_path
            mock_settings.project_root.return_value = tmp_path

            run_cohort(project_id="test", compute=True)

        mock_ra.assert_called_once()
        call_kwargs = mock_ra.call_args
        assert call_kwargs.kwargs.get("compute") is True
        assert call_kwargs.kwargs.get("use_package", True) is not False, \
            "run_cohort must not pass use_package=False"


# ---------------------------------------------------------------------------
# Test 4: MCP tool entry point
# ---------------------------------------------------------------------------
class TestMCPRouting:
    def test_align_run_tool_propagates_compute(self):
        """the MCP align_run tool builds the align graph and invokes it
        with compute=True in the state."""
        from omicstra.mcp.server import align_run

        fake_out = {
            "report": {
                "status": "computed",
                "n_arms": 1,
                "runs": ["R1_v3"],
                "method": "package",
            }
        }

        with patch("omicstra.mcp.server._runner") as mock_runner, \
             patch("omicstra.graph._load_cohort", return_value={}), \
             patch("omicstra.tracing.assert_may_trace"):
            mock_app = MagicMock()
            mock_app.invoke.return_value = fake_out
            mock_app.get_state.return_value = MagicMock(
                values=fake_out, next=(), tasks=()
            )
            mock_runner.return_value = (mock_app, None)

            result = align_run(project_id="test", compute=True)

        mock_app.invoke.assert_called_once()
        invoke_args = mock_app.invoke.call_args[0][0]
        assert invoke_args["compute"] is True