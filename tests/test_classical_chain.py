"""classical baseline chain: execution, dispatch, upstream failure, and parity.

these tests exercise the four-step CLASSICAL_BASELINE_STEPS chain
(load_data | apply_reference_split | fit_project | evaluate_and_save)
with synthetic data, verifying that the chain produces the expected
artifacts and that upstream failures propagate correctly.

no torch dependency. the classical chain uses numpy/sklearn only.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from omicstra.contracts.project import ProjectConfig
from omicstra.protocols.align import (
    CLASSICAL_BASELINE_STEPS,
    NicheJoin,
    _resolve_run_config,
    _train_classical_chain,
    run_align,
)
from omicstra.protocols import build_protocol


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
N_SPOTS = 20
HE_DIM = 32
ST_DIM = 16
N_SUBARRAYS = 6
N_PATIENTS = 4


def _write_synthetic_parquets(niches_dir: Path) -> None:
    """write minimal niche-join parquets that load_subarray can read."""
    rng = np.random.default_rng(42)
    patients = list(range(1, N_PATIENTS + 1))
    subs_per_patient = N_SUBARRAYS // N_PATIENTS

    sub_idx = 0
    for pid in patients:
        for _ in range(subs_per_patient):
            sub_idx += 1
            sub_name = f"sub_{sub_idx:03d}"
            rows = []
            for j in range(N_SPOTS):
                rows.append({
                    "spot_id": f"{sub_name}_s{j}",
                    "subarray": sub_name,
                    "patient_id": pid,
                    "archetype": pid % 3,
                    "compartment": "tumor" if j % 2 == 0 else "stroma",
                    "virchow2_niche": rng.normal(size=HE_DIM).astype(np.float32).tolist(),
                    "virchow2_cell_tokens": rng.normal(size=7 * 1280).astype(np.float32).tolist(),
                    "novae_niche": rng.normal(size=ST_DIM).astype(np.float32).tolist(),
                    "gpath2vec_niche": np.zeros(0, dtype=np.float32).tolist(),
                    "mc_weights_niche": rng.dirichlet(np.ones(5)).astype(np.float32).tolist(),
                })
            df = pd.DataFrame(rows).set_index("spot_id")
            df.to_parquet(niches_dir / f"{sub_name}.parquet")
    # remaining subarrays for odd division
    while sub_idx < N_SUBARRAYS:
        sub_idx += 1
        pid = patients[sub_idx % len(patients)]
        sub_name = f"sub_{sub_idx:03d}"
        rows = []
        for j in range(N_SPOTS):
            rows.append({
                "spot_id": f"{sub_name}_s{j}",
                "subarray": sub_name,
                "patient_id": pid,
                "archetype": pid % 3,
                "compartment": "tumor",
                "virchow2_niche": rng.normal(size=HE_DIM).astype(np.float32).tolist(),
                "virchow2_cell_tokens": rng.normal(size=7 * 1280).astype(np.float32).tolist(),
                "novae_niche": rng.normal(size=ST_DIM).astype(np.float32).tolist(),
                "gpath2vec_niche": np.zeros(0, dtype=np.float32).tolist(),
                "mc_weights_niche": rng.dirichlet(np.ones(5)).astype(np.float32).tolist(),
            })
        df = pd.DataFrame(rows).set_index("spot_id")
        df.to_parquet(niches_dir / f"{sub_name}.parquet")


def _write_reference_split(split_path: Path) -> None:
    patients = list(range(1, N_PATIENTS + 1))
    split = {
        "train_patients": patients[:2],
        "val_patients": [patients[2]],
        "test_patients": [patients[3]],
        "seed": 42,
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2))


@pytest.fixture
def classical_env(tmp_path):
    """set up synthetic niches + reference split for classical chain tests."""
    niches = tmp_path / "niches"
    niches.mkdir()
    _write_synthetic_parquets(niches)

    ref_run_dir = tmp_path / "runs" / "R1_v3"
    ref_run_dir.mkdir(parents=True)
    _write_reference_split(ref_run_dir / "split.json")

    return {
        "niches_dir": str(niches),
        "ref_split": ref_run_dir / "split.json",
        "runs_root": tmp_path / "runs",
        "tmp_path": tmp_path,
    }


# ---------------------------------------------------------------------------
# 1. chain execution: each method runs and produces correct artifacts
# ---------------------------------------------------------------------------
class TestClassicalChainExecution:
    @pytest.mark.parametrize("method", ["cca", "procrustes", "unaligned"])
    def test_chain_produces_artifacts(self, classical_env, method):
        """the four-step chain runs end-to-end and writes the expected files."""
        run_dir = classical_env["runs_root"] / f"test_{method}"
        run_cfg = {
            "run_id": f"test_{method}",
            "method": method,
            "shared_dim": 8,
            "st_features": ["novae_niche"],
        }

        records = _train_classical_chain(
            run_cfg, classical_env["niches_dir"],
            classical_env["ref_split"], run_dir)

        assert (run_dir / "embeddings_test.parquet").is_file()
        assert (run_dir / "metrics_h1_raw.json").is_file()
        assert (run_dir / "split.json").is_file()
        assert (run_dir / "run_config.json").is_file()

        for step_id in ("load_data", "apply_reference_split",
                        "fit_project", "evaluate_and_save"):
            assert step_id in records, f"missing record for {step_id}"
            assert records[step_id]["status"] == "pass"

    @pytest.mark.parametrize("method", ["cca", "procrustes", "unaligned"])
    def test_metrics_schema(self, classical_env, method):
        """metrics_h1_raw.json has the expected keys including baseline-specific fields."""
        run_dir = classical_env["runs_root"] / f"test_{method}_schema"
        run_cfg = {
            "run_id": f"test_{method}_schema",
            "method": method,
            "shared_dim": 8,
            "st_features": ["novae_niche"],
        }

        _train_classical_chain(
            run_cfg, classical_env["niches_dir"],
            classical_env["ref_split"], run_dir)

        metrics = json.loads((run_dir / "metrics_h1_raw.json").read_text())
        assert "shared_dim_used" in metrics
        assert metrics["baseline"] == method
        assert "cka_before" in metrics
        assert "cka_after" in metrics

    def test_reference_split_is_copied(self, classical_env):
        """the chain copies the reference split, not a fresh one."""
        run_dir = classical_env["runs_root"] / "test_split_copy"
        run_cfg = {
            "run_id": "test_split_copy",
            "method": "unaligned",
            "shared_dim": 8,
            "st_features": ["novae_niche"],
        }

        _train_classical_chain(
            run_cfg, classical_env["niches_dir"],
            classical_env["ref_split"], run_dir)

        ref = json.loads(classical_env["ref_split"].read_text())
        copied = json.loads((run_dir / "split.json").read_text())
        assert ref == copied

    def test_embeddings_row_count_matches_metadata(self, classical_env):
        """emb_df rows, z_he rows and z_st rows all match."""
        run_dir = classical_env["runs_root"] / "test_row_match"
        run_cfg = {
            "run_id": "test_row_match",
            "method": "procrustes",
            "shared_dim": 8,
            "st_features": ["novae_niche"],
        }

        _train_classical_chain(
            run_cfg, classical_env["niches_dir"],
            classical_env["ref_split"], run_dir)

        emb_df = pd.read_parquet(run_dir / "embeddings_test.parquet")
        z_he = np.stack(emb_df["z_he"].values)
        z_st = np.stack(emb_df["z_st"].values)
        assert len(emb_df) == z_he.shape[0] == z_st.shape[0]
        assert len(emb_df) > 0

    def test_provenance_separates_method_and_backend(self, classical_env):
        """run_config.json records 'method' (scientific) and 'backend' (execution) separately."""
        run_dir = classical_env["runs_root"] / "test_provenance"
        run_cfg = {
            "run_id": "test_provenance",
            "method": "cca",
            "shared_dim": 8,
            "st_features": ["novae_niche"],
        }

        _train_classical_chain(
            run_cfg, classical_env["niches_dir"],
            classical_env["ref_split"], run_dir)

        cfg = json.loads((run_dir / "run_config.json").read_text())
        assert cfg["method"] == "cca"
        assert cfg["backend"] == "package"


# ---------------------------------------------------------------------------
# 2. CCA reg passthrough
# ---------------------------------------------------------------------------
class TestCCAReg:
    def test_reg_reaches_run_cca(self, classical_env):
        """configured reg value is passed through the chain to run_cca."""
        from unittest.mock import patch as _patch
        from omicstra.models.alignment_classical import BASELINES, run_cca

        run_dir = classical_env["runs_root"] / "test_reg"
        run_cfg = {
            "run_id": "test_reg",
            "method": "cca",
            "shared_dim": 8,
            "st_features": ["novae_niche"],
            "reg": 1e-3,
        }

        spy = MagicMock(wraps=run_cca)
        with _patch.dict(BASELINES, {"cca": spy}):
            _train_classical_chain(
                run_cfg, classical_env["niches_dir"],
                classical_env["ref_split"], run_dir)

        spy.assert_called_once()
        assert spy.call_args.kwargs.get("reg") == 1e-3


# ---------------------------------------------------------------------------
# 3. upstream failure propagation
# ---------------------------------------------------------------------------
class TestUpstreamFailure:
    def test_load_data_failure_propagates(self, classical_env):
        """if load_data finds no parquets, downstream steps are not_run."""
        empty_dir = classical_env["tmp_path"] / "empty_niches"
        empty_dir.mkdir()
        run_dir = classical_env["runs_root"] / "test_fail"
        run_cfg = {
            "run_id": "test_fail",
            "method": "cca",
            "shared_dim": 8,
            "st_features": ["novae_niche"],
        }

        records = _train_classical_chain(
            run_cfg, str(empty_dir),
            classical_env["ref_split"], run_dir)

        assert records["load_data"]["status"] == "fail"
        for step in ("apply_reference_split", "fit_project", "evaluate_and_save"):
            assert records[step]["status"] == "not_run"


# ---------------------------------------------------------------------------
# 4. dispatch routing: run_align reaches the classical chain
# ---------------------------------------------------------------------------
CLASSICAL_STEP_IDS = ("load_data", "apply_reference_split",
                      "fit_project", "evaluate_and_save")

FAKE_CLASSICAL_RECORDS = {
    sid: {"status": "pass", "duration_s": 0.1, "params": {}}
    for sid in CLASSICAL_STEP_IDS
}


def _make_classical_fixture(tmp_path, run_id="B1_v3"):
    """fixture that triggers the classical compute path for one baseline run."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    # reference contrastive run with a split
    ref_dir = runs_root / "R1_v3"
    ref_dir.mkdir()
    _write_reference_split(ref_dir / "split.json")
    # mark the reference as already computed
    (ref_dir / "embeddings_test.parquet").write_bytes(b"")

    niches = tmp_path / "niches"
    niches.mkdir()

    cfg = MagicMock(spec=ProjectConfig)
    cfg.runs_dir = str(runs_root)
    cfg.contrastive_runs = ["R1_v3"]
    cfg.classical_runs = [run_id]

    nj = NicheJoin(manifest_ref="test", niches_dir=str(niches))
    return cfg, nj, runs_root


class TestClassicalDispatch:
    def test_classical_uses_chain_by_default(self, tmp_path):
        """run_align dispatches B1 to _train_classical_chain when use_package=True."""
        cfg, nj, runs_root = _make_classical_fixture(tmp_path, "B1_v3")

        with patch("omicstra.protocols.align._train_classical_chain",
                   return_value=FAKE_CLASSICAL_RECORDS) as mock_chain, \
             patch("omicstra.protocols.align._run_script") as mock_script, \
             patch("omicstra.protocols.align._refuse_protected"), \
             patch("omicstra.protocols.align._project_rel", return_value=runs_root):
            grid, rec = run_align(cfg, nj, compute=True,
                                  runs_root=runs_root, run_ids=["B1_v3"])

        mock_chain.assert_called_once()
        mock_script.assert_not_called()
        assert set(rec.params["chain_records"]["B1_v3"]) == set(CLASSICAL_STEP_IDS)

    def test_classical_uses_script_when_false(self, tmp_path):
        """run_align dispatches B1 to script when use_package=False."""
        cfg, nj, runs_root = _make_classical_fixture(tmp_path, "B1_v3")

        with patch("omicstra.protocols.align._train_classical_chain") as mock_chain, \
             patch("omicstra.protocols.align._run_script") as mock_script, \
             patch("omicstra.protocols.align._refuse_protected"), \
             patch("omicstra.protocols.align._project_rel", return_value=runs_root):
            grid, rec = run_align(cfg, nj, compute=True,
                                  runs_root=runs_root, use_package=False,
                                  run_ids=["B1_v3"])

        mock_script.assert_called_once()
        mock_chain.assert_not_called()

    @pytest.mark.parametrize("run_id,expected_method", [
        ("B1_v3", "cca"),
        ("B2_v3", "procrustes"),
        ("B3_v3", "unaligned"),
    ])
    def test_method_resolved_from_config(self, run_id, expected_method):
        """_resolve_run_config reads method from the config file."""
        run_cfg = _resolve_run_config(run_id)
        assert run_cfg["method"] == expected_method

    def test_contrastive_inferred_from_legacy_config(self):
        """R1_v3 config has no method field; resolver infers contrastive from loss."""
        run_cfg = _resolve_run_config("R1_v3")
        assert run_cfg["method"] == "contrastive"

    def test_random_init_dispatches_to_script(self, tmp_path):
        """B4 (random_init) always dispatches to script regardless of use_package."""
        cfg, nj, runs_root = _make_classical_fixture(tmp_path, "B4_v3")

        with patch("omicstra.protocols.align._train_classical_chain") as mock_chain, \
             patch("omicstra.protocols.align._run_script") as mock_script, \
             patch("omicstra.protocols.align._refuse_protected"), \
             patch("omicstra.protocols.align._project_rel", return_value=runs_root):
            grid, rec = run_align(cfg, nj, compute=True,
                                  runs_root=runs_root, run_ids=["B4_v3"])

        mock_script.assert_called_once()
        assert "align_b4.py" in mock_script.call_args[0][0]
        mock_chain.assert_not_called()


# ---------------------------------------------------------------------------
# 5. chain structure: build_protocol accepts the step list
# ---------------------------------------------------------------------------
class TestChainStructure:
    def test_classical_chain_builds(self):
        """CLASSICAL_BASELINE_STEPS composes through build_protocol without error."""
        chain = build_protocol(CLASSICAL_BASELINE_STEPS, "classical_baseline")
        assert chain is not None

    def test_step_ids_are_distinct(self):
        ids = [s.id for s in CLASSICAL_BASELINE_STEPS]
        assert len(ids) == len(set(ids))

    def test_produces_requires_chain(self):
        """each step's requires is a subset of what earlier steps produce."""
        produced = set()
        for step in CLASSICAL_BASELINE_STEPS:
            assert step.requires <= produced, (
                f"{step.id} requires {step.requires - produced} "
                f"not yet produced")
            produced |= step.produces