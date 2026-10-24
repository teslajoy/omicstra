"""tests for the five-step LCEL alignment training chain.

verifies: chain construction, per-step records with timing, upstream failure
propagation, and step isolation. does NOT require niche data on disk - uses
synthetic tensors.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("torch", reason="requires optional dependency torch")

from omicstra.protocols import build_protocol
from omicstra.protocols.align import ALIGN_TRAIN_STEPS, _train_contrastive_chain

SEED = 1234


class TestChainConstruction:
    def test_builds_without_error(self):
        chain = build_protocol(ALIGN_TRAIN_STEPS, "align_train")
        assert chain is not None

    def test_step_order_valid(self):
        """requires/produces are satisfied by earlier steps."""
        seen: set[str] = set()
        for s in ALIGN_TRAIN_STEPS:
            assert s.requires <= seen, f"{s.id} requires {s.requires - seen}"
            seen |= s.produces

    def test_step_ids(self):
        ids = [s.id for s in ALIGN_TRAIN_STEPS]
        assert ids == ["load_data", "split", "train", "embed_test", "evaluate_and_save"]


class TestUpstreamFailurePropagation:
    """when load_data fails, all downstream steps are not_run."""

    def test_downstream_not_run_on_load_failure(self):
        chain = build_protocol(ALIGN_TRAIN_STEPS, "align_train")
        records = chain.invoke({
            "params": {
                "load_data": {
                    "niches_dir": "/nonexistent/path",
                    "st_features": ("novae_niche",),
                },
                "split": {"val_frac": 0.15, "test_frac": 0.15, "seed": 42, "run_dir": "/tmp"},
                "train": {"loss": "infonce", "seed": 42, "run_dir": "/tmp"},
                "embed_test": {"device": "cpu"},
                "evaluate_and_save": {"run_dir": "/tmp"},
            }
        })
        assert records["load_data"]["status"] == "fail"
        for step_id in ("split", "train", "embed_test", "evaluate_and_save"):
            assert records[step_id]["status"] == "not_run", \
                f"{step_id} should be not_run when load_data fails"


class TestStepIsolation:
    """each step fn, given a valid ctx, returns (DiagnosticRecord, dict)."""

    @pytest.fixture
    def synthetic_subarrays(self):
        rng = np.random.RandomState(SEED)
        subs = []
        for i in range(6):
            n = 20
            subs.append({
                "spot_id": np.arange(i * n, (i + 1) * n),
                "patient_id": i // 2,
                "archetype": i % 3,
                "compartment": np.array(["stroma"] * n),
                "subarray": f"SA{i}",
                "x_he_niche": rng.randn(n, 1280).astype(np.float32),
                "x_he_tokens": rng.randn(n, 7, 1280).astype(np.float32),
                "x_st": rng.randn(n, 576).astype(np.float32),
                "mc_w": rng.dirichlet(np.ones(14), n).astype(np.float32),
                "n_dropped_nan": 0,
            })
        return subs

    def test_step_split(self, synthetic_subarrays):
        from omicstra.protocols.align import _step_split
        with tempfile.TemporaryDirectory() as td:
            ctx = {
                "subarrays": synthetic_subarrays,
                "run_dir": Path(td),
                "params": {"val_frac": 0.15, "test_frac": 0.15,
                           "seed": SEED, "run_dir": td},
            }
            rec, updates = _step_split(ctx)
            assert rec.status == "pass"
            assert "split" in updates
            assert set(updates["split"]) >= {"train", "val", "test"}
            split_file = Path(td) / "split.json"
            assert split_file.is_file()
            saved = json.loads(split_file.read_text())
            assert sorted(saved["train_patients"]) == sorted(updates["split"]["train_patients"])

    def test_step_train(self, synthetic_subarrays):
        from omicstra.protocols.align import _step_train
        with tempfile.TemporaryDirectory() as td:
            train = synthetic_subarrays[:4]
            val = synthetic_subarrays[4:]
            ctx = {
                "split": {"train": train, "val": val, "test": []},
                "run_dir": Path(td),
                "params": {
                    "fusion": "late", "loss": "infonce", "seed": SEED,
                    "epochs": 2, "patience": 2, "run_dir": td,
                    "device": "cpu",
                },
            }
            rec, updates = _step_train(ctx)
            assert rec.status == "pass"
            assert "model" in updates
            assert (Path(td) / "checkpoint.pt").is_file()
            assert (Path(td) / "training_log.csv").is_file()

    def test_step_embed_test(self, synthetic_subarrays):
        from omicstra.protocols.align import _step_embed_test, _step_train
        with tempfile.TemporaryDirectory() as td:
            train = synthetic_subarrays[:4]
            val = synthetic_subarrays[4:5]
            test = synthetic_subarrays[5:]
            ctx_train = {
                "split": {"train": train, "val": val, "test": test},
                "run_dir": Path(td),
                "params": {
                    "fusion": "late", "loss": "infonce", "seed": SEED,
                    "epochs": 2, "patience": 2, "run_dir": td,
                    "device": "cpu",
                },
            }
            _, train_updates = _step_train(ctx_train)
            ctx_embed = {
                "model": train_updates["model"],
                "split": {"test": test},
                "params": {"device": "cpu"},
            }
            rec, updates = _step_embed_test(ctx_embed)
            assert rec.status == "pass"
            assert "embeddings" in updates
            assert updates["embeddings"]["Z_he"].shape[1] == 512

    def test_step_evaluate_and_save(self, synthetic_subarrays):
        from omicstra.protocols.align import (
            _step_embed_test,
            _step_evaluate_and_save,
            _step_train,
        )
        with tempfile.TemporaryDirectory() as td:
            train = synthetic_subarrays[:4]
            val = synthetic_subarrays[4:5]
            test = synthetic_subarrays[5:]
            _, train_up = _step_train({
                "split": {"train": train, "val": val, "test": test},
                "run_dir": Path(td),
                "params": {"fusion": "late", "loss": "infonce", "seed": SEED,
                           "epochs": 2, "patience": 2, "run_dir": td, "device": "cpu"},
            })
            _, embed_up = _step_embed_test({
                "model": train_up["model"], "split": {"test": test},
                "params": {"device": "cpu"},
            })
            rec, updates = _step_evaluate_and_save({
                "embeddings": embed_up["embeddings"],
                "run_dir": Path(td),
                "params": {"run_dir": td, "ks": [1, 5, 10],
                           "negative_ratio": 10, "negative_seed": 0},
            })
            assert rec.status == "pass"
            assert "metrics" in updates
            assert (Path(td) / "embeddings_test.parquet").is_file()
            assert (Path(td) / "metrics_h1_raw.json").is_file()


class TestFullChainSynthetic:
    """run the full five-step chain on synthetic data."""

    def _make_parquets(self, td: str, n_subs: int = 6) -> Path:
        """write synthetic niche-join parquets matching load_subarray's expectations."""
        import pandas as pd

        niches = Path(td) / "niches"
        niches.mkdir()
        rng = np.random.RandomState(SEED)
        for i in range(n_subs):
            n = 20
            df = pd.DataFrame({
                "virchow2_niche": [rng.randn(1280).astype(np.float32).tolist() for _ in range(n)],
                "virchow2_cell_tokens": [rng.randn(7 * 1280).astype(np.float32).tolist() for _ in range(n)],
                "novae_niche": [rng.randn(64).astype(np.float32).tolist() for _ in range(n)],
                "gpath2vec_niche": [rng.randn(512).astype(np.float32).tolist() for _ in range(n)],
                "mc_weights_niche": [rng.dirichlet(np.ones(14)).astype(np.float32).tolist() for _ in range(n)],
                "patient_id": [i // 2] * n,
                "archetype": [i % 3] * n,
                "compartment": ["stroma"] * n,
                "subarray": [f"SA{i}"] * n,
            }, index=[f"spot_{i}_{j}" for j in range(n)])
            df.to_parquet(niches / f"SA{i}.parquet")
        return niches

    def test_chain_produces_all_records(self):
        with tempfile.TemporaryDirectory() as td:
            niches = self._make_parquets(td)
            run_dir = Path(td) / "run"
            run_cfg = {
                "run_id": "test_run",
                "loss": "infonce",
                "fusion": "late",
                "st_features": ["novae_niche", "gpath2vec_niche"],
                "seed": SEED,
                "epochs": 2,
                "patience": 2,
                "val_frac": 0.15,
                "test_frac": 0.15,
                "shared_dim": 512,
                "dropout": 0.3,
                "tau": 0.07,
                "tau_target": 0.1,
                "barlow_lambda": 0.005,
                "device": "cpu",
            }
            records = _train_contrastive_chain(run_cfg, str(niches), run_dir)

            expected = {"load_data", "split", "train", "embed_test", "evaluate_and_save"}
            assert set(records.keys()) == expected

            for step_id in expected:
                r = records[step_id]
                assert r["status"] == "pass", f"{step_id}: {r.get('result')}"
                assert r.get("duration_s") is not None and r["duration_s"] >= 0
                assert r.get("params") is not None

            assert (run_dir / "split.json").is_file()
            assert (run_dir / "checkpoint.pt").is_file()
            assert (run_dir / "training_log.csv").is_file()
            assert (run_dir / "embeddings_test.parquet").is_file()
            assert (run_dir / "metrics_h1_raw.json").is_file()
            assert (run_dir / "run_config.json").is_file()

    def test_each_step_executes_once(self):
        """verify no step runs more than once."""
        call_counts: dict[str, int] = {}
        original_fns = {s.id: s.fn for s in ALIGN_TRAIN_STEPS}

        def counting_wrapper(step_id, original_fn):
            def wrapper(ctx):
                call_counts[step_id] = call_counts.get(step_id, 0) + 1
                return original_fn(ctx)
            return wrapper

        with tempfile.TemporaryDirectory() as td:
            niches = self._make_parquets(td)
            run_dir = Path(td) / "run"

            patched_steps = []
            for s in ALIGN_TRAIN_STEPS:
                patched_steps.append(type(s)(
                    id=s.id,
                    fn=counting_wrapper(s.id, s.fn),
                    produces=s.produces,
                    requires=s.requires,
                    params_space=s.params_space,
                ))

            chain = build_protocol(patched_steps, "align_train")
            run_cfg = {
                "loss": "infonce", "fusion": "late",
                "st_features": ["novae_niche", "gpath2vec_niche"],
                "seed": SEED, "epochs": 2, "patience": 2,
                "val_frac": 0.15, "test_frac": 0.15, "shared_dim": 512,
                "dropout": 0.3, "tau": 0.07, "tau_target": 0.1,
                "barlow_lambda": 0.005, "device": "cpu",
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            chain.invoke({
                "params": {
                    "load_data": {"niches_dir": str(niches),
                                  "st_features": ("novae_niche", "gpath2vec_niche")},
                    "split": {"val_frac": 0.15, "test_frac": 0.15,
                              "seed": SEED, "run_dir": str(run_dir)},
                    "train": {"fusion": "late", "loss": "infonce", "seed": SEED,
                              "epochs": 2, "patience": 2, "run_dir": str(run_dir),
                              "device": "cpu"},
                    "embed_test": {"device": "cpu"},
                    "evaluate_and_save": {"run_dir": str(run_dir)},
                },
                "run_dir": run_dir,
            })

            for step_id in ("load_data", "split", "train", "embed_test", "evaluate_and_save"):
                assert call_counts.get(step_id) == 1, \
                    f"{step_id} called {call_counts.get(step_id, 0)} times"


class TestSplitParity:
    """_step_split produces the same patient lists as split_by_patient."""

    def test_same_split(self):
        from omicstra.adapters.alignment_data import split_by_patient
        from omicstra.protocols.align import _step_split

        rng = np.random.RandomState(SEED)
        subs = []
        for i in range(30):
            subs.append({
                "spot_id": np.arange(20),
                "patient_id": i,
                "archetype": i % 3,
                "compartment": np.array(["s"] * 20),
                "subarray": f"SA{i}",
                "x_he_niche": rng.randn(20, 1280).astype(np.float32),
                "x_he_tokens": rng.randn(20, 7, 1280).astype(np.float32),
                "x_st": rng.randn(20, 576).astype(np.float32),
                "mc_w": rng.dirichlet(np.ones(14), 20).astype(np.float32),
                "n_dropped_nan": 0,
            })

        direct_train, direct_val, direct_test = split_by_patient(
            subs, 0.15, 0.15, SEED)

        with tempfile.TemporaryDirectory() as td:
            rec, updates = _step_split({
                "subarrays": subs, "run_dir": Path(td),
                "params": {"val_frac": 0.15, "test_frac": 0.15,
                           "seed": SEED, "run_dir": td},
            })

        assert sorted(updates["split"]["train_patients"]) == sorted(direct_train)
        assert sorted(updates["split"]["val_patients"]) == sorted(direct_val)
        assert sorted(updates["split"]["test_patients"]) == sorted(direct_test)