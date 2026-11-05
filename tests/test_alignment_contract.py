"""test the declaration-driven alignment adapter.

two fixtures: seed (virchow2/1280d/k=6/patient_id) and alternate
(uni2/1024d/k=4/donor_id). no paid calls - all data is synthetic numpy.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from omicstra.adapters.alignment_data import (
    AlignmentContract,
    AlignmentDeclarationMissing,
    build_matrices,
    discover_subarrays,
    load_subarray,
)


SEED_CONTRACT = AlignmentContract.for_fixture(
    he_niche_column="virchow2_niche",
    he_tokens_column="virchow2_cell_tokens",
    he_dim=1280,
    k_plus_1=7,
    supervision_column="mc_weights_niche",
    subject_column="patient_id",
)

ALT_CONTRACT = AlignmentContract.for_fixture(
    he_niche_column="uni2_niche",
    he_tokens_column="uni2_cell_tokens",
    he_dim=1024,
    k_plus_1=5,
    supervision_column="label_weights",
    subject_column="donor_id",
    stratification_column="site",
    unit_label_columns=("cell_type",),
)


def _make_parquet(path, *, he_dim, k_plus_1, n_mc, n_spots=20,
                  he_niche_col, he_tokens_col, supervision_col,
                  subject_col, subject_val, sample_name,
                  extra_cols=None):
    rng = np.random.default_rng(42)
    rows = {}
    rows[he_niche_col] = [rng.standard_normal(he_dim).tolist() for _ in range(n_spots)]
    rows[he_tokens_col] = [rng.standard_normal(k_plus_1 * he_dim).tolist() for _ in range(n_spots)]
    rows[supervision_col] = [rng.dirichlet(np.ones(n_mc)).tolist() for _ in range(n_spots)]
    rows[subject_col] = [subject_val] * n_spots
    rows["subarray"] = [sample_name] * n_spots
    if extra_cols:
        for col, vals in extra_cols.items():
            rows[col] = vals
    df = pd.DataFrame(rows, index=[f"spot_{i}" for i in range(n_spots)])
    df.to_parquet(path)
    return df


@pytest.fixture
def seed_parquets(tmp_path):
    d = tmp_path / "niches"
    d.mkdir()
    for i, pid in enumerate([1, 2, 3]):
        _make_parquet(
            d / f"sub_{i}.parquet",
            he_dim=1280, k_plus_1=7, n_mc=14, n_spots=15,
            he_niche_col="virchow2_niche",
            he_tokens_col="virchow2_cell_tokens",
            supervision_col="mc_weights_niche",
            subject_col="patient_id",
            subject_val=pid,
            sample_name=f"subarray_{i}",
            extra_cols={"novae_niche": [np.random.default_rng(i).standard_normal(64).tolist() for _ in range(15)]},
        )
    return d


@pytest.fixture
def alt_parquets(tmp_path):
    d = tmp_path / "niches_alt"
    d.mkdir()
    for i, did in enumerate(["D001", "D002"]):
        _make_parquet(
            d / f"alt_{i}.parquet",
            he_dim=1024, k_plus_1=5, n_mc=8, n_spots=10,
            he_niche_col="uni2_niche",
            he_tokens_col="uni2_cell_tokens",
            supervision_col="label_weights",
            subject_col="donor_id",
            subject_val=did,
            sample_name=f"sample_{i}",
            extra_cols={
                "site": ["kidney"] * 10,
                "cell_type": [f"type_{j % 3}" for j in range(10)],
                "mol_embed": [np.random.default_rng(i).standard_normal(128).tolist() for _ in range(10)],
            },
        )
    return d


class TestAlignmentContract:
    def test_for_fixture_seed(self):
        c = SEED_CONTRACT
        assert c.he_dim == 1280
        assert c.k_plus_1 == 7
        assert c.supervision_column == "mc_weights_niche"
        assert c.subject_column == "patient_id"
        assert c.stratification_column is None

    def test_for_fixture_alt(self):
        c = ALT_CONTRACT
        assert c.he_dim == 1024
        assert c.k_plus_1 == 5
        assert c.supervision_column == "label_weights"
        assert c.subject_column == "donor_id"
        assert c.stratification_column == "site"
        assert c.unit_label_columns == ("cell_type",)

    def test_frozen(self):
        with pytest.raises(AttributeError):
            SEED_CONTRACT.he_dim = 999

    def test_from_project_dir_seed(self, tmp_path):
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "project.json").write_text(json.dumps({
            "encoders": [
                {"name": "virchow2", "dim": 1280, "role": "morphology"},
                {"name": "novae", "dim": 64, "role": "molecular"},
            ],
            "k_neighbors": 6,
            "supervision": "mc_weights_niche",
        }))
        (proj / "cohort.json").write_text(json.dumps({
            "subject_id_column": "patient_id",
        }))
        c = AlignmentContract.from_project_dir(proj)
        assert c.he_niche_column == "virchow2_niche"
        assert c.he_tokens_column == "virchow2_cell_tokens"
        assert c.he_dim == 1280
        assert c.k_plus_1 == 7
        assert c.subject_column == "patient_id"

    def test_from_project_dir_alt(self, tmp_path):
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "project.json").write_text(json.dumps({
            "encoders": [
                {"name": "uni2", "dim": 1024, "role": "primary"},
            ],
            "k_neighbors": 4,
            "supervision": "label_weights",
        }))
        (proj / "cohort.json").write_text(json.dumps({
            "subject_id_column": "donor_id",
        }))
        c = AlignmentContract.from_project_dir(proj)
        assert c.he_niche_column == "uni2_niche"
        assert c.he_tokens_column == "uni2_cell_tokens"
        assert c.he_dim == 1024
        assert c.k_plus_1 == 5
        assert c.subject_column == "donor_id"


class TestDeclarationRefusal:
    def test_missing_encoders(self, tmp_path):
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "project.json").write_text(json.dumps({
            "k_neighbors": 6,
            "supervision": "mc_weights_niche",
        }))
        (proj / "cohort.json").write_text(json.dumps({
            "subject_id_column": "patient_id",
        }))
        with pytest.raises(AlignmentDeclarationMissing, match="no encoders declared"):
            AlignmentContract.from_project_dir(proj)

    def test_missing_multiple(self, tmp_path):
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "project.json").write_text(json.dumps({"project_id": "test"}))
        with pytest.raises(AlignmentDeclarationMissing, match="declaration") as exc_info:
            AlignmentContract.from_project_dir(proj)
        msg = str(exc_info.value)
        assert "no encoders declared" in msg
        assert "no k_neighbors declared" in msg
        assert "no supervision declared" in msg
        assert "no subject_id_column declared" in msg

    def test_no_morphology_role(self, tmp_path):
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "project.json").write_text(json.dumps({
            "encoders": [{"name": "novae", "dim": 64, "role": "molecular"}],
            "k_neighbors": 6,
            "supervision": "mc_weights_niche",
        }))
        (proj / "cohort.json").write_text(json.dumps({
            "subject_id_column": "patient_id",
        }))
        with pytest.raises(AlignmentDeclarationMissing, match="no encoder with role"):
            AlignmentContract.from_project_dir(proj)

    def test_missing_dim(self, tmp_path):
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "project.json").write_text(json.dumps({
            "encoders": [{"name": "virchow2", "role": "morphology"}],
            "k_neighbors": 6,
            "supervision": "mc_weights_niche",
        }))
        (proj / "cohort.json").write_text(json.dumps({
            "subject_id_column": "patient_id",
        }))
        with pytest.raises(AlignmentDeclarationMissing, match="declares no dim"):
            AlignmentContract.from_project_dir(proj)

    def test_no_project_dir(self):
        with pytest.raises(AlignmentDeclarationMissing, match="no directory"):
            AlignmentContract.from_project_dir("/nonexistent/path")

    def test_load_subarray_no_contract_refused(self, seed_parquets):
        """no contract or project_dir is refused, not silently defaulted."""
        paths = discover_subarrays(seed_parquets)
        with pytest.raises(AlignmentDeclarationMissing, match="requires either a contract"):
            load_subarray(paths[0], ("novae_niche",))


class TestLoadSubarray:
    def test_seed_shapes(self, seed_parquets):
        paths = discover_subarrays(seed_parquets)
        assert len(paths) == 3
        s = load_subarray(paths[0], ("novae_niche",), contract=SEED_CONTRACT)
        assert s["x_he_niche"].shape[1] == 1280
        assert s["x_he_tokens"].shape == (15, 7, 1280)
        assert s["x_st"].shape[1] == 64
        assert s["mc_w"].shape[1] == 14
        assert s["patient_id"] == 1

    def test_alt_shapes(self, alt_parquets):
        paths = discover_subarrays(alt_parquets)
        assert len(paths) == 2
        s = load_subarray(paths[0], ("mol_embed",), contract=ALT_CONTRACT)
        assert s["x_he_niche"].shape[1] == 1024
        assert s["x_he_tokens"].shape == (10, 5, 1024)
        assert s["x_st"].shape[1] == 128
        assert s["mc_w"].shape[1] == 8
        assert s["patient_id"] == "D001"
        assert s["archetype"] == "kidney"
        assert "cell_type" in s

    def test_dim_mismatch_refused(self, seed_parquets):
        """contract he_dim doesn't match data - refused, not silently accepted."""
        mismatched = AlignmentContract.for_fixture(
            he_niche_column="virchow2_niche",
            he_tokens_column="virchow2_cell_tokens",
            he_dim=512,
            k_plus_1=7,
            supervision_column="mc_weights_niche",
            subject_column="patient_id",
        )
        paths = discover_subarrays(seed_parquets)
        with pytest.raises(AlignmentDeclarationMissing, match="declared he_dim=512"):
            load_subarray(paths[0], ("novae_niche",), contract=mismatched)

    def test_missing_column_refused(self, seed_parquets):
        wrong = AlignmentContract.for_fixture(
            he_niche_column="nonexistent_col",
            he_tokens_column="virchow2_cell_tokens",
            he_dim=1280,
            k_plus_1=7,
            supervision_column="mc_weights_niche",
            subject_column="patient_id",
        )
        paths = discover_subarrays(seed_parquets)
        with pytest.raises(AlignmentDeclarationMissing, match="missing columns.*nonexistent_col"):
            load_subarray(paths[0], ("novae_niche",), contract=wrong)

    def test_missing_st_feature_refused(self, seed_parquets):
        paths = discover_subarrays(seed_parquets)
        with pytest.raises(AlignmentDeclarationMissing, match="not_a_feature.*not in columns"):
            load_subarray(paths[0], ("not_a_feature",), contract=SEED_CONTRACT)

    def test_spot_ids_preserved(self, seed_parquets):
        paths = discover_subarrays(seed_parquets)
        s = load_subarray(paths[0], ("novae_niche",), contract=SEED_CONTRACT)
        assert all(sid.startswith("spot_") for sid in s["spot_id"])
        assert len(s["spot_id"]) == 15

    def test_nan_handling(self, tmp_path):
        d = tmp_path / "niches_nan"
        d.mkdir()
        rng = np.random.default_rng(99)
        n = 10
        he_niche = [rng.standard_normal(1280).tolist() for _ in range(n)]
        he_niche[3] = [float("nan")] * 1280
        rows = {
            "virchow2_niche": he_niche,
            "virchow2_cell_tokens": [rng.standard_normal(7 * 1280).tolist() for _ in range(n)],
            "mc_weights_niche": [rng.dirichlet(np.ones(14)).tolist() for _ in range(n)],
            "patient_id": [1] * n,
            "subarray": ["s0"] * n,
            "novae_niche": [rng.standard_normal(64).tolist() for _ in range(n)],
        }
        df = pd.DataFrame(rows, index=[f"spot_{i}" for i in range(n)])
        df.to_parquet(d / "nan_test.parquet")

        s = load_subarray(d / "nan_test.parquet", ("novae_niche",), contract=SEED_CONTRACT)
        assert s["n_dropped_nan"] == 1
        assert len(s["spot_id"]) == 9


class TestBuildMatrices:
    def test_stacking(self, seed_parquets):
        paths = discover_subarrays(seed_parquets)
        subs = [load_subarray(p, ("novae_niche",), contract=SEED_CONTRACT) for p in paths]
        X_he, X_st, meta = build_matrices(subs)
        total = sum(len(s["spot_id"]) for s in subs)
        assert X_he.shape == (total, 1280)
        assert X_st.shape == (total, 64)
        assert len(meta) == total
        assert set(meta.columns) >= {"spot_id", "subarray", "patient_id"}
