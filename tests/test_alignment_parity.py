"""parity tests: package models vs reference scripts.

tests forward outputs, losses, gradients, attention, and anisotropic
scale using identical weights, inputs, model modes, and controlled
randomness. reports discrepancies without changing the reference science.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="requires optional dependency torch")

import torch
import torch.nn.functional as F

# --- import package models ----------------------------------------------------
from omicstra.models.alignment import (
    LateFusion as PkgLateFusion,
    CrossAttnFusion as PkgCrossAttnFusion,
    MLPBlock as PkgMLPBlock,
    build_model as pkg_build_model,
    infonce_loss as pkg_infonce,
    aninfonce_loss as pkg_aninfonce,
    supcon_loss as pkg_supcon,
    barlow_loss as pkg_barlow,
    compute_loss as pkg_compute_loss,
)
from omicstra.models.alignment_classical import (
    run_cca as pkg_cca,
    run_procrustes as pkg_procrustes,
    run_unaligned as pkg_unaligned,
    l2_normalize as pkg_l2_normalize,
)

# --- import script models (test-only dependency) -----------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import align as script_align  # noqa: E402
from align import (  # noqa: E402
    LateFusion as ScriptLateFusion,
    CrossAttnFusion as ScriptCrossAttnFusion,
    infonce_loss as script_infonce,
    aninfonce_loss as script_aninfonce,
    supcon_loss as script_supcon,
    barlow_loss as script_barlow,
    compute_loss as script_compute_loss,
    cka_linear as script_cka,
)
from align_classical import (  # noqa: E402
    run_cca as script_cca,
    run_procrustes as script_procrustes,
    run_unaligned as script_unaligned,
    l2_normalize as script_l2_normalize,
)

# dims matching the seed cohort
HE_DIM = 1280
HE_TILE_DIM = 1280
ST_DIM = 576
SHARED_DIM = 512
DROPOUT = 0.3
N = 32   # batch size for tests
N_TILES = 7
MC_DIM = 14


SEED = 1234


@pytest.fixture
def seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return SEED


@pytest.fixture
def inputs(seed):
    x_he_niche = torch.randn(N, HE_DIM)
    x_he_tokens = torch.randn(N, N_TILES, HE_TILE_DIM)
    x_st = torch.randn(N, ST_DIM)
    mc_w = torch.rand(N, MC_DIM)
    return x_he_niche, x_he_tokens, x_st, mc_w


def _copy_weights(src: torch.nn.Module, dst: torch.nn.Module):
    dst.load_state_dict(src.state_dict())


# =============================================================================
# architecture parity - LateFusion eval mode
# =============================================================================
class TestLateFusionEval:
    def test_forward_parity(self, seed, inputs):
        x_he_niche, x_he_tokens, x_st, _ = inputs
        torch.manual_seed(seed)
        script_model = ScriptLateFusion(HE_DIM, ST_DIM, SHARED_DIM, DROPOUT)
        pkg_model = PkgLateFusion(HE_DIM, ST_DIM, SHARED_DIM, DROPOUT)
        _copy_weights(script_model, pkg_model)

        script_model.eval()
        pkg_model.eval()
        with torch.no_grad():
            sz_he, sz_st, s_attn = script_model(x_he_niche, x_he_tokens, x_st)
            pz_he, pz_st, p_attn = pkg_model(x_he_niche, x_he_tokens, x_st)

        assert torch.equal(sz_he, pz_he), f"z_he max diff: {(sz_he - pz_he).abs().max()}"
        assert torch.equal(sz_st, pz_st), f"z_st max diff: {(sz_st - pz_st).abs().max()}"
        assert s_attn is None and p_attn is None

    def test_state_dict_keys_match(self, seed):
        torch.manual_seed(seed)
        s = ScriptLateFusion(HE_DIM, ST_DIM, SHARED_DIM, DROPOUT)
        p = PkgLateFusion(HE_DIM, ST_DIM, SHARED_DIM, DROPOUT)
        assert set(s.state_dict().keys()) == set(p.state_dict().keys())


# =============================================================================
# architecture parity - LateFusion with anisotropic
# =============================================================================
class TestLateFusionAnisotropic:
    def test_forward_parity(self, seed, inputs):
        x_he_niche, x_he_tokens, x_st, _ = inputs
        torch.manual_seed(seed)
        script_model = ScriptLateFusion(HE_DIM, ST_DIM, SHARED_DIM, DROPOUT,
                                        anisotropic=True)
        pkg_model = PkgLateFusion(HE_DIM, ST_DIM, SHARED_DIM, DROPOUT,
                                  anisotropic=True)
        _copy_weights(script_model, pkg_model)

        script_model.eval()
        pkg_model.eval()
        with torch.no_grad():
            sz_he, sz_st, _ = script_model(x_he_niche, x_he_tokens, x_st)
            pz_he, pz_st, _ = pkg_model(x_he_niche, x_he_tokens, x_st)

        assert torch.equal(sz_he, pz_he)
        assert torch.equal(sz_st, pz_st)

    def test_aniso_log_scale_present(self, seed):
        torch.manual_seed(seed)
        pkg = PkgLateFusion(HE_DIM, ST_DIM, SHARED_DIM, DROPOUT, anisotropic=True)
        assert pkg.aniso_log_scale is not None
        assert pkg.aniso_log_scale.shape == (SHARED_DIM,)
        assert torch.equal(pkg.aniso_log_scale.data, torch.zeros(SHARED_DIM))

    def test_aniso_absent_when_false(self, seed):
        torch.manual_seed(seed)
        pkg = PkgLateFusion(HE_DIM, ST_DIM, SHARED_DIM, DROPOUT, anisotropic=False)
        assert pkg.aniso_log_scale is None

    def test_state_dict_keys(self, seed):
        torch.manual_seed(seed)
        s = ScriptLateFusion(HE_DIM, ST_DIM, SHARED_DIM, DROPOUT, anisotropic=True)
        p = PkgLateFusion(HE_DIM, ST_DIM, SHARED_DIM, DROPOUT, anisotropic=True)
        assert set(s.state_dict().keys()) == set(p.state_dict().keys())
        assert "aniso_log_scale" in p.state_dict()


# =============================================================================
# architecture parity - CrossAttnFusion eval mode
# =============================================================================
class TestCrossAttnFusionEval:
    def test_forward_parity(self, seed, inputs):
        x_he_niche, x_he_tokens, x_st, _ = inputs
        torch.manual_seed(seed)
        script_model = ScriptCrossAttnFusion(HE_TILE_DIM, ST_DIM, SHARED_DIM, DROPOUT)
        pkg_model = PkgCrossAttnFusion(HE_TILE_DIM, ST_DIM, SHARED_DIM, DROPOUT)
        _copy_weights(script_model, pkg_model)

        script_model.eval()
        pkg_model.eval()
        with torch.no_grad():
            sz_he, sz_st, s_attn = script_model(x_he_niche, x_he_tokens, x_st)
            pz_he, pz_st, p_attn = pkg_model(x_he_niche, x_he_tokens, x_st)

        assert torch.equal(sz_he, pz_he), f"z_he max diff: {(sz_he - pz_he).abs().max()}"
        assert torch.equal(sz_st, pz_st), f"z_st max diff: {(sz_st - pz_st).abs().max()}"
        assert torch.equal(s_attn, p_attn), f"attn max diff: {(s_attn - p_attn).abs().max()}"

    def test_attention_shape(self, seed, inputs):
        x_he_niche, x_he_tokens, x_st, _ = inputs
        torch.manual_seed(seed)
        pkg = PkgCrossAttnFusion(HE_TILE_DIM, ST_DIM, SHARED_DIM, DROPOUT)
        pkg.eval()
        with torch.no_grad():
            _, _, attn = pkg(x_he_niche, x_he_tokens, x_st)
        assert attn.shape == (N, N_TILES)
        assert torch.allclose(attn.sum(dim=-1), torch.ones(N), atol=1e-6)

    def test_state_dict_keys_match(self, seed):
        torch.manual_seed(seed)
        s = ScriptCrossAttnFusion(HE_TILE_DIM, ST_DIM, SHARED_DIM, DROPOUT)
        p = PkgCrossAttnFusion(HE_TILE_DIM, ST_DIM, SHARED_DIM, DROPOUT)
        assert set(s.state_dict().keys()) == set(p.state_dict().keys())


# =============================================================================
# architecture parity - train mode (BN running stats update)
# =============================================================================
class TestTrainMode:
    def test_late_fusion_train(self, seed, inputs):
        x_he_niche, x_he_tokens, x_st, _ = inputs
        torch.manual_seed(seed)
        script_model = ScriptLateFusion(HE_DIM, ST_DIM, SHARED_DIM, DROPOUT)
        pkg_model = PkgLateFusion(HE_DIM, ST_DIM, SHARED_DIM, DROPOUT)
        _copy_weights(script_model, pkg_model)

        script_model.train()
        pkg_model.train()
        torch.manual_seed(SEED)
        sz_he, sz_st, _ = script_model(x_he_niche, x_he_tokens, x_st)
        torch.manual_seed(SEED)
        pz_he, pz_st, _ = pkg_model(x_he_niche, x_he_tokens, x_st)
        assert torch.equal(sz_he, pz_he), f"train z_he max diff: {(sz_he - pz_he).abs().max()}"
        assert torch.equal(sz_st, pz_st), f"train z_st max diff: {(sz_st - pz_st).abs().max()}"

    def test_cross_attn_train(self, seed, inputs):
        x_he_niche, x_he_tokens, x_st, _ = inputs
        torch.manual_seed(seed)
        script_model = ScriptCrossAttnFusion(HE_TILE_DIM, ST_DIM, SHARED_DIM, DROPOUT)
        pkg_model = PkgCrossAttnFusion(HE_TILE_DIM, ST_DIM, SHARED_DIM, DROPOUT)
        _copy_weights(script_model, pkg_model)

        script_model.train()
        pkg_model.train()
        torch.manual_seed(SEED)
        sz_he, sz_st, s_attn = script_model(x_he_niche, x_he_tokens, x_st)
        torch.manual_seed(SEED)
        pz_he, pz_st, p_attn = pkg_model(x_he_niche, x_he_tokens, x_st)
        assert torch.equal(sz_he, pz_he)
        assert torch.equal(sz_st, pz_st)
        assert torch.equal(s_attn, p_attn)


# =============================================================================
# loss parity
# =============================================================================
class TestLossParity:
    @pytest.fixture
    def z_pair(self, seed):
        torch.manual_seed(seed)
        z_he = F.normalize(torch.randn(N, SHARED_DIM), dim=-1)
        z_st = F.normalize(torch.randn(N, SHARED_DIM), dim=-1)
        return z_he, z_st

    def test_infonce(self, z_pair):
        z_he, z_st = z_pair
        s = script_infonce(z_he, z_st, tau=0.07)
        p = pkg_infonce(z_he, z_st, tau=0.07)
        assert torch.equal(s, p), f"infonce diff: {abs(s.item() - p.item())}"

    def test_aninfonce(self, z_pair, seed):
        z_he, z_st = z_pair
        log_scale = torch.randn(SHARED_DIM)
        s = script_aninfonce(z_he, z_st, log_scale, tau=0.07)
        p = pkg_aninfonce(z_he, z_st, log_scale, tau=0.07)
        assert torch.equal(s, p), f"aninfonce diff: {abs(s.item() - p.item())}"

    def test_supcon(self, z_pair, seed):
        z_he, z_st = z_pair
        mc_w = torch.rand(N, MC_DIM)
        s = script_supcon(z_he, z_st, mc_w, tau=0.07, tau_target=0.1)
        p = pkg_supcon(z_he, z_st, mc_w, tau=0.07, tau_target=0.1)
        assert torch.allclose(s, p, atol=1e-6), f"supcon diff: {abs(s.item() - p.item())}"

    def test_barlow(self, z_pair):
        z_he, z_st = z_pair
        s = script_barlow(z_he, z_st, lam=5e-3)
        p = pkg_barlow(z_he, z_st, lam=5e-3)
        assert torch.equal(s, p), f"barlow diff: {abs(s.item() - p.item())}"


# =============================================================================
# gradient parity
# =============================================================================
class TestGradientParity:
    def test_infonce_gradient(self, seed, inputs):
        x_he_niche, x_he_tokens, x_st, _ = inputs
        torch.manual_seed(seed)
        script_model = ScriptLateFusion(HE_DIM, ST_DIM, SHARED_DIM, DROPOUT)
        pkg_model = PkgLateFusion(HE_DIM, ST_DIM, SHARED_DIM, DROPOUT)
        _copy_weights(script_model, pkg_model)

        script_model.train()
        pkg_model.train()

        torch.manual_seed(SEED)
        sz_he, sz_st, _ = script_model(x_he_niche, x_he_tokens, x_st)
        torch.manual_seed(SEED)
        pz_he, pz_st, _ = pkg_model(x_he_niche, x_he_tokens, x_st)

        s_loss = script_infonce(sz_he, sz_st, tau=0.07)
        p_loss = pkg_infonce(pz_he, pz_st, tau=0.07)

        s_loss.backward()
        p_loss.backward()

        for (sn, sp), (pn, pp) in zip(script_model.named_parameters(),
                                       pkg_model.named_parameters()):
            assert sn == pn, f"param order mismatch: {sn} vs {pn}"
            if sp.grad is not None and pp.grad is not None:
                max_diff = (sp.grad - pp.grad).abs().max().item()
                assert max_diff < 1e-6, f"{sn} grad max diff: {max_diff}"
            else:
                assert sp.grad is None and pp.grad is None, f"{sn} grad None mismatch"

    def test_barlow_gradient(self, seed, inputs):
        x_he_niche, x_he_tokens, x_st, _ = inputs
        torch.manual_seed(seed)
        script_model = ScriptLateFusion(HE_DIM, ST_DIM, SHARED_DIM, DROPOUT)
        pkg_model = PkgLateFusion(HE_DIM, ST_DIM, SHARED_DIM, DROPOUT)
        _copy_weights(script_model, pkg_model)

        script_model.train()
        pkg_model.train()

        torch.manual_seed(SEED)
        sz_he, sz_st, _ = script_model(x_he_niche, x_he_tokens, x_st)
        torch.manual_seed(SEED)
        pz_he, pz_st, _ = pkg_model(x_he_niche, x_he_tokens, x_st)

        s_loss = script_barlow(sz_he, sz_st, lam=5e-3)
        p_loss = pkg_barlow(pz_he, pz_st, lam=5e-3)

        s_loss.backward()
        p_loss.backward()

        for (sn, sp), (pn, pp) in zip(script_model.named_parameters(),
                                       pkg_model.named_parameters()):
            assert sn == pn
            if sp.grad is not None and pp.grad is not None:
                max_diff = (sp.grad - pp.grad).abs().max().item()
                assert max_diff < 1e-6, f"{sn} grad max diff: {max_diff}"

    def test_aninfonce_gradient(self, seed, inputs):
        x_he_niche, x_he_tokens, x_st, _ = inputs
        torch.manual_seed(seed)
        script_model = ScriptLateFusion(HE_DIM, ST_DIM, SHARED_DIM, DROPOUT,
                                        anisotropic=True)
        pkg_model = PkgLateFusion(HE_DIM, ST_DIM, SHARED_DIM, DROPOUT,
                                  anisotropic=True)
        _copy_weights(script_model, pkg_model)

        script_model.train()
        pkg_model.train()

        torch.manual_seed(SEED)
        sz_he, sz_st, _ = script_model(x_he_niche, x_he_tokens, x_st)
        torch.manual_seed(SEED)
        pz_he, pz_st, _ = pkg_model(x_he_niche, x_he_tokens, x_st)

        s_loss = script_aninfonce(sz_he, sz_st, script_model.aniso_log_scale, tau=0.07)
        p_loss = pkg_aninfonce(pz_he, pz_st, pkg_model.aniso_log_scale, tau=0.07)

        s_loss.backward()
        p_loss.backward()

        for (sn, sp), (pn, pp) in zip(script_model.named_parameters(),
                                       pkg_model.named_parameters()):
            assert sn == pn
            if sp.grad is not None and pp.grad is not None:
                max_diff = (sp.grad - pp.grad).abs().max().item()
                assert max_diff < 1e-6, f"{sn} grad max diff: {max_diff}"

    def test_supcon_gradient(self, seed, inputs):
        x_he_niche, x_he_tokens, x_st, mc_w = inputs
        torch.manual_seed(seed)
        script_model = ScriptLateFusion(HE_DIM, ST_DIM, SHARED_DIM, DROPOUT)
        pkg_model = PkgLateFusion(HE_DIM, ST_DIM, SHARED_DIM, DROPOUT)
        _copy_weights(script_model, pkg_model)

        script_model.train()
        pkg_model.train()

        torch.manual_seed(SEED)
        sz_he, sz_st, _ = script_model(x_he_niche, x_he_tokens, x_st)
        torch.manual_seed(SEED)
        pz_he, pz_st, _ = pkg_model(x_he_niche, x_he_tokens, x_st)

        s_loss = script_supcon(sz_he, sz_st, mc_w, tau=0.07, tau_target=0.1)
        p_loss = pkg_supcon(pz_he, pz_st, mc_w, tau=0.07, tau_target=0.1)

        s_loss.backward()
        p_loss.backward()

        for (sn, sp), (pn, pp) in zip(script_model.named_parameters(),
                                       pkg_model.named_parameters()):
            assert sn == pn
            if sp.grad is not None and pp.grad is not None:
                max_diff = (sp.grad - pp.grad).abs().max().item()
                assert max_diff < 1e-6, f"{sn} grad max diff: {max_diff}"


# =============================================================================
# loss-to-run label verification
# =============================================================================
class TestLossRunLabels:
    CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "v3"

    EXPECTED = {
        "R1_v3": "infonce", "R2_v3": "supcon", "R3_v3": "barlow",
        "R4_v3": "infonce", "R5_v3": "aninfonce", "R6_v3": "infonce",
    }

    @pytest.mark.parametrize("run_id,expected_loss", list(EXPECTED.items()))
    def test_config_loss_matches(self, run_id, expected_loss):
        cfg_path = self.CONFIGS_DIR / f"{run_id}.json"
        if not cfg_path.is_file():
            pytest.skip(f"config not found: {cfg_path}")
        cfg = json.loads(cfg_path.read_text())
        assert cfg["loss"] == expected_loss, (
            f"{run_id}: config says {cfg['loss']!r}, expected {expected_loss!r}")

    @pytest.mark.parametrize("loss_name", ["infonce", "supcon", "barlow", "aninfonce"])
    def test_compute_loss_dispatches(self, loss_name):
        torch.manual_seed(SEED)
        z = F.normalize(torch.randn(4, 8), dim=-1)
        kwargs = {"tau": 0.07, "tau_target": 0.1, "lam": 5e-3}
        if loss_name == "supcon":
            kwargs["mc_w"] = torch.rand(4, 3)
        if loss_name == "aninfonce":
            model = PkgLateFusion(8, 8, 8, 0.0, anisotropic=True)
            kwargs["model"] = model
        result = pkg_compute_loss(loss_name, z, z, **kwargs)
        assert result.shape == ()
        assert torch.isfinite(result)


# =============================================================================
# classical baseline parity
# =============================================================================
class TestClassicalParity:
    @pytest.fixture
    def data(self):
        np.random.seed(SEED)
        he_tr = np.random.randn(100, 50).astype(np.float64)
        st_tr = np.random.randn(100, 30).astype(np.float64)
        he_te = np.random.randn(40, 50).astype(np.float64)
        st_te = np.random.randn(40, 30).astype(np.float64)
        return he_tr, st_tr, he_te, st_te

    def test_cca_parity(self, data):
        he_tr, st_tr, he_te, st_te = data
        np.random.seed(SEED)
        sz_he, sz_st, sk = script_cca(he_tr, st_tr, he_te, st_te, 20, reg=1e-4)
        np.random.seed(SEED)
        pz_he, pz_st, pk = pkg_cca(he_tr, st_tr, he_te, st_te, 20, reg=1e-4)
        assert sk == pk, f"k_used: script={sk}, package={pk}"
        np.testing.assert_array_equal(sz_he, pz_he)
        np.testing.assert_array_equal(sz_st, pz_st)

    def test_procrustes_parity(self, data):
        he_tr, st_tr, he_te, st_te = data
        np.random.seed(SEED)
        sz_he, sz_st, sk = script_procrustes(he_tr, st_tr, he_te, st_te, 20)
        np.random.seed(SEED)
        pz_he, pz_st, pk = pkg_procrustes(he_tr, st_tr, he_te, st_te, 20)
        assert sk == pk
        np.testing.assert_array_equal(sz_he, pz_he)
        np.testing.assert_array_equal(sz_st, pz_st)

    def test_unaligned_parity(self, data):
        he_tr, st_tr, he_te, st_te = data
        np.random.seed(SEED)
        sz_he, sz_st, sk = script_unaligned(he_tr, st_tr, he_te, st_te, 20)
        np.random.seed(SEED)
        pz_he, pz_st, pk = pkg_unaligned(he_tr, st_tr, he_te, st_te, 20)
        assert sk == pk
        np.testing.assert_array_equal(sz_he, pz_he)
        np.testing.assert_array_equal(sz_st, pz_st)

    def test_l2_normalize_parity(self):
        x = np.random.randn(10, 5).astype(np.float32)
        np.testing.assert_array_equal(script_l2_normalize(x), pkg_l2_normalize(x))


# =============================================================================
# checkpoint loading (skipped if artifacts unavailable)
# =============================================================================
class TestCheckpointLoading:
    RUNS_DIR = Path(__file__).resolve().parent.parent / "runs" / "tnbc-92_v3"

    def test_r1_checkpoint_loads(self):
        ckpt = self.RUNS_DIR / "R1_v3" / "checkpoint.pt"
        if not ckpt.is_file():
            pytest.skip(f"SKIPPED (unverified): R1_v3 checkpoint not found at {ckpt}")
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "v3" / "R1_v3.json"
        cfg = json.loads(cfg_path.read_text())
        model = pkg_build_model(
            cfg.get("fusion", "late"),
            he_dim=HE_DIM, st_dim=ST_DIM, shared_dim=cfg.get("shared_dim", 512),
            dropout=cfg.get("dropout", 0.3),
            anisotropic=cfg.get("anisotropic", False),
        )
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
