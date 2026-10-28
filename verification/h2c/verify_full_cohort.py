"""full-cohort verification: package measures vs historical artifacts.

encodes ALL ~260 subarrays from the model checkpoint (torch forward pass),
feeds the same cosine matrices to the SCRIPT's functions and the PACKAGE's
measures, and compares both against the historical biology.parquet and
biology_rho.parquet field by field.

this is the verification that the test-split comparison cannot provide:
it operates on the same population as the published artifacts.

requires: torch, scripts/align.py (Config, build_model, load_subarray,
discover_subarrays), scripts/eval_alignment_biology.py (rho_with_null,
biology_perm_test, patient_perm_test, load_tnbc_labels, subarray_mean_pool,
cosine_matrix).

does NOT modify any scientific parameters or historical artifacts.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RUN_ID = "R4_v3"
RUNS_DIR = ROOT / "runs" / "tnbc-92_v3"
NICHES_DIR = ROOT / "data" / "embeddings" / "niches_v3"
HIST_BIO = RUNS_DIR / RUN_ID / "eval" / "biology.parquet"
HIST_RHO = RUNS_DIR / RUN_ID / "eval" / "biology_rho.parquet"
OUT_DIR = Path(__file__).resolve().parent

from align import Config, build_model, discover_subarrays, load_subarray  # noqa: E402
import eval_alignment_biology as _eab  # noqa: E402
_eab.RUNS_ROOT = RUNS_DIR
_eab.NICHES_DIR = NICHES_DIR
from eval_alignment_biology import (  # noqa: E402
    biology_perm_test as script_bio_perm,
    cosine_matrix as script_cosine,
    encode_trained as script_encode,
    load_tnbc_labels as script_load_labels,
    patient_perm_test as script_patient_perm,
    reconstruct_trained_model,
    rho_with_null as script_rho,
    subarray_mean_pool as script_mean_pool,
)

from omicstra.measures.biology import (  # noqa: E402
    biology_perm_test as pkg_bio_perm,
    cosine_sim_matrix as pkg_cosine,
    geometric_faithfulness as pkg_geo_faith,
    mean_pool_l2 as pkg_mean_pool,
    patient_perm_test as pkg_patient_perm,
)

N_PERM = 10_000
N_RHO = 1000
SEED = 42
ST_FEATURES = ("novae_niche", "gpath2vec_niche")


def _ser(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"cannot serialise {type(obj)}")


def main():
    report: dict = {"run_id": RUN_ID, "checks": [], "scope": {}}
    ok, fail = 0, 0

    def check(name, passed, detail=""):
        nonlocal ok, fail
        status = "PASS" if passed else "FAIL"
        if passed:
            ok += 1
        else:
            fail += 1
        report["checks"].append({"name": name, "status": status, "detail": detail})
        mark = "OK" if passed else "!!"
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))

    # ----------------------------------------------------------------
    # 1. encode full cohort from checkpoint
    # ----------------------------------------------------------------
    t0 = time.time()
    print(f"loading {len(list(NICHES_DIR.glob('*.parquet')))} niche parquets...")
    paths = discover_subarrays(NICHES_DIR)
    subs_raw = [load_subarray(p, ST_FEATURES) for p in paths]
    subs = [s for s in subs_raw
            if s["patient_id"] is not None and s["archetype"] is not None]
    print(f"  {len(subs)} subarrays after dropping missing patient_id/archetype")

    print("reconstructing model from checkpoint...")
    import torch
    model, _ = reconstruct_trained_model(RUN_ID, subs[0])

    print("encoding + mean-pooling...")
    keys, mean_he, mean_st, mean_raw_he, mean_raw_st = [], [], [], [], []
    for sub in subs:
        z_he, z_st = script_encode(model, sub)
        keys.append(sub["subarray"])
        mean_he.append(script_mean_pool(z_he))
        mean_st.append(script_mean_pool(z_st))
        mean_raw_he.append(script_mean_pool(sub["x_he_niche"]))
        mean_raw_st.append(script_mean_pool(sub["x_st"]))

    Z_he = np.stack(mean_he)
    Z_st = np.stack(mean_st)
    Z_mean = np.linalg.norm(0.5 * (Z_he + Z_st), axis=-1, keepdims=True)
    Z_mean = 0.5 * (Z_he + Z_st) / np.clip(Z_mean, 1e-12, None)
    R_he = np.stack(mean_raw_he)
    R_st = np.stack(mean_raw_st)
    n_subs = len(keys)
    t_encode = time.time() - t0
    print(f"  {n_subs} subarrays encoded in {t_encode:.1f}s")

    check("n_subarrays", n_subs > 250,
          f"{n_subs} subarrays (expected ~258-261)")

    # cosine matrices
    cos_z_he = script_cosine(Z_he)
    cos_z_st = script_cosine(Z_st)
    cos_z_mean = script_cosine(Z_mean)
    cos_raw_he = script_cosine(R_he)
    cos_raw_st = script_cosine(R_st)

    # labels
    patients, labels_by_key = script_load_labels(keys)
    n_patients = len(set(patients))
    print(f"  {n_patients} patients, labels: {list(labels_by_key.keys())}")

    # ----------------------------------------------------------------
    # 2. rho: package geometric_faithfulness vs script rho_with_null
    # ----------------------------------------------------------------
    print("\n--- rho: script vs package vs historical ---")
    hist_rho = pd.read_parquet(HIST_RHO)

    for emb_name, emb_sim in [("z_he", cos_z_he), ("z_st", cos_z_st),
                                ("z_mean", cos_z_mean)]:
        for raw_name, raw_sim in [("raw_he", cos_raw_he), ("raw_st", cos_raw_st)]:
            # script
            s_rho = script_rho(emb_sim, raw_sim, N_RHO, SEED)
            # package
            p_rho = pkg_geo_faith(emb_sim, raw_sim,
                                   n_permutations=N_RHO, seed=SEED)

            tag = f"rho_{emb_name}_vs_{raw_name}"
            for field in ("rho", "pearson", "z", "p_value", "n_pairs"):
                sv = s_rho[field]
                pv = p_rho[field]
                diff = abs(float(sv) - float(pv))
                check(f"{tag}_{field}_parity", diff < 1e-10,
                      f"script={float(sv):.10f}, package={float(pv):.10f}, "
                      f"diff={diff:.2e}")

            # vs historical
            hist_row = hist_rho[(hist_rho["emb_view"] == emb_name) &
                                (hist_rho["raw_baseline"] == raw_name)]
            if len(hist_row) == 1:
                hr = hist_row.iloc[0]
                for field in ("rho", "pearson", "z", "p_value", "n_pairs"):
                    hv = hr[field]
                    pv = p_rho[field]
                    diff = abs(float(hv) - float(pv))
                    check(f"{tag}_{field}_vs_historical", diff < 1e-6,
                          f"historical={float(hv):.10f}, package={float(pv):.10f}, "
                          f"diff={diff:.2e}")

    # ----------------------------------------------------------------
    # 3. permutation tests: package vs historical biology.parquet
    # ----------------------------------------------------------------
    print("\n--- permutation tests: package vs historical ---")
    hist_bio = pd.read_parquet(HIST_BIO)

    # the historical file used n_permutations=100000, but that takes ~30min.
    # we use N_PERM (10000) for script-vs-package parity, then compare
    # against historical for the SEED-INDEPENDENT fields only (mean_same,
    # mean_diff, n_same_pairs, n_diff_pairs) which depend on cosine matrices
    # and labels, not on permutation count.
    for view_name, view_sim in [("z_he", cos_z_he), ("z_st", cos_z_st),
                                 ("z_mean", cos_z_mean),
                                 ("raw_he", cos_raw_he), ("raw_st", cos_raw_st)]:
        # biology tests
        for key, labs in labels_by_key.items():
            s_bio = script_bio_perm(view_sim, patients, labs, N_PERM, SEED)
            s_obs, s_nm, s_ns, s_p, s_nsame, s_ndiff, s_msame, s_mdiff = s_bio

            p_bio = pkg_bio_perm(view_sim, patients, labs,
                                  n_permutations=N_PERM, seed=SEED)

            tag = f"bio_{view_name}_{key}"
            fields = [
                ("mean_same", s_msame, p_bio["mean_same"]),
                ("mean_diff", s_mdiff, p_bio["mean_diff"]),
                ("delta", s_obs, p_bio["delta"]),
                ("z", s_obs / s_ns if s_ns > 0 else float("nan"), p_bio["z"]),
                ("p_value", s_p, p_bio["p_value"]),
                ("n_same_pairs", s_nsame, p_bio["n_same_pairs"]),
                ("n_diff_pairs", s_ndiff, p_bio["n_diff_pairs"]),
            ]
            for fname, sv, pv in fields:
                diff = abs(float(sv) - float(pv))
                check(f"{tag}_{fname}_parity", diff < 1e-10,
                      f"script={float(sv):.10f}, package={float(pv):.10f}, "
                      f"diff={diff:.2e}")

            # vs historical (seed-independent fields)
            hist_row = hist_bio[(hist_bio["view"] == view_name) &
                                (hist_bio["label"] == key) &
                                (hist_bio["test_type"] == "biology")]
            if len(hist_row) == 1:
                hr = hist_row.iloc[0]
                for fname in ("mean_same", "mean_diff",
                              "n_same_pairs", "n_diff_pairs"):
                    hv = hr[fname]
                    pv = p_bio[fname] if fname != "mean_same" else s_msame
                    diff = abs(float(hv) - float(pv))
                    check(f"{tag}_{fname}_vs_historical", diff < 1e-6,
                          f"historical={float(hv):.10f}, this_run={float(pv):.10f}, "
                          f"diff={diff:.2e}")

        # patient test
        s_pt = script_patient_perm(view_sim, patients, N_PERM, SEED)
        s_obs, s_nm, s_ns, s_p, s_nsame, s_ndiff, s_msame, s_mdiff = s_pt

        p_pt = pkg_patient_perm(view_sim, patients,
                                 n_permutations=N_PERM, seed=SEED)

        tag = f"patient_{view_name}"
        fields = [
            ("mean_same", s_msame, p_pt["mean_same"]),
            ("mean_diff", s_mdiff, p_pt["mean_diff"]),
            ("delta", s_obs, p_pt["delta"]),
            ("z", s_obs / s_ns if s_ns > 0 else float("nan"), p_pt["z"]),
            ("p_value", s_p, p_pt["p_value"]),
            ("n_same_pairs", s_nsame, p_pt["n_same_pairs"]),
            ("n_diff_pairs", s_ndiff, p_pt["n_diff_pairs"]),
        ]
        for fname, sv, pv in fields:
            diff = abs(float(sv) - float(pv))
            check(f"{tag}_{fname}_parity", diff < 1e-10,
                  f"script={float(sv):.10f}, package={float(pv):.10f}, "
                  f"diff={diff:.2e}")

        # vs historical
        hist_row = hist_bio[(hist_bio["view"] == view_name) &
                            (hist_bio["test_type"] == "patient")]
        if len(hist_row) == 1:
            hr = hist_row.iloc[0]
            for fname in ("mean_same", "mean_diff",
                          "n_same_pairs", "n_diff_pairs"):
                hv = hr[fname]
                pv = p_pt[fname] if fname != "mean_same" else s_msame
                diff = abs(float(hv) - float(pv))
                check(f"{tag}_{fname}_vs_historical", diff < 1e-6,
                      f"historical={float(hv):.10f}, this_run={float(pv):.10f}, "
                      f"diff={diff:.2e}")

    # ----------------------------------------------------------------
    # 4. scope
    # ----------------------------------------------------------------
    # use a patient test row (no NA filtering) for the subarray count
    hist_pt_row = hist_bio[hist_bio["test_type"] == "patient"].iloc[0]
    hist_n_pairs = int(hist_pt_row["n_same_pairs"] + hist_pt_row["n_diff_pairs"])
    hist_n_subs = int(np.round((1 + np.sqrt(1 + 8 * hist_n_pairs)) / 2))
    n_pairs = int(Z_he.shape[0] * (Z_he.shape[0] - 1) / 2)

    report["scope"] = {
        "this_run": f"{n_subs} subarrays, {n_patients} patients, {n_pairs} pairs",
        "historical": f"~{hist_n_subs} subarrays",
        "n_perm_this_run": N_PERM,
        "n_perm_historical": 100_000,
        "n_rho_this_run": N_RHO,
        "n_rho_historical": 1000,
        "seed": SEED,
        "note": "seed-dependent fields (z, p_value, null_mean, null_std) differ "
                "from historical when n_permutations differs. "
                "seed-independent fields (mean_same, mean_diff, n_same_pairs, "
                "n_diff_pairs) should match exactly if the same subarrays are encoded",
        "encode_time_s": round(t_encode, 1),
    }

    check("scope_subarrays_match_historical",
          n_subs == hist_n_subs,
          f"this run: {n_subs}, historical: {hist_n_subs}")

    # ----------------------------------------------------------------
    # save
    # ----------------------------------------------------------------
    report["summary"] = {"ok": ok, "fail": fail, "total": ok + fail}
    out_path = OUT_DIR / "full_cohort_report_R4_v3.json"
    out_path.write_text(json.dumps(report, indent=2, default=_ser))

    print(f"\n{'='*60}")
    print(f"  {ok} PASS, {fail} FAIL (of {ok + fail})")
    print(f"  -> {out_path.relative_to(ROOT)}")
    return fail


if __name__ == "__main__":
    rc = main()
    sys.exit(1 if rc > 0 else 0)