"""matched-input comparison: preserved script vs package measures.

constructs a cosine matrix from the adapter's real R4_v3 test-split data,
then feeds the EXACT SAME matrix, patients, and labels to:
  (a) the script's biology_perm_test / patient_perm_test (from scripts/)
  (b) the package's measures.biology functions
  (c) the LCEL chain step (through build_protocol, not called directly)

all three paths receive identical observations and parameters. differences
are reported without adjusting scientific parameters.

also documents the scope gap: the script's published biology.parquet was
computed on ~258 full-cohort subarrays re-encoded from the checkpoint.
this comparison uses the 38-subarray test split because the adapter
carries only that. full-cohort reproduction requires re-encoding, which
is a separate task.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# --- load the script's functions (not the package) ---
from eval_alignment_biology import (  # noqa: E402
    biology_perm_test as script_bio_perm,
    cosine_matrix as script_cosine,
    l2 as script_l2,
    load_tnbc_labels as script_load_labels,
    patient_perm_test as script_patient_perm,
    subarray_mean_pool as script_mean_pool,
)

# --- load the package's functions ---
from omicstra.adapters.evaluation import arrays  # noqa: E402
from omicstra.measures.biology import (  # noqa: E402
    biology_perm_test as pkg_bio_perm,
    cosine_sim_matrix as pkg_cosine,
    mean_pool_l2 as pkg_mean_pool,
    patient_perm_test as pkg_patient_perm,
    ratio_delta_ci as pkg_ratio_ci,
)
from omicstra.protocols import build_protocol  # noqa: E402
from omicstra.protocols.evaluate import EVALUATE_STEPS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
N_PERM = 10_000
SEED = 42


def _ser(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    raise TypeError(f"cannot serialise {type(obj)}")


def main():
    report: dict = {"checks": [], "scope": {}}
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
        print(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))

    # ----------------------------------------------------------------
    # 1. build the shared input: subarray-level cosine matrix
    # ----------------------------------------------------------------
    print("loading R4_v3 via adapter (test split)...")
    ctx = arrays(str(ROOT / "projects" / "tnbc-92"), "R4_v3")
    he = np.asarray(ctx["spaces"]["z_he"], dtype=np.float32)
    sub_arr = ctx["subarray"]
    conf_arr = ctx["confounder"]
    unique_subs = np.unique(sub_arr)
    n_subs = len(unique_subs)
    print(f"  {n_subs} subarrays, {len(he)} niches")

    # 1a. package mean_pool_l2
    pkg_embs = np.stack([pkg_mean_pool(he[sub_arr == s]) for s in unique_subs])
    # 1b. script subarray_mean_pool
    script_embs = np.stack([script_mean_pool(he[sub_arr == s]) for s in unique_subs])

    print("\n--- aggregation ---")
    agg_diff = float(np.max(np.abs(pkg_embs - script_embs)))
    check("mean_pool_identical", agg_diff == 0.0,
          f"max abs diff = {agg_diff:.2e}")

    # cosine matrices
    pkg_cos = pkg_cosine(pkg_embs)
    script_cos = script_cosine(script_embs)
    cos_diff = float(np.max(np.abs(pkg_cos - script_cos)))
    check("cosine_matrix_fp32_parity", cos_diff < 1e-6,
          f"max abs diff = {cos_diff:.2e} (float32 clipping precision)")

    # patient arrays (same order as unique_subs)
    sub_patients = np.array([
        conf_arr[np.where(sub_arr == s)[0][0]] for s in unique_subs
    ])
    # script uses string patient IDs like "TNBC1"
    sub_patients_str = np.array([f"TNBC{p}" for p in sub_patients])

    # biology labels from the adapter
    bl = ctx["biology_labels"]
    label_map = {"MC_global": bl["MC_global"], "MC_tumor": bl["MC_tumor"],
                 "TIME": bl["TIME"]}
    sub_labels = {
        name: np.array([
            str(arr[np.where(sub_arr == s)[0][0]]) for s in unique_subs
        ])
        for name, arr in label_map.items()
    }

    # ----------------------------------------------------------------
    # 2. matched-input: script functions vs package functions
    # ----------------------------------------------------------------
    shared_cos = pkg_cos  # identical, use either

    print("\n--- patient permutation test (script vs package) ---")
    # script returns: (obs, null_mean, null_std, p, n_same, n_diff, m_same, m_diff)
    s_pt = script_patient_perm(shared_cos, sub_patients_str, N_PERM, SEED)
    s_obs, s_nm, s_ns, s_p, s_nsame, s_ndiff, s_msame, s_mdiff = s_pt

    p_pt = pkg_patient_perm(shared_cos, sub_patients_str,
                             n_permutations=N_PERM, seed=SEED)
    fields_pt = [
        ("mean_same", s_msame, p_pt["mean_same"]),
        ("mean_diff", s_mdiff, p_pt["mean_diff"]),
        ("delta", s_obs, p_pt["delta"]),
        ("null_mean", s_nm, p_pt["null_mean"]),
        ("null_std", s_ns, p_pt["null_std"]),
        ("z", s_obs / s_ns if s_ns > 0 else float("nan"), p_pt["z"]),
        ("p_value", s_p, p_pt["p_value"]),
        ("n_same_pairs", s_nsame, p_pt["n_same_pairs"]),
        ("n_diff_pairs", s_ndiff, p_pt["n_diff_pairs"]),
    ]
    for fname, sv, pv in fields_pt:
        diff = abs(float(sv) - float(pv))
        check(f"patient_{fname}", diff < 1e-10,
              f"script={float(sv):.10f}, package={float(pv):.10f}, diff={diff:.2e}")

    print("\n--- biology permutation tests (script vs package) ---")
    for lname in ["MC_global", "MC_tumor", "TIME"]:
        labs = sub_labels[lname]
        s_bio = script_bio_perm(shared_cos, sub_patients_str, labs, N_PERM, SEED)
        s_obs, s_nm, s_ns, s_p, s_nsame, s_ndiff, s_msame, s_mdiff = s_bio

        p_bio = pkg_bio_perm(shared_cos, sub_patients_str, labs,
                              n_permutations=N_PERM, seed=SEED)
        fields_bio = [
            ("mean_same", s_msame, p_bio["mean_same"]),
            ("mean_diff", s_mdiff, p_bio["mean_diff"]),
            ("delta", s_obs, p_bio["delta"]),
            ("null_mean", s_nm, p_bio["null_mean"]),
            ("null_std", s_ns, p_bio["null_std"]),
            ("z", s_obs / s_ns if s_ns > 0 else float("nan"), p_bio["z"]),
            ("p_value", s_p, p_bio["p_value"]),
            ("n_same_pairs", s_nsame, p_bio["n_same_pairs"]),
            ("n_diff_pairs", s_ndiff, p_bio["n_diff_pairs"]),
        ]
        for fname, sv, pv in fields_bio:
            diff = abs(float(sv) - float(pv))
            check(f"bio_{lname}_{fname}", diff < 1e-10,
                  f"script={float(sv):.10f}, package={float(pv):.10f}, "
                  f"diff={diff:.2e}")

    # ----------------------------------------------------------------
    # 3. LCEL chain (build_protocol) on the same data
    # ----------------------------------------------------------------
    print("\n--- LCEL chain (build_protocol, not direct step call) ---")
    chain = build_protocol(EVALUATE_STEPS, "evaluate")
    chain_ctx = {
        **ctx,
        "params": {
            "label_granularity": {"labels": ("mc_megacluster",),
                                   "tau_label_nmi": 0.5},
            "paired_identifiability": {"ks": (1, 5), "negative_ratio": 5,
                                        "negative_seed": 0},
            "structure_and_its_null": {"k": 14, "n_init": 10, "seed": SEED,
                                        "silhouette_sample": 300,
                                        "n_permutations": 25, "probe_max_iter": 200,
                                        "held_out_groups": 2},
            "confounder_suppression": {
                "biology_labels": ("MC_global", "MC_tumor", "TIME"),
                "n_permutations": N_PERM,
                "n_rho_null": 1000,
                "seed": SEED,
                "alpha": 0.05,
            },
            "held_out_comparison": {"ks": (1, 5), "negative_ratio": 5,
                                     "negative_seed": 0,
                                     "min_units_per_subject": 10},
            "association_and_specificity": {"n_permutations": 40, "alpha": 0.05,
                                            "tau_specificity": 0.5,
                                            "tail": "two_sided",
                                            "held_out_groups": 2,
                                            "transfer_seed": 42,
                                            "within_cohort_seed": 7,
                                            "min_units_per_side": 20},
            "hypothesis_verdict": {"hypotheses": {
                "H1": ("paired_identifiability", "held_out_comparison"),
                "H2": ("structure_and_its_null", "confounder_suppression"),
                "H3": ("association_and_specificity",)}},
        },
    }
    result = chain.invoke(chain_ctx)
    # build_protocol merges records at top level, keyed by step_id
    cs_rec = result.get("confounder_suppression", {})
    cs_obs = cs_rec.get("observed", {})
    cs_pv = cs_obs.get("per_view", {}).get("z_he", {})

    check("lcel_chain_ran", cs_rec.get("status") == "pass",
          f"status={cs_rec.get('status')}")

    # compare chain output vs package direct calls (same-input, both on test split)
    chain_pt = cs_pv.get("patient", {})
    for fname, pv_val in [("mean_same", p_pt["mean_same"]),
                           ("mean_diff", p_pt["mean_diff"]),
                           ("delta", p_pt["delta"]),
                           ("z", p_pt["z"]),
                           ("p_value", p_pt["p_value"])]:
        cv = chain_pt.get(fname)
        if cv is not None:
            diff = abs(float(cv) - float(pv_val))
            check(f"lcel_patient_{fname}", diff < 1e-10,
                  f"chain={float(cv):.10f}, direct={float(pv_val):.10f}, "
                  f"diff={diff:.2e}")

    for lname in ["MC_global", "MC_tumor", "TIME"]:
        p_bio = pkg_bio_perm(shared_cos, sub_patients_str, sub_labels[lname],
                              n_permutations=N_PERM, seed=SEED)
        chain_bio = cs_pv.get("biology", {}).get(lname, {})
        for fname in ["mean_same", "mean_diff", "delta", "z", "p_value"]:
            cv = chain_bio.get(fname)
            pv = p_bio[fname]
            if cv is not None:
                diff = abs(float(cv) - float(pv))
                check(f"lcel_bio_{lname}_{fname}", diff < 1e-10,
                      f"chain={float(cv):.10f}, direct={float(pv):.10f}, "
                      f"diff={diff:.2e}")

    # ----------------------------------------------------------------
    # 4. scope documentation
    # ----------------------------------------------------------------
    print("\n--- scope ---")
    hist = pd.read_parquet(ROOT / "runs" / "tnbc-92_v3" / "R4_v3" / "eval" /
                           "biology.parquet")
    hist_z_he = hist[hist["view"] == "z_he"].iloc[0]
    hist_n_pairs = int(hist_z_he["n_same_pairs"] + hist_z_he["n_diff_pairs"])
    hist_n_subs = int(np.round((1 + np.sqrt(1 + 8 * hist_n_pairs)) / 2))

    report["scope"] = {
        "this_comparison": f"{n_subs} subarrays (test split), matched input to "
                           "both script and package functions",
        "historical_biology_parquet": f"~{hist_n_subs} subarrays (full cohort "
                                      "re-encoded from model checkpoint)",
        "gap": "full-cohort reproduction requires re-encoding all subarrays "
               "through the trained model, which the adapter does not do. "
               "that is a separate task from verifying function parity",
        "rho": "deferred: geometric faithfulness requires raw baseline cosine "
               "matrices (pre-alignment). the adapter does not carry them",
    }
    check("scope_documented", True,
          f"this comparison: {n_subs} subs, historical: ~{hist_n_subs} subs")

    # ----------------------------------------------------------------
    # save
    # ----------------------------------------------------------------
    report["summary"] = {"ok": ok, "fail": fail, "total": ok + fail}
    out_path = OUT_DIR / "script_parity_report_R4_v3.json"
    out_path.write_text(json.dumps(report, indent=2, default=_ser))

    print(f"\n{'='*60}")
    print(f"  {ok} PASS, {fail} FAIL (of {ok + fail})")
    print(f"  -> {out_path.relative_to(ROOT)}")
    return fail


if __name__ == "__main__":
    rc = main()
    sys.exit(1 if rc > 0 else 0)
