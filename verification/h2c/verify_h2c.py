"""verify H2-C chain step against historical script output.

runs the confounder_suppression step on real R4_v3 data and compares
against the published biology.parquet. writes a verification report
and the chain output to verification/h2c/.

the chain runs on the TEST SPLIT (38 subarrays from embeddings_test.parquet).
the historical script re-encodes the FULL COHORT (260 subarrays) from the
model checkpoint. this is a scope difference, not a parameter difference -
the numbers are expected to differ.

verifications performed:
  1. scope documentation: chain vs script populations
  2. aggregation parity: mean_pool_l2 on the chain's own data matches
     a direct call to the measures function
  3. cosine matrix parity: chain's matrix matches direct computation
  4. permutation parity: chain's permutation test matches direct call
  5. ratio and CI parity: chain's delta-method CI matches direct call
  6. FDR: BH-FDR runs (statsmodels is a hard import, no silent fallback)
  7. label propagation: biology labels match the clinical file's values

does NOT modify any scientific parameters.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from omicstra.adapters.evaluation import arrays, biology_labels
from omicstra.measures.biology import (
    biology_perm_test,
    cosine_sim_matrix,
    mean_pool_l2,
    patient_perm_test,
    ratio_delta_ci,
)
from omicstra.protocols.evaluate import confounder_suppression

RUN_ID = "R4_v3"
PROJECT_DIR = ROOT / "projects" / "tnbc-92"
HISTORICAL = ROOT / "runs" / "tnbc-92_v3" / RUN_ID / "eval" / "biology.parquet"
OUT_DIR = Path(__file__).resolve().parent


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
    report: dict = {"run_id": RUN_ID, "checks": []}
    ok, fail = 0, 0

    def check(name, passed, detail=""):
        nonlocal ok, fail
        status = "PASS" if passed else "FAIL"
        if passed:
            ok += 1
        else:
            fail += 1
        report["checks"].append({"name": name, "status": status, "detail": detail})
        mark = "OK" if passed else "FAIL"
        print(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))

    # --- load data ---
    print(f"loading {RUN_ID} via adapter...")
    ctx = arrays(str(PROJECT_DIR), RUN_ID)
    n_niches = ctx["n_units_labelled"]
    n_subs = ctx["sample_count"]
    print(f"  {n_niches} niches, {n_subs} subarrays (test split)")

    bl = ctx["biology_labels"]
    bl_report = ctx["biology_labels_report"]
    print(f"  biology labels: {list(bl.keys())}")
    print(f"  patients in labels file: {bl_report['n_labels_file']}, "
          f"patients in test split: {bl_report['n_patients']}")

    # --- check 1: label propagation from clinical file ---
    print("\n1. label propagation")
    clin = pd.read_csv(
        PROJECT_DIR / "../../data/embeddings/biological_signals/clinical_labels.tsv",
        sep="\t")
    col_map = {"MC_global": "molecular_subtype_global",
               "MC_tumor": "molecular_subtype_tumor",
               "TIME": "TIME_expression_global"}
    sub_arr = ctx["subarray"]
    conf_arr = ctx["confounder"]

    for alias, src_col in col_map.items():
        label_arr = bl[alias]
        mismatches = 0
        for s in np.unique(sub_arr):
            mask = sub_arr == s
            pat = conf_arr[mask][0]
            expected = str(clin.loc[clin["patient_id"] == pat, src_col].iloc[0])
            actual_vals = np.unique(label_arr[mask])
            if len(actual_vals) != 1 or actual_vals[0] != expected:
                mismatches += 1
        check(f"label_{alias}_propagation",
              mismatches == 0,
              f"{mismatches} mismatches across {len(np.unique(sub_arr))} subarrays")

    # intra-subarray consistency
    for alias in col_map:
        inconsistent = 0
        for s in np.unique(sub_arr):
            vals = np.unique(bl[alias][sub_arr == s])
            if len(vals) > 1:
                inconsistent += 1
        check(f"label_{alias}_intra_subarray_consistent",
              inconsistent == 0,
              f"{inconsistent} subarrays have mixed labels")

    # --- check 2: run the chain step ---
    print("\n2. chain step execution")
    params = {
        "biology_labels": ("MC_global", "MC_tumor", "TIME"),
        "n_permutations": 10_000,
        "n_rho_null": 1000,
        "seed": 42,
        "alpha": 0.05,
    }
    step_ctx = {**ctx, "params": params}
    rec, updates = confounder_suppression(step_ctx)

    check("step_status_pass", rec.status == "pass",
          f"status={rec.status}: {rec.result[:100]}")
    check("step_has_updates", "confounder_suppression" in updates)

    cs = updates.get("confounder_suppression", {})
    check("step_n_subarrays", cs.get("n_subarrays") == n_subs,
          f"chain={cs.get('n_subarrays')}, expected={n_subs}")
    check("step_n_patients", cs.get("n_patients") == bl_report["n_patients"],
          f"chain={cs.get('n_patients')}, expected={bl_report['n_patients']}")

    # --- check 3: aggregation parity ---
    print("\n3. aggregation parity (mean_pool_l2)")
    he = np.asarray(ctx["spaces"]["z_he"], dtype=np.float32)
    unique_subs = np.unique(sub_arr)
    direct_embs = np.stack([mean_pool_l2(he[sub_arr == s]) for s in unique_subs])
    sub_patients = np.array([conf_arr[np.where(sub_arr == s)[0][0]]
                             for s in unique_subs])

    direct_cos = cosine_sim_matrix(direct_embs)

    # the chain computes the same aggregation internally - verify via the
    # patient test which depends on the cosine matrix
    direct_pt = patient_perm_test(direct_cos, sub_patients,
                                   n_permutations=10_000, seed=42)
    chain_pt = cs["per_view"]["z_he"]["patient"]

    for field in ["mean_same", "mean_diff", "delta", "z", "p_value"]:
        d_val = direct_pt[field]
        c_val = chain_pt[field]
        diff = abs(float(d_val) - float(c_val))
        check(f"patient_{field}_parity", diff < 1e-10,
              f"direct={d_val:.10f}, chain={c_val:.10f}, diff={diff:.2e}")

    # --- check 4: biology permutation parity ---
    print("\n4. biology permutation parity")
    for alias in ["MC_global", "MC_tumor", "TIME"]:
        sub_labels = np.array([
            str(bl[alias][np.where(sub_arr == s)[0][0]]) for s in unique_subs
        ])
        direct_bio = biology_perm_test(direct_cos, sub_patients, sub_labels,
                                        n_permutations=10_000, seed=42)
        chain_bio = cs["per_view"]["z_he"]["biology"][alias]
        for field in ["mean_same", "mean_diff", "delta", "z", "p_value"]:
            d_val = direct_bio[field]
            c_val = chain_bio[field]
            diff = abs(float(d_val) - float(c_val))
            check(f"bio_{alias}_{field}_parity", diff < 1e-10,
                  f"direct={d_val:.10f}, chain={c_val:.10f}, diff={diff:.2e}")

    # --- check 5: ratio and CI parity ---
    print("\n5. ratio and CI parity")
    for alias in ["MC_global", "MC_tumor", "TIME"]:
        bio_z = cs["per_view"]["z_he"]["biology"][alias]["z"]
        pt_z = cs["per_view"]["z_he"]["patient"]["z"]
        direct_ci = ratio_delta_ci(bio_z, pt_z)
        chain_ci = cs["per_view"]["z_he"]["ratio"][alias]
        for field in ["ratio", "ratio_se", "ratio_ci_low", "ratio_ci_high"]:
            d_val = direct_ci[field]
            c_val = chain_ci[field]
            diff = abs(float(d_val) - float(c_val))
            check(f"ratio_{alias}_{field}_parity", diff < 1e-10,
                  f"direct={d_val:.10f}, chain={c_val:.10f}, diff={diff:.2e}")

    # --- check 6: FDR was applied ---
    print("\n6. BH-FDR applied")
    for alias in ["MC_global", "MC_tumor", "TIME"]:
        chain_bio = cs["per_view"]["z_he"]["biology"][alias]
        has_q = "q_bh" in chain_bio
        check(f"fdr_{alias}_present", has_q,
              f"q_bh={'present' if has_q else 'MISSING'}")

    # --- check 7: rho recorded as not_implemented ---
    print("\n7. geometric faithfulness status")
    rho = cs["per_view"]["z_he"]["rho"]
    check("rho_status_explicit",
          rho.get("status") == "not_implemented",
          f"status={rho.get('status')}")

    # --- check 8: G3 guard wording ---
    print("\n8. G3 wording")
    guards = rec.observed.get("guards", [])
    g3 = [g for g in guards if g.get("guard") == "G3_bio_vs_confounder"]
    check("g3_guard_present", len(g3) == 1)
    decision_text = rec.decision
    check("g3_no_amplification_wording",
          "amplif" not in decision_text.lower(),
          f"decision text: {decision_text[:80]}")
    check("g3_correct_wording",
          "biology z is lower" in decision_text or "ratio" in decision_text,
          f"decision: {decision_text[:80]}")

    # --- check 9: scope documentation ---
    print("\n9. scope comparison with historical")
    hist = pd.read_parquet(HISTORICAL)
    hist_z_he = hist[hist["view"] == "z_he"]
    hist_n_pairs = hist_z_he.iloc[0]["n_same_pairs"] + hist_z_he.iloc[0]["n_diff_pairs"]
    chain_n_pairs = (chain_pt["n_same_pairs"] + chain_pt["n_diff_pairs"])
    hist_n_subs = int(np.round(
        (1 + np.sqrt(1 + 8 * hist_n_pairs)) / 2))
    check("scope_documented",
          hist_n_subs != n_subs,
          f"historical: ~{hist_n_subs} subarrays (full cohort), "
          f"chain: {n_subs} subarrays (test split). "
          "different populations, numbers expected to differ")

    # z convention
    z_conv = rec.observed.get("z_convention", "")
    check("z_convention_historical",
          "historical" in z_conv,
          f"z_convention: {z_conv[:60]}")

    # --- save outputs ---
    out_path = OUT_DIR / "chain_output_R4_v3.json"
    chain_data = {"record": rec.model_dump(), "updates": updates}
    out_path.write_text(json.dumps(chain_data, indent=2, default=_ser))

    report["summary"] = {
        "ok": ok, "fail": fail, "total": ok + fail,
        "scope_note": (
            f"the chain evaluates the TEST SPLIT ({n_subs} subarrays, "
            f"{bl_report['n_patients']} patients). the historical script "
            f"re-encodes the FULL COHORT (~{hist_n_subs} subarrays). the "
            "numerical operations (mean_pool, cosine, permutation, ratio CI) "
            "are verified to be identical on the same data. the population "
            "difference is documented, not a discrepancy"),
    }

    report_path = OUT_DIR / "verification_report_R4_v3.json"
    report_path.write_text(json.dumps(report, indent=2, default=_ser))
    print(f"\n{'='*60}")
    print(f"  {ok} PASS, {fail} FAIL (of {ok + fail})")
    print(f"  -> {out_path.relative_to(ROOT)}")
    print(f"  -> {report_path.relative_to(ROOT)}")

    return fail


if __name__ == "__main__":
    rc = main()
    sys.exit(1 if rc > 0 else 0)