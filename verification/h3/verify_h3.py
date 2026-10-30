"""verify H3 pathway CCA: LCEL chain output against script provenance.

five tiers:

  1. held-out patient identity - compare the chain record's held_out_subjects
     with the script's shuffle+slice

  2. computation parity - same inputs -> same outputs for CCA and transfer,
     no permutations, no tolerance

  3. LCEL chain execution - build_protocol(EVALUATE_STEPS) invoked as a
     Runnable, n_permutations=50

  4. permutation null parity - run the script's cached permutation at 50 perms
     with the same seed, compare null statistics numerically with the chain.
     no arbitrary z/p allowances - matched draws should match

  5. historical artifact comparison - chain values vs per_pathway_cca.parquet

exit code: fail + unresolved > 0 means verification failed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

RUN_ID = "R4_v3"
PROJECT_DIR = ROOT / "projects" / "tnbc-92"
RUNS_DIR = ROOT / "runs" / "tnbc-92_v3"
HIST_DIR = RUNS_DIR / "eval" / "H3" / "pathway_cca"
AUCELL_PATH = ROOT / "data/embeddings/biological_signals/niche_aucell_5targets.parquet"
OUT_DIR = Path(__file__).resolve().parent

PATHWAYS = ["TGF-beta_Signaling", "Immune_System", "ECM_Organization",
            "Cell_Cycle", "Programmed_Cell_Death"]
VIEWS = ["z_he", "z_st", "z_mean"]
SEED = 42
N_PERMS = 50


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


def univariate_cca(Z, y):
    """the script's CCA: center in float32, lstsq in float64."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    w, *_ = np.linalg.lstsq(Zc.astype(np.float64), yc.astype(np.float64),
                              rcond=None)
    norm = float(np.linalg.norm(w))
    if norm < 1e-12:
        return float("nan"), np.zeros_like(w)
    w = w / norm
    z_proj = Zc @ w
    r, _ = pearsonr(z_proj, yc)
    return float(r), w


def script_perm_option_A(Z_tr, Z_te, y_tr, y_te, n_perms, seed):
    """script's cached-pinv permutation null for the transfer test."""
    Z_tr_c = Z_tr - Z_tr.mean(axis=0, keepdims=True)
    Z_te_c = Z_te - Z_te.mean(axis=0, keepdims=True)
    Z_tr_pinv = np.linalg.pinv(Z_tr_c.astype(np.float64))
    y_te_c = y_te.astype(np.float64) - y_te.mean()
    yn_te = float(np.linalg.norm(y_te_c))
    if yn_te < 1e-12:
        return np.full(n_perms, np.nan)
    y_tr_c = y_tr.astype(np.float64) - y_tr.mean()
    rng = np.random.default_rng(seed + 1)
    nulls = np.empty(n_perms)
    for i in range(n_perms):
        y_tr_shuf = rng.permutation(y_tr_c)
        w_shuf = Z_tr_pinv @ y_tr_shuf
        norm = float(np.linalg.norm(w_shuf))
        if norm < 1e-12:
            nulls[i] = float("nan")
            continue
        w_shuf = w_shuf / norm
        z_proj_te = Z_te_c @ w_shuf
        zn = float(np.linalg.norm(z_proj_te))
        if zn < 1e-12:
            nulls[i] = float("nan")
            continue
        nulls[i] = float(z_proj_te @ y_te_c) / (zn * yn_te)
    return nulls


def main():
    report: dict = {"run_id": RUN_ID, "checks": []}
    ok, fail, n_unresolved = 0, 0, 0

    def check(name, passed, detail=""):
        nonlocal ok, fail
        if passed:
            ok += 1
        else:
            fail += 1
        report["checks"].append({"name": name, "status": "PASS" if passed else "FAIL",
                                  "detail": detail})
        print(f"  [{'OK' if passed else 'FAIL'}] {name}" +
              (f"  {detail}" if detail else ""))

    def unresolved(name, detail):
        nonlocal n_unresolved
        n_unresolved += 1
        report["checks"].append({"name": name, "status": "UNRESOLVED",
                                  "detail": detail})
        print(f"  [UNRESOLVED] {name}  {detail}")

    # --- load data ---
    print("loading data...")
    t0 = time.time()
    emb = pd.read_parquet(RUNS_DIR / RUN_ID / "embeddings_test.parquet").reset_index()
    emb["niche_id"] = emb["subarray"].astype(str) + "::" + emb["spot_id"].astype(str)
    aucell = pd.read_parquet(AUCELL_PATH)
    merged = emb.merge(aucell, left_on="niche_id", right_index=True, how="left",
                        suffixes=("", "_au"))
    ok_mask = merged[PATHWAYS].notna().all(axis=1)
    merged = merged[ok_mask].reset_index(drop=True)
    patients = merged["patient_id"].astype(int).values
    n_full = len(merged)
    hist_cca = pd.read_parquet(HIST_DIR / "per_pathway_cca.parquet")
    print(f"  {n_full} niches, loaded in {time.time() - t0:.1f}s")

    # script split
    unique_pats = np.unique(patients)
    rng_s = np.random.default_rng(SEED)
    rng_s.shuffle(unique_pats)
    script_test_pats = sorted(unique_pats[:3].tolist())
    script_train_mask = np.array([p not in set(script_test_pats) for p in patients])
    script_test_mask = ~script_train_mask

    def _stack(col):
        return np.stack([np.asarray(v, dtype=np.float32) for v in col])

    views_data = {}
    for v in VIEWS:
        if v == "z_mean":
            views_data[v] = 0.5 * (views_data["z_he"] + views_data["z_st"])
        else:
            views_data[v] = _stack(merged[v])

    from omicstra.measures.evaluation import (
        top_canonical_correlation,
        transfer_correlation,
    )

    # ================================================================
    # TIER 2: computation parity
    # ================================================================
    print("\n--- tier 2: computation parity ---")

    h_row = hist_cca[hist_cca.run_id == RUN_ID].iloc[0]
    check("n_full", n_full == int(h_row["n_full"]),
          f"{n_full} vs hist {int(h_row['n_full'])}")

    for view in VIEWS:
        Z = views_data[view]
        for pname in PATHWAYS:
            tag = f"{view}/{pname}"
            y = merged[pname].astype(np.float64).values
            hr = hist_cca[(hist_cca.run_id == RUN_ID) &
                          (hist_cca.view == view) &
                          (hist_cca.pathway_name == pname)].iloc[0]

            # within-cohort: script vs historical (should be exact)
            r_script, _ = univariate_cca(Z, y)
            d_sh = abs(r_script - float(hr["corr_full"]))
            check(f"cca_full_script_{tag}", d_sh < 1e-8,
                  f"script={r_script:.10f} hist={float(hr['corr_full']):.10f} d={d_sh:.2e}")

            # within-cohort: package vs historical
            r_pkg = top_canonical_correlation(Z, y[:, None])
            d_ph = abs(r_pkg - float(hr["corr_full"]))
            if d_ph < 1e-8:
                check(f"cca_full_pkg_{tag}", True, f"d={d_ph:.2e}")
            else:
                unresolved(f"cca_full_pkg_{tag}",
                    f"pkg={r_pkg:.10f} hist={float(hr['corr_full']):.10f} "
                    f"d={d_ph:.2e} (float32 vs float64 centering)")

            # transfer on script split: script vs historical (should be exact)
            Z_tr, Z_te = Z[script_train_mask], Z[script_test_mask]
            y_tr, y_te = y[script_train_mask], y[script_test_mask]

            Zc_tr = Z_tr - Z_tr.mean(axis=0, keepdims=True)
            yc_tr = (y_tr - y_tr.mean()).astype(np.float64)
            w_tr, *_ = np.linalg.lstsq(Zc_tr.astype(np.float64), yc_tr, rcond=None)
            n_w = float(np.linalg.norm(w_tr))
            w_tr = w_tr / n_w if n_w > 1e-12 else w_tr
            Zc_te = Z_te - Z_te.mean(axis=0, keepdims=True)
            yc_te = y_te - y_te.mean()
            r_s_te, _ = pearsonr((Zc_te @ w_tr).ravel(), yc_te)

            dh_te = abs(float(r_s_te) - float(hr["corr_test"]))
            check(f"transfer_script_{tag}", dh_te < 1e-8,
                  f"script={r_s_te:.8f} hist={float(hr['corr_test']):.8f} d={dh_te:.2e}")

            # transfer: package vs script
            r_te_pkg, r_tr_pkg, _, _ = transfer_correlation(
                Z_tr, y_tr[:, None], Z_te, y_te[:, None])
            d_pkg_te = abs(float(r_te_pkg) - float(r_s_te))
            if d_pkg_te < 1e-8:
                check(f"transfer_pkg_{tag}", True, f"d={d_pkg_te:.2e}")
            else:
                unresolved(f"transfer_pkg_{tag}",
                    f"pkg={r_te_pkg:.8f} script={r_s_te:.8f} "
                    f"d={d_pkg_te:.2e} (float32 vs float64 centering)")

    # ================================================================
    # TIER 3: LCEL chain execution
    # ================================================================
    print("\n--- tier 3: LCEL chain execution (n_perm=50) ---")

    from omicstra.adapters.evaluation import arrays
    from omicstra.protocols import build_protocol
    from omicstra.protocols.evaluate import EVALUATE_STEPS

    ctx = arrays(str(PROJECT_DIR), RUN_ID,
                 extra_views=("z_he", "z_st", "z_mean"),
                 signals=(),
                 include_pathway_scores=True)

    check("adapter_scores", len(ctx.get("scores", {})) == 5,
          f"keys={sorted(ctx.get('scores', {}).keys())}")
    check("adapter_views", all(v in ctx["spaces"] for v in VIEWS),
          f"spaces={sorted(ctx['spaces'].keys())}")

    chain = build_protocol(EVALUATE_STEPS, "evaluate")
    chain_params = {
        "association_and_specificity": {
            "n_permutations": N_PERMS,
            "alpha": 0.05,
            "tau_specificity": 0.8,
            "tail": "greater",
            "held_out_groups": 3,
            "transfer_seed": SEED,
            "within_cohort_seed": SEED,
            "min_units_per_side": 100,
        },
    }
    chain_ctx = {**ctx, "params": chain_params}

    t1 = time.time()
    records = chain.invoke(chain_ctx)
    elapsed = time.time() - t1
    print(f"  chain completed in {elapsed:.1f}s")

    as_rec = records.get("association_and_specificity", {})
    check("chain_step_pass", as_rec.get("status") == "pass",
          f"status={as_rec.get('status')}")

    obs = as_rec.get("observed", {})
    per_view = obs.get("per_view", {})

    # ================================================================
    # TIER 1: held-out patient identity (from chain record)
    # ================================================================
    print("\n--- tier 1: held-out patient identity ---")
    chain_held = sorted(obs.get("held_out_subjects", []))
    script_held = sorted(str(p) for p in script_test_pats)
    check("held_out_match", chain_held == script_held,
          f"chain={chain_held} script={script_held}")
    check("n_train_match", obs.get("n_train") == int(script_train_mask.sum()),
          f"chain={obs.get('n_train')} script={int(script_train_mask.sum())}")
    check("n_test_match", obs.get("n_test") == int(script_test_mask.sum()),
          f"chain={obs.get('n_test')} script={int(script_test_mask.sum())}")

    # structure
    for view in VIEWS:
        blk = per_view.get(view, {})
        evaluable = [s for s in blk.get("per_signal", [])
                     if not s.get("unevaluable")]
        check(f"n_eval_{view}", len(evaluable) == 5,
              f"evaluable={len(evaluable)}")

    # FDR family: should now be global (3 views x 5 pathways = 15)
    for view in VIEWS:
        blk = per_view.get(view, {})
        ff = blk.get("fdr_family")
        check(f"fdr_family_{view}", ff == 15,
              f"fdr_family={ff} (expected 15 = 3 views x 5 pathways)")

    # specificity guard
    for view in VIEWS:
        blk = per_view.get(view, {})
        s = blk.get("specificity_status", "missing")
        check(f"spec_{view}", s in ("pass", "fail"), f"status={s}")

    # ================================================================
    # TIER 4: permutation null parity (script vs chain, 50 perms)
    # ================================================================
    print("\n--- tier 4: permutation null parity ---")

    for view in VIEWS:
        Z = views_data[view]
        Z_tr, Z_te = Z[script_train_mask], Z[script_test_mask]
        chain_blk = per_view.get(view, {})

        for pname in PATHWAYS:
            tag = f"{view}/{pname}"
            y = merged[pname].astype(np.float64).values
            y_tr, y_te = y[script_train_mask], y[script_test_mask]

            # script transfer null at 50 perms
            script_nulls = script_perm_option_A(
                Z_tr, Z_te, y_tr, y_te, N_PERMS, SEED)

            chain_row = None
            for r in chain_blk.get("per_signal", []):
                if r.get("signal") == pname:
                    chain_row = r
                    break
            if chain_row is None or chain_row.get("unevaluable"):
                check(f"perm_{tag}", False, "no chain row")
                continue

            chain_z = chain_row.get("z")
            chain_p = chain_row.get("p")
            chain_r_te = chain_row.get("r_test")
            chain_n_null = chain_row.get("n_null", 0)

            check(f"n_null_{tag}", chain_n_null == N_PERMS,
                  f"chain={chain_n_null}")

            # the script's observed r_test for this split
            hr = hist_cca[(hist_cca.run_id == RUN_ID) &
                          (hist_cca.view == view) &
                          (hist_cca.pathway_name == pname)].iloc[0]
            s_obs = float(hr["corr_test"])

            # compute script's z and p from its nulls, using the chain's
            # conservative (1+k)/(1+n) formula for matched comparison
            valid = ~np.isnan(script_nulls)
            n_valid = int(valid.sum())
            s_mean = float(np.mean(script_nulls[valid]))
            s_std = float(np.std(script_nulls[valid]))
            s_z = (s_obs - s_mean) / s_std if s_std > 0 else float("nan")
            s_beyond = int((script_nulls[valid] >= s_obs).sum())
            s_p = float((1 + s_beyond) / (1 + n_valid))

            # observed r_test: chain vs script (centering precision)
            d_obs = abs(float(chain_r_te) - s_obs)
            if d_obs < 1e-8:
                check(f"r_te_{tag}", True, f"d={d_obs:.2e}")
            else:
                unresolved(f"r_te_{tag}",
                    f"chain={chain_r_te:.8f} script={s_obs:.8f} d={d_obs:.2e} "
                    "(float32 vs float64 centering)")

            # z-score: same null draws, centering difference propagates
            d_z = abs(float(chain_z) - s_z)
            if d_z < 1e-4:
                check(f"z_{tag}", True,
                      f"chain={chain_z:.4f} script={s_z:.4f} d={d_z:.2e}")
            else:
                unresolved(f"z_{tag}",
                    f"chain={chain_z:.4f} script={s_z:.4f} d={d_z:.2e}")

            # p-value: matched formula (1+k)/(1+n)
            d_p = abs(float(chain_p) - s_p)
            if d_p < 1e-8:
                check(f"p_{tag}", True,
                      f"chain={chain_p:.6f} script={s_p:.6f} d={d_p:.2e}")
            else:
                unresolved(f"p_{tag}",
                    f"chain={chain_p:.6f} script={s_p:.6f} d={d_p:.2e}")

            # within-cohort r vs historical corr_full
            wc = chain_row.get("within_cohort", {})
            if wc:
                wc_r = float(wc.get("r", float("nan")))
                h_full = float(hr["corr_full"])
                d_wc = abs(wc_r - h_full)
                if d_wc < 1e-8:
                    check(f"wc_r_{tag}", True, f"d={d_wc:.2e}")
                else:
                    unresolved(f"wc_r_{tag}",
                        f"chain={wc_r:.10f} hist={h_full:.10f} d={d_wc:.2e}")

    # ================================================================
    # TIER 5: historical transfer comparison (from chain)
    # ================================================================
    print("\n--- tier 5: historical transfer comparison ---")

    for view in VIEWS:
        chain_blk = per_view.get(view, {})
        for row in chain_blk.get("per_signal", []):
            if row.get("unevaluable"):
                continue
            sname = row["signal"]
            tag = f"{view}/{sname}"
            r_te = row.get("r_test")
            r_tr = row.get("r_train")
            hr = hist_cca[(hist_cca.run_id == RUN_ID) &
                          (hist_cca.view == view) &
                          (hist_cca.pathway_name == sname)]
            if hr.empty:
                check(f"hist_{tag}", False, "no hist row")
                continue
            h_te = float(hr.iloc[0]["corr_test"])
            h_tr = float(hr.iloc[0]["corr_train"])
            d_te = abs(float(r_te) - h_te)
            d_tr = abs(float(r_tr) - h_tr)
            if d_te < 1e-8 and d_tr < 1e-8:
                check(f"hist_{tag}", True,
                      f"d_te={d_te:.2e} d_tr={d_tr:.2e}")
            else:
                unresolved(f"hist_{tag}",
                    f"chain r_te={r_te:.8f} hist={h_te:.8f} d_te={d_te:.2e}; "
                    f"chain r_tr={r_tr:.8f} hist={h_tr:.8f} d_tr={d_tr:.2e}")

    # --- save ---
    out_path = OUT_DIR / "chain_output_R4_v3.json"
    out_path.write_text(json.dumps({"records": records}, indent=2, default=_ser))

    report["summary"] = {
        "ok": ok, "fail": fail, "unresolved": n_unresolved,
        "total": ok + fail + n_unresolved,
        "elapsed_chain_s": round(elapsed, 1),
        "verdict": "PASS" if (fail + n_unresolved) == 0 else "FAIL",
    }
    report_path = OUT_DIR / "verification_report_R4_v3.json"
    report_path.write_text(json.dumps(report, indent=2, default=_ser))

    print(f"\n{'=' * 60}")
    total_bad = fail + n_unresolved
    print(f"  {ok} PASS, {fail} FAIL, {n_unresolved} UNRESOLVED")
    print(f"  verdict: {'PASS' if total_bad == 0 else 'FAIL'}")
    print(f"  -> {out_path.relative_to(ROOT)}")
    print(f"  -> {report_path.relative_to(ROOT)}")
    return 1 if total_bad > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
