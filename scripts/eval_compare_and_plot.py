#!/usr/bin/env python
"""eval_compare_and_plot.py - distribution-first comparison + figures for the
H1/H2/H3 alignment grid.

consumes per-run artifacts produced by align.py, align_classical.py, and
eval_alignment_biology.py. produces:

  runs/tnbc-92/eval/compare/
    paired_tests.parquet         pairwise (R_a vs R_b) perm tests on
                                 bio_delta and patient_delta; uses shared
                                 seed=42 null index so arrays align
    effective_rank.parquet       per-run participation ratio, variance spectrum
                                 top-3 fraction, cosine-pair distribution
                                 summary statistics
    asymmetric_compression.parquet
                                 (bio_delta - patient_delta) per run + bootstrap
                                 95% CI (patient-level resample)

  runs/tnbc-92/eval/figures/
    fig1_bio_vs_patient_z.pdf     bio_z vs patient_z scatter, one point per run
    fig2_rk_curves.pdf            R@K curves (log-y, K on x, chance floor)
    fig3_effrank_vs_cka.pdf       effective_rank vs CKA_after scatter
    fig6_null_histograms.pdf      per-label perm-null histograms + observed
    fig9_asymmetric_compression.pdf
                                 bar of (bio_delta - patient_delta) with
                                 bootstrap CIs, per run

usage:
    python scripts/eval_compare_and_plot.py --runs R1 R4 B1 R6
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_theme(style='ticks', palette='Set2', context='notebook')

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / 'runs' / 'tnbc-92'
NICHES_DIR = ROOT / 'data' / 'embeddings' / 'niches'
OUT_DIR = RUNS_ROOT / 'eval' / 'compare'
FIG_DIR = RUNS_ROOT / 'eval' / 'figures'

RUN_COLORS = {
    'R1': '#1f77b4', 'R2': '#aec7e8', 'R3': '#9467bd', 'R4': '#d62728',
    'R6': '#ff7f0e', 'B1': '#2ca02c', 'B2': '#98df8a', 'B3': '#888888',
}
RUN_LABELS = {
    'R1': 'R1 infonce+late', 'R2': 'R2 supcon+late', 'R3': 'R3 barlow+late',
    'R4': 'R4 infonce+cross_attn', 'R6': 'R6 infonce+late (novae-only ST)',
    'B1': 'B1 CCA', 'B2': 'B2 Procrustes', 'B3': 'B3 Unaligned PCA',
}


# -------------------------------------------------------------------------- #
# loaders
# -------------------------------------------------------------------------- #

def load_biology_summary(run_id: str) -> pd.DataFrame:
    return pd.read_parquet(RUNS_ROOT / run_id / 'eval' / 'biology.parquet')


def load_biology_nulls(run_id: str) -> pd.DataFrame:
    p = RUNS_ROOT / run_id / 'eval' / 'biology_nulls.parquet'
    return pd.read_parquet(p) if p.exists() else None


def load_h1_metrics(run_id: str) -> dict:
    return json.loads((RUNS_ROOT / run_id / 'metrics_h1_raw.json').read_text())


def load_embeddings(run_id: str) -> pd.DataFrame:
    return pd.read_parquet(RUNS_ROOT / run_id / 'embeddings_test.parquet')


# -------------------------------------------------------------------------- #
# paired-perm test: R_a vs R_b on bio_delta (and patient_delta)
# -------------------------------------------------------------------------- #

def paired_perm_test(run_a: str, run_b: str, view: str, label: str,
                     test_type: str) -> dict:
    """paired permutation test on (delta_a - delta_b) under shared-seed null.

    nulls are loaded from biology_nulls.parquet for both runs; because both
    runs used seed=42 with identical permutation scheme, null perm index i
    corresponds to the SAME shuffled labels across runs. we can take
    (null_a[i] - null_b[i]) as the null for the observed difference.
    """
    nulls_a = load_biology_nulls(run_a)
    nulls_b = load_biology_nulls(run_b)
    if nulls_a is None or nulls_b is None:
        return {'status': 'nulls_missing'}

    def null_row(df):
        row = df[(df['view'] == view) & (df['label'] == label) &
                 (df['test_type'] == test_type)]
        if len(row) == 0:
            return None
        perm_cols = [c for c in df.columns if c.startswith('p')]
        return row[perm_cols].values[0].astype(np.float32)

    na, nb = null_row(nulls_a), null_row(nulls_b)
    if na is None or nb is None:
        return {'status': 'test_not_found'}

    summary_a = load_biology_summary(run_a)
    summary_b = load_biology_summary(run_b)
    obs_a = summary_a[(summary_a['view'] == view) &
                       (summary_a['label'] == label) &
                       (summary_a['test_type'] == test_type)]['delta'].values[0]
    obs_b = summary_b[(summary_b['view'] == view) &
                       (summary_b['label'] == label) &
                       (summary_b['test_type'] == test_type)]['delta'].values[0]

    obs_diff = float(obs_a - obs_b)
    null_diff = na - nb
    p_two = float((np.abs(null_diff) >= abs(obs_diff)).sum() + 1) / (len(null_diff) + 1)
    return {
        'run_a': run_a, 'run_b': run_b, 'view': view, 'label': label,
        'test_type': test_type,
        'obs_a': float(obs_a), 'obs_b': float(obs_b),
        'obs_diff': obs_diff,
        'null_diff_mean': float(null_diff.mean()),
        'null_diff_std': float(null_diff.std()),
        'p_two_sided': p_two,
        'n_perm': int(len(null_diff)),
        'status': 'ok',
    }


def run_paired_tests(runs: list[str]) -> pd.DataFrame:
    """all pairwise paired-perm tests for (bio, patient) on views of interest."""
    views = ['z_he', 'z_st', 'z_mean']
    bio_labels = ['MC_global', 'MC_tumor', 'TIME']
    rows = []
    for i, ra in enumerate(runs):
        for rb in runs[i+1:]:
            for v in views:
                for lb in bio_labels:
                    rows.append(paired_perm_test(ra, rb, v, lb, 'biology'))
                rows.append(paired_perm_test(ra, rb, v, 'patient_id', 'patient'))
    return pd.DataFrame([r for r in rows if r.get('status') == 'ok'])


# -------------------------------------------------------------------------- #
# effective rank + cosine distributions
# -------------------------------------------------------------------------- #

def participation_ratio(X: np.ndarray) -> float:
    Xc = X - X.mean(0, keepdims=True)
    C = Xc.T @ Xc / max(len(Xc), 1)
    evals = np.linalg.eigvalsh(C)
    evals = evals[evals > 1e-12]
    return float((evals.sum() ** 2) / ((evals ** 2).sum() + 1e-20))


def variance_spectrum(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0, keepdims=True)
    C = Xc.T @ Xc / max(len(Xc), 1)
    evals = np.linalg.eigvalsh(C)
    return np.sort(evals)[::-1]


def compute_rank_table(runs: list[str]) -> pd.DataFrame:
    rows = []
    for r in runs:
        emb = load_embeddings(r)
        z_he = np.stack([np.asarray(v) for v in emb['z_he']])
        z_st = np.stack([np.asarray(v) for v in emb['z_st']])
        # pair cosine distribution: 500-sample matched vs mismatched
        rng = np.random.default_rng(42)
        n = min(500, len(z_he))
        idx = rng.choice(len(z_he), n, replace=False)
        zh, zs = z_he[idx], z_st[idx]
        matched = (zh * zs).sum(-1)
        mis = zh @ zs.T
        mis = mis[~np.eye(n, dtype=bool)]
        h1 = load_h1_metrics(r)
        rows.append({
            'run_id': r,
            'pr_z_he': participation_ratio(z_he),
            'pr_z_st': participation_ratio(z_st),
            'matched_cos_mean': float(matched.mean()),
            'matched_cos_std': float(matched.std()),
            'mismatched_cos_mean': float(mis.mean()),
            'mismatched_cos_std': float(mis.std()),
            'cka_after': h1.get('cka_after', float('nan')),
            'cka_before': h1.get('cka_before', float('nan')),
            'auc': h1.get('auc', float('nan')),
            'r1': h1.get('r1', float('nan')),
            'r5': h1.get('r5', float('nan')),
            'r10': h1.get('r10', float('nan')),
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------- #
# asymmetric compression with patient-bootstrap CI
# -------------------------------------------------------------------------- #

def asymmetric_compression_ci(run_id: str, views: tuple[str, ...] = ('z_he',),
                               bio_label: str = 'TIME',
                               n_boot: int = 1000) -> list[dict]:
    """bootstrap 95% CI on (bio_delta - patient_delta) for one run, multi-view.

    resample patients with replacement, recompute deltas per view, collect.
    encoding/pooling is amortized; one bootstrap pass covers all views.
    returns one row per view.
    """
    from eval_alignment_biology import (
        load_tnbc_labels, ST_FEATURES, cosine_matrix, subarray_mean_pool, l2,
        reconstruct_trained_model, encode_trained, fit_classical, encode_classical,
        _biology_delta, _patient_delta, _pair_indices_and_same_patient_mask,
    )
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from align import load_subarray, discover_subarrays  # noqa: E402

    run_dir = RUNS_ROOT / run_id
    run_cfg = json.loads((run_dir / 'run_config.json').read_text())
    is_classical = 'baseline' in run_cfg
    st_features = tuple(run_cfg.get('st_features', ST_FEATURES))

    paths = discover_subarrays()
    subs_raw = [load_subarray(p, st_features) for p in paths]
    subs = [s for s in subs_raw if s['patient_id'] is not None and s['archetype'] is not None]

    if is_classical:
        state = fit_classical(run_id, subs)
        encode = lambda sub: encode_classical(state, sub)
    else:
        model, _ = reconstruct_trained_model(run_id, subs[0])
        encode = lambda sub: encode_trained(model, sub)

    # encode + pool to subarray-level (once)
    keys = []
    zh_rows, zs_rows = [], []
    for sub in subs:
        z_he, z_st = encode(sub)
        keys.append(sub['subarray'])
        zh_rows.append(subarray_mean_pool(z_he))
        zs_rows.append(subarray_mean_pool(z_st))
    Z_he = np.stack(zh_rows)
    Z_st = np.stack(zs_rows)
    Z_mean = l2(0.5 * (Z_he + Z_st))
    view_mats = {'z_he': Z_he, 'z_st': Z_st, 'z_mean': Z_mean}

    patients, labels_by_key = load_tnbc_labels(keys)
    labels = labels_by_key[bio_label]

    def deltas(sub_idx, view_mat):
        pats = patients[sub_idx]
        labs = labels[sub_idx]
        V = view_mat[sub_idx]
        S = cosine_matrix(V)
        iu, same_pt = _pair_indices_and_same_patient_mask(pats)
        pairs = S[iu]
        valid = ((labs != 'NA')[:, None] & (labs != 'NA')[None, :])[iu]
        bio, *_ = _biology_delta(pairs, same_pt, labs, iu, valid)
        pat, *_ = _patient_delta(pairs, pats, iu)
        return bio, pat

    # observed deltas per view
    all_idx = np.arange(len(keys))
    obs_per_view = {}
    for v in views:
        bio_obs, pat_obs = deltas(all_idx, view_mats[v])
        obs_per_view[v] = (bio_obs, pat_obs, bio_obs - pat_obs)

    # bootstrap: one resample sequence, deltas for each view in the inner loop
    unique_patients = np.unique(patients)
    patient_to_subs = {p: np.where(patients == p)[0] for p in unique_patients}
    rng = np.random.default_rng(42)
    boot_per_view = {v: {'asym': np.empty(n_boot, dtype=np.float32),
                         'bio':  np.empty(n_boot, dtype=np.float32),
                         'pat':  np.empty(n_boot, dtype=np.float32)} for v in views}
    for i in range(n_boot):
        sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        sampled_idx = np.concatenate([patient_to_subs[p] for p in sampled_patients])
        for v in views:
            bio_i, pat_i = deltas(sampled_idx, view_mats[v])
            boot_per_view[v]['bio'][i] = bio_i
            boot_per_view[v]['pat'][i] = pat_i
            boot_per_view[v]['asym'][i] = bio_i - pat_i

    rows = []
    for v in views:
        bio_obs, pat_obs, asym_obs = obs_per_view[v]
        b = boot_per_view[v]
        rows.append({
            'run_id': run_id, 'view': v, 'bio_label': bio_label,
            'bio_obs': float(bio_obs), 'patient_obs': float(pat_obs),
            'asym_obs': float(asym_obs),
            'asym_boot_lo': float(np.quantile(b['asym'], 0.025)),
            'asym_boot_hi': float(np.quantile(b['asym'], 0.975)),
            'bio_boot_lo': float(np.quantile(b['bio'], 0.025)),
            'bio_boot_hi': float(np.quantile(b['bio'], 0.975)),
            'patient_boot_lo': float(np.quantile(b['pat'], 0.025)),
            'patient_boot_hi': float(np.quantile(b['pat'], 0.975)),
            'n_boot': int(n_boot),
        })
    return rows


# -------------------------------------------------------------------------- #
# plots
# -------------------------------------------------------------------------- #

def fig1_bio_vs_patient(runs: list[str], view: str = 'z_he',
                         bio_label: str = 'TIME') -> plt.Figure:
    """bio_z vs patient_z scatter, one point per run; raw_he/raw_st marked."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for r in runs:
        bio = load_biology_summary(r)
        row_bio = bio[(bio['view'] == view) & (bio['label'] == bio_label)
                      & (bio['test_type'] == 'biology')]
        row_pat = bio[(bio['view'] == view) & (bio['label'] == 'patient_id')
                      & (bio['test_type'] == 'patient')]
        if len(row_bio) == 0 or len(row_pat) == 0:
            continue
        x = float(row_pat['z'].iloc[0])
        y = float(row_bio['z'].iloc[0])
        c = RUN_COLORS.get(r, '#444')
        ax.scatter(x, y, s=140, c=c, edgecolor='white', linewidth=1.5, zorder=3)
        ax.annotate(r, (x, y), xytext=(6, 6), textcoords='offset points',
                    fontsize=11, fontweight='bold', color=c)
    # raw reference from R1 (or first available)
    if runs:
        bio = load_biology_summary(runs[0])
        for raw_view, marker in [('raw_he', 's'), ('raw_st', '^')]:
            rb = bio[(bio['view'] == raw_view) & (bio['label'] == bio_label)
                     & (bio['test_type'] == 'biology')]
            rp = bio[(bio['view'] == raw_view) & (bio['label'] == 'patient_id')
                     & (bio['test_type'] == 'patient')]
            if len(rb) and len(rp):
                ax.scatter(float(rp['z'].iloc[0]), float(rb['z'].iloc[0]),
                           s=180, marker=marker, facecolor='none',
                           edgecolor='#444', linewidth=2, label=raw_view)
    # unit-slope reference (bio == patient)
    lim = 150
    ax.plot([0, lim], [0, lim], '--', color='#aaa', alpha=0.6, lw=1, zorder=1)
    ax.annotate('ideal: bio ≫ patient', (5, lim * 0.92), fontsize=10, color='#555')
    ax.annotate('bio = patient', (lim * 0.75, lim * 0.78), fontsize=9,
                color='#888', rotation=45, ha='center')
    ax.set_xlabel(f'patient_id z  (higher = more patient leakage)')
    ax.set_ylabel(f'{bio_label} biology z  (higher = more biology preserved)')
    ax.set_title(f'biology vs patient signal (subarray-level perm z, view={view})')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(-5, lim)
    ax.set_ylim(-1, 20)
    sns.despine(ax=ax)
    return fig


def fig2_rk_curves(runs: list[str]) -> plt.Figure:
    """R@K curves with chance floor."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ks = [1, 5, 10]
    for r in runs:
        h = load_h1_metrics(r)
        ys = [h[f'r{k}'] for k in ks]
        c = RUN_COLORS.get(r, '#444')
        ax.plot(ks, ys, 'o-', color=c, lw=2, markersize=7,
                label=RUN_LABELS.get(r, r), markeredgecolor='white',
                markeredgewidth=0.8)
    # chance floor
    n_test = load_h1_metrics(runs[0]).get('test_niches', 40000)
    chance = np.array(ks) / n_test
    ax.plot(ks, chance, ':', color='#aaa', lw=1.5, label=f'chance (N={n_test})')
    ax.set_xlabel('K')
    ax.set_ylabel('R@K (log)')
    ax.set_yscale('log')
    ax.set_xticks(ks)
    ax.set_title('cross-subarray cross-modal retrieval R@K')
    ax.legend(loc='lower right', fontsize=8)
    sns.despine(ax=ax)
    return fig


def fig3_effrank_vs_cka(rank_tbl: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    for _, row in rank_tbl.iterrows():
        r = row['run_id']
        c = RUN_COLORS.get(r, '#444')
        x = row['pr_z_he']
        y = row['cka_after']
        ax.scatter(x, y, s=140, c=c, edgecolor='white', linewidth=1.5, zorder=3)
        ax.annotate(r, (x, y), xytext=(6, 6), textcoords='offset points',
                    fontsize=11, fontweight='bold', color=c)
    ax.set_xlabel('effective rank (z_he, participation ratio)')
    ax.set_ylabel('CKA after projection')
    ax.set_title('does CKA come from collapse? (lower rank + higher CKA = yes)')
    ax.set_xscale('log')
    sns.despine(ax=ax)
    return fig


def fig6_null_histograms(runs: list[str], label: str = 'TIME',
                          view: str = 'z_he', test_type: str = 'biology',
                          n_plot_perms: int = 5000) -> plt.Figure:
    """per-run permutation null histograms with observed delta marked."""
    fig, axes = plt.subplots(1, len(runs), figsize=(3.2 * len(runs), 3.5),
                              sharey=True)
    if len(runs) == 1:
        axes = [axes]
    for ax, r in zip(axes, runs):
        nulls = load_biology_nulls(r)
        if nulls is None:
            ax.set_title(f'{r} (no nulls saved)')
            continue
        row = nulls[(nulls['view'] == view) & (nulls['label'] == label)
                     & (nulls['test_type'] == test_type)]
        perm_cols = [c for c in row.columns if c.startswith('p')]
        null_arr = row[perm_cols].values[0].astype(np.float32)
        if len(null_arr) > n_plot_perms:
            null_arr = np.random.default_rng(42).choice(null_arr, n_plot_perms, replace=False)
        summary = load_biology_summary(r)
        obs = summary[(summary['view'] == view) & (summary['label'] == label)
                       & (summary['test_type'] == test_type)]['delta'].values[0]
        c = RUN_COLORS.get(r, '#444')
        ax.hist(null_arr, bins=50, color=c, alpha=0.5, edgecolor='white', lw=0.3)
        ax.axvline(obs, color=c, lw=2.5, zorder=5)
        ax.axvline(0, color='#aaa', ls=':', lw=1)
        ax.set_title(f'{r}')
        ax.set_xlabel(f'{label} delta')
    axes[0].set_ylabel('null perm count')
    fig.suptitle(f'permutation null distribution with observed delta ({test_type}, {view})',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    return fig


def fig9_asymmetric_compression(asym_df: pd.DataFrame, view: str | None = None) -> plt.Figure:
    """bar plot of (bio - patient) delta with bootstrap CIs.

    if view is given, df is assumed already filtered to that view; title gets the tag.
    if view is None, plots whatever is in df (assumes single-view).

    NOTE: when the observed value lies outside the bootstrap CI (ex CCA's
    geometry is fragile to patient resampling), errs would be negative which
    matplotlib rejects. we clamp to 0 and annotate the bar so the reader can
    see the artifact rather than have it hidden.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    runs = asym_df['run_id'].tolist()
    x = np.arange(len(runs))
    obs = asym_df['asym_obs'].values
    lo = asym_df['asym_boot_lo'].values
    hi = asym_df['asym_boot_hi'].values
    err_lo = np.maximum(obs - lo, 0)   # clamp to >= 0
    err_hi = np.maximum(hi - obs, 0)
    err = np.vstack([err_lo, err_hi])
    outside_ci = (obs < lo) | (obs > hi)
    colors = [RUN_COLORS.get(r, '#444') for r in runs]
    ax.bar(x, obs, yerr=err, color=colors, alpha=0.85, capsize=5,
           edgecolor='white', linewidth=0.8)
    # mark CI midpoint with cross when observed is outside CI (artifact flag)
    for i, flag in enumerate(outside_ci):
        if flag:
            ci_mid = 0.5 * (lo[i] + hi[i])
            ax.scatter(i, ci_mid, marker='x', color='#222', s=80, zorder=5)
            ax.annotate('CI off', (i, ci_mid),
                        xytext=(0, -14), textcoords='offset points',
                        ha='center', fontsize=8, color='#222')
    ax.axhline(0, color='#aaa', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(runs, fontsize=10)
    ax.set_ylabel('bio_delta − patient_delta  (higher = more asymmetric compression)')
    suffix = f' ({view})' if view else ''
    ax.set_title(f'asymmetric compression with 95% bootstrap CI{suffix}')
    sns.despine(ax=ax)
    return fig


# -------------------------------------------------------------------------- #
# main
# -------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--bio-label', default='TIME')
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--skip-asym', action='store_true',
                    help='skip bootstrap asym compression (expensive; ~1 min/run)')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f'=== comparison + plots for runs {args.runs} ===\n')

    # paired permutation tests on bio + patient deltas
    print('  paired permutation tests...')
    paired = run_paired_tests(args.runs)
    if len(paired):
        paired.to_parquet(OUT_DIR / 'paired_tests.parquet')
        print(f'    -> {(OUT_DIR / "paired_tests.parquet").relative_to(ROOT)}')
    else:
        print('    WARN: no paired tests produced (nulls missing?)')

    # effective rank + cosine distributions
    print('\n  effective rank + cosine spread...')
    rank_tbl = compute_rank_table(args.runs)
    rank_tbl.to_parquet(OUT_DIR / 'effective_rank.parquet')
    print(rank_tbl.round(4).to_string(index=False))
    print(f'    -> {(OUT_DIR / "effective_rank.parquet").relative_to(ROOT)}')

    # asymmetric compression with bootstrap CI - z_he and z_mean in one pass
    asym_df = None
    if not args.skip_asym:
        print(f'\n  asymmetric compression bootstrap ({args.n_boot} resamples per run, views: z_he + z_mean)...')
        asym_rows = []
        for r in args.runs:
            print(f'    {r}...')
            asym_rows.extend(asymmetric_compression_ci(
                r, views=('z_he', 'z_mean'),
                bio_label=args.bio_label, n_boot=args.n_boot,
            ))
        asym_df = pd.DataFrame(asym_rows)
        asym_df.to_parquet(OUT_DIR / 'asymmetric_compression.parquet')
        print(asym_df[['run_id', 'view', 'asym_obs', 'asym_boot_lo', 'asym_boot_hi']].round(4).to_string(index=False))
        print(f'    -> {(OUT_DIR / "asymmetric_compression.parquet").relative_to(ROOT)}')

    # figures
    print('\n  figures...')
    for title, fig in [
        ('fig1_bio_vs_patient_z', fig1_bio_vs_patient(args.runs)),
        ('fig2_rk_curves', fig2_rk_curves(args.runs)),
        ('fig3_effrank_vs_cka', fig3_effrank_vs_cka(rank_tbl)),
        ('fig6_null_histograms', fig6_null_histograms(args.runs, label=args.bio_label)),
    ]:
        out = FIG_DIR / f'{title}.pdf'
        fig.savefig(out, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f'    {out.relative_to(ROOT)}')
    if asym_df is not None:
        for view in sorted(asym_df['view'].unique()):
            view_df = asym_df[asym_df['view'] == view].reset_index(drop=True)
            fig = fig9_asymmetric_compression(view_df, view=view)
            out = FIG_DIR / f'fig9_asym_compression_{view}.pdf'
            fig.savefig(out, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f'    {out.relative_to(ROOT)}')

    print('\ndone.')


if __name__ == '__main__':
    main()