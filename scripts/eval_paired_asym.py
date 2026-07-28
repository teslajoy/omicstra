#!/usr/bin/env python
"""eval_paired_asym.py - paired-permutation test on (bio_delta - patient_delta) difference.

question this answers:
    the report rests on "R1 disentangles biology vs patient more than R4."
    individual paired tests on bio_delta and patient_delta separately tell us
    each delta is significant per-run. but to claim "R_a has more asymmetric
    compression than R_b" we need a test on the difference of the difference:
        H0:  asym_a == asym_b   where  asym = bio_delta - patient_delta
        Test statistic:  obs_diff = asym_a_obs - asym_b_obs
        Null:  paired permutation using shared seed=42 perm indices

how this works:
    biology_nulls.parquet stores 1e5 perm draws per run for each
    (view, label, test_type). because all runs share seed=42 the perm-index i
    corresponds to the same shuffled labels (or matched-null patient
    assignment) across runs. that gives us paired nulls:
        bio_diff_null[i]  = bio_null_a[i]  - bio_null_b[i]
        pat_diff_null[i]  = pat_null_a[i]  - pat_null_b[i]
    and by linearity of expectation on the difference-of-differences:
        asym_diff_null[i] = bio_diff_null[i] - pat_diff_null[i]
    p-value: (|asym_diff_null| >= |obs_diff|).sum() + 1) / (n_perm + 1)

this is nonparametric. no Gaussian assumption on the null. consistent with
the distribution-first pivot already in place.

usage:
    python scripts/eval_paired_asym.py
        --runs R1 R2 R3 R4 R6 B1 B2
        --views z_he z_mean
        --bio-label TIME
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / 'runs' / 'tnbc-92'


def load_summary(run_id: str) -> pd.DataFrame:
    return pd.read_parquet(RUNS_ROOT / run_id / 'eval' / 'biology.parquet')


def load_nulls(run_id: str) -> pd.DataFrame | None:
    p = RUNS_ROOT / run_id / 'eval' / 'biology_nulls.parquet'
    return pd.read_parquet(p) if p.exists() else None


def get_null_array(nulls_df: pd.DataFrame, view: str, label: str,
                   test_type: str) -> np.ndarray | None:
    row = nulls_df[(nulls_df['view'] == view) & (nulls_df['label'] == label)
                   & (nulls_df['test_type'] == test_type)]
    if len(row) == 0:
        return None
    perm_cols = [c for c in nulls_df.columns if c.startswith('p')]
    return row[perm_cols].values[0].astype(np.float32)


def get_obs_delta(summary: pd.DataFrame, view: str, label: str,
                   test_type: str) -> float | None:
    row = summary[(summary['view'] == view) & (summary['label'] == label)
                  & (summary['test_type'] == test_type)]
    if len(row) == 0:
        return None
    return float(row['delta'].iloc[0])


def per_run_vs_raw_asym(run_id: str, view: str,
                         raw_view: str = 'raw_he',
                         bio_label: str = 'TIME') -> dict:
    """one-sample paired-perm test: does this run's (bio - patient) asym differ
    from raw_he's asym? uses the same shared-seed null structure.

    test stat:  obs_diff = (run_asym_obs) - (raw_asym_obs)
    null:       (run_null[i] - raw_null[i]) under shared seed
    direction:  sign of obs_diff says move-from-raw direction
                positive = ran moved toward symmetric
                negative = run moved AWAY from raw (more asymmetric)
    """
    nulls = load_nulls(run_id)
    if nulls is None:
        return {'status': 'nulls_missing', 'run_id': run_id, 'view': view}

    summary = load_summary(run_id)

    bio_run = get_null_array(nulls, view, bio_label, 'biology')
    pat_run = get_null_array(nulls, view, 'patient_id', 'patient')
    bio_raw = get_null_array(nulls, raw_view, bio_label, 'biology')
    pat_raw = get_null_array(nulls, raw_view, 'patient_id', 'patient')
    if any(x is None for x in [bio_run, pat_run, bio_raw, pat_raw]):
        return {'status': 'test_not_found', 'run_id': run_id, 'view': view}

    bio_obs_run = get_obs_delta(summary, view, bio_label, 'biology')
    pat_obs_run = get_obs_delta(summary, view, 'patient_id', 'patient')
    bio_obs_raw = get_obs_delta(summary, raw_view, bio_label, 'biology')
    pat_obs_raw = get_obs_delta(summary, raw_view, 'patient_id', 'patient')

    asym_run = bio_obs_run - pat_obs_run
    asym_raw = bio_obs_raw - pat_obs_raw
    obs_diff = asym_run - asym_raw

    # null: paired (asym_run_null - asym_raw_null) under shared seed
    asym_run_null = bio_run - pat_run
    asym_raw_null = bio_raw - pat_raw
    null_diff = asym_run_null - asym_raw_null

    n_perm = len(null_diff)
    p_two = float((np.abs(null_diff) >= abs(obs_diff)).sum() + 1) / (n_perm + 1)

    return {
        'run_id': run_id, 'view': view, 'raw_view': raw_view, 'bio_label': bio_label,
        'asym_run_obs': float(asym_run),
        'asym_raw_obs': float(asym_raw),
        'obs_diff': float(obs_diff),
        'direction': 'toward_symmetric' if obs_diff > 0 else 'away_from_raw',
        'null_diff_mean': float(null_diff.mean()),
        'null_diff_std': float(null_diff.std()),
        'p_two_sided': p_two,
        'n_perm': int(n_perm),
        'status': 'ok',
    }


def paired_asym_test(run_a: str, run_b: str, view: str,
                      bio_label: str = 'TIME') -> dict:
    """paired-perm test on (bio_delta - patient_delta) difference between two runs.

    returns dict with status='ok' and stats, OR status with diagnostic.
    """
    nulls_a = load_nulls(run_a)
    nulls_b = load_nulls(run_b)
    if nulls_a is None or nulls_b is None:
        return {'status': 'nulls_missing', 'run_a': run_a, 'run_b': run_b,
                'view': view, 'bio_label': bio_label}

    summary_a = load_summary(run_a)
    summary_b = load_summary(run_b)

    bio_null_a = get_null_array(nulls_a, view, bio_label, 'biology')
    bio_null_b = get_null_array(nulls_b, view, bio_label, 'biology')
    pat_null_a = get_null_array(nulls_a, view, 'patient_id', 'patient')
    pat_null_b = get_null_array(nulls_b, view, 'patient_id', 'patient')
    if any(x is None for x in [bio_null_a, bio_null_b, pat_null_a, pat_null_b]):
        return {'status': 'test_not_found', 'run_a': run_a, 'run_b': run_b,
                'view': view, 'bio_label': bio_label}

    bio_obs_a = get_obs_delta(summary_a, view, bio_label, 'biology')
    bio_obs_b = get_obs_delta(summary_b, view, bio_label, 'biology')
    pat_obs_a = get_obs_delta(summary_a, view, 'patient_id', 'patient')
    pat_obs_b = get_obs_delta(summary_b, view, 'patient_id', 'patient')

    asym_obs_a = bio_obs_a - pat_obs_a
    asym_obs_b = bio_obs_b - pat_obs_b
    obs_diff = asym_obs_a - asym_obs_b

    # paired null on asym diff (shared seed)
    bio_diff_null = bio_null_a - bio_null_b
    pat_diff_null = pat_null_a - pat_null_b
    asym_diff_null = bio_diff_null - pat_diff_null

    n_perm = len(asym_diff_null)
    p_two = float((np.abs(asym_diff_null) >= abs(obs_diff)).sum() + 1) / (n_perm + 1)

    return {
        'run_a': run_a, 'run_b': run_b, 'view': view, 'bio_label': bio_label,
        'asym_obs_a': float(asym_obs_a),
        'asym_obs_b': float(asym_obs_b),
        'obs_diff': float(obs_diff),
        'null_diff_mean': float(asym_diff_null.mean()),
        'null_diff_std': float(asym_diff_null.std()),
        'p_two_sided': p_two,
        'n_perm': int(n_perm),
        'status': 'ok',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+',
                    default=['R1', 'R2', 'R3', 'R4', 'R6', 'B1', 'B2'])
    ap.add_argument('--views', nargs='+', default=['z_he', 'z_mean'])
    ap.add_argument('--bio-label', default='TIME')
    args = ap.parse_args()

    print(f'=== paired-perm test on (bio - patient) delta difference ===')
    print(f'  runs:  {args.runs}')
    print(f'  views: {args.views}')
    print(f'  label: {args.bio_label}\n')

    rows = []
    for ra, rb in combinations(args.runs, 2):
        for view in args.views:
            r = paired_asym_test(ra, rb, view, args.bio_label)
            rows.append(r)

    df = pd.DataFrame(rows)
    ok = df[df['status'] == 'ok'].reset_index(drop=True)
    bad = df[df['status'] != 'ok']
    if len(bad):
        print(f'WARN: {len(bad)} pairs skipped (missing nulls / test): '
              f'{sorted(bad["status"].unique())}')

    if len(ok) == 0:
        print('no tests produced - check that biology_nulls.parquet exists for all runs.')
        return

    # sort by p_two for headline reading
    ok_sorted = ok.sort_values('p_two_sided').reset_index(drop=True)
    print('top 10 most-asymmetric-compression-different pairs:')
    cols = ['run_a', 'run_b', 'view', 'asym_obs_a', 'asym_obs_b',
            'obs_diff', 'p_two_sided', 'n_perm']
    print(ok_sorted[cols].head(10).round(5).to_string(index=False))
    print()

    # the load-bearing pair: R1 vs R4 on z_he and z_mean
    print('headline: R1 vs R4 (the report\'s mechanism claim):')
    headline = ok[((ok['run_a'] == 'R1') & (ok['run_b'] == 'R4')) |
                   ((ok['run_a'] == 'R4') & (ok['run_b'] == 'R1'))]
    print(headline[cols].round(5).to_string(index=False))
    print()

    out_path = RUNS_ROOT / 'eval' / 'compare' / 'paired_asym_tests.parquet'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok.to_parquet(out_path)
    print(f'saved -> {out_path.relative_to(ROOT)}')

    # per-run vs raw_he and vs raw_st (one-sample paired-perm)
    print('\n=== per-run vs raw asym (one-sample, paired-perm) ===')
    raw_rows = []
    for r in args.runs:
        for view in args.views:
            for raw_view in ['raw_he', 'raw_st']:
                raw_rows.append(per_run_vs_raw_asym(r, view, raw_view, args.bio_label))
    raw_df = pd.DataFrame(raw_rows)
    raw_ok = raw_df[raw_df['status'] == 'ok'].reset_index(drop=True)
    cols2 = ['run_id', 'view', 'raw_view', 'asym_run_obs', 'asym_raw_obs',
             'obs_diff', 'direction', 'p_two_sided']
    print('z_he view, raw_he reference (the load-bearing comparison):')
    sel = raw_ok[(raw_ok['view'] == 'z_he') & (raw_ok['raw_view'] == 'raw_he')]
    print(sel[cols2].round(5).to_string(index=False))
    print()
    print('z_mean view, raw_he reference:')
    sel = raw_ok[(raw_ok['view'] == 'z_mean') & (raw_ok['raw_view'] == 'raw_he')]
    print(sel[cols2].round(5).to_string(index=False))

    out_path2 = RUNS_ROOT / 'eval' / 'compare' / 'per_run_vs_raw_asym.parquet'
    raw_ok.to_parquet(out_path2)
    print(f'\nsaved -> {out_path2.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
