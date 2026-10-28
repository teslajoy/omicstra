# H2-C verification summary

## what was verified

three verification scripts, exercising the full H2-C scoring workflow
from raw features through historical artifact reproduction.

### 1. step-vs-measures consistency (`verify_h2c.py`, 63 checks)

runs the `confounder_suppression` step on real R4_v3 test-split data and
compares against direct calls to the underlying `measures/biology.py`
functions. covers label propagation, aggregation, permutation tests,
ratio/CI, BH-FDR, rho (geometric faithfulness), and G3 wording.

### 2. script-vs-package-vs-chain parity (`verify_script_parity.py`, 60 checks)

feeds the SAME cosine matrix to three code paths:
  (a) the preserved script functions (`scripts/eval_alignment_biology.py`)
  (b) the package measures (`measures/biology.py`)
  (c) the LCEL chain (`build_protocol(EVALUATE_STEPS)`)

every field (mean_same, mean_diff, delta, null_mean, null_std, z, p_value,
n_same_pairs, n_diff_pairs) is exact (diff=0) across all three paths.

### 3. full-cohort historical artifact reproduction (`verify_full_cohort.py`, 282 checks)

encodes ALL 259 subarrays from the model checkpoint (torch forward pass),
computes all 5 cosine matrices + 2 raw baselines, then compares:
  - **rho**: 6 tests (3 views x 2 raw baselines) — script vs package: exact.
    package vs `biology_rho.parquet`: exact (diff=0 on rho, pearson, z, p, n_pairs)
  - **permutation tests**: 25 tests (5 views x (3 labels + 1 patient)) —
    script vs package: exact. package vs `biology.parquet`: exact on all
    seed-independent fields (mean_same, mean_diff, n_same_pairs, n_diff_pairs)
  - seed-dependent fields (z, p, null_mean, null_std) are verified between
    script and package (exact at n_perm=10,000) but not against historical
    (which used n_perm=100,000). the cosine matrices are identical, so the
    only source of difference is the longer null distribution.

## geometric faithfulness (rho)

rho is now wired end-to-end. the adapter (`raw_baseline_cosines`) loads raw
features from niche parquets, mean-pools per subarray, and returns cosine
matrices. the step calls `geometric_faithfulness` for each (view, raw_baseline)
pair. test-split rho for z_he: 0.561 vs raw_he (z=7.7), 0.312 vs raw_st (z=4.7).

full-cohort rho matches `biology_rho.parquet` exactly.

## known precision difference

the cosine matrix differs by 7.15e-07 between the script (float32 matmul)
and the package (float64 matmul then cast to float32). documented precision,
not a logic error. the permutation tests receive the same matrix and agree
exactly.

## adapter

`biology_labels()` in `adapters/evaluation.py` reads a cohort-declared
column mapping from `project.json`. it validates intra-subarray consistency
(one patient per subarray) and reports missing patients by count rather than
silently dropping observations. 8 adapter tests use non-cohort-specific names.

`raw_baseline_cosines()` loads raw features (virchow2_niche, novae+gpath2vec)
from niche parquets for the same niches that survived label filtering. returns
None gracefully when the required columns or run config are absent.

## files

| file | role |
|---|---|
| `verify_h2c.py` | step-vs-measures internal consistency (63 checks) |
| `verify_script_parity.py` | script-vs-package-vs-chain matched-input parity (60 checks) |
| `verify_full_cohort.py` | full-cohort historical artifact reproduction (282 checks) |
| `chain_output_R4_v3.json` | chain step output on test split (gitignored) |
| `verification_report_R4_v3.json` | consistency report (gitignored) |
| `script_parity_report_R4_v3.json` | parity report (gitignored) |
| `full_cohort_report_R4_v3.json` | full-cohort report (gitignored) |