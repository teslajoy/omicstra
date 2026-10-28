# H2-C verification summary

## what was verified

matched-input comparison of the preserved script (`scripts/eval_alignment_biology.py`)
against the package measures (`measures/biology.py`) and the LCEL chain
(`build_protocol(EVALUATE_STEPS)`). all three paths receive the same
cosine matrix, patient array, and biology labels.

**60/60 checks pass.** every field (mean_same, mean_diff, delta, null_mean,
null_std, z, p_value, n_same_pairs, n_diff_pairs) is exact (diff=0) across
all three code paths on matched input.

## what is not verified

- **full-cohort reproduction.** the historical `biology.parquet` was computed
  on ~258 subarrays re-encoded from the model checkpoint. this comparison uses
  the 38-subarray test split because the adapter carries only `embeddings_test.parquet`.
  full-cohort reproduction requires re-encoding and is a separate task.

- **geometric faithfulness (rho).** deferred. requires raw baseline cosine matrices
  (pre-alignment) that the adapter does not carry. the step records
  `status: not_implemented` with a reason, not an unexplained empty dict.

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

## files

| file | role |
|---|---|
| `verify_h2c.py` | step-vs-measures internal consistency (51 checks) |
| `verify_script_parity.py` | script-vs-package-vs-chain matched-input parity (60 checks) |
| `chain_output_R4_v3.json` | chain step output on test split (gitignored) |
| `verification_report_R4_v3.json` | consistency report (gitignored) |
| `script_parity_report_R4_v3.json` | parity report (gitignored) |