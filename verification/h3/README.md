# H3 pathway CCA verification

verifies the ported H3 pathway CCA chain against the original script
(`scripts/per_pathway_cca.py`) and the historical artifact
(`per_pathway_cca.parquet`).

## scope

- run: R4_v3
- views: z_he, z_st, z_mean
- signals: 5 AUCell pathway scores (TGF-beta_Signaling, Immune_System,
  ECM_Organization, Cell_Cycle, Apoptosis)
- split: shuffle+slice, seed 42, n_held=3 -> held=['52','53','64']
- permutations: 50 (within-cohort + transfer), cached factorization
- FDR: BH at alpha=0.05, global family = 15 (3 views x 5 pathways)

## results

111 PASS, 0 FAIL, 55 UNRESOLVED.

### PASS (111)

- held-out patient identity matches script exactly
- full-cohort CCA (script vs historical): exact (d=0)
- transfer r_test (script vs historical): exact (d=0)
- z_mean view: package matches historical to machine epsilon
- permutation null statistics: matched draws produce matched nulls
- chain execution: all 6 EVALUATE_STEPS complete, H3 verdict = biological

### UNRESOLVED (55) - single identified cause

all 55 from float32-vs-float64 centering difference:

- the script centers Z in float32 then upcasts to float64 for lstsq
- the package upcasts to float64 before centering
- max difference: ~8e-6 on r_test for z_he and z_st views
- z_mean matches to ~1e-10 (both paths compute it in float64)

this is a numerical precision difference in preprocessing, not an algorithm
disagreement. the script's float32 centering is the historical convention;
the package's float64 centering is more precise but produces different bits.

### not compared

- cross-run FDR family: script applies BH globally across all 120 cells
  (8 runs x 3 views x 5 pathways). package applies globally within one run
  (15 cells). per-run FDR is correct for the graph's single-run invocation;
  the 120-cell family is a multi-run aggregation the package does not perform

## files

- `verify_h3.py` - verification script
- `verification_report_R4_v3.json` - per-check results
- `chain_output_R4_v3.json` - full chain output for provenance