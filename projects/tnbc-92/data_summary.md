# tnbc-92 - data summary (v3 grid)

scaffold populated 2026-06-02 from `eda_summary.json`, niche-join manifests, and `runs/tnbc-92_v3/eval/*`. prose voice pass TBD.

## cohort

| field | value | source |
|---|---|---|
| dataset | Wang et al. 2024 TNBC | DOI 10.5281/zenodo.8135721 |
| platform | original Spatial Transcriptomics (Stahl 2016, KTH/STAB; not 10x Visium) | CLAUDE.md |
| patients on disk | 94 | `eda_summary.json:patients` |
| patients in publication | 92 (IDs 17/18 QC-failed) | `eda_summary.json:patients_in_publication` |
| samples (subarrays) | 282 | `eda_summary.json:samples` |
| subarrays with ST counts | 281 (1 excluded: CN26/D1, 8 selection spots) | `eda_summary.json:samples_with_counts` |
| matched H&E + ST subarrays | 260 (Wang co-registration) | demo / results_summary |
| matched scale-embeddings | 249 (intersection: Virchow2 niche ∩ Novae) | `eda_summary.json:scale_embeddings.matched.n_subarrays` |
| post-v3-gpath2vec niches | 208,786 (67,131 dropped: no significant pathway under MAD/mean gene selection) | niche-join manifest |
| held-out test patients | 14 | patient-stratified 85/15, seed 42 |
| held-out test subarrays | 38 | same split |
| held-out test niches | 35,594 | `runs/tnbc-92_v3/*/embeddings_test.parquet` |
| spot geometry | 100 um diameter, 200 um center-to-center, 1934 spots/array (full grid), ~1075 tissue-selected per subarray | CLAUDE.md |
| niche definition | center spot + 6 spatial neighbors (~1200 cells, ~500 um footprint) | program.md / methods.md |

## subarray funnel (the 282 -> 35,594 trace)

```
282  samples
 -1  CN26/D1 excluded (8 selection spots)
281  with ST counts
 -21 missing matched H&E or co-registration metadata
260  matched H&E + ST (the proposal cohort)
 -11 missing scale-embedding outputs (Virchow2 niche or Novae)
249  with both scale-embeddings (eda_summary.scale_embeddings.matched)
        |
        v
208,786 niches (post-v3-gpath2vec intersection; 67,131 niches dropped)
 -173,192 train (80 patients)
35,594  test (14 patients, 38 subarrays)
```

## modalities and encoders (v3 grid)

| modality | encoder | dim | role |
|---|---|---|---|
| H&E | Virchow2 (MSK Paige, frozen FM) | 1280 (per spot) | primary; chosen via niche-level MC linear probe vs UNI2 |
| H&E (alternative) | UNI2-h | 1536 | pluggable swap |
| H&E (R4 only) | Virchow2 per-spot tokens | 7 x 1280 | preserves within-niche detail for cross-attention |
| ST expression | Novae GNN (MICS-Lab, frozen) | 64 | spatially-aware niche-mean, z-scored per subarray |
| ST pathway | gpath2vec (ours) | 512 | metapath2vec on niche-Reactome graph; v3 build `fisher_madmean_low_dim512_e5_s1234`, sha-locked |
| biological labels | Wang clinical / mc_megacluster / mc_weights / TLS / compartment / archetype | various | see `tnbc92_methods.md` glossary |

## QC gates passed (EDA)

| check | value | gate |
|---|---|---|
| triple-negative confirmed | ESR1 2.6%, PGR 0.9% expression | PASS |
| Moran's I (top markers) | 5/5 markers >= 0.3 | PASS |
| batch silhouettes | plate -0.076, lane -0.081 (both negative -> no batch artifact) | PASS |
| count matrix raw status | log-normalized, NOT raw | flagged but usable |
| cross-modal registration | spot-H&E overlay confirmed on CN1/C1 | PASS |
| platform mismatch (Novae) | trained on MERSCOPE/Xenium/CosMX, applied to original ST | known risk; validated empirically (Spearman rho -0.38, comparable to Virchow2 -0.41) |

## v3 vs v1 - what changed

| dimension | v1 | v3 |
|---|---|---|
| gpath2vec build | irreproducible v1_legacy | `fisher_madmean_low_dim512_e5_s1234` (sha 13985cbd...) |
| niche-join coverage | 286k niches | 208,786 niches (intersection with sha-locked gpath2vec) |
| alignment grid | 9 runs (R1-R4, R6, B1-B4) | 10 runs (+ R5 AnInfoNCE + late) |
| eval scripts | argparse | click; `--runs-dir`/`--niches-dir`/`--runs` flags |
| CI on H1 metrics | not computed | Hanley-McNeil (AUC) + B=200 bootstrap (CKA), seed 42 |

## see also

- `eda_summary.json` - machine-readable EDA artifact (authoritative)
- `docs/v2/tnbc92_results_summary.md` - per-hypothesis verdicts and numbers
- `docs/v2/tnbc92_methods.md` - what every metric measures (teaching)
- `docs/v2/tnbc92_provenance.md` - script + artifact lineage per metric
- `docs/v2/tnbc92_routing_matrix.md` - task-conditional dispatch table