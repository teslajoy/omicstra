# omicstra · TNBC-92 v3 - calculation provenance map

the technical companion to `tnbc92_methods.md` (which teaches the *concepts*). this doc records **where every number comes from**: the script + function that computes each metric, the config that defines each run, and the artifact each result lands in. it exists because `scripts/` is gitignored (QA workspace, per project convention) - so this tracked doc is the durable lineage record for review and reproduction.

**grid:** `runs/tnbc-92_v3/` · **niche-join:** `data/embeddings/niches_v3/` (sha-locked gpath2vec `13985cbd...`) · **split:** patient-stratified 85/15, seed 42, shared across runs via `split.json`.

---

## stage 1 · niche-join (per-niche feature table)

| what | where |
|---|---|
| build script | `scripts/build_niche_join.py` |
| key functions | `process_subarray()` (per-niche row build), `parse_niche_id()` (subarray::spot key), `niche_mean_pool()` |
| inputs | `data/embeddings/{virchow2_niche, virchow2_cell, novae_niche_full}/`, gpath2vec parquet, `biological_signals/{mc_labels,mc_weights,tls_scores,morphology_labels}.tsv`, `clinical/Clinical.RDS` |
| output | `data/embeddings/niches_v3/{subarray}.parquet` + `manifest.json` (records gpath2vec sha256, git commit, mc_label coverage) |
| invocation | `python scripts/build_niche_join.py --gpath2vec-parquet data/embeddings/gpath2vec/fisher_madmean_low_dim512_e5_s1234/fisher_madmean_low_cluster_embeddings.parquet --out-dir data/embeddings/niches_v3` |

## stage 2 · alignment runs

| run | script | config | output artifacts |
|---|---|---|---|
| R1-R4, R6 | `scripts/align.py` | `configs/v3/{R1,R2,R3,R4,R6}_v3.json` | `runs/tnbc-92_v3/{run}_v3/{embeddings_test.parquet, metrics_h1_raw.json, run_config.json, split.json, training_log.csv, checkpoint.pt}` |
| **R5** (AnInfoNCE) | `scripts/align.py` | `configs/v3/R5_v3.json` | same; AnInfoNCE wiring = `Config.anisotropic`, `aninfonce_loss()`, `LateFusion(anisotropic=)` |
| B1/B2/B3 | `scripts/align_classical.py` | `--baseline cca\|procrustes\|unaligned` | same artifact schema, closed-form (no checkpoint/training_log) |
| B4 | `scripts/align_b4.py` | `--reference-split runs/tnbc-92_v3/R1_v3/split.json` | random-init MLP, `run_config.json` records `training_steps: 0` |
| model classes | `align.py`: `LateFusion`, `CrossAttnFusion` (R4), `MLPBlock` | | `attention_weights` column written by `CrossAttnFusion` only (R4) |
| losses | `align.py`: `infonce_loss`, `supcon_loss`, `barlow_loss`, `aninfonce_loss` | | |

## stage 3 · metrics - function -> algorithm -> artifact

| metric | function (file) | algorithm | lands in |
|---|---|---|---|
| **AUC** | `retrieval_metrics()` (`align.py`) | sklearn `roc_auc_score` on matched-diagonal vs sampled-mismatched cosines | `runs/.../{run}/metrics_h1_raw.json` |
| **R@K, MRR, median rank, alignment gap** | `retrieval_metrics()` (`align.py`) | argsort ranking of self in cosine matrix | same json |
| **CKA** | `cka_linear()` (`align.py`) | linear-kernel HSIC ratio on centered z_he vs z_st | same json (`cka_before/after`) |
| **H2-A archetype ARI + silhouette** | `cluster_quality()` (`eval.py`) | sklearn `KMeans(9 or 14, n_init=10, seed42)` -> `adjusted_rand_score` + `silhouette_score` | `runs/tnbc-92_v3/eval/H2/summary.json` |
| **H2-A mc_megacluster ARI** | `cluster_quality()` (`scripts/_scratch/eval_h2_mc_coherence.py`) | KMeans(14) vs `mc_megacluster`, z_he | `runs/tnbc-92_v3/eval/H2/mc_coherence.parquet` |
| **H2-A kNN purity** | `knn_purity()` (`scripts/_scratch/eval_h2_mc_diagnostics.py`) | sklearn `NearestNeighbors` (cosine via L2+euclidean), fraction of k-NN sharing mc | `runs/tnbc-92_v3/eval/H2/mc_diagnostics/knn_purity.parquet` |
| **H2-A linear probe** (R4 rescue) | `linear_probe()` (`eval_h2_mc_diagnostics.py`) | sklearn `LogisticRegression` (lbfgs, C=1, 14-class), patient-stratified 11/3, `StandardScaler` | `.../mc_diagnostics/linear_probe.parquet` |
| **H2-A rank-matched KMeans** | `run_rank_matched_diagnostic()` (`eval_h2_mc_diagnostics.py`) | project z_he to top-3 PCs (SVD), recompute ARI | `.../mc_diagnostics/rank_matched_kmeans.parquet` |
| **H2-B.1 compartment Welch z** | `welch_z()` (`scripts/_scratch/eval_h2_compartment_welch.py`) | unpooled-variance z on per-compartment matched-pair cosine aggregates | `runs/tnbc-92_v3/eval/H2/compartment_welch_contrasts.parquet` (from `compartment_cosine.parquet`, written by `eval.py::eval_h2_compartment_cosine`) |
| **H2-B.2 TLS-signature z_TLS** | `univariate_cca_proj()` + perm loop (`scripts/_scratch/eval_h2_tls_signature.py`) | lstsq canonical direction on z_he vs continuous `tls` score, cross-patient 11/3, 500-perm null, `bh_fdr()` | `runs/tnbc-92_v3/eval/H2/tls_signature_cca.parquet` |
| **H2-C bio/patient ratio** | `eval_alignment_biology.py` (perm z-tests) | matched-null permutation z on same-vs-diff-label cosine deltas for TIME/MC_global/MC_tumor and patient_id | `runs/tnbc-92_v3/{run}/eval/biology.parquet` (+ `biology_rho`, `biology_nulls`) |
| **H3 pathway CCA z_A (AUCell arm)** | `univariate_cca()` + perms (`scripts/eval_h3_pathway_cca.py`, `_perms.py`) | lstsq+pearsonr, option-A cross-patient 11/3, perm null, BH-FDR | `runs/tnbc-92_v3/eval/H3/pathway_cca/{per_pathway_cca,perm_nulls}.parquet` |
| **H3 pathway CCA z_A (gpath2vec arm, headline)** | `scripts/eval_h3_pathway_cca_gpath2vec_v2.py` | niche-cosine to embedded Reactome subtree, same option-A protocol; sha-locked embeddings | `runs/tnbc-92_v3/eval/H3/pathway_cca_gpath2vec_v3/{per_pathway_cca,perm_nulls,provenance}.parquet/json` |
| **BH-FDR** | `bh_fdr()` (in the tls / perms scripts) | Benjamini-Hochberg over each declared test family | inline in the above parquets (`q_bh`, `fdr_A`, `sig_*` columns) |

## the scorecard + ROC figures (notebook §11, §12)

| figure | reads from | built by |
|---|---|---|
| §11 scorecard (5 panels) | the 5 eval sources above (H1 json, mc_coherence, tls_signature_cca, biology.parquet, pathway_cca_gpath2vec_v3) | `notebooks/final/05_summary_umaps_v3.ipynb` §11 cell (reproducible via `scripts/_scratch/build_v3_summary_notebook.py`) |
| §12 H1 ROC curves | each run's `embeddings_test.parquet` (recompute matched/mismatched cosines) + AUC from `metrics_h1_raw.json` | §12 cell, same build script |

## CLI conventions (all eval scripts)

every eval script takes `--runs-dir runs/tnbc-92_v3 --niches-dir data/embeddings/niches_v3 --runs R1_v3,R2_v3,...` (click-based, per memory `feedback_cli_use_click`). v1 defaults are preserved when args are omitted, so the same scripts reproduce the v1 grid.

## reproduce a number end-to-end (worked example: R4 AUC 0.859)

1. `data/embeddings/niches_v3/` exists (stage 1) ->
2. `python scripts/align.py --config configs/v3/R4_v3.json --runs-root runs/tnbc-92_v3` writes `runs/tnbc-92_v3/R4_v3/metrics_h1_raw.json` ->
3. `["auc"]` field = 0.8591, computed by `retrieval_metrics()` via `roc_auc_score` on matched vs sampled-mismatched cosines of `embeddings_test.parquet`'s `z_he`/`z_st`. the ROC curve for it is notebook §12.

---

see `tnbc92_methods.md` for what these metrics *mean* and `tnbc92_results_summary.md` for the *values*.