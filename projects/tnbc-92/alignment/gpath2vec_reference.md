# gpath2vec validation reference

reference doc for the upstream gpath2vec validation framework that we are
about to mirror at the alignment layer. this is the template for
`scripts/eval_alignment_biology.py`.

## repo boundary

- **scripts** live in `/Users/sanati/BForePC/gpath2vec/scripts/` (use gpath2vec internals)
- **outputs** live here in omicstra at `data/embeddings/biological_signals/gpath2vec_output/full_cohort_tf_low/`
- gpath2vec is the method; omicstra is the dataset + application

## scripts in /Users/sanati/BForePC/gpath2vec/scripts/

| script | purpose |
|---|---|
| `resume_full_cohort.py` | resume full-cohort embedder training from network.pkl after the crash. runs walks + word2vec + saves embeddings.pkl, model.pt, cluster_embeddings.parquet. CLI args for epochs/dims/walks |
| `umap_cluster_embeddings.py` | 2D UMAP of 286k niches colored by patient. matplotlib PNG. first pass |
| `umap_pathway_category_2d.py` | 2D/3D UMAP colored by pathway_category (reactome parent of lowest-fdr sig pathway, filtered to variable categories). `--n-components 2 or 3`. plotly HTML |
| `similarity_heatmap.py` | 280x280 subarray (or 94x94 patient) cosine-sim heatmap with hierarchical clustering, top+side dendrograms via plotly.figure_factory, patient color strip. `--group-by subarray or patient` |
| `he_umap_linked.py` | linked H&E + UMAP per subarray. hover either plot to highlight corresponding niche on the other. legend click coupling, dashed niche circles at centroids, hover shows fdr + oddsratio. `--subarray` (multi), `--umap-dim 2/3`, `--color-by top_pathway/pathway_category` |
| `patient_mc_overlap.py` | 94 patient centroids UMAP colored by Bareche MC + TIME, silhouette scores for MC/TIME/archetype labels. the "is there MC signal at patient centroid level" test |
| `subarray_biology_check.py` | clustered subarray heatmap with patient + MC + TIME annotation strips, plus first pass of the biology permutation test (1k perms) |
| `embedding_vs_rawea_biology.py` | **the refactored validation tool**. 10^5 perms, cohort presets (TNBC configured, HEST/Ravi stubbed), rho with shuffled-matrix null, input validation, TNBC regression check, bonferroni |
| `null_distribution_check.py` | normality diagnostics on all 8 permutation nulls. skew/kurtosis/JB/KS, histogram + normal-fit PNG |

## package code we modified

`gpath2vec/embedder.py` - rewrote `train_embeddings()` to stream (target, context) pairs per walk chunk instead of materializing all ~2 billion pairs upfront. memory went from OOM to ~4-5GB peak. this is what let the full-cohort run complete.

## outputs at data/embeddings/biological_signals/gpath2vec_output/full_cohort_tf_low/

- **embeddings**: `embeddings.pkl`, `model.pt`, `cluster_embeddings.parquet` (286,233 x 512)
- **pre-embedding**: `network.pkl`, `enrichment.parquet`, `ea_matrix_fdr.parquet`, `ea_matrix_oddsratio.parquet`, `gene_sets.json`
- **UMAP coords**: `umap_cluster_embeddings_2d_coords.parquet`, `umap_pathway_category_2d_coords.parquet`, `umap_pathway_category_3d_coords.parquet`
- **UMAP plots**: `umap_pathway_category_2d.html`, `umap_pathway_category_3d.html`, `umap_cluster_embeddings_2d.png`
- **heatmaps**: `similarity_heatmap.html`, `similarity_heatmap_matrix.parquet`, `subarray_biology_heatmap.html`, `subarray_biology_heatmap_tests.parquet`
- **linked H&E**: `he_umap_linked_TNBC1_CN1_C1_lowpw.html`, `he_umap_linked_TNBC1_CN1_C2_lowpw.html`, `he_umap_linked_TNBC3_CN2_C1_lowpw.html`, `he_umap_linked_TNBC83_CN42_C1_lowpw.html`
- **patient-level MC**: `patient_mc_umap.html`, `patient_mc_umap_table.parquet`, `patient_mc_umap_silhouette.parquet`
- **validation results**: `embedding_vs_rawea_biology.parquet`, `embedding_vs_rawea_biology_rho.parquet`, `null_distribution_check.parquet`, `null_distribution_check.png`
- **logs**: `resume.log`, `umap.log`, `similarity_heatmap.log`, `he_umap_linked_batch.log`, `embedding_vs_rawea_biology.log`

## validation finding (recap)

gpath2vec embedding is faithful to raw pathway biology on TNBC cohort.

- geometric agreement: spearman rho = **0.955** (vs raw pathway activity)
- per-label biology z (10^5 perms, label-shuffle null, bonferroni alpha = 0.05/8):

| label | raw EA z | embedding z | direction |
|---|---|---|---|
| MC_global | 4.84 | **5.67** | biology up |
| MC_tumor | 4.08 | **4.53** | biology up |
| TIME | 4.84 | **6.52** | biology up |
| patient_id | 34.30 | **29.63** | patient down (-14%) |

asymmetric smoothing (raw delta ratio, emb / raw, lower = more compression):

| label | ratio |
|---|---|
| MC_global | 0.47 |
| MC_tumor | 0.45 |
| TIME | 0.54 |
| patient_id | **0.34** (compressed most) |

permutation nulls are mildly non-normal (biology nulls right-skewed +0.28-0.39, mild excess kurtosis +0.14-0.34). use permutation p, not parametric p. report z as standardized effect size only.

## what this gives us at the alignment layer

mirror `embedding_vs_rawea_biology.py` for `runs/tnbc-92/{run_id}/embeddings_test.parquet`. inputs:

- `z_he`, `z_st`, or `z_mean = (z_he + z_st) / 2` per niche (as the embedding under test)
- subarray-level cosine matrix derived from these
- biology labels (MC_global, MC_tumor, TIME, archetype) propagated from clinical / mc_labels
- patient_id propagated from mc_weights / clinical

outputs (per alignment run):

- biology z (one per label)
- patient z (matched-null permutation, preserves cluster sizes)
- emb / raw ratio per label (the asymmetric-smoothing diagnostic)
- comparable across runs -> direct ranking on disentanglement axis

planned location of the new script: `scripts/eval_alignment_biology.py` (omicstra side, not gpath2vec side - alignment is a downstream consumer of gpath2vec embeddings)