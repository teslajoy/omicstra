# MODELS.md

training provenance, architecture, tissue coverage, and validation status for
encoders and modules used in omicstra.

this file exists so that agents and users running the pipeline on new tissue
types can assess whether an encoder is appropriate before trusting its
embeddings. without this, the pipeline will produce embeddings that cluster
by slide noise rather than biology, with no diagnostic signal.

---

## foundation model encoders

### UNI2-h (H&E histopathology)

| field | value |
|---|---|
| output dim | 1536 |
| architecture | ViT-H/14 (DINOv2, 350k WSIs) |
| model size | 681M params |
| training data | Mass General Brigham + TCGA, 200M+ pathology images, 100k+ WSIs |
| tissue coverage | broad pan-cancer: breast, lung, colon, prostate, kidney, liver, skin |
| known gaps | rare cancers, non-oncology tissue, pediatric |
| TNBC status | supported - TCGA-BRCA in training |
| robustness (PathoROB) | mid-tier: RI=0.836 (TCGA), 0.544 (Camelyon), 0.923 (Tolkach). after ComBat+Reinhard: 0.870, 0.931, 0.963. encodes center signatures in early PCs. Komen et al. 2025 |
| L2 normalization | none - raw CLS tokens |
| role in omicstra | primary H&E encoder; histology agent tool |
| reference | Chen et al. 2024, Nat. Med. 30, 850-862 |

### Virchow2 (H&E histopathology, ablation)

| field | value |
|---|---|
| output dim | 1280 |
| architecture | ViT-H/14 (DINOv2, 3.1M WSIs) |
| model size | 632M params |
| training data | 3.1M slides, Memorial Sloan Kettering |
| tissue coverage | broad pan-cancer, MSK case mix weighted toward adult solid tumors |
| known gaps | non-cancer tissue, rare cancers, pediatric |
| TNBC status | supported - MSK breast cases in training |
| robustness (PathoROB) | Pareto-optimal (with Atlas): RI=0.848 (TCGA), 0.806 (Camelyon), 0.955 (Tolkach). most robust SSL model tested. Komen et al. 2025 |
| role in omicstra | drop-in ablation encoder via pluggable agent tool interface (H1 encoder sensitivity). robustness-motivated: if UNI2-h alignment shows center-driven clustering, Virchow2 swap is justified |
| reference | Vorontsov et al. 2024, arXiv:2408.00738 |

### Novae GNN (spatial transcriptomics)

| field | value |
|---|---|
| output dim | 64 (novae_latent) |
| architecture | CellEmbedder (512-d) -> GAT encoder (64-d) -> SwavHead (training only) |
| model size | 32M params |
| training data | image-based ST only: MERSCOPE, Xenium, CosMX - ~78 slides, ~30M cells, 18 tissues |
| tissue coverage | brain, intestine, liver, lymph node, skin - NOT Visium/VisiumHD |
| known gaps | trained on subcellular-resolution image-based ST, not spot-based platforms |
| TNBC status | validated_with_constraints (see validation results below) |
| input requirement | raw counts preferred; gene symbols (not Ensembl); obsm['spatial'] in microns |
| L2 normalization | none - raw GAT output (mean norm=2.618, std=0.52). LayerNorm required at MLP input |
| batch correction | native - suppresses patient/batch signal by design. within/between gap is not a valid QC metric |
| spatial graph | novae.spatial_neighbors(adata) - Delaunay by default. consider radius cutoff for spot-based data |
| multimodal capability | natively supports H&E foundation model features as node-level attributes (early fusion); omicstra externalizes integration strategy to the orchestration layer, treating fusion mode as an experimental variable |
| role in omicstra | primary ST encoder; spatial transcriptomics agent tool |
| reference | Blampey et al. 2025, Nat. Methods; doi:10.1038/s41592-025-02899-6 |
| HuggingFace | MICS-Lab/novae-human-0 |

#### TNBC validation results (Wang et al. dataset)

```
novae_tnbc_status: validated_with_constraints
spatial_criterion: PASS (mean rho=-0.380, all p~0)
gap_criterion: not_applicable (batch_correction_active)
alignment_training_set: [TNBC1_CN1_C1, TNBC3_CN2_C1, TNBC83_CN42_C1]
excluded: TNBC55_CN28_C1 (hvg_pca_fallback, 196 median genes)
deprioritized: TNBC68_CN34_D2 (low_gene_count=873, rho=-0.254)
platform_mismatch: 200um spots vs subcellular training - documented, not blocking
```

---

## non-foundation-model encoders

### scVI (ST fallback)

| field | value |
|---|---|
| output dim | configurable (default 10-128) |
| type | variational autoencoder for scRNA-seq / ST |
| tissue coverage | tissue-agnostic — trained per dataset, not pretrained |
| input requirement | raw counts preferred; can handle normalized |
| use case | fallback ST encoder when spatial_autocorrelation_pass: false or Novae embeddings collapse |
| reference | Lopez et al. 2018 |
| note | not in original proposal — added during EDA as empirical fallback |

### PCA on HVGs (baseline)

| field | value |
|---|---|
| output dim | configurable (typically 50) |
| type | linear dimensionality reduction on highly variable genes |
| tissue coverage | tissue-agnostic |
| input requirement | normalized counts |
| use case | minimum viable ST baseline for ablation; fallback if Novae and scVI both fail |

### gpath2vec (pathway embeddings)

| field | value                                                                                                                        |
|---|------------------------------------------------------------------------------------------------------------------------------|
| output dim | configurable                                                                                                                 |
| type | gene-set to pathway-level embedding vectors encoding enrichment strength and inter-pathway network topology                  |
| source | Reactome functional interactions w upper / lower level pathways                                                              |
| target pathways (H3) | TGF-β Signaling, Immune System, Extracellular Matrix Organization, Cell Cycle, Programmed Cell Death                         |
| use case | H3 evaluation — correlate pathway embeddings with shared latent dimensions via CCA; generate pathway-morphology spatial maps |
| contingency | if spot-level embeddings too noisy, aggregate to spatial neighborhood level (k=15 neighbors)                                 |
| reference | Sanati 2024, github.com/teslajoy/gpath2vec                                                                                   |

---

## alignment module

### InfoNCE contrastive alignment (2-layer MLP projection heads)

not an encoder — operates on encoder outputs. documented here for provenance.

| field | value |
|---|---|
| shared latent dim | 512 |
| H&E projection | Linear(1536, 512) -> GELU -> Linear(512, 512) |
| ST projection | LayerNorm(64) -> Linear(64, 512) -> GELU -> Linear(512, 512) |
| loss | InfoNCE, temperature τ = 0.07, symmetric, averaged across directions |
| positive pairs | co-registered H&E tile + ST spot from same tissue location |
| hard negatives | within-slide, different tissue compartments; cross-patient negatives excluded |
| training | batch size 256, max 100 epochs, early stopping on validation cosine similarity (patience=10) |
| split | 85/15 train/test, stratified by patient |
| contingency | if below expected range, increase to 3-layer MLP with curriculum learning |
| precedent | CONCH (Lu et al. 2024) — InfoNCE for pathology image + clinical text, 14 downstream tasks; Lazard et al. 2025 — contrastive alignment in unimodal histopathology |
| references | Oord et al. 2018 arXiv:1807.03748; Lu et al. 2024 Nat. Med. 30, 863-874; Lazard et al. 2025 arXiv:2508.05084 |

---

## H1 baselines

all four must be computed before writing winner.json.

| baseline | description |
|---|---|
| InfoNCE contrastive | primary strategy — late interaction MLP projection heads |
| CCA projection | canonical correlation analysis projection into shared space |
| late fusion concatenation | concatenate normalized H&E + ST embeddings, no alignment |
| unaligned concatenation | raw concatenation of H&E (1536-d) + ST (64-d) = 1600-d, no normalization |

---

## tissue-model compatibility matrix

| tissue | UNI2-h / Virchow2 (H&E) | Novae (ST) | verdict |
|---|---|---|---|
| breast (TNBC, ER+) | validated | uncertain | seed cohort — validate Novae UMAP first |
| colon / CRC | validated | validated (intestine in training) | safer combination |
| brain / GBM | validated | validated (brain in training) | safer combination |
| lung | validated | uncertain | validate Novae |
| pancreas | validated | uncertain | validate Novae; pilot cosine >0.70 on separate H&E-ST dataset |
| liver | validated | validated (liver in training) | safer combination |
| lymph node | validated | validated (in training) | safer combination |
| skin | validated | validated (in training) | safer combination |
| rare / pediatric | risky | risky | out of scope v1 |

"validated" = tissue type confirmed in encoder training data.
"uncertain" = tissue not confirmed; must run validation protocol before alignment.
"risky" = tissue unlikely to be represented; expect poor embeddings.

---

## validation protocol for uncertain tissue-model combinations

when model_tissue_fit[encoder].fit == "uncertain":

1. run encoder on the dataset
2. compute UMAP of embeddings colored by known biological labels (subtype, archetype, condition)
3. check for embedding collapse (all points within 2 std of centroid)
4. check for batch-driven clustering (silhouette by technical variable > biological variable)
5. if collapse or batch dominance: do not use this encoder; fall back per PLAN_RULES.md section 4.1
6. if embeddings show meaningful biological structure: upgrade status to "validated" in this table
7. record result in projects/{project_id}/eda_summary.json under model_tissue_fit