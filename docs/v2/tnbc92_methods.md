# omicstra · methods (teaching companion)
## TNBC-92 v3 - how every run and every number is produced

companion to `tnbc92_results_summary.md`. that doc reports *what the results are*; this doc teaches *what each run is*, *what every biological label means*, and *how each metric works* - the concept, the intuition, and the math - so someone new to spatial biology or to this project can follow the pipeline end to end and understand every number. (where the code lives is a one-line pointer at the bottom; this doc is for learning the methods, not navigating the repo.)

> **draft-for-voice-pass.** Claude-drafted scaffolding (per memory `feedback_writing_voice`); code references verified against the scripts in `scripts/`. voice pass before publishing.

---

## biological data H2 evaluates against (glossary)

H2 asks whether the aligned space preserves biological structure. "biological structure" is operationalized by several ground-truth labels from Wang et al. 2024, **at different biological resolutions**. the central subtlety (the evaluation-question concern) is that some labels are **patient-level** (one value per whole tumor) and some are **niche/spot-level** (vary within a tumor) - a niche-level objective must be tested with a niche-level label, or the metric silently measures patient identity.

### the ground-truth labels at a glance

| label | what it is, biologically | resolution | how derived | role in H2 |
|---|---|---|---|---|
| **9 spatial archetypes (SA)** | a whole-tumor profile classification, mappable from bulk-RNA molecular subtype (see SA table below) | **patient-level** (1 per patient) | cluster the 92 patients' global-pseudobulk ST into 9 groups | Part A **proposal-literal** label - diagnostic only |
| **mc_megacluster** (14 classes) | a recurring transcriptomic *tissue state* of a niche (the local cell-state program) | **per-spot / niche** | NMF, 14 factors, on the spot×gene matrix; each spot = its dominant factor | Part A **audit-correct** label |
| **17 morphological categories** | what the tissue *looks like* under H&E to a pathologist | **per-spot** | expert annotation (12 populated in test set: Tumor, Necrosis, High/Low-TIL stroma, Lymphoid nodule [TLS], Acellular stroma, Stroma cell, Lymphocyte, Fat tissue, Lactiferous duct, Vessels, in-situ) | Part B.1 |
| **TLS gene signature** | continuous "TLS-ness" - strength of the tertiary-lymphoid-structure program (Wang's 30-gene ST signature) | **per-spot / niche** (continuous) | gene-signature scoring on ST counts (`tls_scores.tsv`) | Part B.2 |
| **Bareche TIME / MC_global / MC_tumor** | patient molecular/immune subtypes from bulk RNA (TIME = tumor-immune-microenvironment class) | **patient-level** | Bareche bulk-RNA subtyping, propagated to all the patient's niches | Part C |
| **mc_weights** (14-d) | the *soft* version of mc_megacluster - the full NMF loading mixture per niche | per-niche (continuous) | the 14 NMF loadings, mean-pooled | **supervision only** (R2 positive pairs); never an H2 label or model input |
| **patient_id** | which of the 94 patients a niche came from | per-patient | metadata | the **confound axis** the construct-validity tests check against |

### the 9 spatial archetypes (Wang Fig 9) - and why they are patient-level

an archetype is built by collapsing all of a patient's spatial spots into one average (global pseudobulk) and clustering the 92 patients, so the spatial dimension is averaged away before clustering. Wang Fig 9 shows the SAs as a refinement of the **5 bulk-RNA molecular subtypes** (each SA is mappable from bulk RNA). consequence: **every niche in patient P inherits the same archetype A_P** (verified: 14/14 held-out patients carry exactly one archetype; NMI(archetype, patient_id) = 0.89). clustering niches by archetype therefore measures "which patient is this niche from," not "what tissue state is this niche." -> diagnostic only; the audit-correct Part A label is `mc_megacluster`.

| SA | characterization (Wang Fig 9) | bulk subtype origin | prognosis | actionable target |
|---|---|---|---|---|
| **SA1** | high immune, DNA repair deficiency, *KRAS*-dependent | IM | - | ICBs, PARPi |
| **SA2** | high proliferation, high *ERBB3* expression | IM | - | ICBs, ADCs (Trop-2, *ERBB3*) |
| **SA3** | high proliferation, high immune with *PD-L1* expression | BL | - | ICBs |
| **SA4** | high immune with **TLS signature** + *PD-L1*, DNA repair deficiency | BL | **good** | ICBs, PARPi, de-escalation |
| **SA5** | LAR-enriched, high metabolic pathways, high *ERBB2* | M / LAR | - | antiandrogens, PIK3CA/mTOR inh., ADCs (*ERBB2*) |
| **SA6** | high stromal, high proliferation, high *PDGF* | M | - | targeting PDGF signaling |
| **SA7** | high stromal, high EMT, high angiogenesis + adenosine, high proliferation | MSL | - | anti-CD73, antiangiogenic, targeting EMT |
| **SA8** | low immune, high *NECTIN4* expression | LAR | **poor** | ADCs (*NECTIN4*) |
| **SA9** | high stromal, low immune | LAR | - | targeting stroma |

patient counts (this cohort): SA4=16, SA7=15, SA2=14, SA5=13, SA6/SA9/SA3=8 each, SA8/SA1=6 each. **SA4 is the TLS archetype** (organized immune response, good prognosis - Wang's headline, and the tie-in to H2 Part B); **SA8** (NECTIN4-high) is the poor-prognosis archetype.

**"bulk subtype origin"** - the 5 pre-existing TNBC molecular subtypes defined from bulk RNA-seq (no spatial info); Fig 9 is a Sankey flowing each tumor from its bulk subtype to its SA. the mapping is **not 1:1** - one bulk subtype splits across several SAs (e.g. LAR -> SA5/SA8/SA9), and that splitting is the added value of the spatial archetypes. the column shows the dominant flow, not a strict assignment.

| abbr | bulk molecular subtype |
|---|---|
| **IM** | Immunomodulatory |
| **BL** | Basal-Like |
| **M** | Mesenchymal |
| **MSL** | Mesenchymal Stem-Like |
| **LAR** | Luminal Androgen Receptor |

**"prognosis"** - from Wang Fig 8g (iBCFS multivariate survival, combined cohort, adjusted for age/size/nodal status). only two SAs reached significance: **SA4 good** (HR 0.60, FDR 0.047), **SA8 poor** (HR 1.8, FDR 0.039); the other seven are "**-**" (no significant outcome association).

**"actionable target"** - Wang Fig 9 "therapeutic perspectives": the precision-medicine strategy each archetype's molecular features point to.

| abbr | therapy | rationale |
|---|---|---|
| **ICBs** | immune checkpoint blockade (anti-PD-1/PD-L1) | high-immune / PD-L1+ archetypes (SA1-4) |
| **PARPi** | PARP inhibitors | DNA-repair-deficient archetypes (SA1, SA4) |
| **ADCs** | antibody-drug conjugates | target a surface protein: Trop-2 / *ERBB3* (SA2), *ERBB2* (SA5), *NECTIN4* (SA8) |
| antiandrogens, PIK3CA/mTOR inh. | hormone + pathway inhibitors | LAR / AR+ archetype (SA5) |
| anti-CD73, antiangiogenic, EMT-targeting | stromal / mesenchymal therapies | SA7 |
| PDGF / stroma targeting | stromal therapies | SA6, SA9 |

these are all **whole-tumor** characterizations (origin, outcome, therapy) - none is a spatial/regional property, which is again why the archetype label is patient-level.

### TLS - tertiary lymphoid structure (the one genuine FTU)

an organized ectopic lymphoid aggregate inside the tumor: a B-cell follicle + adjacent T-cell zone + high endothelial venules, performing local antigen presentation and antibody maturation - an immune "mini lymph node" the tumor grows. it is a **functional tissue unit (FTU)**: recurring, sharply bounded, defined function. clinically, TLS density associates with immunotherapy response in TNBC (Wang derived a 30-gene TLS signature for exactly this). the dataset captures it two ways - the categorical **"Lymphoid nodule"** pathologist annotation (Part B.1) and the continuous **TLS gene signature** (Part B.2) - and it is the one compartment where H2 Part B's directional prediction held.

### mc_megacluster - the audit-correct niche-level biology

Wang's per-spot 14-class NMF. NMF decomposes the spot × gene matrix into 14 non-negative factors; each factor is a recurring cell-state program, and each spot is assigned the factor it loads on most. unlike archetype, these states **recur within and across patients** - a single tumor section contains several megaclusters - so clustering niches by `mc_megacluster` is a real niche-level biology test. (exact factor identities: Wang's mc characterization, `data/inputs/clustering/clustPrototypes/`.)

---

## 0 · the pipeline in one picture

```
 raw inputs                  per-niche features                shared 512-d space            evaluation
 ----------                  -----------------                 ------------------            ----------
 H&E WSI  --Virchow2-->  virchow2_niche (1280-d)  --\
 (frozen FM)             virchow2_cell_tokens (7x1280)  \                                    H1: retrieval (AUC, R@K, CKA)
                                                          >-- align.py --> z_he, z_st  -->   H2: clustering (ARI, probe, cosine)
 ST counts --Novae GNN-> novae_niche (64-d) ----------/   (MLP or cross-attn)                H3: pathway CCA (z_A, BH-FDR)
 (frozen FM)             gpath2vec_niche (512-d) ---/      InfoNCE / SupCon / Barlow
                         mc_weights_niche (14-d, supervision only)
                         tls (1-d), mc_megacluster (label), compartment (label)
```

**stage 1 - niche-join** (`scripts/build_niche_join.py`): for each subarray, build one row per niche (center spot + 6 spatial neighbors). mean-pool Novae over the 7 spots then z-score per subarray; keep Virchow2 niche-mean AND the 7 raw tile tokens; attach the v3 gpath2vec embedding; attach labels (mc_megacluster, compartment, tls, mc_weights). output: `data/embeddings/niches_v3/{subarray}.parquet`. niches with no v3 gpath2vec coverage are dropped (intersection split) -> 208,786 niches.

**stage 2 - alignment** (`scripts/align.py` for contrastive R-runs, `scripts/align_classical.py` for B1-B3, `scripts/align_b4.py` for B4): project both modalities into a shared 512-d space. patient-stratified 85/15 split (seed 42) -> train on 80 patients, test on 14. output per run: `runs/tnbc-92_v3/{run}/embeddings_test.parquet` with per-niche `z_he`, `z_st` (and `attention_weights` for R4).

**stage 3 - evaluation** (`scripts/eval.py` + the per-hypothesis scripts): compute H1/H2/H3 metrics on the 35,594 held-out test niches.

### the input features defined (what feeds the towers)

a **niche** = a center spot + its 6 spatial neighbors = 7 spots (~1200 cells). every run sees the same four per-niche features; they differ only in which they consume and how they fuse them.

| feature | dim | what it is | what it captures |
|---|---|---|---|
| **virchow2_niche** | 1280 | the 7 spots' Virchow2 foundation-model embeddings **mean-pooled** into one vector | the *average* H&E morphology of the niche. used by every run except R4. |
| **virchow2_cell_tokens** | 7 x 1280 | the same 7 per-spot Virchow2 embeddings kept **separate** (not pooled) | within-niche morphological detail - lets R4's cross-attention weight each spot instead of averaging. R4 only. |
| **novae_niche** | 64 | the Novae GNN (GAT) latent, mean-pooled over the 7 spots then z-scored per subarray | spatially-aware ST *expression*: what the niche expresses given its spatial transcriptomic neighborhood. |
| **gpath2vec_niche** | 512 | a metapath2vec embedding of the niche's pathway-activation profile (v3 build `fisher_madmean_low_dim512_e5_s1234`, sha-locked) | *which Reactome biological processes are on* in the niche. built by Fisher-enriching each niche's expression against Reactome, then metapath2vec random walks over the niche<->pathway graph - so niches with similar active-pathway programs land close together. complements novae (program-level, not raw-expression-level). |

**virchow2_niche vs the 7 tile tokens** is the single H&E difference across runs: pooling (everyone) throws away which of the 7 spots drove the signal; R4 keeps them addressable so attention can recover it. that trade is R4's whole bet, and it wins H1 + H3.

**the two `+` in the run table are different operations.** when a run's input reads `Virchow2 niche + novae+gpath2vec`:

- the **inner `+` (novae + gpath2vec)** is a true **concatenation** -> one 576-d ST feature vector (64 + 512). (`align.py:146`, `x_st = np.concatenate(...)`; written `⊕` in the run tables below.)
- the **outer `+` (Virchow2 ... + novae+gpath2vec)** is **not** a concatenation. the H&E vector and the ST vector are the two *sides* of the contrastive pair: each passes through its **own MLP tower** into the shared 512-d space, where InfoNCE/SupCon/Barlow pulls matched (H&E, ST) niches together. the two modalities are never stacked into a single vector.

---

## 1 · run definitions (all 10)

every run maps the same two inputs (H&E side, ST side) into a shared 512-d space; they differ in **loss**, **fusion architecture**, and **ST input composition**.

### the shared building block: `MLPBlock` (`align.py`)

**what an MLP is.** an **MLP** (multi-layer perceptron) is the simplest learnable neural network: a stack of `Linear` layers (each a learned weight matrix) with nonlinearities (ReLU) between them. an MLP turns one vector into another vector by applying `x -> Linear -> nonlinearity -> Linear -> ...`; the weights are learned from data. an **MLP tower** is one such stack used as a column that processes one modality end-to-end. in "late fusion" each modality gets its **own** tower with independent weights (no sharing): the H&E vector goes through `MLPBlock_HE`, the ST vector through `MLPBlock_ST`, each produces a 512-d output, and the contrastive loss aligns them in that shared space.

the specific `MLPBlock` used here:

```
LayerNorm(in) -> Linear(in, 512) -> ReLU -> BatchNorm1d(512) -> Dropout(0.3) -> Linear(512, 512) -> L2-normalize
```

- two `Linear` layers = two learned weight matrices.
- `LayerNorm` / `BatchNorm1d` stabilize training by normalizing activations.
- `Dropout(0.3)` randomly zeros 30% of activations during training (regularization).
- `L2-normalize` at the end puts every output on the unit sphere, so downstream "similarity" = cosine = dot product.

every learned run uses this block (one per modality for late fusion; R4 replaces the H&E tower with `CrossAttnFusion`).

### contrastive runs (trained with SGD, `align.py`)

| run | loss | fusion | ST input | plain-English what it is |
|---|---|---|---|---|
| **R1** | InfoNCE | late (2 independent MLPBlocks) | novae(64) ⊕ gpath2vec(512) = 576-d | the standard CLIP-style baseline. H&E niche-mean -> MLP; ST -> MLP; pull matched (H&E, ST) niche pairs together, push cross-patient mismatches apart. |
| **R2** | SupCon | late | novae ⊕ gpath2vec | like R1 but the "correct match" target is *soft*: niches with similar `mc_weights` (Wang's 14-d cell-state mixture) are treated as partial positives, not just the exact diagonal. |
| **R3** | Barlow Twins | late | novae ⊕ gpath2vec | no negatives at all. drives the H&E×ST cross-correlation matrix toward the identity (matched dimensions correlated, off-diagonals decorrelated). |
| **R4** | InfoNCE | **cross-attention** | novae ⊕ gpath2vec | the only non-late run. the ST embedding *queries* the 7 raw H&E tile tokens (center + 6 neighbors) via attention, producing a weighted-sum H&E representation conditioned on ST. captures which tiles matter instead of mean-pooling them. |
| **R5** | **AnInfoNCE** | late | novae ⊕ gpath2vec | R1 + a learnable per-dimension temperature on the bilinear similarity (diagonal Mahalanobis). lets the loss weight some axes of the shared space harder than others. `init = 0 -> identical to InfoNCE at step 0`. |
| **R6** | InfoNCE | late | **novae only (64-d)** | the ablation: drop gpath2vec from the ST input. tests whether the H&E pathway signal needs a pathway anchor on the ST side. |

### baselines

| run | what it is | trained? | why it exists |
|---|---|---|---|
| **B1** | Canonical Correlation Analysis (CCA) - closed-form. find linear projections of H&E and ST with maximum cross-modal correlation in whitened space. | closed-form (no SGD) | the proposal's named "CCA projection" baseline. does classical linear correlation suffice? |
| **B2** | Orthogonal Procrustes - PCA each modality to 512-d, then solve for the orthogonal rotation minimizing paired Frobenius distance. | closed-form | is a rigid rotation enough, without whitening? |
| **B3** | Unaligned PCA - independent per-modality PCA to 512-d, L2-norm, **no cross-modal step**. | closed-form | the proposal's "unaligned concatenation" sanity floor. any real method must beat this; if B3 scores high, the test is broken. |
| **B4** | Random-init MLP - R1's exact architecture, Xavier init, **zero training steps**. | not trained | the strict control: separates "is R1's gain from the *loss*?" from "is it from the *architecture shape*?" if B4 ≈ chance, the loss is doing the work. |

### the circular-dependency guard (why mc_weights is supervision-only)

`mc_weights_niche` (Wang's 14-d cell-state mixture) defines the *positive pairs* for R2's SupCon loss. it is therefore **never** put in the ST input vector - if it were both the input and the target, the model would predict its own input. `align.py` hard-asserts `mc_weights_niche ∉ st_features` (`program.md` constraint 3).

---

## 2 · how each metric works (concept, intuition, math)

each metric below is explained as a *method* - what it measures, the intuition, and the computation - not as a code location. (the functions that implement them are listed once at the end of this section for anyone who wants to read the source.)

### H1 retrieval metrics

H1 asks: can an H&E niche find its matched ST niche among all the others, in patients the model never trained on? all H1 numbers operate on the L2-normalized `z_he` and `z_st` matrices (N × 512) - every embedding is a unit vector, so similarity = cosine = dot product.

```
sim = z_he @ z_st.T            # (N, N) cosine-similarity matrix; entry (i,j) = cos(H&E_i, ST_j)
ranks = (-sim).argsort(axis=1).argsort(axis=1)[diag]   # 0-indexed rank of the matched ST for each H&E query
```

- **R@1 / R@5 / R@10** = fraction of queries whose matched ST falls in the top 1 / 5 / 10. `(ranks < K).mean()`.
- **MRR** (mean reciprocal rank) = `mean(1 / (ranks + 1))`.
- **median rank** = `median(ranks + 1)`. random baseline ≈ N/2 = 17,797.
- **alignment gap** = `mean(diagonal cosines) - mean(off-diagonal cosines)`. positive = matched pairs systematically closer.
- **AUC** - the headline H1 number. computed with **scikit-learn `roc_auc_score`**, NOT a nearest-neighbor method:
  ```
  matched     = diag(sim)                      # cosine of each true (H&E, ST) pair
  mismatched  = sim[off-diagonal], sampled to 10x len(matched) for speed
  labels      = [1]*len(matched) + [0]*len(mismatched)
  scores      = concat(matched, mismatched)
  auc         = roc_auc_score(labels, scores)
  ```
  interpretation: probability that a random matched pair has higher cosine than a random mismatched pair. 0.5 = chance, 1.0 = perfect. it's a ranking metric over cosines - no clustering, no kNN.

### H1 representation structure (CKA)

**CKA** (Centered Kernel Alignment, linear kernel) measures whether the two modality embeddings have the same *pairwise-similarity geometry*, even if individual pairs aren't matched:
```
Xc = X - X.mean(0); Yc = Y - Y.mean(0)          # center
num = ||Xc.T @ Yc||_F^2                          # HSIC(X, Y)
den = ||Xc.T @ Xc||_F * ||Yc.T @ Yc||_F
CKA = num / den                                  # 0 = unrelated geometry, 1 = identical
```
`cka_before` is computed on the raw inputs (virchow2_niche vs raw ST features); `cka_after` on the aligned `z_he` vs `z_st`.

### H2 Part A clustering (ARI, silhouette)

**ARI** (Adjusted Rand Index) on KMeans clusters vs a biology label:
```
km    = KMeans(n_clusters=14, n_init=10, random_state=42)   # sklearn
preds = km.fit_predict(z_he)
ari   = adjusted_rand_score(y_true_label, preds)             # sklearn; chance-corrected, 0=random 1=perfect
```
the label `y` is `mc_megacluster` (audit-correct) or `archetype` (proposal-literal, diagnostic only). **silhouette** = `silhouette_score` on a 5,000-niche subsample (cosine geometry). this is a clustering metric - KMeans assumes globular clusters, which is why R4's tight non-globular manifold scores low here even though its biology is linearly recoverable (see linear probe).

### H2 Part A diagnostics (kNN purity, linear probe, rank-matched KMeans)

three metric-independent cross-checks on the same `mc_megacluster` label:

- **kNN purity** (`knn_purity()`): L2-normalize `z_he`, fit **sklearn `NearestNeighbors`** (euclidean on unit vectors = cosine), for each niche check what fraction of its k nearest neighbors share its `mc_megacluster`. reported at k = 5/10/25. chance = 1/14 = 0.071. caveat: dominated by within-patient niche similarity - not patient-stratified, so it's patient-confounded.
- **linear probe** (`linear_probe()`): the metric-independent biology test. split the 14 test patients into 11 probe-train / 3 probe-test (patient-level, so no patient leaks across the split). `StandardScaler` on train, `LogisticRegression(max_iter=300, lbfgs, C=1.0)` 14-class, score accuracy + macro-F1 on the held-out 3 patients. chance = 0.071. **this is what shows R4 encodes mc biology at parity** even though its KMeans ARI is low - LogReg needs only linear separability, not globular clusters.
- **rank-matched KMeans**: project `z_he` to its top-3 PCs (SVD), recompute ARI. controls for R4's low effective dimension - if a wider run at rank-3 still beats R4 full, R4's deficit is encoding-shape, not just dimensionality.

### H2 Part B.1 compartment contrasts (Welch z-test)

per (run, compartment), `eval.py` aggregates matched-pair cosine `cos(z_he_i, z_st_i)` to `{n, mean, std}`. the **Welch z-test** (unpooled variance) on the 6 named contrasts:
```
z = (mean_distinct - mean_ambiguous) / sqrt(std_distinct^2 / n_distinct + std_ambiguous^2 / n_ambiguous)
```
pass = z ≥ +1.96 (distinct > ambiguous, the proposal direction); inverted = z ≤ -1.96. Welch (not pooled-variance) because compartments have very different cell counts and variance.

### H2 Part B.2 TLS gene-signature (univariate CCA)

direct test against the continuous per-niche TLS score (`tls_scores.tsv`, mean-pooled to niche). **univariate CCA** between `z_he` (512-d) and the 1-d TLS score:
```
w        = lstsq(Z_centered, y_centered)     # normal-equations solution for the canonical direction
w        = w / ||w||                          # unit-norm
proj     = Z_centered @ w
r        = pearsonr(proj, y_centered)         # the canonical correlation (1-d target -> reduces to pearson r)
```
fit on 11 train patients, evaluate the *frozen* `w` on 3 held-out patients. **permutation null**: shuffle the held-out TLS scores 500×, recompute r each time; `z_TLS = (r_test - null_mean) / null_std`. significance by **BH-FDR** across the run grid.

### H3 pathway CCA (cross-patient transfer)

same univariate-CCA machinery as B.2, but the 1-d target is a *pathway* signal (AUCell score, or gpath2vec niche-cosine-to-pathway-subtree). **two operationalizations**:
- **option B (fit-on-all)**: fit and evaluate on all test niches. proposal-literal but universally positive (doesn't discriminate).
- **option A (cross-patient)**: fit on 11 train patients, evaluate frozen `w` on 3 held-out patients. **this is the discriminating test.** z_A = held-out r in permutation-null std units.

**permutation null** (`eval_h3_pathway_cca_perms.py`): for speed, the canonical correlation = sqrt(R²) of the OLS regression, and R² = ||Qᵀy||² / ||y||² where Z = QR (skinny QR). each permutation is one matmul `Qᵀ y_shuffled`, amortizing Q across the 5 pathways. 500-1000 perms per cell. **BH-FDR** applied within each declared test family (e.g., the testable z_he pathways).

### BH-FDR (multiple-testing correction)

Benjamini-Hochberg: sort p-values ascending, multiply the k-th by n/k, enforce monotonicity from the top, clip to [0,1]. controls the false-discovery rate across a family of tests so that "X/N significant" means something across many simultaneous comparisons (e.g. the 120-cell H3 grid, or the per-run TLS grid).

### implementations (for the code-curious)

all metrics use standard scikit-learn (`roc_auc_score`, `KMeans`, `adjusted_rand_score`, `LogisticRegression`, `NearestNeighbors`) and scipy (`pearsonr`) - auditable building blocks. **the full calculation-provenance map - which script + function computes each metric, and which artifact each result lands in - is its own doc: [`tnbc92_provenance.md`](tnbc92_provenance.md).** (kept separate so this doc stays about the methods, not the repo layout.)

---

## 3 · the two pathway representations (biology, not code)

the project uses two different per-niche pathway encodings, and they capture different biology - this matters for reading H3.

| | **gpath2vec** (ST input feature) | **AUCell** (H3 evaluation readout) |
|---|---|---|
| pipeline | TF-gene filter -> MAD/mean variable-gene selection per niche -> Fisher enrichment vs Reactome -> metapath2vec random walks on the (niche, pathway) graph -> 512-d | rank all genes in a niche by expression -> for each Reactome gene-set, area under the recovery curve over the top ~5% (`decoupler dc.mt.aucell`) |
| biology captured | **regulatory-program neighborhood structure** - which TF-driven pathways co-fire, embedded so niches with similar regulatory state sit close | **direct pathway activity** - how strongly a specific gene-set's genes are over-expressed in a niche |
| gene selection | TF-restricted + most-variable (MAD/mean) | none; all genes ranked, gene-set decides what counts |
| emphasizes | discriminative regulatory state | abundant pathway-gene expression (rank-based) |
| per-niche output | 512-d dense embedding (geometry meaningful, dims not individually interpretable) | k-d score vector, one interpretable activity score per named pathway |

gpath2vec is what the H&E side learns to align *to* (it's on the ST input). AUCell is one of the *readouts* H3 tests the aligned space against. The H3 hypothesis is only meaningful because these are different views - the alignment is informative only if organizing one (gpath2vec on ST) produces a space that predicts the other (AUCell / TLS signature) in held-out patients.

---

## 4 · what each hypothesis tests

| hypothesis | the question (plain English) | metric |
|---|---|---|
| **H1** | can H&E find its matched ST niche in patients the model never saw? | AUC, R@K, MRR, median rank, alignment gap, CKA |
| **H2 Part A** | does the aligned space group niches by transcriptomic tissue state? | KMeans ARI + silhouette (vs mc_megacluster); kNN purity, linear-probe accuracy, rank-matched ARI |
| **H2 Part B.1** | do morphologically distinct compartments cohere more than ambiguous stroma? | Welch z on 6 named contrasts |
| **H2 Part B.2** | does the H&E view predict the continuous TLS gene signature cross-patient? | univariate CCA z_TLS + permutation null + BH-FDR |
| **H2 Part C** | does the space encode biology more than patient identity? | bio-z / patient-z ratio (permutation z-tests) |
| **H3** | do named Reactome pathways correlate with shared-latent directions, and transfer across held-out patients? | univariate CCA z_A (cross-patient) + permutation null + BH-FDR |

---

## 5 · ground-truth labels (the four proposal-named sources)

| proposal ground truth | dataset asset | resolution | used in |
|---|---|---|---|
| 9 spatial archetypes | Wang `Spatial archetypes_defined_on_ST_global_pseudobulk` | **patient-level** (1 per patient) | H2 Part A literal - diagnostic only (patient-confounded) |
| TNBC molecular subtypes per spot | `mc_labels.megacluster` (14-class NMF) | **per-spot / niche** | H2 Part A audit-correct (KMeans ARI, linear probe) |
| 17 morphological categories | `morphology_labels.tsv` (pathologist) | **per-spot** | H2 Part B.1 (compartment cosine) |
| TLS gene signatures scored spatially | `tls_scores.tsv` (continuous) | **per-spot / niche** | H2 Part B.2 (TLS signature CCA) |

the construct-validity correction in H2 Part A (archetype -> mc_megacluster) is a switch between two *proposal-named* labels at different resolutions - proposal-compliant, not a deviation.

---

## pointers

results companion: [`tnbc92_results_summary.md`](tnbc92_results_summary.md). construct-validity audit: `projects/tnbc-92/evaluation_question_audit.md`. agent design: `docs/mcp_agent_design.md`. code: `scripts/{build_niche_join,align,align_classical,align_b4,eval}.py` + `scripts/_scratch/eval_h2_*.py` + `scripts/eval_h3_pathway_cca*.py`.
