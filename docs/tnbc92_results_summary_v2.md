# omicstra · project data: TNBC-92 (v2)

**Cohort**: Wang et al. 2024 TNBC (92 published patients, 94 on disk).
**Niche unit**: center spot + 6 spatial neighbors (~1,200 cells; ~100µm footprint).
**Niche-join geometry**: 260 subarrays, 286,250 niches (intersection of Virchow2-niche × Novae-niche × gpath2vec_niche).
**Evaluation split**: subarray-level, patient-stratified 85/15 → test = **45,661 niches / 14 held-out patients / 38 subarrays**.
**FDR control**: BH-FDR over the test families described per hypothesis.
**Version note**: v2 of this summary applies a numbers audit (May 2026) against the underlying parquets; previously reported "9.9% / 9480 per run" H3-DAG figures and "collagen biosynthesis (5.28)" headline hit are corrected here. v1 of this doc is superseded.

---

## method definitions

| Method | Definition | What it fits to the data | What it's there to falsify |
|---|---|---|---|
| R4 | InfoNCE cross-attention alignment. ST queries attend over H&E tile tokens (center + 6 spatial neighbors). Produces a weighted-sum H&E representation conditioned on ST. | learned MLP + cross-attention weights via SGD | "does tile-level interaction beat pooled late fusion?" |
| R1 | InfoNCE late-fusion alignment with independent modality projections (H&E niche → MLP, novae + gpath2vec → MLP). Standard CLIP-style contrastive baseline. | learned MLP weights via SGD | "does standard contrastive alignment work?" |
| R6 | R1 ablation: drops gpath2vec from the ST input (novae 64-d only). Tests whether H&E pathway signal depends on the gpath2vec anchor being present on ST. | learned MLP weights via SGD, ST input dim 64 instead of 576 | "does H&E pathway signal need a pathway anchor on ST?" |
| R2 | Supervised contrastive late-fusion using soft mc_weights similarity as targets. | learned MLP weights via SGD | "does biology-graded supervision beat 1-hot pairs?" |
| R3 | Barlow Twins late-fusion: redundancy-reduction loss, no explicit negatives. | learned MLP weights via SGD | "does negatives-free decorrelation match InfoNCE?" |
| B1 | Canonical Correlation Analysis. Classical linear, closed-form, whitened-space. | fits closed-form correlation directions to data | "do we need a learned model at all, or does CCA suffice?" |
| B2 | Orthogonal Procrustes. Per-modality PCA to shared_dim, then orthogonal rotation. | fits per-modality PCA + closed-form rotation | "is a closed-form rotation enough?" |
| B3 | Unaligned PCA. Independent per-modality PCAs with L2-norm, no cross-modal objective. Sanity floor. | fits per-modality PCA only; no cross-modal step | "sanity floor — is any number above this real?" |
| B4 | Random-initialized untrained MLPs. R1 architecture, Xavier init, no training. Strict baseline that isolates the contribution of contrastive training from the architecture alone. | **nothing — random projection, no fit** | "is R1's gain from the *loss*, or from the *MLP shape*?" |

Reading across the **"what it fits"** column: B1 fits cross-modal correlation, B2 fits orthogonal rotation, B3 fits per-modality variance only, R1/R2/R3/R4/R6 fit via SGD on contrastive objectives, **B4 fits nothing**. B4 is the only row in the grid whose encoder weights at inference time are exactly the random init — the unique control for separating "is the architecture doing the work?" from "is the contrastive loss doing the work?"

ST input composition (verified from each `runs/tnbc-92/{run}/run_config.json`, `st_features` field): R1, R2, R3, R4, B1, B2, B3 all ingest `novae_niche` (64-d) ⊕ `gpath2vec_niche` (512-d) = 576-d ST input. **R6 ingests `novae_niche` only** — the no-gpath2vec ablation. Full methods detail (why this composition, how niches are aggregated, how the streams are concatenated, gpath2vec v1 build, metric definitions) is in the **Methods** section at the end of this document.

---

## H1 — Cross-Modal Alignment

| Field | Detail |
|---|---|
| Proposal alignment | Aligning H&E and ST embeddings within a shared latent space via contrastive learning. |
| Evaluation question | Can H&E embeddings retrieve matched ST niches more accurately than unaligned or classical alignment? |
| Observed result | **R4 strongest: AUC 0.851, CKA 0.564.** R1 and R6 exceed the proposal target range (AUC 0.733–0.741). Classical baselines span 0.443–0.703; **B4 untrained sits at chance (0.491)**, confirming the architecture alone contributes none of the H1 gain. |
| Interpretation | Contrastive cross-attention contributes genuine cross-modal biological alignment beyond manifold compression. A rank-matched CKA control (R1 projected to top-3 PCs reaches CKA 0.234 vs R4 full 0.564) rejects the "R4 wins by collapsing the manifold" explanation. |
| Scientific conclusion | Contrastive training does the bulk of the work (B4 at chance); cross-attention adds the rest. |
| Primary metrics | R@K, MRR, AUC, CKA, alignment gap |
| Strong methods | R4 (best), R1, R6 |
| Baselines | B1, B2, B3, B4 |

## H1 — Structural Agreement

| Field | Detail |
|---|---|
| Observed result | Contrastive runs separate cleanly from classical baselines on CKA (range 0.120–0.564 vs 0.086–0.124, no overlap). **R4 alignment gap +0.195 vs B3 −0.049** (sanity-floor anti-correlation). |
| Scientific conclusion | Contrastive objectives improved latent-geometry alignment beyond linear correlation methods. |
| Strong methods | R4 |
| Baselines | B1, B3 |

---

## H2 — Structural Coherence

| Field | Detail |
|---|---|
| Evaluation question | Do the proposal's nine spatial archetypes cluster more coherently in the aligned space? |
| Observed result | **Construct-validity caveat:** the proposal's `archetype` label is patient-level pseudobulk (`Spatial archetypes_defined_on_ST_global_pseudobulk`) — verified: 14/14 held-out test patients have a single archetype across all their niches; NMI(archetype, patient_id) ≈ 0.90. ARI on archetype is mechanically a 9-class patient classification at niche resolution, so classical baselines that amplify patient identity score highest (B1 archetype ARI 0.307). On the proper niche-level test (Wang's `mc_megacluster`, 14-class per-spot NMF), KMeans ARI: **R6 0.251, R1 0.231, R4 0.098**. R4's low KMeans is rescued by a patient-stratified linear probe at parity (R4 LP 0.230 vs R1 0.260, raw H&E 0.239). |
| Scientific conclusion | R1 and R6 are right for grouping niches into biologically distinct tissue regions (immune-rich, stromal, necrotic, TLS). R4 trades cluster geometry for a tighter manifold that enables cross-patient transfer (see H3). Fusion-axis trade-off, not a single winner. |
| Primary metrics | ARI, silhouette, LP accuracy, kNN purity |
| Strong methods | R1, R6 (KMeans + LP); R4 (LP only) |
| Baselines | B1, B3 (lose 31–37% of ARI on the niche-level resolution shift; that drop quantifies the patient-leakage component) |

## H2 — Compartment Contrasts

| Field | Detail |
|---|---|
| Evaluation question | Do morphologically distinct compartments (tumor, TLS, necrosis) show higher matched-pair cosine similarity than ambiguous stromal compartments (high-TIL, low-TIL)? |
| Observed result | Only the TLS-vs-TIL-stroma contrasts consistently match the directional prediction (passes in R1/R2/R6, Welch z = +3.2 to +5.3). Tumor and necrosis contrasts invert decisively (z ≤ −2.8). |
| Interpretation | The large Tumor (n=4,384) and Necrosis (n=1,178) annotations pool subtypes and dilute matched-pair similarity; small well-curated TIL annotations (n=105–354) inflate cosine. This is an annotation-vocabulary issue, not a biology-absence issue. |
| Coverage (verified) | Compartment annotations cover **13/38 held-out-test subarrays, 12 classes populated** (34.9% of test niches, 15,953/45,661). An 18-class compartment re-annotation on the annotated subset is queued. |
| Strong methods | R1, R6 (on the TLS contrasts) |
| Baselines | B1 (all six contrasts non-significant) |

## H2 — Biology vs Patient

| Field | Detail |
|---|---|
| Observed result | **R4 has the lowest patient z (22.97) in the grid** — strongest patient suppression on z_he. **B1 amplifies patient identity 3.3×** (patient z 122.13 vs raw 37.25), making it actively anti-helpful for cohort-level interpretation. |
| Scientific conclusion | Shared embeddings preserved biological organization while suppressing patient-specific signal. R4's patient suppression is what enables H3 cross-patient transfer. B1 should be avoided for any interpretation task requiring cross-patient validity. |
| Primary metrics | Patient-ID decoding, bio z / patient z ratio |
| Strong methods | R4 (patient suppression); R6 (bio/patient ratio 0.548, highest); R1 (0.517) |
| Baselines | B1 (ratio 0.064, below raw-modality floor 0.129 — anti-helpful) |

---

## H3 — Pathway Interpretability (AUCell)

| Field | Detail |
|---|---|
| Proposal alignment | The proposal H3 specifies pathway embeddings via gpath2vec correlated with shared-latent directions via CCA, for five Reactome targets (TGF-β, Immune System, ECM Organization, Cell Cycle, Programmed Cell Death). **In this summary the per-niche pathway signal is AUCell**, a documented substitution (cleaner per-niche pathway scoring, decoupler `dc.mt.aucell` over Reactome gene sets normalized 1e4 → log1p). The gpath2vec-with-AUCell-weights all-level rerun is in progress (above); the proposal-literal gpath2vec arm is excluded from this v2 summary pending that rerun. |
| Evaluation question | Do the five Reactome target pathways correlate with directions in the shared latent space, and does that signal transfer across held-out patients? |
| Test grid | 5-parent (5 pathways × 3 views = 15 cells per run) and DAG-resolution (395 sub-pathway nodes under the 5 parents × 3 views = **1,185 tests per run**; 9,480 across the 8-run grid). Cross-patient option-A test: train CCA on within-test train patients, evaluate on a held-out 3-patient sub-split. Permutation null + BH-FDR per run. |
| Observed result (DAG, % BH-FDR-significant, verified from `per_node_cca.parquet`) | **R4 78.8%**, B2 74.2%, B1 70.2%, B3 64.4%, R6 36.9%, R1 27.1%, R2 23.0%, R3 18.4%. R4 leads overall; classical baselines are **competitive at DAG resolution**; late-fusion contrastive (R1/R2/R3) trail well below classical. |
| Per-parent (R4 vs leading classical, % significant DAG-tests) | Cell Cycle: R4 69.4 / B2 66.7. Immune: R4 83.5 / B2 79.8. PCD: R4 74.6 / B1 59.6. TGF-β: R4 81.0 / B2 81.0 (tie). **ECM: B1 98.3, B2 98.3, R4 93.3 — classical exceeds R4 on ECM.** |
| R4 top sub-pathway hits (verified top-8 by z_A) | Co-inhibition by BTLA (z=6.92, immune checkpoint); Interleukin receptor SHC signaling (6.80); Integrin cell surface interactions (6.21); ECM proteoglycans (5.73); Regulation of IFNA/IFNB signaling (5.71); Extracellular matrix organization (5.49); Degradation of the extracellular matrix (5.47); Generation of second messengers (5.45). Clinically interpretable sub-programs spanning immune-checkpoint biology and stromal/ECM remodeling. |
| Interpretation | The 5-parent fit-on-all test gives saturated correlations across all runs and does not discriminate. The cross-patient sub-split at DAG resolution does: R4 leads overall and on the immune / Cell Cycle / PCD parents; classical (B1/B2) overtake R4 on ECM (the most over-represented parent in the embedding's pathway coverage). R1/R2/R3 fail cross-patient transfer despite winning H2 niche-level clustering — the same fusion-axis trade-off seen in H2. |
| Scientific conclusion | Cross-attention alignment improves pathway-level cross-patient transferability across biologically distinct sub-programs (immune-checkpoint and stromal/ECM remodeling — both clinically relevant to TNBC therapy response and TME architecture). Classical CCA/Procrustes remain competitive at fine-grained ECM detection, consistent with their closed-form sensitivity to dense linear gene-coexpression structure. |
| Primary metrics | Held-out cross-patient univariate CCA, permutation null + BH-FDR over the 1,185-test-per-run DAG grid |
| Strong methods | R4 (overall lead, wins Immune / Cell Cycle / PCD, ties on TGF-β); B1, B2 (lead on ECM) |
| Baselines | B1/B2/B3 competitive at DAG resolution; R1, R6 well below classical; R2, R3 lowest |

---

## Systems-oriented evaluation

| Field | Detail |
|---|---|
| Evaluation question | Does one alignment strategy dominate every biological objective simultaneously? |
| Observed result | No. R4 wins H1 retrieval and H3 cross-patient pathway transfer overall. R1 and R6 win H2 niche-level region clustering (mc_megacluster KMeans ARI). Classical baselines are competitive at fine-grained pathway detection (and exceed R4 on ECM) but fail H1 retrieval; B1 actively amplifies patient identity and is anti-helpful for cohort-level interpretation. |
| Scientific conclusion | Different alignment architectures optimize different biological objectives. Niche-level cluster geometry, cross-modal retrieval, and cross-patient pathway transfer are not co-monotonic. The framework's value is in exposing the trade-offs and supporting task-conditional routing rather than a single globally optimal embedding strategy. |
| Strong methods | R4 (retrieval, cross-patient pathway transfer); R1, R6 (region clustering on tissue) |
| Baselines | All baselines lose at least one hypothesis family decisively. |

## task-conditional routing rule

| If the question is | Route to | Evidence |
|---|---|---|
| Cross-modal retrieval (find matched ST niche from H&E) | **R4** | H1 AUC 0.851, CKA 0.564; rank-matched control passes |
| Group niches by shared tumor-microenvironment state on a tissue section | **R1 or R6** | H2 KMeans on mc_megacluster (Wang 14-class niche-level NMF): R6 ARI 0.251, R1 ARI 0.231; R1 LP 0.260 |
| Amplify biology, suppress patient identity | **R6 or R1**; R4 for leakage minimization | bio/patient ratio: R6 0.548, R1 0.517; R4 patient z 22.97 (lowest) |
| Find pathway-similar niches across new patients | **R4** | H3 DAG cross-patient: R4 78.8% significant tests (leader) |
| Detect fine-grained sub-pathway biology in dense linear structure (ECM) | **B1 or B2** (preferred), R4 (competitive) | H3 DAG ECM: B1/B2 98.3% vs R4 93.3% |
| Avoid for cohort-level biology interpretation | **B1 (CCA)** | Amplifies patient identity 3.3×; bio/patient ratio 0.064 below raw-modality floor 0.129 |

No single method dominates every objective; the framework's value is in surfacing the trade-offs so downstream analyses route the right question to the right method. As this evaluation runs on additional cohorts (HEST, future TNBC studies, other tumor types), a routing pattern accumulates: which question types map reliably to which method families, where the trade-offs hold, and where they shift with cohort characteristics. The MCP server's value compounds with each cohort the alignment-agent absorbs. TNBC-92 is the first row.

---

## consolidated proposal-alignment matrix

| Hypothesis | Evaluation question | Observed result | Scientific conclusion | Strong methods | Baselines |
|---|---|---|---|---|---|
| H1 Cross-Modal Alignment | Can H&E retrieve matched ST niches better than unaligned/classical? | R4 AUC 0.851, CKA 0.564. R1/R6 0.733–0.741. B4 untrained 0.491 (chance). | Contrastive training does the bulk; cross-attention adds the rest. | R4, R1, R6 | B1, B2, B3, B4 |
| H1 Structural Agreement | Stronger cross-modal structural agreement than baselines? | Contrastive 0.120–0.564 vs classical 0.086–0.124 CKA, no overlap. R4 gap +0.195 vs B3 −0.049. | Contrastive objectives improve latent-geometry alignment beyond linear methods. | R4 | B1, B3 |
| H2 Structural Coherence | Do the 9 spatial archetypes cluster better in the aligned space? | Archetype is patient-level pseudobulk → classical baselines win the patient-leakage proxy (B1 ARI 0.307). On niche-level mc_megacluster: R6 0.251, R1 0.231; R4 KMeans 0.098 but LP at parity (0.230). | R1/R6 for tissue-region grouping; R4 trades cluster geometry for cross-patient transferability. | R1, R6 (KMeans + LP); R4 (LP only) | B1, B3 |
| H2 Compartment Contrasts | Distinct compartments > ambiguous stroma on matched-pair cosine? | TLS-vs-TIL passes (z +3.2 to +5.3 in R1/R2/R6); tumor/necrosis invert (z ≤ −2.8). Coverage: 13/38 test subarrays, 12 classes. | Annotation granularity / morphological purity drive the result; 18-class re-annotation queued. | R1, R6 (TLS contrasts) | B1 |
| H2 Biology vs Patient | Does the aligned space encode biology more than patient identity? | R4 patient z 22.97 (lowest). B1 amplifies patient 3.3× (122.13 vs raw 37.25). | Aligned embeddings preserve biology while suppressing patient signal; B1 is anti-helpful for cohort-level use. | R4 (suppression); R6 (ratio 0.548); R1 (0.517) | B1 (ratio 0.064 below raw floor 0.129) |
| H3 Pathway Interpretability (AUCell, DAG resolution) | Do the 5 Reactome targets correlate with shared-latent directions and transfer cross-patient? | **R4 78.8%** significant DAG-tests; B2 74.2%, B1 70.2%, B3 64.4%; R6 36.9%, R1 27.1%, R2 23.0%, R3 18.4%. **Classical exceeds R4 on ECM (B1/B2 98.3 vs R4 93.3)**; R4 wins Immune (83.5), Cell Cycle (69.4), PCD (74.6); ties B2 on TGF-β (81.0). R4 top hits: BTLA 6.92, IL/SHC 6.80, integrin 6.21, ECM proteoglycans 5.73, IFN-α/β 5.71, ECM organization 5.49, ECM degradation 5.47, second messengers 5.45. | Cross-attention improves pathway-level cross-patient transferability across biologically distinct sub-programs (immune-checkpoint, stromal/ECM remodeling). Classical methods remain the right choice for fine ECM detection. | R4 (overall lead); B1, B2 (ECM lead) | R1, R6 (well below classical); R2, R3 (lowest) |
| Systems-oriented | Does one strategy dominate every objective? | No — R4 wins H1 + H3-overall, R1/R6 win H2 region clustering, classical exceed R4 on ECM and fail H1. | Task-conditional routing, not a single global winner. | R4, R1, R6 (each wins different families) | All baselines lose at least one family decisively. |

---

## Methods

### Cohort and niche unit

92 published TNBC patients (94 on disk) from Wang et al. 2024, original Spatial Transcriptomics platform (Ståhl 2016; **not** 10x Visium), 281 subarrays with counts, 270,310 spots post-QC. The atomic analysis unit is a **niche** — one center spot plus its 6 spatial neighbors (`K_NEIGHBORS = 6` in `scripts/build_niche_join.py:64`), a 7-spot footprint ~100 µm wide containing ~1,200 cells. Boundary niches that lack 6 valid neighbors in the H&E meta repeat the center spot to keep the 7-token stack fixed-size (`build_niche_join.py:169-174`).

The niche-join parquet set at `data/embeddings/niches/{TNBC*}.parquet` is the intersection of `virchow2_niche × novae_niche_full × gpath2vec_output` coverage = **260 subarrays, 286,250 niches**.

### Evaluation split

Subarray-level, patient-stratified 85/15 split (`align.py:166-192`), seed 42, archetype-stratified when class counts allow. Test set = **45,661 niches from 14 held-out patients across 38 subarrays**. All 8 runs (R1, R2, R3, R4, R6, B1, B2, B3) share this split via `runs/tnbc-92/R1/split.json` referenced from each `run_config.json`.

### Method-grid axes

The method-definitions table at the top of the document lists every run's loss, architecture, supervision, and what it falsifies. Three orthogonal axes vary across the grid:

1. **Family** — contrastive learned (R1, R2, R3, R4, R6), classical closed-form (B1, B2, B3), or random control (B4).
2. **Fusion** — late (independent MLPs per modality: R1, R2, R3, R6, B4) vs cross-attention (ST queries 7 H&E tiles: R4) vs closed-form linear (B1, B2, B3).
3. **ST input** — `novae ⊕ gpath2vec` (R1, R2, R3, R4, B1, B2, B3) vs `novae` only (R6 ablation).

### Why `novae ⊕ gpath2vec` for the ST stream

The composition is the only one that satisfies all of the following constraints simultaneously:

- proposal §3 requires the ST agent to use the Novae GNN as its primary encoder → `novae_niche` (64-d) covers the spatial transcriptomic manifold.
- proposal H3 requires per-pathway interpretability via gpath2vec on Reactome → `gpath2vec_niche` (512-d) provides the pathway-graph signal directly on the ST side.
- `mc_weights` is supervision-only per `program.md` hard constraint #3 (circular dependency if used as input AND positive-pair definition).
- single-d signals (`tls`, `cytotrace`) are too sparse / not validated for ST applicability → flagged HOLD in `program.md`.
- per-niche Fisher significance counts (12-d) were considered as a pathway proxy but downgraded to "exploration/visualization only" per `TODO_fusion_experiments.md` constraint #6.

R6 is the targeted ablation of this choice (drop `gpath2vec_niche` from the ST stream, keep everything else identical) and the H3 cross-patient DAG drop from R4's 78.8% to R6's 36.9% is the empirical evidence that the gpath2vec anchor is load-bearing for cross-patient pathway transfer.

### Niche aggregation (stage 1, `build_niche_join.py`)

| feature | aggregation in `build_niche_join.py` | dim |
|---|---|---|
| `virchow2_niche` | already mean-pooled at extraction time (`extract_virchow2_niche.py`) | 1280 |
| `virchow2_cell_tokens` | **per-tile preserved**: stacked (7, 1280) flattened to 8960 (cross-attention input for R4) | 8960 |
| `novae_niche` | **mean-pool** over center + 6 neighbors, then **per-subarray z-score** (after pooling) | 64 |
| `gpath2vec_niche` | **already per-niche** at gpath2vec output stage (graph built over `niche × pathway` nodes); no further pooling | 512 |
| `mc_weights_niche` | mean-pool over center + 6 neighbors → **supervision only, never input** | 14 |
| `tls` | mean-pool over center + 6 neighbors (1-d, sparse coverage) | 1 |
| boundary niches | neighbors not in virchow2 meta replaced by self-repeats to keep fixed 7-token stack | — |

The per-subarray z-score on novae is the only feature-level normalization at this stage. Done **after** niche aggregation, **per subarray** (`build_niche_join.py:234-240`); reflects the 2026-04 finding that novae z-score gives +0.11 silhouette while Virchow2 z-score gives −13% probe accuracy (Virchow2 is already stain-invariant per PathoROB).

### ST concat and MLP (stage 2, `align.py`)

ST concat in `align.py:117-121` is plain `np.concatenate` along the feature axis:

```
st_parts = [novae_niche, gpath2vec_niche]
x_st = np.concatenate(st_parts, axis=1)   # (N, 576) = (N, 64) | (N, 512)
```

No re-scaling between the 64-d novae block and the 512-d gpath2vec block before concat. The **`LayerNorm(576)` at the input of the ST MLP** (`align.py:203, 222-223`) handles per-niche feature normalization implicitly — without it, gpath2vec's 512 dimensions would dominate the input gradient by count vs novae's 64.

ST MLP block (identical structure for R1, R2, R3, R6 — R6 with input dim 64 instead of 576):

```
x_st (576)
  → LayerNorm(576)
  → Linear(576, 512) + ReLU
  → BatchNorm1d(512)
  → Dropout(p=0.3)
  → Linear(512, 512)
  → F.normalize(dim=-1)
  = z_st (512-d unit-sphere)
```

### Cross-attention variant (R4)

R4 uses the same `novae ⊕ gpath2vec` ST input but the H&E side stays **per-tile** at (7, 1280) rather than mean-pooled; the ST signal is the **query** in cross-attention against unpooled H&E tile tokens. The ST → 576 → 512 projection structure is identical to R1; what changes is how H&E is consumed.

```
ST query:     x_st (576) → LayerNorm → Linear(576, 512) → q (1, 512)
H&E key/val:  x_he_tokens (7, 1280) → Linear(1280, 512) → k, v (7, 512)
attn_out          = softmax(q @ k.T / sqrt(512)) @ v   # (1, 512), tile-weighted H&E
attention_weights = softmax(q @ k.T / sqrt(512))       # (1, 7), per-tile attention
z_he, z_st        → L2-norm to unit sphere
```

### Loss functions

| run | loss | supervision | reference |
|---|---|---|---|
| R1 | InfoNCE (hard CLIP-style symmetric) | none (paired niches = positives) | `align.py` |
| R2 | soft SupCon (mc_weights cosine targets) | `mc_weights_niche` similarity matrix | `align.py` |
| R3 | Barlow Twins (redundancy reduction) | none (off-diagonal decorrelation) | `align.py` |
| R4 | InfoNCE | none | `align.py` |
| R6 | InfoNCE | none (ablation of R1: drops gpath2vec) | `align.py` |
| B1 | closed-form CCA (no SGD) | none | `align_classical.py` |
| B2 | closed-form orthogonal Procrustes (no SGD) | none | `align_classical.py` |
| B3 | none (per-modality PCA + L2-norm only) | none | `align_classical.py` |
| B4 | none (random Xavier init, no training pass) | none | `scripts/_scratch/b4_random_late_fusion_concat.py` |

Hyperparameters fixed across contrastive runs (per each `run_config.json`): `shared_dim=512`, `tau=0.07`, `tau_target=0.1`, `barlow_lambda=0.005`, `dropout=0.3`, `lr=5e-4`, `weight_decay=1e-3`, `epochs=50`, `patience=10`, `seed=42`.

### gpath2vec v1 — how the ST-side pathway feature was computed

The `gpath2vec_niche` feature consumed by the alignment runs above is the **v1** gpath2vec build, located at `data/embeddings/biological_signals/gpath2vec_output/full_cohort_tf_low/` (files since renamed with `_v1_legacy` suffix). It is the build the alignment runs were trained against on ST. It is **not regenerable** from the current code (produced before the F1 dedup / F2 directional fixes); the v1 outputs survive only as the frozen `_v1_legacy` files on disk. The exact pipeline (from `arm_A.log` / `LEGACY_V1_NOTES.md` / `gpath2vec/cli.py niche-pipeline`, package branch `feature/redesign`, commits `9f9d091` → `b9b01f9`):

1. **Per-niche gene selection (Fisher path)**: rank each niche's pseudobulk expression (niche-aggregated counts); take the **top-100 most-expressed genes** per niche (`np.argsort(row)[-top_genes:]`, `--top-genes` default 100, zeros dropped). There is no MAD/mean/variability filter — only top-N-by-expression.
2. **Pathway universe**: Reactome at `level=low` (SBGN, sub-pathway nodes only), filtered to pathways overlapping a TF gene list (1,658 transcription factors).
3. **Enrichment**: Fisher's exact test (per niche × pathway), per-niche BH-FDR, min-FDR aggregation (post-F1; v1 used first-FDR aggregation, a known irreproducibility now retired).
4. **Graph**: unified niche × pathway graph (286,913 nodes, 11,419,852 edges; 286,233 niche clusters).
5. **Embedding**: metapath2vec, dim 512, lr 0.005, 5 epochs, ~2.87 M random walks; ~5–8 hours wall time.
6. **Per-niche output**: a 512-d vector per niche (`cluster_embeddings_v1_legacy.parquet`), joined into the niche-join parquets by `build_niche_join.py` and consumed as `gpath2vec_niche` on the ST input side.

**In progress (not reported here):** gpath2vec recomputed via the new AUCell→graph path at `level=all` (no per-niche top-N gene selection; pathway scoring via decoupler AUCell across all Reactome levels) is being run for further investigation. Results not included in this summary.

### B4 artifact gap (transparency)

B4 was generated by `scripts/_scratch/b4_random_late_fusion_concat.py`, which seeds `torch.manual_seed(42)` and reads R1's `split.json` at runtime to match the test split. The script wrote `runs/tnbc-92/B4/metrics_h1_raw.json` (442 bytes) but **did not persist** `embeddings_test.parquet`, `run_config.json`, `split.json`, `run.log`, or an `eval/` subdirectory. Consequently B4 appears in `metrics_h1_raw.json` and the v2 text but **is not in `runs/tnbc-92/eval/H1/summary.json`** (runs there: R1–R4, R6, B1–B3) and **is not in `runs/tnbc-92/eval/H3/pathway_cca/dag_full/per_node_cca.parquet`** (same 8-run set). The H1 number is reproducible from the seeded random init; an artifact regeneration to add embeddings + downstream eval is queued.

### Metric definitions (unchanged from v1)

- **H1 family**: R@K (Recall at K), MRR (mean reciprocal rank), median rank, AUC (matched vs mismatched pair separability), alignment gap (`mean[same cos] − mean[diff cos]`), CKA (centered kernel alignment, before/after projection).
- **H2 family**: ARI (adjusted Rand index, KMeans clusters vs reference labels), silhouette score, linear probe accuracy (held-out patients), kNN purity, matched-pair cosine per compartment, bio z and patient z (permutation z-scores on label-shuffle nulls).
- **H3 family**: univariate CCA correlation (one canonical direction per pathway-view, computed via `scipy.stats.pearsonr` since the pathway side is 1-d AUCell), percent significant DAG-tests with BH-FDR < 0.05, permutation null (1,000 perms per (run, pathway, view)).

---

## v1 → v2 corrections applied (transparency)

- H3-DAG per-run percentages corrected from the wrong "9.9% / 9,480 cells per run / ~8 cell-types" framing to the verified per-run 1,185-test, R4 78.8% (and the rest of the per-run grid).
- "R4 wins or ties on every named parent" corrected to "R4 leads overall; classical (B1/B2) exceeds R4 on ECM".
- R4 top-8 hits replaced with the verified ranking; "collagen biosynthesis (5.28)", "elastic fibre formation (5.33)", "FLT3 signaling (5.15)", "NFAT (5.10)" removed (below the top-8 cutoff in `per_node_cca.parquet`).
- Compartment coverage corrected from "90/260 subarrays, 15 classes" to the verified "13/38 test subarrays, 12 classes (34.9% of test niches)".
- Cohort line clarified (test split 45,661/14/38 stated explicitly).
- H3 method label now states explicitly that the per-niche pathway signal is AUCell (documented substitution); the gpath2vec-AUCell-all-levels rerun is flagged as in-progress.
- gpath2vec v1 (the build the runs ingested on ST) methods box added; v1 validation metrics (`rho 0.9548`, MC z 5.67, TIME z 6.52) are not cited — v1 is irreproducible and superseded.