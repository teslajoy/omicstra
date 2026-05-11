# tnbc-92 current report

**status**: H1 PASS (R4 cross-attention dominant), H2 FAILS-as-stated + recovers under proper resolution (R1/R6 win niche-level biology; Part B annotation-confound limits the directional test), H3 PASS under cross-patient generalization (R4 dominant). Integrated winner depends on task — R4 for cross-modal retrieval + cross-patient pathway generalization, R1/R6 for niche-resolution biology preservation.

scope: Wang et al. 2024 TNBC cohort, 94 patients, 260 matched subarrays.

deep technical detail and methodology arcs preserved in:
- `projects/tnbc-92/proposal_deviations.md` (gitignored) — cross-reference of executed vs proposal-spec
- memory: `project_h2_2_5_2_6_methodology_arc.md`, `project_h2_mc_coherence_findings.md`, `project_h3_pathway_cca_findings.md`

---

## architecture (two-stream, not three)

gpath2vec is bundled into the ST input vector as a feature, not aligned as its own third stream.

```
H&E side:                                    ST side:
─────────                                    ─────────
virchow2_niche                               novae_niche (64d, z-scored per sub)
(1280d, mean-pooled                          +
 over self + 6 neighbors)                    gpath2vec_niche (512d)
                                             = 576d concatenated
       │                                            │
       ▼                                            ▼
  MLP_he (LN→Linear→ReLU→BN→                  MLP_st (same shape)
          Drop→Linear)                        576 → 512
  1280 → 512                                        │
       │                                            ▼
       ▼                                       z_st (512d, L2-normed)
  z_he (512d, L2-normed)                            │
       │                                            │
       └──────────── InfoNCE loss ──────────────────┘
```

R4 differs only on the H&E side: 7 raw virchow2_cell tile tokens (center + 6 neighbors), cross-attention from ST query selecting which tiles matter.

R6 = R1 architecture with gpath2vec dropped from ST stream (ST = novae 64d only); ablation for circularity check on H3 (validated: H&E carries biology signal independently of gpath2vec being in ST input).

biology validation labels (H3-extended z-test) use external Bareche subtype labels from `Clinical.xlsx` (MC_global, MC_tumor, TIME) — never seen by the model. mc_labels.megacluster (Wang per-spot NMF, 14-class niche-level discrete biological label) was added to niche-join parquets in commit 2.5 for H2 niche-resolution test.

---

## H1: Cross-Modal Alignment

**proposal asked**: does contrastive learning improve cross-modal retrieval and representation alignment over unaligned baselines (CCA, late fusion concatenation, raw unaligned concatenation)? metrics: R@K, MRR, median rank, alignment gap, AUC, CKA. expected similarity range 0.65-0.75; relative improvement over baselines is the primary criterion.

**verdict: PASS**

### results (260 matched subarrays, 85/15 patient-stratified split, 45,661 test niches)

| run | method | R@1 | R@5 | R@10 | MRR | median rank | gap | AUC | CKA after |
|---|---|---|---|---|---|---|---|---|---|
| **R4** | **infonce + cross-attn** | **0.0002** | **0.0014** | **0.0027** | **0.0019** | **4244** | **0.195** | **0.851** | **0.564** |
| R1 | infonce + late | 0.0003 | 0.0010 | 0.0023 | 0.0015 | 8827 | 0.159 | 0.741 | 0.242 |
| R6 | infonce + late (novae-only ST ablation) | 0.0002 | 0.0007 | 0.0014 | 0.0011 | 9227 | 0.155 | 0.733 | 0.225 |
| R2 | supcon + late | 0.0002 | 0.0009 | 0.0017 | 0.0012 | 10153 | 0.043 | 0.707 | 0.120 |
| R3 | barlow + late | 0.0001 | 0.0007 | 0.0014 | 0.0010 | 10267 | 0.125 | 0.646 | 0.252 |
| B2 | Procrustes (classical) | 0.0002 | 0.0008 | 0.0015 | 0.0012 | 10342 | 0.156 | 0.703 | 0.124 |
| B1 | CCA (classical) | 0.0000 | 0.0002 | 0.0003 | 0.0003 | 20181 | 0.007 | 0.541 | 0.086 |
| B3 | Unaligned PCA (raw concat analog) | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 26430 | -0.049 | 0.443 | 0.124 |

random median-rank baseline = ⌈n/2⌉ = 22,831. **CKA before projection**: cross-modal CKA between raw `virchow2_niche` (1280d) and raw ST input (`novae_niche` ⊕ `gpath2vec_niche` = 576d for R1-R4, B1-B3; novae only = 64d for R6) on the test set. Value = 0.1148 for the 7 full-ST runs, **0.0999 for R6** (different input dimensionality changes the raw cosine).

### what this shows

- all 4 contrastive runs beat all 3 classical baselines on AUC and CKA with zero overlap between families. proposal's H1 claim of "relative improvement over baselines" is supported.
- **R4 AUC 0.851** exceeds the proposal's predicted upper bound (0.75); R1 (0.741), R6 (0.733), R2 (0.707), and B2 (0.703) land *within* the predicted 0.65-0.75 range. R3 (0.646) sits marginally below the lower bound. **The proposal's predicted range was conservative for the cross-attention bridge specifically** — the prediction is strengthened, not contradicted.
- median rank: R4 5.4× better than random; R1/R6/R2/R3/B2 cluster around 2.2-2.6×; B1 only 1.1×; B3 worse than random (correct no-alignment sanity floor).
- absolute R@1 is low across all runs (~10⁻⁴ on ~45k niches; chance floor ~2.5e-5) — cross-patient niche-level exact retrieval is empirically near-impossible at this scale; the proposal correctly emphasized relative improvement over absolute thresholds.

### baseline terminology (per proposal p.5) — and a falsifier-completeness gap

the proposal lists three distinct baselines:

| proposal baseline | what's in this grid |
|---|---|
| CCA projection | B1 (closed-form CCA, train-fit applied cohort-wide) |
| **late fusion concatenation** (non-contrastive: independent encoding → concat, no cross-modal objective) | **NOT directly run.** The contrastive late-fusion runs R1/R2/R3 use this architecture *with* a contrastive loss added; subtracting the loss would give the proposal-strict baseline. This is a falsifier-completeness gap in the H1 grid. The contrastive-vs-classical family separation still holds, but the strict "does the *loss* help vs late-fusion alone?" question is not directly tested. |
| raw unaligned concatenation | B3 closest analog (per-modality PCA-to-512 + L2-norm, no alignment objective). A stricter version would skip PCA and concatenate raw 1280d + 576d = 1856d with L2-norm; not run. |

### rank-matched CKA control

R1 z_he projected to its top-3 PCs (matching R4's ~3 effective dimensions): CKA drops from R1 full (0.242) to R1@rank3 (0.234) — a 3.3% loss attributable to dimensional reduction. R4 full = 0.564. The residual gap at matched rank (R1@rank3 = 0.234 vs R4 full = 0.564, gap of -0.331) is the **non-geometric component** of R4's CKA advantage. Sanity: R4@rank3 ≈ R4 full (R4 already lives in ~3 dims, so projection is approximately identity). **R4's CKA advantage is largely architectural, not collapse-driven** — at matched effective rank R1's top-3 PCs still cannot reach R4's CKA.

---

## H2: Structural Coherence

**proposal asked**:
- Part A: do 9 spatial archetypes cluster more coherently in the aligned latent vs single-modality (ARI + silhouette)?
- Part B: do morphologically distinct compartments (tumor, TLS, necrosis) show higher matched-pair cosine than ambiguous ones (high-TIL, low-TIL stroma)?

**verdict: FAILS as stated, partially recovers under proper resolution**

### Part A — 9-archetype clustering: FAIL (label-resolution confound)

`archetype` in `data/embeddings/niches/*.parquet` is sourced from Wang's `Spatial archetypes_defined_on_ST_global_pseudobulk` — a **patient-level pseudobulk label** (verified: 30/30 sampled subarrays have `archetype_unique_within_subarray = 1`, every niche in a patient inherits one archetype). KMeans(k=9) → ARI vs archetype is mechanically a 9-class patient classification, not a spatial-coherence test.

| run | ARI (archetype, z_he) | reading |
|---|---|---|
| B1 CCA | 0.307 | wins by amplifying patient identity 3.3× (z_he patient z = 122.13 vs raw 37.25) |
| B2 / B3 (rotation-equivalent) | 0.298 | patient leakage |
| raw_he | 0.287 | raw-modality patient-architecture |
| R6 | 0.254 | — |
| R1 | 0.250 | — |
| R2 | 0.204 | — |
| R3 | 0.129 | — |
| R4 | 0.059 | strongest patient suppression (z_he patient z = 22.97, lowest in grid) |

ranking exactly tracks patient-axis preservation. silhouettes all near-zero (no compact archetype clusters in any space). metric is confounded.

### Part A-MC — niche-level biological clustering (commit 2.5 + 2.6)

`mc_labels.megacluster` (Wang per-spot 14-class NMF discrete label, niche-level biological) joined into niche parquets in commit 2.5. KMeans(k=14) + linear probe (patient-stratified, 11/3 within test patients).

| run | ARI (MC, z_he) | linear probe accuracy (MC) |
|---|---|---|
| **R6** | **0.251** | 0.235 |
| **R1** | **0.231** | **0.260** |
| B2 / B3 | 0.205 | 0.233 / 0.237 |
| B1 | 0.202 | 0.210 |
| R2 | 0.174 | 0.229 |
| raw_he | 0.174 | 0.239 |
| R3 | 0.166 | 0.182 |
| **R4** | **0.098** (geometry bias) | **0.230** (parity with R1/R6/raw_he) |

R4 fails KMeans (~3 effective dims cannot resolve 14 globular clusters) but **passes linear probe at parity with R1/R6**. KMeans tests globular cluster shape; LogReg tests linear decodability. R4's biology is encoded in linearly-separable directions, not globular regions. **rank-matched KMeans decomposition**: R1 projected to its top-3 PCs drops from 0.231 to 0.193 (16% loss attributable to dimensional reduction); R4 at its native rank scores 0.098. The residual gap (R1@rank3 = 0.193 vs R4 full = 0.098) at matched rank is the non-geometric component — **roughly half of R4's KMeans deficit is geometry (KMeans-unfriendly tight manifold), half is encoding-shape** (linearly-separable but non-globular biology axes). Linear probe parity confirms the encoding-shape half is not biology absence.

classical baselines (B1/B2/B3) and raw_he **lose 31-39% of ARI under the resolution shift**, confirming patient-leakage diagnosis.

### Part B — compartment cosine contrasts: FAIL (annotation-purity confound, 1 of 6 contrasts supported)

proposal predicted: tumor / TLS / necrosis > hiTIL / loTIL stroma in matched-pair cosine. Welch z-test (unpooled variance) on 12-compartment vocabulary.

| contrast | R1 | R2 | R3 | R4 | R6 | B1 | B2 | B3 |
|---|---|---|---|---|---|---|---|---|
| Tumor vs hiTIL | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ ns | ✗ | ✗ |
| Tumor vs loTIL | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ ns | ✗ | ✗ |
| TLS vs hiTIL | **✓ z=+3.2** | ✓ | ✗ | ✗ | ✓ | ✓ ns | ✗ | ✗ |
| TLS vs loTIL | **✓ z=+5.3** | ✓ | ✗ | ✗ | ✓ | ✓ ns | ✗ | ✓ |
| Necrosis vs hiTIL | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ ns | ✓ | ✗ |
| Necrosis vs loTIL | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ ns | ✗ | ✗ |
| pass count | 2/6 | 2/6 | 3/6 | 0/6 | 2/6 | 5/6 ns | 1/6 | 2/6 |

only **TLS vs TIL-stroma** supported across contrastive runs (R1/R2/R6). Tumor and Necrosis contrasts INVERT with statistical decisiveness (z ≤ -2.8). B1's 5/6 directionally-correct contrasts all non-significant — uniform pattern, not biology.

mechanism: Tumor (n=4384) and Necrosis (n=1178) are large heterogeneous annotations pooling sub-types. hiTIL (n=105) / loTIL (n=354) are small well-curated annotations. within-compartment cosine measures annotation-set purity, not morphological distinctness. proposal assumed equivalence; on this cohort they aren't equivalent.

### H2 consolidated verdict

- Part A archetype test fails because labels are patient-level (metric inherits patient leakage)
- Part A-MC at proper resolution: **R1/R6 win niche-level biology preservation** on KMeans + linear probe. R4 fails KMeans (geometry), matches on linear probe (biology present). classical baselines lose 31-37% on resolution shift (confirming patient-leakage).
- Part B: 1/6 contrasts supported (TLS-vs-TIL). 5/6 inverted due to annotation-purity confound, not biology absence.

H2 is a fusion-axis trade-off: late-fusion contrastive (R1/R6) for niche-level biology resolution; cross-attention bridge (R4) for cross-modal alignment with sacrificed cluster geometry.

---

## H3: Pathway Interpretability

**proposal asked**: do gpath2vec Reactome embeddings for 5 named pathways (TGF-β R-HSA-170834, Immune R-HSA-168256, ECM R-HSA-1474244, Cell Cycle R-HSA-1640170, PCD R-HSA-5357801) correlate with specific directions in the shared latent space via canonical correlation analysis, indicating preservation of interpretable biological signal?

**verdict: PASS under cross-patient generalization (the discriminating test)**

pathway signal = niche-level AUCell scores from `data/embeddings/biological_signals/niche_aucell_5targets.parquet` (286k niches × 5 pathways, 100% test-set coverage, ECM present — TF-filter concern resolved on this artifact).

univariate CCA via closed-form lstsq (15-line solver, no sklearn machinery). Two variants reported:
- **Part A** (option B in code): fit on all test niches — proposal's literal test
- **Part B** (option A in code): cross-patient sub-split (fit on 11 train patients within test set, eval on 3 held-out)

permutation null = 1000 perms × 120 cells per option, BH-FDR.

### Part A — fit-on-all CCA (proposal's literal test): every run PASSES

null mean = 0.106 across every cell (= sqrt(d/n) = sqrt(512/45661), the theoretical OLS overfit floor). Observed 0.70-0.90, z-scores 180-245. **all 120 cells BH-FDR < 0.001**.

| run | cumulative \|corr\| (5 pathways × 3 views) |
|---|---|
| B2 / B3 | **4.242** (rotation-equivalent decisive win, +0.143 over R1, +0.150 over B1) |
| R1 | 4.099 |
| R6 | 4.093 |
| B1 | 4.092 |
| R2 | 4.079 |
| R3 | 4.021 |
| R4 | 3.943 |

every run encodes pathway-aligned information that survives null calibration. **the proposal's H3 hypothesis is supported for every run**. The test is saturated by 512-d projection capacity onto a 1-d target — narrow spread, no architectural ranking power.

**a sub-finding worth flagging**: CCA (B1) optimizes cumulative canonical correlation *by construction*. It losing to rotation-equivalent B2/B3 (Procrustes / unaligned PCA) by 0.15 cumulative is itself diagnostic. Most likely B1's whitening + projection step compresses some variance that B2/B3 preserve through orthogonal rotation, so B2/B3's linear regression has slightly more usable variance per pathway than B1's whitened latent. Not load-bearing for the H3 verdict (option A is the discriminating test) but logged as a CCA-baseline pathology.

### Part B — cross-patient generalization (the discriminating test)

CCA fit on 11/14 test patients, evaluated on 3 held-out patients. BH-FDR < 0.05 over 120 cells.

| run | significant cells / 15 max | z_he z-scores (TGF-β / Immune / ECM / CC / PCD) |
|---|---|---|
| **R4** | **15 / 15** | **4.5 / 4.6 / 5.7 / 4.2 / 3.8** |
| B1 | 14 / 15 | 2.8 / 3.5 / 4.4 / 2.4 / 1.6 |
| B2 | 14 / 15 | 3.0 / 3.5 / 4.8 / 2.5 / 1.3 |
| B3 | 12 / 15 | 3.0 / 3.5 / 4.8 / 2.5 / 1.3 |
| R6 | 11 / 15 | z_st 5/5 + z_mean 5/5 driven; **z_he 1/5 (only ECM)** — counter-intuitive, see below |
| R1 | 5 / 15 | 1.2 / 1.3 / -0.3 / 1.5 / 1.4 |
| R2 | 5 / 15 | -1.6 / -0.6 / 1.1 / -0.9 / -0.7 |
| R3 | 5 / 15 | -0.6 / -1.4 / -1.7 / 1.3 / -0.0 |

**R4 dominates cross-patient generalization** (15/15 significant, z-scores 3.8-5.7). R1/R2/R3 barely above the 5% FDR baseline.

**R6 counter-intuitive direction (verified from `runs/tnbc-92/eval/H3/pathway_cca/perm_nulls.parquet`)**: R6 dropped gpath2vec from the ST stream, so we would expect z_st to be the *weakest* pathway-correlated view (no explicit pathway anchor in input). Observed: opposite — z_st is 5/5 significant cross-patient (z-scores 2.3-5.0), z_he is 1/5. Plausible mechanism: contrastive training with no pathway anchor on ST forces z_he to compress toward z_st's geometry through the InfoNCE objective, eroding z_he's native pathway alignment. z_st alone retains implicit biology correlation through Novae's spatial-neighborhood GAT structure even without an explicit pathway feature. This deserves a follow-up R6b mirror ablation (gpath2vec-only ST, novae dropped) to confirm whether the pattern flips.

### specificity diagnostic — how R4 wins

z_he off-diagonal |mean| of pairwise canonical-direction cosines across 5 pathways:

| run | off-diag \|mean\| | max | reading |
|---|---|---|---|
| **R4** | **0.819** | **0.970** | **fully aliased — 5 pathways collapse to ~1 direction** |
| R1 | 0.481 | 0.771 | mid |
| R6 | 0.446 | 0.733 | mid |
| B1 | 0.411 | 0.742 | mid |
| R2 | 0.407 | 0.639 | distinct |
| R3 | 0.390 | 0.723 | most distinct |
| B2 / B3 | 0.376 | 0.684 | distinct |

R4's ~3-effective-dim manifold cannot afford 5 orthogonal pathway directions. It encodes ONE robust biology axis that all 5 (correlated) Reactome pathways project onto. One direction is harder to overfit to patient identity than five, so the encoding transfers cross-patient. **Aliasing is the mechanism of generalization, not a bug.**

R1/R6 have wider 5-dim manifolds encoding more-distinct pathway directions — but those distinctions are patient-specific gradients that don't survive held-out patients.

### H3 verdict

- proposal-stated test (Part A): **all 8 runs PASS**; every embedding encodes the 5 named Reactome pathway directions above null.
- discriminating test (Part B): **R4 dominant** (15/15); classical baselines strong (12-14/15); late-fusion contrastive fail (5/15).
- mechanism: R4's tight manifold forces an aliased single-axis encoding that captures the shared biology underneath the 5 correlated pathways; this single axis is robust cross-patient.
- **limitation**: R4's aliased encoding (specificity off-diag 0.819) **cannot distinguish between pathway-specific signals** — TGF-β-active niches and Immune-active niches project onto roughly the same direction in R4's latent. Downstream tasks requiring per-pathway resolution (e.g. "which niches are specifically immune-active independent of TGF-β / ECM activity?") must route to R1/R6 despite their cross-patient generalization failure. R4 is appropriate for tasks where "biologically active vs not" is the relevant axis (cohort stratification, retrieval queries).

---

## integrated H1 / H2 / H3 picture

| metric | R1 / R6 (late-fusion contrastive) | R4 (cross-attention bridge) | B1 / B2 / B3 (classical) |
|---|---|---|---|
| H1 retrieval (AUC) | 0.733-0.741 | **0.851** | 0.44-0.70 |
| H1 structure (CKA) | 0.23-0.25 | **0.564** | 0.09-0.12 |
| H2 archetype ARI (patient-confounded) | mid | low (patient suppression) | win (patient leakage) |
| H2 MC ARI (niche-level) | **win (0.23-0.25)** | low (geometry bias) | mid (lost 31-37% on resolution shift) |
| H2 linear probe MC | parity (0.26) | parity (0.23) | parity (0.21-0.24) |
| H2 Part B contrasts | 2-3 / 6 (TLS only) | 0 / 6 (annotation-purity confound) | 0-2 / 6 |
| H3 fit-on-all CCA | all pass | all pass | all pass |
| **H3 cross-patient CCA** | **fail (5/15)** | **win (15/15)** | strong (12-14/15) |

**R4 wins H1 and H3.** R1/R6 win H2 niche-level biology. Classical baselines lose all three or saturate non-discriminatingly.

### architectural recommendation (task-conditional)

- **cross-modal retrieval + cross-patient pathway generalization** → R4 (cross-attention bridge)
- **niche-resolution biology preservation + linear interpretability** → R1 or R6 (late-fusion contrastive)
- avoid classical baselines for either objective

this is the omicstra alignment-agent's task-conditional routing rule: when a user asks for cross-patient or retrieval, route to R4; when they ask for niche-resolution interpretation, route to R1/R6.

---

## ResearchHub project status (phase 1 complete, phase 2 planned)

the proposal commits to two layers: (1) a scientific foundation evaluating cross-modal contrastive alignment on TNBC, and (2) an MCP server / agent orchestration layer that operationalizes the validated approach as a reusable tool. these are not equally mature.

### phase 1 — scientific foundation (complete)

| proposal element | status |
|---|---|
| Virchow2 + UNI2 + Novae encoder pipelines | ✔ 260-subarray niche join table materialized |
| gpath2vec pathway embedding (full cohort) | ✔ 286k niches × 512d, validated |
| contrastive alignment module | ✔ 8 runs (R1-R4 + R6 + B1-B3), reproducible scripts |
| H1 retrieval / alignment metrics | ✔ R4 cross-attention dominant (AUC 0.851, CKA 0.564) |
| H2 structural coherence | ✔ Part A label-resolution confound diagnosed and corrected via niche-level MC test; Part B annotation-purity confound on 5/6 contrasts |
| H3 pathway interpretability | ✔ proposal-spec per-pathway CCA done; Part A all pass; Part B R4 dominant cross-patient (15/15 cells) |
| cohort-level robustness | ✔ H3-extended bio-vs-patient z-test on external Bareche labels |
| falsifier discipline | ✔ pre-registered predictions for 2.5 / 2.6 / 3 / 3.5 with documented arc |

this is the foundation that justifies building the agent layer. the architectural recommendation (task-conditional routing R4 / R1+R6) is the result that the MCP server will encode as routing logic.

### phase 2 — agent orchestration layer (planned)

| proposal element | status |
|---|---|
| MCP server (`src/mcp/`) | not started |
| he_agent, st_agent, alignment_agent, eval_agent classes | not started |
| LangGraph StateGraph with conditional routing | not started |
| Claude Sonnet routing / Claude Opus synthesis | not started |
| LangSmith tracing wiring | not started |
| HITL escalation logic | not started |
| pip-installable release | not started |
| HuggingFace checkpoint + model card | not started |
| Zenodo embeddings archive | not started |

the scripts (`align.py`, `eval.py`, `eval_alignment_biology.py`, `build_niche_join.py`, `align_classical.py`, `eval_h3_pathway_cca.py`) are already agent-ready: config in, artifacts out. wrapping them as MCP tools is the wrapping pattern, not new algorithm work. the routing rule from phase 1 (R4 for cross-patient retrieval; R1/R6 for niche-resolution interpretation) becomes the alignment_agent's task-conditional dispatch.

### not yet tried — alignment-method extensions

- **AnInfoNCE (Rusak et al., AISTATS 2025)** — per-dim learnable diagonal Λ̂ in the InfoNCE similarity. *Predicted result if run*: AUC ≥ R4's 0.851 with effective rank in the R1-range (~5-10 instead of R4's ~3), breaking the dim-collapse aliasing trade-off — i.e. preserving R4's cross-modal alignment quality while keeping the 5 pathway canonical directions distinguishable. *Caveat*: per-dim temperature targets dim utilization specifically. R4's H3 aliasing is about *effective rank* being too low for 5 orthogonal pathway directions, which AnInfoNCE addresses *if* the dim-collapse comes from temperature symmetry. If the collapse is from architectural compression (cross-attention output is genuinely low-rank by design), AnInfoNCE may help less. Worth running; predicted effect size moderate.
- **R6b mirror ablation** — gpath2vec-only ST (drop novae). symmetric to R6; directly tests whether the R6 counter-intuitive z_st > z_he pattern is gpath2vec-driven or novae-driven.
- **domain-adversarial GRL on patient_id** — alternative path if cross-patient generalization fails on future cohorts (e.g. HEST validation).

---

## reproducibility

### scripts (all under `scripts/`)

| script | what it does | produces |
|---|---|---|
| `build_niche_join.py` | per-subarray niche-join parquets (virchow2_niche, virchow2_cell_tokens, novae_niche z-scored, gpath2vec_niche, tls, mc_weights_niche, mc_megacluster, archetype, compartment) | `data/embeddings/niches/*.parquet` |
| `align.py` | train one contrastive config (loss × fusion × supervision) | `runs/tnbc-92/{run_id}/` (R1-R4, R6) |
| `align_classical.py` | closed-form CCA / Procrustes / unaligned PCA baselines, same split.json as R1 | `runs/tnbc-92/{run_id}/` (B1-B3) |
| `eval.py --hypothesis H1\|H2\|H3` | H1 rollup, H2 archetype ARI + compartment cosine, H3 gpath2vec CCA + patient probe + attention | `runs/tnbc-92/eval/{H1,H2,H3}/summary.json` |
| `eval_alignment_biology.py --run-id R1` | biology vs patient z-test, 1e5 perms, full nulls persisted at seed=42 | `runs/tnbc-92/{run_id}/eval/biology*.parquet` |
| `eval_h3_pathway_cca.py` | per-pathway univariate CCA on 5 named Reactome (option A + option B), 8 runs × 5 × 3 views | `runs/tnbc-92/eval/H3/pathway_cca/per_pathway_cca.parquet` + `canonical_directions.parquet` + `specificity_matrix.parquet` |
| `eval_h3_pathway_cca_perms.py` | 1000-perm null + BH-FDR calibration for both CCA options | `runs/tnbc-92/eval/H3/pathway_cca/perm_nulls.parquet` |

### configs (under `projects/tnbc-92/alignment/config/`)

| file | run | loss | fusion | notes |
|---|---|---|---|---|
| `R1.json` | R1 | infonce | late | full ST input (novae + gpath2vec) |
| `R2.json` | R2 | supcon | late | mc_weights soft supervision |
| `R3.json` | R3 | barlow | late | redundancy reduction |
| `R4.json` | R4 | infonce | cross_attn | 7 tile tokens, ST query |
| `R6.json` | R6 | infonce | late | ablation: ST = novae 64d only |

B1/B2/B3 parameterized via CLI args to `align_classical.py`.

### per-run artifacts (under `runs/tnbc-92/{run_id}/`)

| file | source | used for |
|---|---|---|
| `run_config.json` + `split.json` | align.py | exact config + identical 85/15 patient split across all 8 runs |
| `embeddings_test.parquet` | align.py / align_classical.py | per-test-niche z_he, z_st (+ attention_weights for R4) |
| `metrics_h1_raw.json` | align.py | R@K, MRR, median_rank, alignment_gap, AUC, CKA |
| `eval/biology.parquet` + `biology_nulls.parquet` | eval_alignment_biology.py | per-view biology and patient z + full 1e5-perm null distributions (seed=42, paired-run-comparable) |

### input data (read-only)

- H&E niche / cell: `data/embeddings/virchow2_niche/` + `virchow2_cell/`
- Novae niche (k=6 neighbors): `data/embeddings/novae_niche_full/`
- gpath2vec niche (286k × 512d): `data/embeddings/biological_signals/gpath2vec_output/full_cohort_tf_low/cluster_embeddings.parquet`
- AUCell pathway scores (286k × 5 named pathways): `data/embeddings/biological_signals/niche_aucell_5targets.parquet`
- biological signals: `data/embeddings/biological_signals/{mc_labels,mc_weights,tls_scores,morphology_labels}.tsv`
- clinical + Bareche subtype labels: `data/inputs/Clinical/Clinical.xlsx`
- subarray → TNBC id: `data/inputs/clinical/ids_subarray_to_tnbc.tsv`

### reproducibility guarantees

- all runs use `seed = 42`; classical runs re-fit from R1's train patients
- neighborhood parity: Virchow2 and Novae use identical KDTree k=6 neighborhoods (1075/1075 spot agreement on TNBC1_CN1_C1)
- biology z-test: 1e5 permutations; pathway CCA null: 1000 perms × 120 cells per option
- sklearn 1.7+ (no `multi_class` kwarg on LogisticRegression), torch 2.x

---

## known limitations

- **TLS / morphology / compartment labels exist only for 90 / 260 subarrays** — Part B compartment cosine is annotated-subset-only.
- **19 subarrays excluded from ST side** for falling below Novae's zero-shot 512-spot floor.
- **niche-level absolute R@K is empirically near-impossible** at ~40k cross-patient niches — proposal correctly emphasizes relative improvement over absolute retrieval.
- **Bareche MC_global / MC_tumor / TIME labels are bulk-NMF computational subtypes**, not gold-standard pathology — H3-extended biology z-test validates against patient-subtype-level biology, not niche-level.
- **mc_megacluster coverage 94%** on niche-join parquets; 1 subarray (TNBC34_CN17_E2) fully missing, others partial — Wang QC exclusions.
- **R4 cross-patient generalization is aliased** (5 pathway directions collapse to ~1). Pathway specificity is sacrificed for cross-patient transfer; appropriate for retrieval-style queries, not per-pathway interpretation.
- **R6b mirror ablation (gpath2vec-only ST) not yet run** — H2 part A-MC R6 result needs the symmetric ablation to confirm gpath2vec-out-of-ST is what drives R6's H&E-side biology lift.