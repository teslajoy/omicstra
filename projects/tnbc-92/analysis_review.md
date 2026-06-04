# omicstra Full Project Review
**Reviewer perspective: senior bioinformatician, code-traced, read-only**
**Scope: EDA gate through final hypothesis verdicts, all notebooks, all docs, all configs**
**Date: 2026-06-01**

---

## 1. Executive Summary

This project asks whether contrastive alignment of H&E histopathology embeddings (Virchow2 1280d) and spatial transcriptomics embeddings (Novae 64d + gpath2vec 512d) into a shared 512d space produces biologically meaningful cross-modal representations for 92 TNBC patients. It tests this across three hypotheses (retrieval, structural coherence, pathway interpretability) using 10 alignment strategies (6 contrastive + 4 baselines).

**Bottom line**: The scientific foundation is strong, rigorously self-audited, and the hypothesis results are reproducible across v1 and v3 retrains. No single method wins everything - this is itself the main finding. The MCP agent layer (phase 2) is entirely unbuilt.

---

## 2. EDA Gate: Thorough and Honest

The EDA gate (`EDA.md` + `eda_summary.json` + two exploration notebooks) is one of the strongest parts of the project.

| EDA Check | Result | Verified In Code |
|---|---|---|
| Cohort: 94 patients, 281 subarrays with counts | PASS | `eda_data_inventory.ipynb` cell 7-8 |
| Spots: 537,069 total, 287,674 selection, 270,310 post-QC | PASS | `eda_biological_signal.ipynb` cell 5 |
| Triple-negative confirmed: ESR1 2.6%, PGR 0.9% | PASS | `eda_biological_signal.ipynb` cell 13 |
| Moran's I >= 0.3 for >= 3/5 top markers | PASS (5/5 pass) | `eda_biological_signal.ipynb` cell 19 |
| Batch silhouettes all negative (plate -0.076, lane -0.081) | PASS | `eda_biological_signal.ipynb` cell 23 |
| Count matrices are log-normalized, NOT raw | Correctly flagged | `eda_data_inventory.ipynb` cell 12 |
| Novae trained on MERSCOPE/Xenium, NOT original ST platform | Correctly flagged as risk | `MODELS.md`, `novae_st_embeddings.ipynb` |

**One concern**: Marker expression statistics (cell 13 of bio_signal notebook) are computed on all 1934 grid spots including off-tissue, not just the 1075 selection spots. The percentages are diluted. This does not change the EDA gate decision (markers still pass/fail correctly), but the reported percentages would be 1.5-2x higher if restricted to on-tissue spots.

---

## 3. Embedding Layer: Strong Foundation, Some Technical Debt

### Encoder selection was empirically driven, not assumed

| Encoder | Dimension | MC Linear Probe | Morph Silhouette | Decision |
|---|---|---|---|---|
| Virchow2 niche | 1280d | 23.6% (3.3x chance) | 0.030 | **Primary** (per encoder_qc_comparison) |
| UNI2 reinhard | 1536d | not reported at niche level | higher morph sil | Alternative |
| Novae zero-shot | 64d | 12.1% (1.7x chance) | n/a | Validated with constraints |
| PCA-HVG | 50d | not run | n/a | Pilot only, never scaled |

This is the right approach: let the data pick the encoder instead of assuming from the literature.

### Embedding quality diagnostics

| Property | Virchow2 niche | Novae | PCA-HVG |
|---|---|---|---|
| Spatial autocorrelation (Spearman rho) | -0.413 | -0.380 | +0.281 (different sign convention) |
| Within/between patient gap | 0.138 | 0.022 (batch-corrected by design) | 0.671 (massive batch) |
| Effective rank (99% variance) | not reported | 11 of 64 dims | 25 of 50 dims |
| Dimension utilization | high | 17% (information bottleneck) | 50% |

Novae's effective rank of 11/64 is the real bottleneck. The 64d -> 512d projection has only ~11 dimensions of actual information. CLAUDE.md correctly identifies this: "ST info bottleneck is Novae's 64d, not shared width."

### Technical debt in embedding notebooks

| Issue | Severity | Where |
|---|---|---|
| Resolution calibration differs between uni2_patch_extraction (0.2774 um/px, 461px patches) and uni2_reinhard (0.3785 um/px, 338px patches) | Medium | Notebooks 1 vs 2 |
| Spot counts differ for same subarrays between notebooks (4514 vs 4413) due to different data sources | Medium | Notebooks 1 vs 2 |
| `niche_aucell_analysis.ipynb` and `niche_pathway_analysis.ipynb` are byte-for-byte identical | Low | Duplicate file |
| Pilot Fisher EA had a vacuous `pathway_adjpvalue` criterion (always true) | Fixed | Full-cohort notebook drops it correctly |
| Novae summary markdown reports stale rho values that don't match code output | Low | `novae_st_embeddings.ipynb` |
| PCA-HVG baseline never built for full cohort | Medium | Only 5-subarray pilot exists |

---

## 4. Alignment Results: The Core Science

### 4.1 H1 - Cross-Modal Retrieval: SUPPORTED

| Run | Type | AUC [95% CI] | CKA [95% CI] | Median Rank | Alignment Gap |
|---|---|---|---|---|---|
| **R4** | **InfoNCE + cross-attn** | **0.859 [0.857, 0.862]** | **0.631 [0.626, 0.636]** | **3,282** | **+0.233** |
| R1 | InfoNCE + late | 0.761 [0.758, 0.764] | 0.312 [0.305, 0.318] | 6,234 | +0.194 |
| R5 | AnInfoNCE + late | 0.761 [0.758, 0.764] | 0.314 [0.306, 0.320] | 6,268 | +0.216 |
| R6 | InfoNCE + late (novae-only) | 0.737 [0.734, 0.740] | 0.239 [0.233, 0.245] | 7,015 | +0.154 |
| R2 | SupCon + late | 0.725 [0.722, 0.728] | 0.101 [0.098, 0.105] | 7,729 | +0.036 |
| B2 | Procrustes (classical) | 0.706 [0.703, 0.710] | 0.125 [0.122, 0.129] | 8,069 | +0.161 |
| R3 | Barlow Twins + late | 0.674 [0.671, 0.677] | 0.306 [0.299, 0.312] | 7,643 | +0.143 |
| B1 | CCA (classical) | 0.544 [0.541, 0.547] | 0.076 *(CI n/a)* | 15,554 | +0.007 |
| B4 | Random-init (chance) | 0.496 [0.493, 0.499] | 0.120 [0.117, 0.124] | 17,936 | -0.000 |
| B3 | Unaligned PCA | 0.440 [0.437, 0.443] | 0.125 [0.122, 0.129] | 20,688 | -0.053 |

**CIs.** AUC: Hanley & McNeil (1982) closed-form (matched to nonparametric bootstrap within 3 decimals); n_pos = 35,594, n_neg = 355,940. CKA: nonparametric bootstrap over test niches, B = 200, seed = 42. B1 CKA CI not estimable by standard bootstrap (CCA's near-diagonal cross-covariance produces structural upward bias; point estimate is correct). source: `scripts/eval_h1_bootstrap_ci.py`; plot-ready arrays at `runs/tnbc-92_v3/{run}/h1_eval_data.npz`. **note**: v3 CKA values differ materially from v1 - R3 CKA 0.306 now exceeds R6 0.239, and B4 random-init CKA 0.120 exceeds R2 SupCon 0.101, consistent with the H2-B.2 finding that some contrastive variants degrade morphology-decodable structure on z_he.

**What this means in plain language**: R4 (cross-attention) has an 85.9% probability that a randomly chosen matched H&E-ST niche pair will have higher cosine similarity than a randomly chosen mismatched pair. This is substantially above chance (50%) and above every baseline. **5 of 6 contrastive runs (R4, R1, R5, R6, R2) beat every classical baseline on AUC with non-overlapping 95% CIs.** R3 Barlow Twins [0.671, 0.677] is the one contrastive run that loses to a classical: it sits below B2 Procrustes [0.703, 0.710] with a 0.026 CI gap. Barlow's redundancy-reduction objective underperforms orthogonal Procrustes on cross-modal retrieval at this scale.

**Important caveat**: Absolute R@1 across ALL runs is ~0.0002 (chance ~0.00002). At 35,594 test niches (v3), exact niche-level retrieval across patients is nearly impossible. The AUC tells you the alignment *orders* matched pairs above mismatched pairs; it does not tell you the alignment can *find* the exact match. This is the difference between "the ranking is good" and "the retrieval works in production." The project pivoted to AUC as the discriminating metric and formally amended PLAN_RULES.md (2026-06-01) to reflect this. Defensible at this scale, but a reviewer will notice R@1 is 10x above chance - modest absolute amplification.

### 4.2 H2 - Structural Coherence: PARTIALLY SUPPORTED (with construct-validity corrections)

This is where the project's self-audit really shines.

**Part A - The archetype trap**: The proposal said to test 9 spatial archetypes. The project discovered that archetypes are patient-level labels (NMI with patient_id = 0.89). Every niche in a patient inherits one archetype. So "archetype ARI" is just patient classification in disguise. B1 CCA wins by amplifying patient identity 3.3x. This is a textbook construct-validity failure, and the project caught it.

**Part A correction - mc_megacluster (14-class NMF, niche-resolution)**:

| Run | KMeans ARI | Linear Probe Acc | Interpretation |
|---|---|---|---|
| R5 | 0.246 | - | Top KMeans cluster |
| R6 | 0.244 | 0.276 | Parsimony winner (novae-only ST) |
| R1 | 0.234 | 0.282 | Top linear probe |
| R4 | 0.101 | 0.240 | KMeans-bad, linear-probe-parity |
| raw ST | 0.046 | - | Baseline floor |

**What R4's KMeans failure means**: R4's cross-attention compresses into a low-dimensional manifold (rank-matched CKA control shows R4@rank3 ~ R4 full, indicating most variance lives in very few dimensions). KMeans assumes globular clusters. In a low-rank manifold you cannot fit 14 globular clusters. But a linear probe (which only needs linear separability, not globular geometry) recovers R4 to parity with R1. The biology is there; it is encoded as linearly-separable directions, not blobs. This is a geometry-of-the-manifold issue, not a biology-absence issue.

**Part B - Compartment contrasts: 1/6 supported (TLS only)**:

Only TLS-vs-TIL stroma contrasts pass. Tumor and Necrosis contrasts INVERT because those annotations are large heterogeneous classes with internal sub-types that dilute within-compartment cosine. The small, well-curated TIL annotations (n=105-354) have higher internal purity. This is an annotation-vocabulary limitation, not a biology failure.

**Part B.2 - TLS gene signature CCA (new in v3)**:

A striking finding: B4 (random-init, untrained MLPs) has the STRONGEST TLS z-score (25.5). This means TLS is directly decodable from raw Virchow2 features - organized lymphoid aggregates have a distinctive morphological signature. Contrastive late-fusion training actually *degrades* this signal on z_he. Contrastive alignment is not free - it trades single-modality decodability for cross-modal coupling.

### 4.3 H3 - Pathway Interpretability: SUPPORTED (cross-patient is the real test)

**Part A (fit-on-all)**: Every run passes. All 120 cells BH-FDR < 0.001. This test is saturated - 512d can always find a 1d direction that correlates with a target. The test does not discriminate between runs. It confirms the hypothesis is trivially true within-cohort.

**Part B (cross-patient, the discriminating test)**:

| Run | Sig pathways /4 | z_A range (z_he) | Off-diag cosine |
|---|---|---|---|
| **R4** | **4/4** | **11.2-25.4** | **0.75 (aliased)** |
| B1 | 4/4 | 7.5-15.2 | 0.41 |
| B2/B3 | 4/4 | 6.7-17.1 | 0.38 |
| R5 | 2/4 | -1.9 to 3.0 | - |
| R1 | 1/4 | -0.2 to 2.7 | 0.48 |
| R6 | 0/4 | 0.3 to 1.3 | 0.45 |
| R2 | 0/4 | -2.2 to -1.1 | 0.41 |

**The aliasing mechanism**: R4's 5 named pathways all project onto roughly the same direction in the shared space (off-diagonal cosine 0.75). This looks like a failure of specificity, but it is actually the *mechanism* of cross-patient generalization. One robust biology axis (correlated across all 5 Reactome pathways) is harder to overfit to patient identity than five patient-specific gradient directions. R1/R6 encode more distinct per-pathway directions - but those distinctions are patient-specific and do not transfer.

**What this means practically**: R4 can tell you "this niche is biologically active" across new patients. It cannot tell you "this niche is specifically immune-active rather than ECM-active" because those pathways alias onto the same direction. For per-pathway resolution, you need B2 (Procrustes) despite its lower z_A magnitude.

---

## 5. Documentation and Provenance: Exemplary Self-Audit, Some Drift

### What is exceptional

| Document | Why It Matters |
|---|---|
| `evaluation_question_audit.md` | Catalogs 5 cases where the evaluation question itself was flawed. This is rare in comp-bio. |
| `proposal_deviations.md` | Rigorous cross-reference of proposal commitments vs execution. Every deviation documented. |
| `05_summary_umaps_REVIEW.md` | A review of the review notebook, with blockers and corrections tracked. |
| v1 -> v3 versioning | Numbers preserved in v1 docs; corrections applied in v2 docs with explicit notes. |
| SHA-locked embeddings | gpath2vec builds pinned by sha256 hash. Prevents silent artifact drift. |

### Documentation drift (stale files)

| File | What's Stale | Impact |
|---|---|---|
| `program.md` | Says shared_dim=256, ST input=65d. Actual: 512d shared, 576d ST input. | Misleading if someone reads this as current spec |
| `MODELS.md` alignment module | Says `Linear(1536, 512)` implying UNI2. Actual: Virchow2 1280d. | Documentation drift from encoder switch |
| `PLAN_RULES.md` | ~~R@1 threshold stale~~ **amended 2026-06-01**: AUC replaces R@1 as stopping criterion; archetype label construct-validity rule added. | Resolved |
| `README.md` | Says "UNI2" as primary H&E encoder. Actual: Virchow2. | Stale from before encoder comparison |
| `data_summary.md` | Empty file (1 byte). | Unused placeholder |
| `PLAN.md` (alignment) | Lists R5 as "early fusion (deferred)". v3 R5 is AnInfoNCE + late fusion. Does not include R6. | Reflects an intermediate plan, never updated |

---

## 6. Cross-Reference Audit: Claims vs Code vs Data

| Claim | Source | Verified Against | Match? |
|---|---|---|---|
| R4 AUC 0.859 (v3) | `tnbc92_results_summary.md` | `05_summary_umaps_v3.ipynb` loads from `metrics_h1_raw.json` | YES |
| R4 AUC 0.851 (v1) | `current_report.md` | `proposal_h1h2h3_evaluation.ipynb` loads from same JSON | YES |
| NMI(archetype, patient_id) = 0.89 | `tnbc92_results_summary.md` | v3 notebook computes 0.893, v1 notebook computes 0.897 | YES (rounding) |
| 260 matched subarrays (v1) | `current_report.md` | `eda_summary.json` says 249 matched | DISCREPANCY (see below) |
| 208,786 niches (v3) | `tnbc92_results_summary.md` | v3 notebook loads from `niches_v3/` | YES |
| B2/B3 produce identical metrics on rotation-invariant measures | Both results summaries | B2=B3 on CKA, archetype ARI, H3 z_A; differ on AUC (0.706 vs 0.441) | EXPLAINED (see below) |
| B4 AUC 0.491 vs 0.496 | `current_report.md` vs `tnbc92_results_summary.md` | Different versions (v1 vs v3) | LIKELY v1/v3 number mixed |

**The 249 vs 260 subarray discrepancy**: `eda_summary.json` records 249 matched subarrays across 89 patients (from the scale embedding run). `current_report.md` says 260 matched subarrays. The difference is likely explained by the niche-join process adding gpath2vec coverage that increased the intersection, but this is not explicitly documented anywhere.

**B2/B3 identity explained**: The results_summary.md correctly calls B2 and B3 "rotation-equivalent." Orthogonal Procrustes (B2) finds the rigid rotation that best aligns B3's independent per-modality PCA outputs. CKA, ARI, and per-pathway CCA are all rotation-invariant metrics - they measure the same pairwise-similarity structure regardless of axis orientation. So B2 and B3 give identical numbers on those metrics by mathematical construction, not by accident. They diverge on AUC (B2 = 0.706 vs B3 = 0.441) because cross-modal cosine similarity IS rotation-sensitive - Procrustes rotates the axes into alignment, which directly improves cosine-based retrieval. This is exactly what the math predicts.

---

## 7. Comprehensive Pros and Cons

### PROS

| Category | Finding | Evidence |
|---|---|---|
| **Methodology** | Empirically-driven encoder selection | Virchow2 chosen by MC linear probe (23.6% vs UNI2), not by citation |
| **Methodology** | Circular dependency caught and quarantined | mc_weights removed from ST input; z_he used as non-circular headline for v3 gpath2vec |
| **Methodology** | Patient-stratified splits throughout | 85/15 patient split, seed=42; early experiment demonstrates collapse without it (R@1 0.13 -> 0.002) |
| **Methodology** | Cross-subarray evaluation discipline | Within-subarray R@1 = 0.667 collapses to 0.000 cross-subarray; correctly identified as leakage |
| **Self-audit** | Construct-validity corrections documented | Archetype = patient-level; H3 fit-on-all saturated; gpath2vec v2 degenerate - all caught in-flight |
| **Self-audit** | Pre-registered predictions for commits 2.5/2.6/3/3.5 | Predictions documented before running, outcomes compared to predictions |
| **Self-audit** | evaluation_question_audit.md | 5 cases where the evaluation question itself was wrong, each with formal diagnosis and rescue |
| **Reproducibility** | SHA-locked embeddings, seed=42, run provenance JSON | Full artifact chain from config to metrics |
| **Results** | v1 -> v3 directional verdict reproduces | Same winners across retrains despite 27% niche reduction |
| **Results** | No-winner finding is itself valuable | Task-conditional routing (R4 for retrieval, R1/R6 for clustering) is the deliverable |
| **Results** | Fisher EA pipeline scaled properly | Pilot vacuous criterion caught, full-cohort pipeline uses proper BH-FDR with QC gates |
| **Novelty** | Cross-attention bridge (R4) finding | Aliasing-as-generalization mechanism: one robust axis transfers better than five patient-specific gradients |
| **Novelty** | TLS B.2 finding | Contrastive training degrades signals already encoded by the foundation model - quantified trade-off |

### CONS

| Category | Finding | Severity | Impact |
|---|---|---|---|
| **Absolute performance** | R@1 ~ 0.0002 across ALL runs (chance 0.00002) | Medium | Niche-level exact retrieval is not practically useful at this scale; only ranking quality is demonstrated |
| **Statistical** | 500-1000 permutations for H3 nulls | Low-Medium | Adequate for z>5, but PCD at z=2.61 (BH-FDR 0.050) is right at the boundary; more perms could shift the call |
| **Statistical** | No confidence intervals on AUC/CKA | Low | With 35k test niches (v3), CIs would be tight; reporting them would strengthen claims |
| **Missing baseline** | PCA-HVG full-cohort never built | Low | Novae passed EDA validation, so PCA-HVG fallback was not triggered per PLAN_RULES.md; nice-to-have for completeness |
| **Documentation drift** | program.md, MODELS.md, PLAN.md, README.md all stale | Medium | Could mislead a new reader about actual architecture |
| **H3 aliasing** | R4 off-diagonal cosine 0.75-0.82 across pathways | Medium | R4 cannot distinguish pathway-specific signals; all 5 pathways -> 1 direction |
| **H2 Part B** | 5/6 compartment contrasts INVERT | Medium | Annotation vocabulary issue, but the proposal predicted the wrong direction for most contrasts |
| **Phase 2** | MCP server, agents, LangGraph, Claude routing - 0% built | High | The proposal commits to an operational agent layer; only scripts exist |
| **Notebook hygiene** | Duplicate notebook (aucell = pathway_analysis), stale markdown headers | Low | Confusing but not scientifically impactful |
| **Novae bottleneck** | 11/64 effective dimensions | Medium | The ST modality brings very little information diversity; 512d projection from 11 effective dims |
| **Compartment coverage** | Only 13/38 test subarrays have annotations (35%) | Medium | H2 Part B results are based on a minority of the test set |
| **PLAN_RULES.md** | R@1 stopping rule replaced by AUC | Resolved | Formally amended 2026-06-01 with honest-mistake framing; archetype label rule also added |

---

## 8. Hypothesis Verdict Summary

| Hypothesis | Proposal Claim | Verdict | Strength of Evidence | Key Caveat |
|---|---|---|---|---|
| **H1** | Contrastive beats unaligned on retrieval | **SUPPORTED** | Strong: clean AUC separation, no family overlap, B4 chance anchor | R@1 absolute performance is near-floor; only ranking quality demonstrated |
| **H2-A** | Archetypes cluster better in aligned space | **SUPPORTED after construct-validity correction** | Strong: 5x lift over raw ST on audit-correct mc_megacluster label | Proposal-literal test fails (archetype = patient-level); correction is principled but changes the target |
| **H2-B.1** | Distinct compartments > ambiguous stroma (categorical) | **Partially supported (TLS only)** | Moderate: TLS passes in 5/9 runs; Tumor/Necrosis invert | 5/6 contrasts fail; annotation vocabulary drives the inversion, not biology absence |
| **H2-B.2** | TLS gene signature decodable from z_he (continuous) | **SUPPORTED for morphology-preserving methods** | Strong: B4 random-init strongest (z=25.5); 6/10 runs sig | Contrastive late-fusion DEGRADES this signal - informative negative finding |
| **H2-C** | Aligned space encodes biology > patient identity | **SUPPORTED for contrastive; B1 fails** | Strong: all 6 contrastive runs above raw floor; B1 below floor | Extension metric not in original proposal |
| **H3** | Pathway CCA generalizes cross-patient | **SUPPORTED for R4 + classical baselines** | Strong: R4 4/4 sig (z 11-25); classical 4/4 with lower z; late-fusion 0-2/4 | R4 achieves this by aliasing 5 pathways onto 1 axis (no per-pathway specificity) |
| **Systems** | No single method dominates all objectives | **SUPPORTED** | Strong: R4 wins H1+H3, R1/R5/R6 win H2; reproduced v1 -> v3 | This is the expected result; the routing table is the deliverable |

---

## 9. What Works Best - The Routing Table

This is the core finding and what the MCP agent should encode:

| If Your Question Is | Route To | Why |
|---|---|---|
| Find the ST niche that matches this H&E patch | **R4** | AUC 0.859, CKA 0.631; cross-attention gives best cross-modal coupling |
| Group niches into TME regions (immune-hot, stromal, etc.) | **R5/R6/R1** | KMeans ARI 0.234-0.246 on mc_megacluster; globular manifold supports clustering |
| Transfer pathway signal to unseen patients | **R4** | 4/4 pathways sig cross-patient (z 11-25); aliased axis transfers |
| Distinguish WHICH pathway is driving a niche | **B2 (Procrustes)** | Lowest off-diagonal aliasing; preserves per-pathway direction separation |
| Minimize patient-identity leakage | **R6** | Highest bio/patient ratio (0.547); drops gpath2vec for parsimony |
| Decode TLS from H&E morphology alone | **B4 or B1** | TLS is already in Virchow2 features; contrastive training hurts this signal |
| Avoid always | **B1 (CCA) for cohort-level interpretation** | Amplifies patient identity 3.3x; bio/patient ratio 0.071 (below raw floor) |

---

## 10. Readiness Assessment for MCP Package

| Component | Readiness | What Exists | What's Missing |
|---|---|---|---|
| Embedding pipelines | **Ready** | 260-subarray niche-join tables, Virchow2 + Novae + gpath2vec | Full-cohort PCA-HVG baseline |
| Alignment configs | **Ready** | R1-R6 + B1-B4 JSON configs, v1 and v3 | Training code is in gitignored `scripts/`, not `src/` |
| Evaluation framework | **Ready** | H1/H2/H3 eval scripts produce JSON/parquet artifacts | Per-subarray reporting not implemented |
| Routing logic | **Ready** | Task-conditional routing table derived from 3 hypotheses | Not encoded as code; lives in markdown only |
| MCP server | **Not started** | `docs/mcp_agent_design.md` spec exists | `src/mcp/` is empty |
| Agent classes | **Not started** | Agent constraints defined in CLAUDE.md | No code in `src/agents/` |
| LangGraph orchestrator | **Not started** | Design doc exists | No implementation |
| Release artifacts | **Not started** | - | No pip package, no HF checkpoint, no Zenodo archive |

**The gap**: Phase 1 (science) is complete and well-documented. Phase 2 (operational tool) is 0% implemented. The scripts that run the pipeline (`align.py`, `eval.py`, `build_niche_join.py`) are already "config in, artifacts out" - wrapping them as MCP tools is engineering, not new science. The routing table above is the intelligence the agent needs to encode.

---

## 11. Teaching Statements (Unambiguous Takeaways)

1. **Cross-attention alignment (R4) is the best cross-modal coupler in this dataset, and it achieves this by compressing the shared space into a low-dimensional manifold** (rank-matched CKA control shows R4@rank3 ~ R4 full). That compression sacrifices cluster geometry (bad for KMeans) but creates a single robust biology axis that transfers across patients (good for generalization). You cannot have both globular clusters and a tight generalizable manifold - pick one per task.

2. **The within-cohort CCA test (H3 Part A) is worthless for method comparison.** Every method passes it. With 512 dimensions and 35,594 test niches (v3), you can always find a 1d direction that correlates with any target above a permutation null. The cross-patient sub-split is what actually discriminates.

3. **Patient identity and biological signal live in the same linear subspace in spatial omics.** The Virchow2 niche ablation (`virchow2_cell_vs_niche_interp.ipynb`) proves this: removing patient-locked PCs from the embedding kills MC biology signal in lockstep. Batch correction and biology preservation are fundamentally in tension at the embedding level. Every method navigates this trade-off differently.

4. **CCA (B1) is anti-helpful for cohort-level biology.** It amplifies patient identity 3.3x over raw H&E and scores below the raw-modality floor on the bio/patient ratio. If you use CCA-aligned embeddings for cohort-level interpretation, you are interpreting patient identity as biology. Procrustes (B2) is the better classical choice.

5. **Contrastive training has a cost on signals already encoded by the foundation model.** TLS morphology is already in Virchow2 features (random-init B4 achieves z=25.5 on TLS CCA). Late-fusion contrastive training degrades this signal while gaining cross-modal coupling. This is a measurable trade-off, not a free upgrade.

6. **The proposal's annotation-based predictions about compartment cosine were wrong for 5/6 contrasts, but for the right reason.** The inversion is an annotation-vocabulary problem (Tumor and Necrosis are heterogeneous aggregates), not a biology problem. The one well-curated annotation (TLS = organized lymphoid structure) passes exactly as predicted.

7. **gpath2vec v3 (Fisher + MAD/mean + 512d + 5 epochs) is a valid pathway embedding; v2 (AUCell + all levels + 128d + 1 epoch) was degenerate.** v2 collapsed to mean cosine 0.878 regardless of label (MC z = -0.13). The noise-floor validation (matched-null permutation z-test on cosine coherence) correctly caught this. Always validate embedding quality before using it as input.

8. **Absolute retrieval at niche-level across patients is effectively impossible at this scale.** R@1 ~ 0.0002 means 1 in 5000 queries finds its exact match in the top position. AUC of 0.85 means the *ranking* is good, but the *retrieval* is not production-ready. For practical cross-modal search, you would need region-level (not niche-level) queries, or within-patient retrieval.