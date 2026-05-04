# tnbc-92 current report

status: H1 sweep + classical baselines + R6 ablation complete; H2 ARI flagged as confounded (proper per-compartment cosine pending); H3 biology vs patient z-test applied to R1/R4/R6/B1 at 1e5 perms; B2/R3/R2 biology validation + rank-matched CKA + per-compartment cosine remain as gating work before the H1/H2 fusion-axis verdict can be adjudicated.
scope: Wang et al. 2024 TNBC cohort, 94 patients, 260 matched subarrays.

## current status by hypothesis

| Hypothesis | What we intended to test | Math / metric used | What actually happened | Correct interpretation |
|---|---|---|---|---|
| **H1 - Alignment (loss axis)** | Do learned contrastive embeddings align H&E ↔ ST better than classical methods? | InfoNCE objective; evaluated with AUC, CKA, R@K | All contrastive (R1–R4) > classical (CCA, Procrustes, PCA) on AUC/CKA | ✔ Strong evidence of shared cross-modal structure (nonlinear, biological signal likely present) |
| **H2 - Structure (fusion axis)** | Does the shared embedding preserve biological structure (archetypes, compartments)? | KMeans + ARI vs archetype labels; silhouette | CCA > raw H&E > R1 > R4 (R4 ~0 ARI) | ❌ ARI conflates biology + patient signal -> cannot distinguish whether structure is biological or confounded |
| **H3 - Biology (interpretability)** | Does the embedding encode meaningful biological programs and reduce confounders? | Subarray-level permutation z on biology labels (Bareche MC_global, MC_tumor, TIME) vs patient_id; matched-null for patient, label-shuffle for biology; 1e5 perms | gpath2vec showed biology ↑, patient ↓; framework ported to alignment and applied to R1/R4/R6/B1. R1/R4/R6 all show biology↑ patient↓; CCA (B1) amplifies patient 3.3×, fails H3 | ✔ Framework working; B2/R3/R2 pending to complete the falsifier table |
| **Observed phenomenon** | Does alignment preserve dimensional richness? | Effective rank (participation ratio), cosine distribution | R4 has ~3 effective dims with matched cos ~0.60 / mismatched ~0.41 (tight manifold); R1 has ~5 dims with matched ~0.24 / mismatched ~0.09 (wider spread); R6 has ~6 dims; B1 ~257 | Architectural trade-off: tight manifold → strong cross-modal alignment (R4 H1 wins) but weak within-space resolution. Not "destruction"; a genuine regime difference. |
| **Methodological issue discovered** | Are we measuring "biology" correctly? | ARI vs archetypes (single score) | ARI high for CCA, low for R4 (contradiction) | ❌ ARI mixes biology + patient -> invalid as primary biological metric |
| **Correct evaluation (from prior work)** | Separate biology vs confound signal | Permutation-based z-scores (biology vs patient) | gpath2vec showed asymmetric smoothing (biology preserved, patient reduced) | ✔ This is the proper disentanglement test and should be reused |

### what actually happened

Initially, the goal was to test whether cross-modal aggregation improves biological interpretability. The alignment experiments (H1) showed strong results: contrastive models significantly outperformed classical baselines in AUC and CKA, indicating that the learned embeddings capture shared cross-modal structure beyond simple linear correlation. This suggests that real underlying signal - likely biological - is being aligned between H&E and spatial transcriptomics.

However, when evaluating biological structure using ARI (H2), the results appeared contradictory: the best alignment model (R4) showed near-zero archetype coherence, while classical CCA - poor at alignment - showed the highest ARI. This discrepancy arises from a methodological issue: ARI does not isolate biological structure, but instead mixes biological variation with patient-specific effects. Since archetypes are defined per patient, high ARI can simply reflect patient clustering, not true biological interpretability. As a result, H2 as currently measured is not a valid test of biological preservation.

The earlier gpath2vec validation provides the correct framework: it separates biological signal (pathways, compartments) from confounding signal (patient identity) using permutation-based z-scores. That analysis showed asymmetric smoothing — biology preserved or enhanced, patient identity reduced — which is the desired outcome. This framework has now been ported to the alignment layer and applied to R1, R4, R6, and B1 at 1e5 perms. R1/R4/R6 all show the target signature (biology z ↑ vs raw_he, patient z ↓ vs raw_he); B1 CCA fails this test (amplifies patient 3.3×). B2, R3, and R2 biology validations remain as the falsifier tests for the mechanism claim (see mechanism section below).

At the same time, an architectural phenomenon emerged: the cross-attention model (R4) lands in a tight low-effective-rank manifold (~3 dims) while late-fusion (R1) occupies a wider manifold (~5 dims). This is a trade-off, not a failure. The tight manifold gives R4 the strongest cross-modal retrieval metrics (AUC, CKA) at the cost of within-space resolution; the wider manifold gives R1 better z_he biology absolute and better dim-richness at the cost of lower AUC and asymmetric stream quality. Both preserve biology above raw and compress patient below raw; which trade-off is preferable depends on the downstream use.

**Putting it together**:

- H1 is validated: all 4 contrastive methods beat all 3 classical baselines on AUC and CKA
- H2 as measured via ARI is mismeasured (ARI cannot distinguish biology from patient when labels are per-patient); proper H2 via per-compartment cosine on the 90-subarray annotated subset is pending
- H3 biology vs patient z-test has been applied: R1, R4, R6 all pass; B1 CCA fails (ratio below raw floor)
- Fusion-axis verdict (R1 vs R4) is held open: rank-matched CKA control, paired-perm test on (bio_delta − patient_delta), and per-compartment cosine are the three measurements needed before committing
- The scientific finding so far: the loss + negative-sampling regime matters more than the projection — contrastive with cross-patient negatives preserves biology and compresses patient; linear methods that preserve input geometry (CCA, likely Procrustes) amplify patient

---


## summary

### what is fused, and how

each run aligns **two streams** via contrastive loss on matched (H&E, ST) niche pairs. streams:

- **H&E stream**: `virchow2_niche` (1280d, mean-pooled over self + 6 spatial neighbors)
- **ST stream**: `[novae_niche(64, z-score per sub) ⊕ gpath2vec_niche(512)]` = 576d concatenated

| run | how the two streams are fused | loss |
|---|---|---|
| R1 | independent MLPs: 1280 -> 512 (H&E), 576 -> 512 (ST). no interaction. | InfoNCE |
| R2 | same as R1 architecture | SupCon (mc_weights soft targets) |
| R3 | same as R1 architecture | Barlow Twins |
| R4 | ST query (576 -> 512) attends over 7 H&E tile tokens (from virchow2_cell). H&E output is pure attention-weighted tiles, no ST residual. | InfoNCE |
| R6 | same as R1 but ST input = novae 64d only (gpath2vec dropped). ablation for circularity check. | InfoNCE |
| B1 | closed-form CCA on train, applied to full cohort. linear. | n/a (no loss) |
| B2 | per-modality PCA to 512 + orthogonal Procrustes. linear. | n/a |
| B3 | per-modality PCA to 512, L2-norm. no alignment. | n/a |

gpath2vec is **inside the ST stream**, not its own stream. (see architecture-clarification section.)

### interim results - trade-offs, not a verdict

H1 and H3 measurements are in; H2 has not been measured correctly yet. Below is the honest trade-off picture across runs. **Crowning a winner is not yet defensible.** See "what is unmeasured" for the gating work.

**H1 (cross-modal alignment quality)**: all 4 contrastive runs > all 3 classical on AUC and CKA. R4 (AUC 0.851, CKA 0.564) leads by a wide margin.

**H2 as currently reported is confounded**: ARI on 9 per-patient archetypes conflates biology with patient identity. CCA topped this metric (0.319) because it amplified patient z by 3.3× on H&E (patient z_he = 122 vs raw 37). Pending a proper H2 (per-compartment cosine on the 90-subarray annotated subset), the fusion axis cannot be adjudicated.

**H3 (biology vs patient, subarray-level permutation z, 1e5 perms, Bareche labels external to the model)**: trade-off table across four runs and two raw references.

| run | z_he TIME | z_he patient | z_he ratio | z_mean TIME | z_mean patient | z_mean ratio | eff rank z_he | dim cost | AUC / CKA |
|---|---|---|---|---|---|---|---|---|---|
| R1 infonce+late | 15.13 | 29.27 | 0.517 | 13.86 | 33.22 | 0.417 | 4.8 | moderate | 0.741 / 0.242 |
| R4 infonce+cross_attn | 10.07 | **22.97** | 0.438 | 11.71 | **22.83** | **0.513** | **3.0** | tight | **0.851 / 0.564** |
| R6 infonce+late (novae-only ST, ablation) | **17.81** | 32.52 | **0.548** | 13.49 | 36.32 | 0.371 | 6.5 | dim-rich | 0.733 / 0.225 |
| B1 CCA (classical) | 7.87 | **122.13** | 0.064 | 17.48 | 80.04 | 0.218 | 257 | uncompressed | 0.541 / 0.086 |
| raw_he (floor) | 4.79 | 37.25 | 0.129 | - | - | - | - | - | - |
| raw_st (floor) | 6.44 | 28.64 | 0.225 | - | - | - | - | - | - |

**What each run does cleanly**:
- R4: best H1 (AUC, CKA), best patient suppression, best z_mean ratio, symmetric across streams - at the cost of ~3 effective dimensions (tight manifold, poor within-space resolution)
- R1: best H1 alignment gap to dim-richness balance, best z_he absolute ratio, asymmetric (z_he strong, z_st weak)
- R6 (gpath2vec dropped from ST): highest z_he biology preservation, most dim-rich - but z_st loses biology signal almost entirely (TIME z = 1.01), so R6 is not a contender for the alignment objective as a whole
- B1 CCA: preserves ST geometry (z_st TIME z = 17.4, highest) but catastrophically amplifies patient on H&E. Cautionary baseline.

**What the trade-off actually is**:
- R4 maximizes cross-modal alignment quality (AUC, CKA, patient suppression, stream balance) while sacrificing dim richness. The low effective rank hurts niche-level retrieval / interpretability but does not hurt the group-level H1/H2/H3 signals the proposal is graded on.
- R1 preserves more dim richness and has higher H&E biology absolute, at the cost of poorer cross-modal alignment and asymmetric stream quality.
- These are genuinely different alignment regimes, both defensible for different downstream uses. The proposal's own objectives (group-level retrieval + interpretability) could be argued either way.

### what is unmeasured (gating work before adjudicating a winner)

| measurement | why it matters | status |
|---|---|---|
| **per-compartment cosine on 90-sub annotated subset** | the honest H2 - tests whether shared space preserves 18-class compartment structure independently of patient. without this we cannot evaluate the fusion axis on H2. | run in eval.py H2; pending writeup |
| **rank-matched CKA control** (project R1 to its top-3 PCs, recompute CKA; compare to R4's CKA_after 0.564) | if R1 at rank 3 reaches R4's CKA, R4's H1 advantage is formally "collapse-driven"; if not, R4 is doing something more than compression. | not done |
| **paired permutation test on (bio_delta − patient_delta) per run** | the only rigorous way to say "R1 disentangles more than R4" or vice versa. uses shared-seed null indices; non-parametric. | paired_tests.parquet has paired tests on bio and patient deltas separately, not on the difference. needs the joint form. |
| **R6b symmetric ablation: gpath2vec-only ST (drop novae)** | R6 dropped gpath2vec from ST. R6b drops novae. without the symmetric ablation we cannot say which ST feature is load-bearing for which downstream property. | not done |
| **AnInfoNCE (R5)** | tests whether R4's alignment gains can be preserved while recovering dim richness. directly addresses the R4-vs-R1 trade-off. | not done |
| **NaN-cleaned SupCon rerun** | current SupCon result is pessimistic due to uniform imputation of NaN mc_weights rows. cleaner rerun would validate or refute "InfoNCE > SupCon". | not done |

Until at least the first three land, the H1/H2 fusion-axis verdict is open. The trade-off table is the honest report.

### what is supported by the data as of now

- **all 4 contrastive methods outperform all 3 classical methods on AUC and CKA**. R4 leads H1 by the widest margin on both.
- **CCA's H2 ARI advantage was patient leakage** (paired-perm p < 1e-5 on patient delta vs raw_he; biology z = 7.87 with patient z = 122.13). The ARI result should not be reported as a biology claim.
- **learned contrastive alignment produces biology preservation that matches or exceeds raw modality z** on every trained run: R1/R4/R6 all show TIME biology z above raw_he z = 4.79 and above raw_st z = 6.44.
- **the cross-attention architecture (R4) achieves this with ~3 effective dimensions**; the late-fusion architecture (R1) achieves it with ~5 effective dimensions at the cost of lower AUC/CKA. This is a real architectural trade-off.
- **R6 ablation shows H&E carries biology signal independently of gpath2vec** (z_he biology z rises without gpath2vec in ST), but at the cost of near-zero z_st biology. Partially addresses the circularity concern; R6b is needed for the symmetric answer.

### biology vs patient by method + embedding (H3, z_he view, subarray-level 1e5-perm z)

| method | H&E input | ST input | how fused | z_he bio (TIME) vs raw | z_he patient vs raw | z_he ratio | verdict |
|---|---|---|---|---|---|---|---|
| raw H&E (floor) | virchow2_niche 1280d | - | - | 4.79 | 37.25 | 0.129 | - |
| raw ST (floor) | - | novae(64)+gpath(512) = 576d | - | 6.44 | 28.64 | 0.225 | - |
| R1 InfoNCE+late | virchow2_niche 1280d | novae+gpath 576d | indep MLPs → 512d each | 15.13 (↑ 3.2×) | 29.27 (↓ 0.79×) | 0.517 | ✅ bio↑ patient↓ |
| R4 InfoNCE+cross_attn | virchow2_cell 7 tokens × 1280d | novae+gpath 576d | ST query attends over 7 H&E tiles | 10.07 (↑ 2.1×) | 22.97 (↓ **0.62×**) | 0.438 | ✅ bio↑ patient↓↓ |
| R6 InfoNCE+late, novae-only ST | virchow2_niche 1280d | novae 64d (no gpath2vec) | indep MLPs → 512d each | **17.81 (↑ 3.7×)** | 32.52 (↓ 0.87×) | **0.548** | ✅ bio↑↑ patient↓ |
| B1 CCA (classical) | virchow2_niche 1280d | novae+gpath 576d | closed-form SVD of whitened cov | 7.87 (↑ 1.6×) | **122.13 (↑ 3.3×)** | **0.064** | ❌ **bio↑ patient↑↑↑** |

H3 PASS criterion: biology z rises above raw AND patient z falls below raw AND bio/patient ratio exceeds raw_he floor 0.129. R1, R4, R6 pass; B1 fails (ratio 0.064 is below the raw_he floor — CCA is anti-helpful, not just unhelpful).

### mechanism (falsifiable prediction)

**CCA fails H3 because it maximizes between-modality linear correlation with no structural mechanism against dominant nuisance variance.** Patient identity is the single biggest shared axis between paired H&E and ST from the same tissue (same tumor architecture shows in both modalities), so CCA writes that patient axis first and biggest. The 1.6× biology amplification vs 3.3× patient amplification is exactly the predicted signature, and the bio/patient ratio dropping below the raw floor is the clean diagnostic.

**InfoNCE succeeds because our training batches use cross-patient negatives.** The softmax over matched vs mismatched pairs implicitly penalizes patient-axis explanation — if patient identity over-explains matched pairs, it also over-explains cross-patient mismatched pairs, so the softmax gradient pushes the model to find features that differentiate matched from mismatched that are not just patient. **This is a property of InfoNCE with our specific negative-sampling regime, not a general property of InfoNCE.** If batches were patient-stratified (negatives drawn within-patient), InfoNCE would have no such bias.

**Predictions for the four unmeasured cells**:

| run | method | predicted H3 verdict | why |
|---|---|---|---|
| **B2 Procrustes** | orthogonal alignment after PCA to 512 | **predicted ❌ fail like CCA** | orthogonal constraint preserves input geometry including dominant patient axis; no mechanism to suppress nuisance variance |
| **R3 Barlow Twins** | cross-correlation matrix penalty toward identity | **predicted outcome informative either way** | Barlow doesn't use softmax over negatives - uses redundancy-reduction via cross-correlation diagonal + off-diagonal penalty. If R3 passes H3, "softmax over negatives" is not the necessary mechanism ("any redundancy-reducing loss" suffices). If R3 fails, the softmax-over-negatives story holds. |
| **R2 SupCon** | InfoNCE variant with mc_weights soft targets | **predicted ✅ pass** | same negative-sampling regime as R1/R4; mc_weights supervision is orthogonal to the softmax mechanism |
| **B3 Unaligned PCA** | per-modality PCA, no cross-modal coupling | **predicted neither amplifies nor compresses** | no loss function, no coupling; should roughly preserve raw_he ratio |

B2 is the **primary falsifier**. Its predicted failure mode is classical-alignment-without-nuisance-suppression. If B2 passes H3 (bio/patient ratio above raw_he floor), the mechanism story is wrong and needs revision - the softmax-vs-no-softmax axis would not be explanatory. If B2 fails as predicted, the mechanism is well-supported across two independent classical methods (CCA and Procrustes) making the same error.

R3 is the **secondary falsifier**. It tests whether softmax-over-negatives is the specific mechanism or whether any redundancy-reducing contrastive loss suffices.

### correlation readout (heuristic)

| property of method | tends to fail H3? | tends to pass H3? |
|---|---|---|
| linear correlation-maximizing without nuisance penalty | **yes** (CCA confirmed; Procrustes predicted via orthogonal-preservation failure mode) | - |
| non-linear projection trained with softmax over cross-patient negatives | - | **yes** (R1, R4, R6 confirmed) |
| redundancy-reducing loss without softmax over negatives (Barlow) | pending (R3 is the informative test) | pending |
| sees patient identity as implicit training signal | **yes, will amplify** | - |
| sees patient identity only via held-out labels | - | **yes, biology wins** |

The falsifiable claim: **classical methods that preserve input geometry directly (including dominant patient axis) will fail H3; non-linear contrastive methods with cross-patient negatives will pass H3.** B2 and R3 runs would either support or refute this within the current experimental grid.

## infrastructure

| component | scope | location |
|---|---|---|
| Virchow2 niche (H&E, 1280d) | 280 subarrays | `data/embeddings/virchow2_niche/` |
| Virchow2 cell (H&E, per-spot, 1280d) | 280 | `data/embeddings/virchow2_cell/` |
| Novae niche (ST, 64d + k=6 neighbor ids) | 262 | `data/embeddings/novae_niche_full/` |
| gpath2vec niche (512d, 286k niches) | 280 | `data/embeddings/biological_signals/gpath2vec_output/full_cohort_tf_low/` |
| matched niche join table | 260 | `data/embeddings/niches/` |

Novae coverage is capped at 262 by the pretrained model's 512-spot
architectural floor; the Virchow2 scope of 280 is unaffected. matched
training scope is **260 subarrays** (intersection).

## validated findings

**gpath2vec pathway embedding preserves biology, compresses patient identity**
(subarray level, TNBC cohort, 10⁵ permutations, Bonferroni α=0.05/8 — family of 8 tests (4 labels × {embedding, raw_ea}). Labels are not strictly independent (MC_global and MC_tumor share the underlying NMF decomposition) but the z-scores (4.53–34.30) survive both Bonferroni and BH-FDR; correction choice is non-load-bearing here. The alignment-side biology validation uses BH-FDR over a correlated 20-test family instead — see "distribution-first statistical pivot" below.):

| label | raw EA z | embedding z | direction |
|---|---|---|---|
| MC_global | 4.84 | **5.67** | biology ↑ |
| MC_tumor | 4.08 | **4.53** | biology ↑ |
| TIME | 4.84 | **6.52** | biology ↑ |
| patient_id | 34.30 | **29.63** | patient ↓ (-14%) |

geometric faithfulness: spearman ρ = 0.955 between embedding and raw EA
cosine similarity matrices (39,060 pairs). asymmetric smoothing: patient
compressed more than biology (raw delta ratios emb/raw: MC_global=0.47,
MC_tumor=0.45, TIME=0.54, patient=0.34). supports using gpath2vec as an
H3 interpretability anchor rather than raw enrichment scores.

## alignment grid (three-axis design)

hypotheses (literature-anchored framing):

| H | question | primary axis | evidence |
|---|---|---|---|
| H1 | does learned contrastive alignment improve cross-modal retrieval and representation structure vs classical and unaligned baselines? | loss | R@{1,5,10}, MRR, alignment gap, AUC, CKA |
| H2 | does the shared manifold preserve 9 spatial archetypes better than single-modality spaces? | fusion | ARI, silhouette, per-compartment cosine |
| H3 | does the shared space carry interpretable biological signal? | readouts | pathway CCA, attention maps, patient-leakage probe |

runs:

| id | method | status |
|---|---|---|
| R1 | infonce + late + mlp (matched-pair only) | **done** |
| R2 | supcon + late + mlp (mc_weights soft targets, cohort-specific) | **done** |
| R3 | barlow + late + mlp (none) | **done** |
| R4 | infonce + cross_attn + mlp (matched-pair only) | **done** |
| R6 | infonce + late + mlp, ST = novae 64d only (gpath2vec dropped); ablation | **done** (2026-04-24) |
| B1 | classical CCA (closed-form SVD) | **done** |
| B2 | classical Procrustes (PCA + orthogonal) | **done** |
| B3 | unaligned per-modality PCA + L2-norm | **done** |

scripts: `scripts/align.py` (trained), `scripts/align_classical.py` (baselines),
`scripts/build_niche_join.py` (data contract). configs under
`projects/tnbc-92/alignment/config/`.

## what is being aligned (architecture clarification)

the alignment is **two-stream**, not three. gpath2vec is bundled into the ST input vector as a feature, not aligned as its own third stream. data flow for R1 (late fusion):

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
              cosine(z_he_i, z_st_i) -> positive (matched niche)
              cosine(z_he_i, z_st_j) -> negative (i ≠ j within batch)
```

R4 differs only on the H&E side: instead of mean-pooled virchow2_niche, it uses 7 raw virchow2_cell tile tokens (center spot + 6 neighbors), with cross-attention from the ST query selecting which tiles matter.

### gpath2vec as input vs gpath2vec as validation

because gpath2vec is in the ST input, any H3 readout that uses gpath2vec as the validation target is partially circular - it would measure whether the model preserves an input feature. to keep the biology validation honest, we use **external Bareche subtype labels** (MC_global, MC_tumor, TIME from `Clinical.xlsx`) rather than gpath2vec for the disentanglement test. those labels were never seen by the model.

R6 = R1 architecture with gpath2vec dropped from the ST stream (ST = novae 64d only) was run 2026-04-24 to test whether z_he biology preservation depends on gpath2vec being in the ST input. Result: z_he biology signal survives (17.81 TIME z, higher than R1's 15.13), showing H&E carries independent biology signal. Mirror ablation R6b (gpath2vec-only ST, novae dropped) is not yet run and remains pending in the unmeasured-gating table.

## H1 full sweep + H2 fusion comparison

260-subarray scope, 85/15 patient-stratified split, same held-out patients
across all runs. ~40k test niches.

| run | method | R@1 | R@5 | R@10 | MRR | gap | AUC | CKA after |
|---|---|---|---|---|---|---|---|---|
| **R4** | **infonce + cross_attn** | **0.0002** | **0.0014** | **0.0027** | **0.0019** | **0.195** | **0.851** | **0.564** |
| R1 | infonce + late | 0.0003 | 0.0010 | 0.0023 | 0.0015 | 0.159 | 0.741 | 0.242 |
| R2 | supcon + late | 0.0002 | 0.0009 | 0.0017 | 0.0012 | 0.043 | 0.707 | 0.120 |
| R3 | barlow + late | 0.0001 | 0.0007 | 0.0014 | 0.0010 | 0.125 | 0.646 | 0.252 |
| B2 | Procrustes | 0.0002 | 0.0008 | 0.0015 | 0.0012 | 0.156 | 0.703 | 0.124 |
| B1 | CCA | 0.0000 | 0.0002 | 0.0003 | 0.0003 | 0.007 | 0.541 | 0.086 |
| B3 | Unaligned PCA | 0.0000 | 0.0000 | 0.0000 | 0.0001 | -0.049 | 0.443 | 0.124 |

CKA before projection: 0.115 (same for all - raw modality CKA is fixed).

### H1 takeaways (loss axis)

- **all four contrastive runs beat all three classical baselines on AUC and CKA-after** - the proposal's H1 claim of "relative improvement over baselines" is supported
- InfoNCE > Barlow Twins > SupCon on CKA; InfoNCE > SupCon > Barlow on AUC
- InfoNCE is preferred here because SupCon's supervision signal (cohort-specific mc_weights) was diluted by boundary-pooling NaNs requiring uniform imputation on ~half of niches. A cleaner SupCon comparison (with NaN niches dropped from the loss rather than imputed) remains open work before the loss-axis verdict is committed

### H2 takeaways (fusion axis) - revised after biology validation

**ARI = Adjusted Rand Index**: a clustering-vs-labels agreement metric. run K-means (k=9) on the embedding, then compare the cluster assignments to the known archetype labels. ARI = 0 means clusters are no better than chance; ARI = 1 means perfect overlap between learned clusters and label structure. widely used as a structural-coherence readout, but it is a **single number** that cannot separate "preserves biology" from "preserves patient identity" when the labels are per-patient (which our 9 archetypes are). that is the confound that bit us below.

the original H2 readout (ARI + CKA on 9 archetypes) put R4 cross-attn on top. the biology vs patient z-test on R1, R4, R6, B1 (see "H3 biology validation" below) reframes this as an architectural trade-off rather than a winner-loser call:

- R4 has ~3 effective dimensions on both streams; R1 has ~5 on z_he / ~10 on z_st; R6 has ~6 / ~11. These are genuinely different alignment regimes.
- R4's tight manifold yields strongest cross-modal alignment (AUC 0.851, CKA 0.564, best patient suppression ~22 across views) and the best z_mean bio/patient ratio (0.513). Cost: low within-space resolution.
- R1's wider manifold yields best z_he biology absolute (TIME z=15.13) and best z_he ratio (0.517), with lower H1 AUC/CKA and asymmetric streams (z_st ratio only 0.155).
- R6 (gpath2vec dropped from ST) has highest z_he biology absolute (17.81) and most dim-richness, but z_st biology collapses to near zero — ablation, not a contender.

fusion axis: **pending adjudication**. The proper H2 (per-compartment cosine on the 90-subarray annotated subset) and a rank-matched CKA control (project R1 to its top-3 PCs and recompute CKA) are the two measurements that would let us commit. Both are listed in the unmeasured-gating section. The cross-attn attention weights still give a free spatial-grounding readout (saved in R4's embeddings_test.parquet).

### H3 biology validation (gpath2vec framework ported)

subarray-level biology vs patient z-test, matched-null permutation, 1e5 perms, 259 subarrays, 92 patients. Bareche/TIME labels from Clinical.xlsx, external to the model.

| run | view | MC_global z | MC_tumor z | TIME z | patient z | TIME / patient |
|---|---|---|---|---|---|---|
| R1 | z_he | 10.19 | 7.69 | 15.13 | 29.27 | **0.517** |
| R1 | z_st | 5.39 | 2.93 | 5.16 | 33.38 | 0.155 |
| R1 | z_mean | 10.64 | 7.42 | 13.86 | 33.22 | 0.417 |
| R4 | z_he | 7.72 | 5.33 | 10.07 | 22.97 | 0.438 |
| R4 | z_st | 6.74 | 4.19 | 8.64 | 21.74 | 0.397 |
| R4 | z_mean | 8.76 | 5.86 | 11.71 | 22.83 | **0.513** |
| B1 | z_he | 7.20 | 4.75 | 7.87 | **122.13** | 0.064 |
| B1 | z_st | 10.19 | 7.55 | 17.38 | 54.74 | 0.317 |
| B1 | z_mean | 10.71 | 7.95 | 17.48 | 80.04 | 0.218 |
| raw_he | - | 5.12 | 5.52 | 4.79 | 37.25 | 0.129 (floor) |
| raw_st | - | 5.58 | 4.21 | 6.44 | 28.64 | 0.225 (floor) |

per-stream interpretation (trade-off framing):

- **R1 (novae+gpath2vec ST)**: amplifies biology on z_he (TIME 4.79 → 15.13, ~3× raw), compresses patient on z_he (37.25 → 29.27). Asymmetric — z_st ratio only 0.155. The H&E MLP carries the disentanglement work.
- **R4 (cross-attention)**: compresses both biology and patient symmetrically across views (patient z ≈ 22, biology z ≈ 10 TIME). Tight manifold with ~3 effective dims. Best z_mean bio/patient ratio (0.513) and best absolute patient suppression; lower biology absolute than R1/R6 because the tight manifold limits how much signal fits.
- **R6 (novae-only ST, ablation)**: highest z_he biology (17.81), most dim-rich (~6), but z_st biology collapses to ~1 because novae alone doesn't carry enough biology signal. Not a contender for the full alignment objective; useful for the circularity check — H&E carries biology independently of gpath2vec being in ST.
- **B1 CCA**: amplifies patient 3.3× on H&E (37.25 → 122.13). Biology on z_st (TIME z = 17.38) is highest of all runs — linear CCA preserves the gpath2vec-carried biology in its ST projection — but z_he is patient-dominated, and the bio/patient ratio (0.064) falls below the raw_he floor (0.129). CCA is anti-helpful on the H&E side, not just unhelpful.

geometric faithfulness (rho vs raw modality, shuffled-matrix null):

- R1 z_he vs raw_he: 0.647; R1 z_st vs raw_st: 0.765
- R4 z_he vs raw_he: 0.632; R4 z_st vs raw_st: 0.846 (cross-attn preserves ST geometry best of the contrastive runs)
- B1 z_he vs raw_he: 0.295 (weak); B1 z_st vs raw_st: 0.727

All three contrastive runs preserve each modality's own geometry strongly (rho 0.6–0.85) while adding cross-modal coupling in the shared space. The fusion-axis verdict remains open pending the rank-matched CKA control and per-compartment cosine measurement.

### classical baseline notes

- Procrustes (B2) is the strongest classical baseline - close to R1 on gap but much weaker on CKA-after (0.124 vs 0.242), confirming contrastive loss adds non-linear structural alignment linear alignment can't capture
- CCA (B1) collapses on AUC (0.541): whitened canonical correlation does not survive L2-normalization into cosine space well
- Unaligned PCA (B3) produces negative alignment gap (-0.049) - correct sanity check, independent PCAs of two modalities should not align cross-modally

## immediate next steps

see "what is unmeasured" table for the full gating list. ordered by leverage:

1. **B2 Procrustes + R3 Barlow biology validation** — the two primary falsifier runs for the mechanism prediction (linear-without-nuisance-penalty fails, softmax-over-cross-patient-negatives passes). B2 predicted to fail like CCA; R3 predicted outcome informative either way. Each takes ~45 min on existing checkpoints.
2. **R2 SupCon biology validation + cleaner SupCon rerun** — completes the loss-axis falsifier row, and a NaN-cleaned SupCon rerun would validate or refute "InfoNCE > SupCon" honestly rather than via boundary-pooling artifact.
3. **rank-matched CKA control** — project R1's z_he to top-3 PCs, recompute CKA vs ST. If R1@rank3 reaches R4's CKA, R4's H1 advantage is formally "collapse-driven"; if not, R4 is doing more than compression.
4. **per-compartment cosine on 90-subarray annotated subset** — the honest H2 metric. script exists (eval.py --hypothesis H2 writes compartment_cosine.parquet); needs a stratified writeup.
5. **paired permutation on (bio_delta − patient_delta) difference per run-pair** — data is in `biology_nulls.parquet` with shared seed indices; ~1 hour of code to add.
6. **AnInfoNCE (R5)** — if the rank-matched CKA control supports the collapse interpretation, AnInfoNCE is the targeted fix (per-dim learnable temperature to preserve dim richness while keeping alignment quality).
7. **synthesize winner.json + final figure set** once B2/R3/R2 land and mechanism prediction is adjudicated.

## known limitations

- TLS scores and morphology-compartment labels only exist for 90/260
  subarrays; planned as annotated-subset-only readouts, not core input.
- 19 subarrays below Novae's zero-shot 512-spot floor are excluded from
  the ST side (H&E-only; dropped from rank2 pairs).
- at niche resolution, absolute R@K across ~40k cross-patient niches is
  expected to be low; the proposal's evaluation emphasizes relative
  improvement over baselines, not absolute retrieval thresholds.
- **Bareche MC_global / MC_tumor / TIME labels are computational subtypes** from bulk-expression NMF decompositions, not gold-standard pathology annotations. The biology validation passes at high z on these labels, but the labels themselves are a statistical construct. This is standard for TNBC subtyping; worth flagging before a reviewer does.
- CKA before projection is fixed at 0.115 across all runs because it's computed on the raw virchow2_niche (1280d) vs raw novae ⊕ gpath2vec (576d) input cosines, which do not depend on the alignment model.
- AnInfoNCE (Rusak et al., AISTATS 2025) is the untried H1 extension that could resolve the rank-vs-alignment trade-off by allowing per-dim learnable temperature in the InfoNCE similarity function. Not yet implemented in our grid; mentioned in the unmeasured-gating section.

## reproducibility

### scripts (all under `scripts/`)

| script | what it does | produces |
|---|---|---|
| `build_niche_join.py` | materialize per-subarray niche-join parquets (virchow2_niche, virchow2_cell_tokens, novae_niche z-scored, gpath2vec_niche, tls, mc_weights_niche, archetype, compartment) | `data/embeddings/niches/*.parquet` + `manifest.json` |
| `align.py` | train one contrastive config (loss × fusion × supervision) | `runs/tnbc-92/{run_id}/` (R1-R4) |
| `align_classical.py` | closed-form CCA / Procrustes / unaligned PCA baselines, same split.json as R1 | `runs/tnbc-92/{run_id}/` (B1-B3) |
| `eval.py --hypothesis H1\|H2\|H3` | read per-run embeddings, compute H1 rollup, H2 archetype ARI + per-compartment cosine, H3 pathway CCA + patient probe + attention | `runs/tnbc-92/eval/{H1,H2,H3}/summary.json` + per-run `runs/tnbc-92/{run_id}/eval/{hypothesis}.json` |
| `eval_alignment_biology.py --run-id R1` | biology vs patient z-test ported from gpath2vec (subarray-level permutation, 1e5 perms, BH-FDR); persists full null distributions at seed=42 for paired-run comparison | `runs/tnbc-92/{run_id}/eval/biology.parquet` + `biology_rho.parquet` + `biology_nulls.parquet` (zstd) |
| `eval_compare_and_plot.py --runs R1 R4 R6 B1` | pairwise paired-permutation tests on shared null indices, effective-rank table, cosine-spread, patient-block bootstrap 95% CI on asymmetric compression, and the five main figures | `runs/tnbc-92/eval/compare/*.parquet` + `runs/tnbc-92/eval/figures/fig{1,2,3,6,9}.pdf` |

### configs (under `projects/tnbc-92/alignment/config/`)

| file | run | loss | fusion | supervision |
|---|---|---|---|---|
| `R1.json` | R1 | infonce | late | none |
| `R2.json` | R2 | supcon | late | mc_weights soft |
| `R3.json` | R3 | barlow | late | none |
| `R4.json` | R4 | infonce | cross_attn | none |
| `R6.json` | R6 | infonce | late | none (ablation: ST = novae 64d only, gpath2vec dropped) |

B1/B2/B3 have no training config; they're parameterized via CLI args to `align_classical.py`.

### per-run artifacts (under `runs/tnbc-92/{run_id}/`)

| file | source | used for |
|---|---|---|
| `run_config.json` | align.py / align_classical.py | reproduce the exact config |
| `split.json` | align.py (R1) or copied from R1 by classical runs | identical 85/15 patient split across all 7 runs |
| `training_log.csv` | align.py | per-epoch train_loss, val_loss, val_cos_sim, val_r1 |
| `checkpoint.pt` | align.py | best-val model weights (trained runs only) |
| `embeddings_test.parquet` | align.py / align_classical.py | per-test-niche z_he, z_st (+ attention_weights for R4) |
| `metrics_h1_raw.json` | align.py / align_classical.py | R@K, MRR, alignment_gap, AUC, CKA |
| `eval/H1.json`, `H2.json`, `H3.json` | eval.py | hypothesis-specific metrics, per-run |
| `eval/biology.parquet` | eval_alignment_biology.py | per-view biology and patient z-scores + BH-FDR q (subarray-level, 1e5 perms) |
| `eval/biology_rho.parquet` | eval_alignment_biology.py | geometric faithfulness rho vs raw_he / raw_st |
| `eval/biology_nulls.parquet` | eval_alignment_biology.py | full 1e5-perm null distributions per test, zstd-compressed; enables paired-run tests at shared seed=42 |
| `run.log` | tee from tmux | stdout + stderr |

### cross-run rollups (under `runs/tnbc-92/eval/`)

| file | source | contents |
|---|---|---|
| `H1/summary.json` | eval.py --hypothesis H1 | per-run R@K / MRR / AUC / CKA table |
| `H2/summary.json` | eval.py --hypothesis H2 | per-run archetype ARI + silhouette on z_he / z_st / z_joint, plus raw reference |
| `H2/compartment_cosine.parquet` | eval.py --hypothesis H2 | per-compartment mean cosine by run (90-subarray annotated subset) |
| `H3/summary.json` | eval.py --hypothesis H3 | per-run CCA(shared, gpath2vec), patient probe accuracy on z_he / z_st / z_mean / raw_he / raw_st |
| `H3/attention_by_compartment.parquet` | eval.py --hypothesis H3 | R4 attention weights aggregated by compartment |
| `compare/paired_tests.parquet` | eval_compare_and_plot.py | pairwise paired-permutation tests (bio, patient) across all run pairs, shared seed=42 null indices |
| `compare/effective_rank.parquet` | eval_compare_and_plot.py | participation ratio + matched/mismatched cosine spread + H1 metrics per run |
| `compare/asymmetric_compression.parquet` | eval_compare_and_plot.py | (bio_delta − patient_delta) per run with 95% patient-block bootstrap CI (effect-size error bars only, not significance) |
| `figures/fig{1,2,3,6,9}.pdf` | eval_compare_and_plot.py | bio vs patient z scatter, R@K curves, effrank vs CKA, permutation null histograms, asymmetric compression bars |

### input data (read-only)

- H&E niche: `data/embeddings/virchow2_niche/{subarray}.npy` + `_meta.tsv`
- H&E cell (per-spot for cross-attn): `data/embeddings/virchow2_cell/{subarray}.npy`
- Novae niche (with k=6 neighbor ids): `data/embeddings/novae_niche_full/{subarray}_novae_embeddings.parquet`
- gpath2vec niche: `data/embeddings/biological_signals/gpath2vec_output/full_cohort_tf_low/cluster_embeddings.parquet` (286k niches × 512d)
- biological signals: `data/embeddings/biological_signals/{mc_labels,mc_weights,tls_scores,morphology_labels}.tsv`
- clinical + archetype: `data/inputs/Clinical/Clinical.xlsx` (Bareche subtype labels for biology z-test)
- subarray → TNBC id mapping: `data/inputs/clinical/ids_subarray_to_tnbc.tsv` (exported from `ids.RDS`)

### reproducibility guarantees

- all runs use `seed = 42` in align.py; classical runs re-fit from R1's train patients
- neighborhood parity verified: Virchow2 and Novae use identical KDTree k=6 neighborhoods (1075/1075 spot agreement on TNBC1_CN1_C1); direct `(subarray, spot_id)` join is valid
- biology z-test uses 1e5 permutations; shuffled-matrix null for rho uses 1e3 perms (p-floor 1e-3)
- sklearn 1.7+ (no `multi_class` kwarg on LogisticRegression) and torch 2.x assumed

---

## distribution-first statistical pivot

landed 2026-04-24. infrastructure now persists full 1e5-perm null distributions per run and supports paired tests via shared seed indices. reviewer instinct: most of what we'd normally add (Bonferroni, bootstrap CIs, z-scores in standard units) assumes the null is Gaussian. the gpath2vec null diagnostics already showed right-skew +0.28-0.39 and excess kurtosis +0.14-0.34 on biology nulls. permutation already gives distribution-free significance; layering Gaussian machinery on top is a category error.

### what we keep and what we drop

| method | keep? | why |
|---|---|---|
| permutation null (1e5 perms, observed delta, p = rank in null) | ✔ primary | distribution-free, already in place |
| Benjamini-Hochberg FDR at q=0.05 | ✔ add | right tool for correlated tests; report raw p, BH-adjusted q, family size |
| Bonferroni α = 0.05/k | ✖ drop | tests are correlated (5 views are derivatives of each other); unjustified precision theater |
| bootstrap CIs (patient-level resample) | ✔ keep for effect-size error bars only | not for significance - permutation handles that |
| z-scores in standard units | ~ subordinate | keep as familiar summary, not primary reporting unit |
| paired permutation test on (bio_delta − patient_delta) | ✔ add | directly operationalizes "asymmetric compression" - the mechanism behind R1 > R4 claim |

### what we look at instead

1. **full permutation null distributions**, saved per test × run × view × label (1e5 values each). plot histogram + observed marked. reader judges extremeness from distribution shape, not z.
2. **per-pair cosine distributions**, 4-way overlay per run × view: same-patient / diff-patient same-bio / diff-patient diff-bio / all pairs. R4's tight manifold = narrow high-mean global distribution. R1's wider spread = larger absolute separation between bio-label groups. B1 CCA = same-patient dominant over everything else.
3. **cumulative eigenvalue spectrum** per run's shared space (not a scalar "effective rank"). R4 flat-lines early, R1 spreads out longer - gradient, not binary.
4. **paired permutation on (bio_delta − patient_delta)** under joint null with shared perm indices (seed=42 across runs). p-value: fraction of null samples where |diff_R1 − diff_R4| ≥ observed difference. non-parametric, directly tests "R1 disentangles more than R4."
5. **KS test on same-bio vs diff-bio cross-patient cosines** per run × view. nonparametric comparison across runs.

### arithmetic reminder that has to be defended, not hidden

| | R1 z_he | R4 z_he | raw_he |
|---|---|---|---|
| TIME biology z | 15.13 | 10.07 | 4.79 |
| patient z | 29.27 | 22.97 | 37.25 |
| bio / patient ratio | **0.517** | 0.438 | 0.129 |

R1 has **more biology AND more patient signal** than R4 in absolute terms. The "R1 disentangles better" claim rests entirely on relative compression (asymmetric smoothing), not on "R1 beats R4 on biology in isolation." The paired-delta-difference test above operationalizes this correctly.

### implementation timeline (~1 day focused work)

**code modifications to `scripts/eval_alignment_biology.py` (~2h)**:
- save per-test null distributions as parquet with zstd compression (~1 MB/run)
- save permutation indices once (seed=42, n_perm × n_subarrays int32) so any pairwise test reuses them
- add BH-FDR adjustment column per family
- add `compare_runs_paired()` function: reads two runs' null deltas, computes paired (bio_delta − patient_delta) difference test

**compute (run in parallel, ~2.5h wall clock)**:
- R6 ablation train + biology eval (~1.5h): new config R6.json with ST stream = novae_niche only (gpath2vec dropped); tests whether z_he's biology amplification is inherited from gpath2vec or learned from H&E
- re-run R1/R4/B1 biology eval with persistence (~2.25h in second tmux)

**plots (~4-6h after artifacts land)**:
- fig 1: bio_z vs patient_z scatter, one point per run (z_he view), raw_he/raw_st marked, ideal arrow top-left
- fig 2: R@K curves with chance floor (log-y)
- fig 3: effective_rank vs CKA_after scatter
- fig 6: permutation null histogram + observed z marked, per label, all runs overlaid
- fig 9: (bio_delta − patient_delta) with bootstrap CIs for effect size only

### refinements to make in the report alongside implementation

- merge H2 status row in Table 1 (current "❌ ARI confounded") with the revised H2 takeaways below into one story
- reframe "InfoNCE > SupCon on this data" as pessimistic (mc_weights NaN dilution not cleaned up); caveat explicitly until a cleaner SupCon rerun
- explain `CKA before = 0.115` (pre-projection CKA computed on raw virchow2_niche vs raw novae⊕gpath2vec; doesn't depend on the run)
- fix Bonferroni claim (real family is 5 views × 4 labels = 20 tests, not 8)
- specify permutation null design explicitly in methods: biology null shuffles label→patient (preserves patient cluster sizes), patient null is matched-null on subarray→patient multiset
- add to known limitations: TIME/MC labels are Bareche-2020 computational subtypes from bulk expression, not gold-standard pathology
- ~~soften the R4-framing language~~ applied 2026-04-24; committed position is the trade-off framing (tight manifold for cross-modal alignment gain; dim-richness cost)
- drop unintroduced AnInfoNCE mention from prose; reintroduce only if R5 is actually run

---

## MCP / ResearchHub goal status

honest accounting against the proposal.

### done - embedding + alignment science

| proposal element | status |
|---|---|
| Virchow2 + UNI2 + Novae encoder pipelines | ✔ done, 260-subarray niche join table |
| gpath2vec pathway embedding | ✔ done, validated |
| contrastive alignment module | ✔ 7 runs (R1-R4 + B1-B3), scripts reusable |
| H1 retrieval / alignment metrics | ✔ done |
| H2 structural coherence | ✔ done (ARI flagged + replaced with bio/patient z) |
| H3 pathway interpretability | ~ partial (biology z done for R1/R4/B1; per-pathway CCA, attention by compartment, patient probe still to report) |
| cohort-level validation (Bareche labels) | ✔ done as the primary H3 readout |

### not done - agents + MCP server (the proposal's primary contribution)

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

### not yet tried - alignment methods

- **AnInfoNCE** - not tried. H1 refinement experiment still on the table. Will only be considered after the distribution-first validation is done (needs a fitness metric first).
- **R6** (gpath2vec out of ST stream) - next up, queued.
- **R7** (three-stream alignment: H&E ↔ novae ↔ gpath2vec with three independent projection heads and pairwise InfoNCE) - conceptual, not yet specced as a config.
- **domain-adversarial GRL on patient_id** - queued post-R6 if patient leakage still dominates.

### honest situation

the grant's primary deliverable is the MCP server / agent orchestration pattern. zero of that is built. current work has been the scientific foundation (alignment experiments, biology validation) - excellent paper material and necessary groundwork, but doesn't cash the grant's agent-orchestration claim.

the scripts we've written (`align.py`, `eval.py`, `eval_alignment_biology.py`, `build_niche_join.py`, `align_classical.py`) are agent-ready - they take config in, return artifacts out. wrapping them as MCP tools is straightforward. hard part is the orchestration logic + Claude integration + LangSmith + HITL.

### recommended sequencing (we are ahead)

1. **finish distribution-first pivot + R6** (~1 day). lands the scientific paper claim cleanly and de-risks the circularity objection.
2. **draft MCP server skeleton** (~2-3 days). one agent = one script wrapper + a tool schema. LangGraph StateGraph connecting them. Claude Sonnet routing, Claude Opus synthesis.
3. **AnInfoNCE / R7 three-stream / GRL** as H1 extension runs invoked via the MCP server itself (eats our own dog food).
4. **package + release**: pip-installable MCP server, HF checkpoint with model card, Zenodo embeddings archive, reproducible scripts.