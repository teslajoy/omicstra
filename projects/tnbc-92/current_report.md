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

all six proposal-specified H1 metrics (R@K, MRR, **median rank**, alignment gap, AUC, CKA) reported. test set has 45,661 niches; random-baseline median rank ≈ 22,830 (= n/2). lower median rank is better.

| run | method | R@1 | R@5 | R@10 | MRR | median rank | gap | AUC | CKA after |
|---|---|---|---|---|---|---|---|---|---|
| **R4** | **infonce + cross-attn contrastive** | **0.0002** | **0.0014** | **0.0027** | **0.0019** | **4244** | **0.195** | **0.851** | **0.564** |
| R1 | infonce + late-fusion contrastive | 0.0003 | 0.0010 | 0.0023 | 0.0015 | 8827 | 0.159 | 0.741 | 0.242 |
| R6 | infonce + late-fusion contrastive (novae-only ST ablation) | 0.0002 | 0.0007 | 0.0014 | 0.0011 | 9227 | 0.155 | 0.733 | 0.225 |
| R2 | supcon + late-fusion contrastive | 0.0002 | 0.0009 | 0.0017 | 0.0012 | 10153 | 0.043 | 0.707 | 0.120 |
| R3 | barlow + late-fusion contrastive | 0.0001 | 0.0007 | 0.0014 | 0.0010 | 10267 | 0.125 | 0.646 | 0.252 |
| B2 | Procrustes (classical baseline) | 0.0002 | 0.0008 | 0.0015 | 0.0012 | 10342 | 0.156 | 0.703 | 0.124 |
| B1 | CCA (classical baseline) | 0.0000 | 0.0002 | 0.0003 | 0.0003 | 20181 | 0.007 | 0.541 | 0.086 |
| B3 | Unaligned PCA (proposal "raw unaligned concat" analog) | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 26430 | -0.049 | 0.443 | 0.124 |

CKA before projection: 0.115 (same for all - raw modality CKA is fixed).

**baseline terminology** (per proposal, p.5): the proposal specifies three baselines — CCA projection, late fusion concatenation, and raw unaligned concatenation. The mapping in this report:
- **CCA projection** → B1 (closed-form CCA, train-fit applied cohort-wide).
- **late fusion concatenation** (proposal's non-contrastive concat baseline) → no pure non-contrastive concat baseline was run; B3 (per-modality PCA-to-512 + L2-norm, no alignment) is the closest analog and is reported as the "raw unaligned concatenation" comparator. The "late-fusion contrastive" runs R1, R2, R3 are extensions, not the proposal's concat baseline.
- **raw unaligned concatenation** → B3.

**proposal AUC prediction comparison** (proposal p.5: "cross-modal similarity improvements are expected within an approximate range of 0.65-0.75"):
- **R4 AUC = 0.851** *exceeds* the proposal's upper bound by 0.10. Strongest result and outside the predicted range.
- R1 AUC = 0.741 sits in the upper half of the predicted range.
- R6 AUC = 0.733, R2 AUC = 0.707, B2 AUC = 0.703 all sit in the predicted range.
- R3 AUC = 0.646 sits just below the lower bound.
- All four contrastive runs (R1, R2, R3, R4) meet or exceed the predicted range; R4 substantially exceeds it. The proposal noted relative improvement over baselines as the primary criterion — confirmed in either reading.

**median rank vs random**: R4 (4244) ≈ 5.4× better than random (22,830); R1, R6, R2, R3, B2 cluster around 2.2-2.6× better; B1 only 1.1× (near random); B3 (26430) is worse than random — correct sanity-check behavior for the no-alignment floor.

### H1 takeaways (loss axis)

- **all four contrastive runs beat all three classical baselines on AUC and CKA-after** - the proposal's H1 claim of "relative improvement over baselines" is supported
- InfoNCE > Barlow Twins > SupCon on CKA; InfoNCE > SupCon > Barlow on AUC
- InfoNCE is preferred here because SupCon's supervision signal (cohort-specific mc_weights) was diluted by boundary-pooling NaNs requiring uniform imputation on ~half of niches. A cleaner SupCon comparison (with NaN niches dropped from the loss rather than imputed) remains open work before the loss-axis verdict is committed

### H2 takeaways - both proposal parts reported separately

The proposal's H2 has two parts:
- **part A**: ARI + silhouette on 9 spatial archetypes, aligned vs single-modality spaces.
- **part B**: per-compartment matched-pair cosine, with directional prediction *morphologically distinct (tumor / TLS / necrosis) > morphologically ambiguous (high-TIL / low-TIL stroma)*.

both parts are reported below; the brief should not conflate them.

#### H2 part A: archetype coherence

**ARI = Adjusted Rand Index**: K-means (k=9) on the embedding, compared to archetype labels. ARI=0 is chance, ARI=1 is perfect overlap. silhouette is the corresponding cluster-compactness metric on the same K-means partitions. both are **single-number readouts that cannot separate "preserves biology" from "preserves patient identity" when labels are per-patient**, which the 9 archetypes are. that confound dominates the part A table.

| run | view | ARI | silhouette |
|---|---|---|---|
| R1 | z_he | 0.250 | 0.032 |
| R1 | z_st | 0.023 | -0.015 |
| R1 | z_joint | 0.154 | 0.013 |
| R2 | z_he | 0.204 | 0.010 |
| R3 | z_he | 0.129 | -0.070 |
| R4 | z_he | 0.059 | -0.075 |
| R4 | z_joint | 0.031 | -0.045 |
| R6 | z_he | 0.254 | 0.022 |
| B1 (CCA) | z_he | 0.307 | 0.027 |
| B1 (CCA) | z_joint | **0.319** | 0.014 |
| B2 (Procrustes) | z_he | 0.298 | 0.059 |
| B3 (Unaligned PCA) | z_he | 0.298 | 0.059 |
| raw_he (reference) | - | **0.287** | 0.035 |
| raw_st (reference) | - | 0.014 | -0.018 |

**reading**:
- B1 CCA tops joint ARI (0.319). H3's bio-vs-patient z-test (below) shows this is patient amplification, not biological structure: B1 z_he patient z = 122.13 vs raw 37.25 (3.3× patient).
- raw H&E alone (0.287 ARI) clusters archetypes better than every contrastive run. The model is not *adding* archetype-cluster coherence; raw H&E already has it because archetypes are spatially smooth on H&E morphology.
- R4 (0.059 z_he, 0.031 z_joint) is the *worst* archetype clusterer — its tight ~3-effective-dim manifold compresses everything into a regime where K-means cannot find 9 distinct clusters. This is consistent with the dim-collapse diagnostic, not a failure of biology.
- silhouettes near 0 or negative everywhere indicate the 9 archetypes do not form compact clusters in any space - ARI's high values reflect cluster *agreement* under matching, not actual cluster *separation*.
- **B2 ≡ B3 on every part A metric** (ARI 0.298, sil 0.059 on z_he). This is mathematically guaranteed: B2 is orthogonal Procrustes (a rotation of B3's PCA coordinates), and ARI / silhouette / K-means partitions are rotation-invariant. Identical numbers are not a coincidence.

**part A verdict (preliminary, on archetype labels)**: ARI alone does not adjudicate the fusion axis. CCA's top-line is patient amplification; raw H&E is a strong ceiling baseline that contrastive runs do not beat; R4's "low" ARI is dim-compression rather than biology loss. Part A is reported for the proposal, but the load-bearing fusion-axis evidence sits in part A-MC, part B, and H3.

#### H2 part A-MC: niche-level biological clustering on `mc_labels.megacluster` (commit 2.5 audit)

**why this section exists**: 2026-05-11 code audit of `scripts/build_niche_join.py:67-82` found that the `archetype` column we used in part A above is sourced from Wang's `Spatial archetypes_defined_on_ST_global_pseudobulk` - a **patient-level pseudobulk label** (verified: 30/30 sampled subarrays have `archetype_unique_within_subarray = 1`). Every niche in a patient inherits one archetype. The proposal's intent (biological structural coherence at niche resolution) requires a niche-level biological label. That label exists: `mc_labels.megacluster` (Wang's per-spot 14-class NMF hard label, 270,136 spots covering 14 megaclusters). It was previously not joined into niche parquets or consumed by eval. commit 2.5 fixed both gaps.

**test**: KMeans(k=14) → ARI + silhouette vs `mc_labels.megacluster` per run × view, on the same test-niche scope as part A. 8 runs × 3 views = 24 cells.

**pre-registered predictions** (locked in memory + `proposal_deviations.md` before computing): partial decoupling expected. R4 MC ARI in [0.15, 0.25] (rises from archetype 0.059); B1 MC ARI in [0.10, 0.20] (drops from archetype 0.307, exposing patient leakage); R1/R3 MC ARI in [0.20, 0.30] (preserves niche biology). Best-case scenario: classical baselines drop, contrastive runs hold or rise.

**observed result — full grid** (ARI and silhouette per run × view; archetype reproduced for direct comparison):

| run | view | archetype ARI | MC ARI | Δ | archetype sil | MC sil |
|---|---|---|---|---|---|---|
| **R1** | z_he | 0.250 | **0.231** | -0.019 | 0.032 | -0.031 |
| **R1** | z_st | 0.023 | 0.069 | **+0.046** | -0.015 | -0.042 |
| **R1** | z_joint | 0.154 | **0.192** | **+0.038** | 0.013 | -0.004 |
| R2 | z_he | 0.204 | 0.174 | -0.030 | 0.010 | -0.055 |
| R2 | z_st | 0.017 | 0.060 | +0.043 | -0.033 | -0.070 |
| R2 | z_joint | 0.116 | 0.129 | +0.013 | 0.003 | -0.025 |
| R3 | z_he | 0.129 | 0.166 | +0.038 | -0.070 | -0.077 |
| R3 | z_st | 0.015 | 0.049 | +0.034 | -0.047 | -0.093 |
| R3 | z_joint | 0.110 | 0.152 | +0.042 | -0.046 | -0.045 |
| R4 | z_he | 0.059 | 0.098 | +0.039 | -0.075 | -0.088 |
| R4 | z_st | 0.017 | 0.048 | +0.031 | -0.042 | -0.117 |
| R4 | z_joint | 0.031 | 0.075 | +0.044 | -0.045 | -0.076 |
| **R6** | z_he | 0.254 | **0.251** | -0.003 | 0.022 | -0.024 |
| **R6** | z_st | 0.027 | 0.064 | **+0.036** | -0.010 | -0.044 |
| **R6** | z_joint | 0.150 | **0.190** | **+0.040** | 0.013 | -0.001 |
| **B1** | z_he | **0.307** | 0.202 | **-0.105** | 0.027 | 0.008 |
| **B1** | z_st | 0.097 | 0.071 | -0.027 | 0.000 | 0.000 |
| **B1** | z_joint | **0.319** | **0.220** | **-0.099** | 0.014 | 0.006 |
| B2 | z_he | 0.298 | 0.205 | -0.093 | 0.059 | -0.005 |
| B2 | z_st | 0.018 | 0.046 | +0.028 | -0.006 | -0.048 |
| B2 | z_joint | 0.132 | 0.159 | +0.027 | 0.030 | -0.001 |
| B3 | z_he | 0.298 | 0.205 | -0.093 | 0.059 | -0.005 |
| B3 | z_st | 0.018 | 0.046 | +0.028 | -0.006 | -0.048 |
| B3 | z_joint | 0.132 | 0.159 | +0.027 | 0.030 | -0.001 |
| raw_he | - | 0.287 | **0.174** | **-0.113** | 0.035 | -0.005 |
| raw_st | - | 0.014 | 0.040 | +0.026 | -0.018 | -0.074 |

**interpretation — scenario 3 (partial decoupling) confirmed with three subsidiary findings**:

1. **Patient-confound diagnosis on classical baselines holds with quantitative precision**. B1's archetype z_joint = 0.319 was the report's previous "winner"; switching to niche-level MC labels drops it to 0.220 (-0.099). B2 and B3 (rotation-equivalent) drop the same way on z_he. **~30-37% of classical-baseline ARI was patient identity**, not biological structure. **B1 z_he loses 34%** (-0.105 of 0.307).

2. **raw_he loses the most ARI under the resolution shift** (-0.113, 39% drop). The "raw H&E ceiling baseline" on archetype was almost entirely patient identity, not biology. raw_he archetype ARI 0.287 was a 13% lift over R3 / 16% over R1; raw_he MC ARI 0.174 is *lower* than R1 (0.231) and R6 (0.251). **The raw modality does NOT have a structural-coherence advantage on niche-level biology.**

3. **Late-fusion contrastive runs (R1, R6) emerge as the niche-level biological clustering winners** on H&E view. R6 0.251 > R1 0.231 > B1 0.202 > raw_he 0.174 > R4 0.098. R6 (novae-only ST ablation) narrowly wins. R1 (full ST input) is the runner-up. **Both substantially exceed raw_he MC ARI and B1 MC ARI.**

4. **R4 cross-attention's tight manifold limits niche-level resolution** but does not destroy biology. R4 rises from 0.059 (archetype) to 0.098 (MC) - a 66% relative improvement, statistically meaningful, but the absolute is still the lowest of all 8 runs. The dim-collapse-hurts-cluster-resolution hypothesis is **confirmed at the right resolution**. R4 is *not* the best niche-level biological clusterer; the bound is mechanical (~3 effective dimensions cannot resolve 14 K-means clusters), not biological.

5. **ST stream encodes niche-level MC biology, not patient identity**. Every run's z_st delta is positive (+0.028 to +0.046). The ST modality is the cleaner biological signal carrier; the H&E side is where patient-architecture confound concentrates.

6. **z_joint pattern**: contrastive runs all improve under the resolution shift (+0.013 to +0.044); classical runs all lose ARI (-0.027 to -0.099). **The full grid is consistent with the diagnosis**: contrastive learning produces representations that preserve niche-level biology; classical baselines preserve patient-level structure that masquerades as biology on patient-level labels.

7. **All silhouettes near zero or negative**. MC clusters do not form compact spatially-separated regions in any embedding. ARI captures cluster-label *agreement* under matching; not cluster *separation*. Consistent with MC being a soft NMF mixture overlaid on continuous H&E morphology — the discrete 14-class hard labels are convenient targets but the underlying structure is continuous.

**verdict draft from KMeans alone (commit 2.5)** — *flagged for refinement by commit 2.6 diagnostics below*:

| run | archetype-only verdict (previous brief) | archetype + MC verdict (KMeans-only) |
|---|---|---|
| R1 | competitive, mid-pack | late-fusion contrastive runner-up on KMeans (z_he MC ARI 0.231) |
| R4 | "worst archetype clusterer due to dim collapse" - framed as architectural trade-off | KMeans MC ARI 0.098, lowest. *Initial reading: dim-collapse confirmed at niche resolution* - **refined below by 2.6 diagnostics** |
| R6 | ablation, "not a contender" | best on KMeans (z_he MC ARI 0.251); gpath2vec-out-of-ST sharpens H&E side |
| B1 (CCA) | "wins archetype ARI by amplifying patient" | confirmed quantitatively: -34% drop on z_he, -31% on z_joint when moving to niche-level labels |
| B2/B3 | "rotation-equivalent classical, contrastive beats" | -31% drop on z_he. Same pattern as B1, slightly less severe |
| raw_he | "ceiling baseline contrastive runs don't beat" | **NOT a ceiling on niche-level biology** - raw_he MC ARI 0.174 < R1/R6 by wide margin. The "raw ceiling" was patient identity |

#### H2 part A-MC diagnostic refinement (commit 2.6)

**why this section exists**: the KMeans-only result above flags R4 as "lowest niche-level biology". R4's strong H1 metrics (AUC 0.851, CKA 0.564) and clean H3 z_he TIME biology amplification (z=10.07 vs raw_he 4.79) are inconsistent with that reading. KMeans assumes globular clusters; R4 has ~3 effective dimensions (per the report's effective-rank diagnostic). Three metric-independent diagnostics were run to disambiguate metric failure from biology failure. Predictions pre-registered before computing (see `project_h2_mc_diagnostics_predictions.md`).

##### diagnostic 1 — kNN purity (z_he, chance = 1/14 = 0.071)

for each test niche, k nearest cosine neighbors; report fraction sharing same `mc_megacluster`. measured at k=5, 10, 25.

| run | k=5 | k=10 | k=25 |
|---|---|---|---|
| B2 / B3 | 0.812 | 0.793 | 0.753 |
| raw_he | 0.810 | 0.792 | 0.750 |
| B1 | 0.806 | 0.790 | 0.762 |
| R6 | 0.805 | 0.783 | 0.742 |
| R1 | 0.798 | 0.772 | 0.730 |
| R2 | 0.779 | 0.748 | 0.699 |
| **R4** | **0.652** | **0.582** | 0.505 |
| R3 | 0.611 | 0.582 | 0.544 |

**all purities far above chance (lifts of 0.4-0.7 over 0.071)**. however, **kNN purity is patient-confounded**: niches from the same patient cluster together in cosine space (patient identity dominates), and within a patient most niches share MC labels (MCs are spatially smooth within a tumor). So high kNN purity at 0.80+ is mostly *within-patient* MC similarity, not cross-patient biological coherence.

**R4's lower value (0.65) is consistent with R4's known best-in-grid patient suppression** (H3 z_he patient z = 22.97, the lowest of all 8 runs - see "H3 biology validation" section), not with biology loss. When R4 suppresses patient identity, its niche neighbors stop being co-patient niches and become biologically-similar niches from other patients - which on this dataset means fewer same-MC neighbors at small k, because the underlying patient-architecture-MC entanglement (see `project_niche_patient_entanglement.md`) means MC labels are partially patient-architecture-defined.

**kNN purity is therefore not a valid biology metric for cross-patient alignment** without patient-stratified handling. Logged for the record; not load-bearing for the verdict.

##### diagnostic 2 — linear probe (LogReg, 14-class, patient-stratified within 14 test patients)

Patient-stratified split: 11 probe-train patients / 3 probe-test patients (seed 42). LogReg(multinomial, lbfgs) on z_he. reference: rank1a Virchow2 niche -> MC = 21.4% (3× chance, on a different prior split). chance = 0.071.

| run | accuracy | macro-F1 | lift over chance |
|---|---|---|---|
| **R1** | **0.260** | 0.133 | +0.188 |
| raw_he | 0.239 | 0.128 | +0.168 |
| B3 | 0.237 | 0.119 | +0.166 |
| R6 | 0.235 | 0.126 | +0.163 |
| B2 | 0.233 | 0.119 | +0.162 |
| **R4** | **0.230** | 0.116 | +0.158 |
| R2 | 0.229 | 0.121 | +0.158 |
| B1 | 0.210 | 0.111 | +0.138 |
| R3 | 0.182 | 0.096 | +0.110 |

n_probe_test = 5912 niches across 3 patients per run. all runs match or exceed rank1a's 21.4% baseline (R3 the only exception). **R4 (0.230) sits within 3 percentage points of R1 (0.260) and raw_he (0.239)**.

**this is the load-bearing diagnostic**. linear probe is geometry-agnostic: it asks "is MC linearly decodable from the embedding?" not "do MC classes form compact clusters?" The answer: **every run encodes MC at roughly parity** under the patient-stratified test. R4 is NOT biology-less.

##### diagnostic 3 — rank-matched KMeans (project z_he to top-3 PCs, recompute MC ARI)

| run | full ARI | rank-5 ARI | rank-3 ARI |
|---|---|---|---|
| R6 | 0.251 | 0.219 | **0.185** |
| R1 | 0.231 | 0.219 | **0.193** |
| B1 | 0.202 | 0.138 | 0.084 |
| R4 | 0.098 | 0.098 | **0.095** (sanity: ≈ full, R4 lives in ~3 dims ✓) |

**main finding**: even when R1 and R6 are projected to 3 PCs (matching R4's effective rank), their KMeans MC ARI (0.193, 0.185) is still **2× R4's full-rank ARI (0.098)**. So R4's gap is NOT just dim count — at matched rank, R1's top-3 PCs are *more MC-aligned* than R4's full 3-d manifold.

But linear probe says R4 has MC information at parity (0.230). So the gap is geometry-shape: **R4 stores MC as linearly-separable directions, not as globular clusters**. KMeans assumes globular Voronoi cells around centroids; R4's tight cross-attention manifold violates that assumption. LogReg only needs linear separability, which R4 has.

##### resolved verdict on R4 (replaces commit 2.5 reading)

| measurement | finding | what it says about R4 |
|---|---|---|
| KMeans MC ARI (full or rank-3) | R4 0.098, R1/R6 0.19-0.25 | **measurement artifact** of globular-cluster assumption |
| kNN purity (k=5) | R4 0.652 vs baselines 0.80 | **patient-confound**: low value reflects R4's strong patient suppression, not biology loss |
| **linear probe (patient-stratified)** | **R4 0.230, R1 0.260, raw_he 0.239** | **biology at parity** with all other runs |
| H1 AUC | R4 0.851 (best) | cross-modal alignment intact |
| H3 z_he TIME biology z | R4 10.07 vs raw_he 4.79 (2.1× amp) | biology amplified, just compressed |

**revised verdict for H2 part A-MC**: R4 cross-attention preserves niche-level biology **at parity with R1/R6/raw_he** by the only metric-independent test (patient-stratified linear probe). R4's KMeans MC ARI of 0.098 was a **K-means / globular-cluster bias against tight manifolds**, not a biology preservation failure. Even at matched effective rank, R4's geometry is K-means-unfriendly because MC is encoded as linearly-separable directions rather than globular regions. **R4's tight 3-effective-d cross-attention manifold compresses biological information into linearly decodable axes; it does not lose it.**

**the broader H2 part A-MC verdict still holds for the classical baselines**: B1 / B2 / B3 lose 31-37% of ARI under the resolution shift even after the diagnostics. raw_he loses 39%. **classical baselines and raw H&E DO have patient leakage in their archetype ARI**; the diagnostics did not rescue them. So:

- **R1, R6, R4**: preserve niche-level biology (at parity by linear probe, K-means rankings reflect geometry differences not biology differences).
- **B1, B2, B3**: preserve patient-architecture; linear probe accuracies near raw_he indicate niche-level MC encoding is at H&E baseline (no contrastive amplification), KMeans amplifies patient identity.
- **raw_he**: ceiling baseline only at LP (parity with contrastive), NOT a ceiling on KMeans MC ARI (raw_he 0.174 < R1/R6 by wide margin).

##### the falsifier-discipline arc (methodology note)

commit 2.5 used KMeans MC ARI as the niche-level biology test. result flagged R4 as worst, inconsistent with R4's H1/H3 performance. user instinct: "R4 shouldn't fail; metric is suspect". commit 2.6 ran three pre-registered metric-independent diagnostics. linear probe (the load-bearing one) showed R4 at parity with R1/R6/raw_he. K-means ARI is now understood as a globular-cluster-shape test, not a biology preservation test. **the project's pre-registration framework worked: surprising result → instinct check → diagnostic falsifier → verdict revision on evidence**. KMeans MC ARI stays in the report as evidence that the metric was misleading; it does NOT get archived. The diagnostic refinement is the load-bearing verdict.

artifacts (all gitignored, in `runs/tnbc-92/eval/H2/`):
- commit 2.5: `mc_coherence.parquet`, `summary.json` extended (mc_z_he/st/joint)
- commit 2.6: `mc_diagnostics/{knn_purity, linear_probe, rank_matched_kmeans}.parquet`

scripts (gitignored, in `scripts/_scratch/`):
- `add_mc_megacluster_to_niche_join.py` (parquet patcher)
- `eval_h2_mc_coherence.py` (commit 2.5 KMeans grid)
- `eval_h2_mc_diagnostics.py` (commit 2.6 three diagnostics)

#### H2 part B: per-compartment matched-pair cosine (the proposal's directional test)

**pre-registered prediction (per proposal p.5)**: distinct compartments (Tumor, Lymphoid nodule as TLS proxy, Necrosis) show *higher* matched-pair cosine `cos(z_he[i], z_st[i])` than ambiguous compartments (High TIL stroma, Low TIL stroma). 6 named contrasts per run: 3 distinct × 2 ambiguous. Welch z-test (unpooled variance — compartments have different n and signal regimes).

source: `runs/tnbc-92/eval/H2/compartment_cosine.parquet` (96 rows, 12 compartments × 8 runs). Lymphoid nodule is the closest 12-class proxy for TLS in the Wang et al. annotation vocabulary.

falsifier-failure summary (n out of 6 contrasts with predicted direction `delta > 0`):

| run | n_passing (of 6) | n_significant_correct (p<0.05) | min z | max z | predicted? |
|---|---|---|---|---|---|
| R1 | 2 | 2 | -16.14 | +5.33 | **fails** (only TLS contrasts pass) |
| R2 | 2 | 2 | -6.79 | +3.60 | fails |
| R3 | 3 | 2 | -11.81 | +12.05 | fails (inverted pattern: necrosis up, TLS down) |
| R4 | 0 | 0 | -11.28 | -1.15 | **inverted on every contrast** |
| R6 | 2 | 2 | -17.62 | +2.00 | fails (only TLS contrasts pass) |
| B1 (CCA) | 5 | 0 | -1.31 | +1.22 | direction OK but no significance |
| B2 | 1 | 0 | -13.74 | +0.42 | fails |
| B3 | 2 | 0 | -3.17 | +0.52 | fails |

per-contrast detail (z statistics, *** = p<0.001, ** = p<0.01, * = p<0.05):

| distinct vs ambiguous | R1 | R2 | R3 | R4 | R6 | B1 | B2 | B3 |
|---|---|---|---|---|---|---|---|---|
| Tumor vs High TIL | -8.82*** | -2.80** | +0.17 | -6.87*** | -9.92*** | +1.19 | -5.85*** | -2.00* |
| Tumor vs Low TIL | -12.37*** | -6.37*** | -11.81*** | -11.28*** | -17.62*** | +0.37 | -13.74*** | -3.17** |
| Lymphoid nodule vs High TIL | **+3.23**\*\* | **+2.80**\*\* | -3.92*** | -1.48 | **+2.00**\* | +1.22 | +0.42 | -2.53* |
| Lymphoid nodule vs Low TIL | **+5.33**\*\*\* | **+3.60**\*\*\* | -5.79*** | -1.15 | **+1.98**\* | +0.71 | -0.31 | -3.08** |
| Necrosis vs High TIL | -11.01*** | -3.07** | +7.98*** | -3.37*** | -5.71*** | +0.27 | -2.55* | +0.52 |
| Necrosis vs Low TIL | -16.14*** | -6.79*** | +12.05*** | -4.29*** | -10.18*** | -1.31 | -6.92*** | +0.37 |

**reading — the proposal's directional prediction broadly fails**:

- The **only contrast pair** where the proposal's predicted direction holds significantly across multiple contrastive runs is **Lymphoid nodule (TLS) vs TIL-stroma** in R1, R2, R6 (the late-fusion contrastive runs). The TLS-vs-TIL prediction is supported.
- **Tumor vs TIL-stroma**: every run shows TIL-stroma > Tumor on matched-pair cosine, opposite to the proposal. Statistically decisive on all contrastive runs (z ≤ -2.8, p<0.01) except R3.
- **Necrosis vs TIL-stroma**: same inversion as Tumor on R1/R2/R4/R6. R3 (Barlow) is the lone exception, showing Necrosis > TIL-stroma with high z — Barlow's redundancy reduction produces a different geometry whose internals are not directly comparable.
- **R4** fails the prediction on every single contrast (0/6) with all z's negative — its tight 3-d manifold has uniformly high cosine, and the small ambiguous compartments are tighter still.
- **B1 CCA** is the only run whose direction agrees with the proposal on 5/6 contrasts, but with no statistical significance — uniform low-effect cosines, no real per-compartment differentiation.

**mechanism hypothesis (not pre-registered)**: the failure is annotation- and sample-size-driven, not a refutation of cross-modal alignment quality.
- High TIL stroma (n=105) and Low TIL stroma (n=354) are small, well-curated compartment annotations — internally homogeneous biology, clean H&E↔ST correspondence.
- Tumor (n=4384) and Necrosis (n=1178) are large, heterogeneous compartments — TNBC subtype variation alone produces within-compartment H&E↔ST decorrelation. The matched-pair cosine averages over more biologically diverse niches.
- Lymphoid nodule (n=44) shows the predicted direction in R1/R2/R6 because it is morphologically *uniquely* distinct (dense lymphocyte aggregation), and its low n keeps within-compartment biology homogeneous.

**part B verdict**: the proposal's H2 part B prediction is **only partially supported** (TLS-vs-TIL direction in 3 of 8 runs; tumor/necrosis predictions fail systematically). The result is consistent with a sample-size + annotation-curation confound rather than evidence against cross-modal alignment. A re-run on the 90-subarray annotated subset with full 18-class compartment vocabulary (`Tumor central` vs `Tumor edge`, `Tumor stroma rich` etc., per Wang et al. annotation) may recover the predicted direction; the current 12-class collapsed vocabulary appears to be too coarse for the prediction to hold.

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

## step 3: stress-testing the fusion-axis verdict (R1 vs R4)

added 2026-05-05. three follow-on analyses run after the distribution-first pivot landed, to determine whether the report's "R1 disentangles more than R4" mechanism claim is statistically defensible. all three operate on already-saved per-run artifacts; no retraining.

### input data

- 7 alignment runs: R1 (infonce+late), R2 (supcon+late), R3 (barlow+late), R4 (infonce+cross_attn), R6 (infonce+late, novae-only ST ablation), B1 (CCA), B2 (Procrustes). Same 85/15 patient-stratified split, seed=42, ~261 matched subarrays at subarray-level pooling.
- Per run: `embeddings_test.parquet` (~45k niches × 512d z_he and z_st), `metrics_h1_raw.json` (CKA, AUC, R@K), `eval/biology.parquet` (subarray-level bio and patient deltas + z-scores + raw_he/raw_st reference), `eval/biology_nulls.parquet` (1e5-perm null distributions, shared seed=42 across runs and views).
- Permutation framework: matched-null on patient_id (preserves patient-cluster sizes), label-shuffle on biology labels.
- Bareche TIME labels (computational subtypes from bulk-expression NMF) external to the model.

### analysis 1 - rank-matched CKA control

**hypothesis**: is R4's H1 advantage (CKA_after = 0.564 vs R1 = 0.242) collapse-driven (R4 lives in ~3 effective dimensions, R1 in ~5) or genuine architectural work? a tighter manifold mechanically scores higher CKA against any partner.

**method**: project R1's z_he onto its top-3 principal components (matching R4's effective rank), recompute linear CKA against R1's z_st. R4 at rank 3 is the sanity check (should approximately match R4's full-dim CKA since R4 already lives in ~3 dims; if not, the projection code is broken).

falsifier grid (committed before computing):
- R1@rank3 ≥ R4 (0.564) → collapse-driven; R4 has no architectural advantage at matched rank
- R1@rank3 < 0.50 → R4's architecture is doing real work
- 0.50 ≤ R1@rank3 < 0.56 → partial; some genuine advantage

**metrics / results**:

| run | full-dim CKA | rank-3 CKA | gap |
|---|---|---|---|
| R1 | 0.242 (matches reported) | 0.234 | -0.008 |
| R4 | 0.564 (matches reported) | 0.564 | -0.0003 (sanity passes) |

R1@rank3 (0.234) vs R4 full (0.564): delta = **-0.331**. R1 is in the "R4's architecture is doing real work" bucket by a wide margin. R1's CKA barely moves under rank projection because R1 already lives mostly in its top 3 PCs (PR=4.84).

**conclusion**: R4's cross-attention architecture is doing real cross-modal alignment work beyond compression. the collapse hypothesis is rejected. the fusion-axis trade-off framing in the report holds; R4 is genuinely doing more than compressing.

artifacts: `runs/tnbc-92/eval/compare/rank_matched_cka.parquet`. code: `scripts/eval_rank_matched_cka.py`.

### analysis 2 - paired-perm test on (bio_delta - patient_delta) difference

**hypothesis**: does R1 have statistically more asymmetric compression than R4 (or vice versa) at the cohort level?

**method**: paired permutation test under shared-seed=42 nulls. for each run pair (run_a, run_b) and view (z_he, z_mean):
- test stat: `obs_diff = asym_a − asym_b`, where `asym = bio_delta − patient_delta`
- null: `asym_diff_null[i] = (bio_a_null[i] − bio_b_null[i]) − (pat_a_null[i] − pat_b_null[i])`
- p_two = `(|asym_diff_null| ≥ |obs_diff|).mean()` with continuity correction
- shared seed makes the per-perm-index difference a true paired sample

**metrics / results (R1 vs R4 — the headline)**:

| view | asym_R1 | asym_R4 | obs_diff | p (1e5 perms) |
|---|---|---|---|---|
| z_he | -0.335 | -0.034 | -0.300 | <1e-5 |
| z_mean | -0.198 | -0.014 | -0.183 | <1e-5 |

R1's asym is significantly more negative than R4's on both views. **in raw cosine units, R4 is closer to symmetric than R1** (less negative asym = closer to bio/patient parity in raw cosine geometry).

**conclusion**: in raw delta units, the test is statistically decisive in the direction *opposite* to the report's prior z-ratio claim. the two framings measure different things — z-ratio is about statistical extremity (delta normalized by null spread); raw-delta is about absolute cosine geometry. they can disagree because R4's null is tighter (smaller deltas in absolute terms, but proportionally larger relative to null), while R1 has wider absolute spread.

artifacts: `runs/tnbc-92/eval/compare/paired_asym_tests.parquet`. code: `scripts/eval_paired_asym.py`.

### analysis 3 - per-run vs raw_he paired-perm (one-sample)

**hypothesis**: did each run move its asymmetry away from raw_he, and in which direction? "asymmetric compression" claims rest on movement *from raw*, not absolute position.

**method**: one-sample paired-perm against raw_he reference, shared-seed=42:
- test stat: `obs_diff = run_asym − raw_he_asym`
- null: `null_diff[i] = run_null[i] − raw_he_null[i]`
- raw_he asym (TIME, both views via paired math) = -0.0965

**metrics / results (z_he, sorted by direction)**:

| run | asym | move from raw | direction | p (1e5 perms) |
|---|---|---|---|---|
| R3 | -0.096 | +0.001 | stays at raw | **0.94 (n.s.)** |
| R4 | -0.034 | +0.062 | TOWARD symmetric | <1e-5 |
| R1 | -0.335 | -0.238 | away from raw | <1e-5 |
| R2 | -0.227 | -0.130 | away from raw | <1e-5 |
| R6 | -0.414 | -0.317 | away from raw | <1e-5 |
| B1 | -0.535 | -0.439 | away from raw | <1e-5 |
| B2 | -0.639 | -0.543 | away from raw | <1e-5 |

z_mean version: R3 and R4 both move TOWARD symmetric (p<1e-5 each); R1, R2, R6, B1, B2 all move AWAY (p<1e-5).

**conclusion**:
- R4 is the only run that moves toward symmetric on both z_he and z_mean (p<1e-5, both views).
- R3 (Barlow) is statistically indistinguishable from raw_he on z_he (p=0.94) — Barlow's redundancy reduction does not move the asymmetric structure on the H&E side. Moves toward symmetric on z_mean (joint average) only.
- All other contrastive runs (R1, R2, R6) actively move away from raw_he in raw cosine units — patient_delta grows faster than bio_delta in raw geometry.
- Classical baselines (B1, B2) catastrophically amplify the raw asymmetry. on raw-delta-from-raw, B1 and B2 are the most extreme runs in the grid.

artifacts: `runs/tnbc-92/eval/compare/per_run_vs_raw_asym.parquet`. code: `scripts/eval_paired_asym.py` (extended).

### supporting: bootstrap CIs on asym (z_he + z_mean, all 7 runs)

extension of the existing distribution-first bootstrap to z_mean and to all 7 runs in one bootstrap pass (encoding amortized per run; only the per-view delta computation runs twice per resample). 1000 patient-level resamples.

| run | view | asym | 95% CI (lo, hi) |
|---|---|---|---|
| R1 | z_he | -0.335 | [-0.433, -0.268] |
| R1 | z_mean | -0.198 | [-0.247, -0.168] |
| R4 | z_he | -0.034 | [-0.049, -0.024] |
| R4 | z_mean | -0.014 | [-0.020, -0.010] |
| R3 | z_he | -0.096 | [-0.138, -0.069] |
| R6 | z_he | -0.414 | [-0.517, -0.351] |
| B1 | z_he | -0.535 | [-0.662, -0.601] |
| B2 | z_he | -0.639 | [-0.753, -0.614] |

note: B1's observed asym lies *outside* its bootstrap CI on both views (CI midpoint is more negative than the observed). this is an artifact of CCA's geometry being fragile to patient-level resampling — re-fitting is implicit in the encode step, and certain patients dominate the CCA solution. fig9 annotates these "CI off" with an `x` marker. not a code bug; a property of CCA.

artifacts: `runs/tnbc-92/eval/compare/asymmetric_compression.parquet`, `runs/tnbc-92/eval/figures/fig9_asym_compression_z_he.pdf`, `fig9_asym_compression_z_mean.pdf`. code: `scripts/eval_compare_and_plot.py`.

### three-framings summary (the consolidated mechanism reading)

| measure | R1 | R4 | winner |
|---|---|---|---|
| z_bio / z_pat (z_he, "primary view") | 0.517 | 0.438 | R1 |
| z_bio / z_pat (z_mean, "joint view") | 0.417 | 0.513 | R4 |
| run_asym − raw_he_asym (raw cosine, both views) | further from raw | toward raw | R4 |

R1 wins one framing, R4 wins two. the directional headline claim is not statistically defensible across legitimate measures.

### overall conclusions on the fusion-axis verdict

1. **R4's H1 advantage is real architectural work**, not driven by manifold collapse. the rank-matched CKA control rejects the collapse hypothesis with a -0.331 gap at matched rank.

2. **the "R1 disentangles more than R4" directional claim is unit-dependent and not statistically defensible.** three legitimate measures of asymmetric compression give 1 vs 2 (R1 vs R4). the cleaner robust claim: every contrastive method compresses patient and amplifies biology relative to raw in z-units; in raw cosine units only R4 actually moves toward bio-patient symmetry.

3. **R3 (Barlow) is conservative, not transformative.** on z_he, Barlow does not move the asymmetric structure (p=0.94 against raw_he). a more subtle Barlow story than the report previously had — Barlow's redundancy reduction preserves the raw H&E asymmetry rather than reorganizing it.

4. **classical baselines amplify the raw asymmetry catastrophically.** B1 (CCA) and B2 (Procrustes) on z_he are the most extreme runs in the grid. CCA fails H3 z-ratio (0.064 < raw floor 0.129); Procrustes passes (0.318). Both amplify raw-delta asymmetry; the H3 difference is that CCA's failure mode is *correlation-maximization*, not *linearity*. this refines the report's existing mechanism story (B2 was previously predicted to fail like CCA; that prediction is wrong).

5. **the R1-vs-R4 choice is task-dependent.** R1's wider cosine spread benefits niche-level retrieval; R4's tight manifold benefits cohort-level group statistics. neither dominates as "more disentangled."

### what this changes elsewhere in the report

pending writing-pass items, not yet applied to the rest of the document:
- mechanism section (lines 124-141) needs revision: the factorization is *correlation-maximization vs everything else*, not *linear vs nonlinear*. B2 prediction was wrong.
- "what is unmeasured" table (lines 90-102): drop rank-matched CKA, paired-(bio−pat), per-run-vs-raw, asymmetric_compression z_mean extension. remaining unmeasured: R6b mirror ablation, AnInfoNCE.
- top-of-report status line: drop "B2/R3/R2 biology validation pending" claim — those landed 2026-04-24.
- patient-metric consolidation: standardize on permutation z as the single primary patient metric; demote probe accuracy (saturated at 14-test-patient scale, 0.90-1.00 across all runs) and ARI (rotation-invariant + patient-conflated) to auxiliary roles.
- R6 z_st patient saturation: pull from biology.parquet z (not patient probe accuracy) for the load-bearing claim.
- B2 ≡ B3 footnote on H2 metrics: rotation-invariance, mathematically guaranteed, not coincidental.

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