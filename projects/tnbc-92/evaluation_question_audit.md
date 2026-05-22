# evaluation-question quality audit · omicstra tnbc-92

written 2026-05-21. catalogs every meta-evaluation finding the project has made: cases where the evaluation **question** the proposal asks is not in fact a faithful proxy for the underlying biological **objective**. companion to `proposal_deviations.md` (scope drift) - this doc is about *question-quality drift*. audience: external reviewer / Cameron / future Nasim.

motivation, plain: Cameron's read, paraphrased: *"the quality of the evaluation is totally dependent on how good of a question your asking is."* this is exactly right, and the project already has a track record of catching its own evaluation-question failures and rescuing them. that track record is below.

source for the proposal H1/H2/H3 wording: `docs/papers/sanati_2025_researchhub.pdf` pp. 5 + 9 (verbatim quoted per hypothesis).

---

## the audit pattern, generalized

every entry below follows the same shape:

| field | meaning |
|---|---|
| **proposal wording (verbatim)** | what the proposal says it tests |
| **the implicit objective** | the underlying biology question the proposal is *really* trying to answer |
| **the original test** | the operationalization that runs as code |
| **what the audit found** | why the test does not faithfully answer the objective |
| **the rescue** | what the test was replaced or augmented with |
| **status under v3** | does v3 change the finding |

a faithful evaluation requires the test to actually move when the objective moves. if the test moves with a confounded axis (patient identity, dimensionality, label vocabulary), the test is invalid even if the test's number looks impressive.

---

## finding #1 - H2 part A: the 9 archetypes are patient-level, not niche-level

**proposal wording (verbatim, p. 5)**: *"The nine spatial archetypes (5) will cluster more coherently in the aligned latent space compared to individual modality spaces, as measured by adjusted Rand index and silhouette score."*

**the implicit objective**: alignment captures **niche-level** biological organization of the tumor microenvironment.

**the original test**: ARI of KMeans-on-aligned-space clusters vs the 9 spatial archetype labels (`archetype` column in `data/embeddings/niches/{TNBC*}.parquet`).

**what the audit found** (2026-05-11): `archetype` is sourced from `Spatial archetypes_defined_on_ST_global_pseudobulk` in Wang's `Clinical.RDS`. it is a **patient-level pseudobulk label** - one value per patient, propagated to every niche of that patient. verified: 14 of 14 held-out test patients have a single archetype across all their niches; NMI(archetype, patient_id) ≈ 0.90 on the test cohort.

mechanically, the H2 ARI test is therefore a 9-class **patient classification at niche resolution**, not a niche-level biology test. higher ARI on archetype = stronger patient identity in the latent space, not better TME biology. classical baseline B1 (CCA) wins this test (ARI 0.307) because CCA amplifies patient identity 3.3x (see finding #3); R4 cross-attention scores lowest (0.059) precisely because it suppresses patient identity.

**the rescue** (commit 2.5, 2026-05-11): rebuild the niche-join with `mc_megacluster` joined (Wang's per-spot 14-class NMF labels from `mc_labels.tsv`, niche-level). add `eval_h2_mc_coherence(run)` to `eval.py`: KMeans k=14 → ARI + silhouette vs `mc_labels.megacluster`. report side-by-side with archetype, with the construct-validity caveat called out in writing.

niche-level result (mc_megacluster ARI, the audit-corrected metric): R6 0.251, R1 0.231; R4 KMeans 0.098 (rescued to 0.230 via patient-stratified linear probe - geometry-induced KMeans failure, not encoding failure).

**status under v3**: the rescue label `mc_megacluster` is unchanged (Wang's labels are external). the v3 alignment retrain re-evaluates niche-level coherence on the same audit-corrected metric. **the construct-validity issue is unresolved by gpath2vec rebuilds; it lives in the label vocabulary, not the embedding.** archetype-ARI will remain a patient-classification proxy under any alignment.

---

## finding #2 - H3: pathway-direction *magnitude* does not test pathway-direction *specificity*

**proposal wording (verbatim, p. 5)**: *"Reactome pathway embeddings computed via gpath2vec for five cancer-relevant pathways (TGF-β Signaling, Immune System, Extracellular Matrix Organization, Cell Cycle, Programmed Cell Death) will correlate with **specific directions** in the shared latent space via canonical correlation analysis, indicating preservation of interpretable biological signal."*

**the implicit objective**: the aligned space encodes **5 separable** pathway-specific axes. interpretability requires distinguishability: "this niche has high Immune signal AND low ECM signal" must be a meaningful claim, which requires Immune and ECM to span different directions.

**the original test**: per-pathway univariate CCA, report z_A (option-A cross-patient transfer) per pathway. BH-FDR across the 5-pathway family.

**what the audit found** (2026-05-21, with v3): R4 z_he gives 4/4 testable pathways significant at BH-FDR 0.004 with z_A 9.6-22.3. but the 5 canonical Z-direction vectors have **off-diagonal pairwise |cos| 0.892 mean, 0.976 max** - the 5 pathway directions essentially all land on the same axis in R4's shared space. z_A tells you "the pathway correlates with *some* direction"; it does not tell you "the 5 pathways correlate with *different* directions". the proposal phrasing "correlate with **specific** directions" implicitly asks the second question.

**the rescue**: pair every published z_A with the **pathway-direction specificity matrix** (off-diagonal |cos| of canonical Z-directions across the 5 pathways). reportable in 3 numbers per run: mean / median / max of the 10-cell off-diagonal. compute the same diagnostic per run.

specificity matrix per run, v3 gpath2vec, z_he (lower = more separable; pre-retrain numbers):

| run | mean off-diag | max | reading |
|---|---:|---:|---|
| R4 | **0.892** | 0.976 | heavy aliasing - z_A magnitude wins, specificity fails |
| B1 (CCA) | 0.895 | 0.970 | heavy aliasing (CCA explicitly optimizes correlation) |
| B2 (Procrustes) | 0.647 | 0.861 | **moderate - z_A high + axes distinguishable** |
| B3 (PCA unaligned) | 0.647 | 0.861 | same as B2 (rotation-equivalent) |
| R1 (InfoNCE late) | **0.395** | 0.660 | cleanest axes, but z_A near zero |
| R6 (R1 no gpath2vec) | 0.435 | 0.721 | clean, near-null |

the routing rule that emerges: **R4 for cross-patient magnitude on any one pathway; B2 for per-pathway distinguishability across multiple pathways.** the manuscript must report both diagnostics or the H3 claim is incomplete.

**status under v3**: the test #3 contribution analysis (Gini 0.32 on R4 z_he, top-10% mass 19.5%) confirms the signal is *broadly distributed across the cohort* (not few-dominant-niches). this strengthens the magnitude claim but doesn't fix the aliasing - the aliasing is a *geometric* property of R4's representation, independent of signal concentration. **post-retrain (phase 2 alignment on v3 ST input) the specificity matrix is re-computed; the routing rule may shift if cross-attention's induced aliasing is sensitive to ST-side feature changes.**

---

## finding #3 - bio z vs patient z: biology coherence is meaningful only relative to patient leakage

**proposal wording** (no direct equivalent): structural coherence is reported in absolute ARI terms across embedding spaces.

**the implicit objective**: aligned space encodes **biology more than patient identity**. patient-only structure is anti-helpful for cross-cohort interpretation.

**the original test**: ARI absolute values per run on the proposal labels.

**what the audit found**: ARI absolute values do not separate "encodes biology" from "encodes patient cluster structure" - they conflate the two. classical CCA (B1) gets ARI 0.307 on archetype while having **patient z = 122.13 vs raw H&E patient z = 37.25** (CCA amplifies patient identity 3.3x). a high ARI in this regime is anti-helpful.

**the rescue**: pair every ARI / silhouette report with a **bio z vs patient z** permutation z-test. compute `delta_obs = mean[same-label cos] - mean[diff-label cos]` for biology labels (mc_megacluster, MC_global, MC_tumor, TIME) AND for `patient_id`, with matched-null shuffles preserving cluster sizes. report the **bio/patient ratio** as a single-number diagnostic. ratio > 1: biology dominates. ratio < 1: patient identity dominates - the run is anti-helpful for cohort-level use, no matter how high its ARI.

v3 noise-floor on the embedding directly (4k stratified niches, 10k perms): MC z = +30.3, patient z = +29.4, ratio = **1.03**. compared to v1 arm A's MC z +20.3, patient z +75.5, ratio 0.27. **v3 alone (before any alignment) flips the patient-leakage dynamic.**

**status under v3**: the bio-vs-patient permutation framework continues to be the load-bearing diagnostic for any "biology coherence" claim. v3's substantial improvement on the ST-side embedding (ratio 1.03 vs v1's 0.27) is the strongest single result we have for "v3 fixes a real problem the v1 stack had." this gets re-measured post-retrain on the aligned latents (z_he / z_st / z_mean).

---

## finding #4 - within-subarray vs cross-subarray retrieval: in-distribution numbers leak the answer

**proposal wording (verbatim, p. 5)**: *"retrieval metrics (Recall@K, MRR, median rank)"* for H1.

**the implicit objective**: the aligned space supports **cross-patient** cross-modal retrieval - finding a matched ST niche from an H&E niche on a *new* patient the model has never seen.

**the original test (early version, retired)**: report R@K on all test pairs without regard to patient-of-origin.

**what the audit found** (cited in memory `feedback_within_vs_cross_subarray_eval.md`): in-distribution within-subarray R@1 = 0.667 collapses to **R@1 = 0.000** on cross-subarray patient-held-out evaluation. CCA (B1) and the InfoNCE-trained R-runs both pass within-subarray (it's the patient-leakage shortcut: niches that share a patient share an H&E slide and a tissue, so retrieval is trivial). cross-subarray is the only honest test.

**the rescue**: cross-subarray patient-held-out evaluation is the **only** primary retrieval metric reported. 14 held-out test patients, 38 test subarrays, ~35k–45k test niches. mentioned as a hard constraint in `program.md` (rule #4): *"NEVER use within-subarray retrieval as the primary evaluation."*

**status under v3**: unchanged. cross-subarray patient-held-out remains the only honest H1 metric. v3 retrain uses the same split protocol.

---

## finding #5 - v2 (level=all) gpath2vec build: degenerate embedding masquerades as broad pathway coverage

**proposal wording (no direct equivalent)**: H3 commits to *"interpretable biological signal"* in the shared latent.

**the implicit objective**: pathway coverage (more pathways embedded → more biology testable) should improve downstream H3 transfer. expanding from Reactome level=low to level=all *should* help.

**the original test**: H3 5-pathway CCA evaluated on the v2 (`aucell_level_all_topk50_dim128_e1_s1234`) gpath2vec build, which embedded 1907 pathways at all hierarchy levels (vs v1's 680 at level=low).

**what the audit found** (2026-05-20, with v2): the v2 embedding **dim-collapsed**. MC z = -0.13 (no biology signal) and patient z = +0.92 (no patient signal either). same-MC cos = diff-MC cos = 0.878 across all label partitions. niches were ~cosine-1 to each other. broader coverage was bought at the cost of representation depth (1 epoch / dim 128 vs v1's 5 epochs / dim 512). under this collapse, the CCA H3 grid would have shown inflated z_A scores from low effective rank, not from meaningful biology.

**the rescue**: the noise-floor permutation test (MC z + patient z) on the embedding **directly** - before running the downstream CCA grid - catches this kind of failure cheaply. now a mandatory gate on any new gpath2vec build before promoting it to canonical. mc/pat ratio < 1.0 + same/diff cos within 0.01 of each other = degenerate, do not ship.

**status under v3**: v3 (Fisher + MAD/mean + level=low + dim 512 / 5 epochs) passes the noise-floor: MC z +30.3, patient z +29.4, ratio 1.03. the rescue procedure (noise-floor before CCA) is now permanent in the workflow. v2 is retired as a "published-quality negative result" demonstrating the diagnostic works.

---

## generalized lessons (audit pattern across findings 1-5)

each finding follows the same shape:
1. proposal commits to an evaluation metric in plain prose.
2. the metric is operationalized in code.
3. an audit discovers the metric measures something **confounded with what it appears to measure** (patient identity, dimensionality, in-distribution shortcut, dim collapse).
4. the rescue adds a paired diagnostic (mc_megacluster instead of archetype; specificity matrix paired with z_A; bio z vs patient z; cross-subarray only; noise-floor before CCA).
5. the headline result is reported only with the paired diagnostic.

**the constructive shape of this audit history is the project's strongest piece of evidence that the evaluation framework is review-defensible.** every named confound has a documented detector. when an external reviewer asks "are you sure your metric isn't measuring X confound", the answer is *"yes, the paired diagnostic is here, and the project caught it when the confound first surfaced."*

---

## where Cameron's biology-validation lane lives

Cameron is offering biology validation. four concrete checks the project needs from a biologist:

1. **MC2 = stromal/mesenchymal? MC11 = proliferative?** v3 R4 z_he canonical directions split as TGFb/Immune/ECM → MC2 (positive direction, n=6,154 niches in MC2 on the test set) vs Cell Cycle / PCD → MC11 (n=1,010). is the MC2-stromal / MC11-proliferative mapping consistent with Wang's MC characterization (`Robjects/clustering/clustPrototypes/`)? if yes, v3 H3 has independent biology validation even with the specificity aliasing.

2. **Compartment-label vocabulary**: H2 part B compartment-cosine result inverted on tumor/necrosis under the 12-class label. the 18-class re-annotation on the 90-subarray annotated subset is queued (Wang's `dominant_18class`). worth a biologist call on whether tumor central vs tumor edge, lymphoid nodule vs high-TIL stroma, etc. are the right granularity.

3. **R6 pathway-vs-novae trade-off**: R6 drops gpath2vec from the ST input. its z_he has clean pathway-direction specificity (off-diag 0.435) but near-null z_A. is this biologically interpretable - "R6 keeps the spatial graph topology Novae encodes, drops the pathway-graph anchor" - or does the empirical pattern not match a clean biology story?

4. **TGF-β R-HSA-170834 vs the family-level R-HSA-9006936**: program.md pins the H3 TGF-β target at R-HSA-170834 ("Signaling by TGF-beta Receptor Complex"), structurally underpowered at n=2 in v1. v3 covers 7 nodes under it. is the **receptor-complex subtree** the right TGF-β scope for TME biology, or should H3 report the family-level supplementary as well?

these are the *biology* questions; the *computational* H1/H2/H3 numbers are review-defensible once paired with their diagnostics. the demo notebook for Cameron (`notebooks/final/cameron_demo.ipynb`, task #13) operationalizes (1)-(4) as concrete questions with the v3 data already loaded.

---

## pointers

| asset | path |
|---|---|
| proposal PDF | `docs/papers/sanati_2025_researchhub.pdf` |
| this audit | `projects/tnbc-92/evaluation_question_audit.md` |
| scope-drift companion | `projects/tnbc-92/proposal_deviations.md` |
| v3 retrain plan | `projects/tnbc-92/v3_phase2_plan.md` |
| v3 H3 results | `notebooks/final/h3_gpath2vec_v3.ipynb` |
| noise-floor harness | `scripts/eval_h3_gpath2vec_noise_floor.py` |
| direction-contribution harness | `scripts/eval_h3_gpath2vec_direction_contribution.py` |
| Wang MC labels (the audit-corrected H2 niche-level label) | `data/embeddings/biological_signals/mc_labels.tsv` |
| Wang MC characterization (for biology validation) | `data/inputs/Robjects/clustering/clustPrototypes/` |