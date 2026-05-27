# tnbc-92 · task-conditional routing matrix (v2 - v3 gpath2vec retrain)

which alignment to use for which question, with v3 evidence. routing-rule **shape** unchanged from v1; only the cited numbers were updated against the v3 retrain.

**v1 baseline** for head-to-head: [`../v1/tnbc92_routing_matrix.md`](../v1/tnbc92_routing_matrix.md).

---

## routing rule (v3 evidence)

| if the question is ... | route to ... | v3 evidence | construct-validity guard |
|---|---|---|---|
| cross-modal retrieval (find matched ST niche from H&E) | **R4_v3** | H1 AUC 0.859, CKA 0.631, alignment gap 0.233. rank-matched control still rejects collapse-driven explanation. | cross-subarray patient-held-out is the only honest retrieval eval (within-subarray = patient leakage). |
| group niches by recurring tumor-microenvironment state | **R1_v3 or R6_v3** | H2 KMeans on `mc_megacluster` (Wang per-spot 14-class NMF, niche-level): R6 ARI 0.244, R1 0.234. | use `mc_megacluster` not `archetype` (archetype is patient-level pseudobulk, NMI(archetype, patient) ≈ 0.89 - any "win" reads patient identity). |
| amplify biology, suppress patient identity | **R6_v3 (best ratio) or R4_v3 (lowest patient z)** | <!-- TODO: recompute v3 bio/patient ratio - v1 ratios: R6 0.548, R1 0.517, R4 patient z 22.97 (lowest in grid) --> | pair every ARI / silhouette claim with a bio-z vs patient-z permutation z-test (`projects/tnbc-92/evaluation_question_audit.md` finding #3). |
| find pathway-similar niches across new patients (proposal-literal) | **R4_v3** | H3 gpath2vec arm option A (cross-patient held-out, BH-FDR<0.05): R4 4/4 testable pathways significant, z_A 11.2-25.4 (Immune 25.4, ECM 16.0, CC 16.5, PCD 11.2). late-fusion contrastive (R1/R2/R3/R6) fail (0-1/4). | report z_he only (`z_st` / `z_mean` circular with v3 gpath2vec on ST input). TGF-β set_size=2 underpowered, excluded from FDR family. |
| detect fine-grained sub-pathway biology, esp. dense linear ECM | **B1_v3 / B2_v3** (classical) or **R4_v3** (competitive) | gpath2vec arm: B2/B1 4/4 testable, ECM z_A 9.2-10.4. R4 wins overall but classical is comparable on ECM. <!-- DAG-resolution decomposition deferred for v3; v1 DAG result: B1/B2 98.3% on ECM vs R4 93.3% --> | classical baselines pass H3 cross-patient on the linear ECM signal; use them as the right tool for that question even though they fail H1. |
| FTU-coherence test (tertiary lymphoid structures) | **R1_v3 / R6_v3 on H2 Part B** | TLS is the one FTU-like compartment where H2 Part B's directional prediction held (v1: R1/R2/R6 passed TLS-vs-TIL with z = +3.2 to +5.3). <!-- TODO: v3 Welch z-test recompute on `compartment_cosine.parquet` --> | other H2 Part B contrasts (Tumor / Necrosis) inverted decisively in v1 due to annotation-granularity confound (large heterogeneous classes); not a biology absence. |
| avoid for cohort-level biology interpretation | **B1_v3 (CCA)** | v3 H1 AUC 0.544 (weakest contrastive-family-comparable). CCA amplifies patient identity by construction (v1: patient z 122.13 vs raw 37.25, ratio 0.064 below raw floor 0.129). | pair every ARI report with a bio-z vs patient-z ratio; ratio < 1 means patient identity dominates - anti-helpful for cohort use. |
| sanity floor / isolate alignment from architecture | **B3_v3** (unaligned PCA) **+ B4_v3** (random-init MLP) | v3 B3 AUC 0.441 sub-chance, gap −0.053 (correct sanity behavior). v3 B4 AUC 0.496 ≈ chance (architecture alone contributes none of H1 gain). | B3 / B4 are required falsifiers - any aligned method must beat both decisively. |
| classical alignment family with rotation-equivalence sanity | **B2_v3** (Procrustes) | v3 AUC 0.706. B2 and B3 are rotation-equivalent on H2 (within-space metrics) and yield identical specificity matrices; they diverge on H1 (cross-modal cosines depend on rotation). | use this divergence to verify the test scoring is correct - if B2 = B3 on a cross-modal metric, something is broken. |

---

## v3 vs v1 routing changes (none)

the routing rule **shape is identical to v1**. v3 retrain confirms every routing recommendation made on v1 evidence, with numbers either improving (R4 H1 AUC 0.851 -> 0.859; R4 H3 gpath2vec arm z_A noticeably higher than v1's AUCell-arm z_A) or unchanged. **no method changed position** on any routing question.

---

## v3-specific guards to encode in the MCP agent

these are construct-validity guards lifted directly from the v3 evidence and the audit doc - the MCP agent must enforce them (see `docs/mcp_agent_design.md` for the design spec):

- **archetype-ARI is patient-leakage diagnostic, not biology** - never report it as a biology coherence claim without paired `mc_megacluster` ARI + patient-z diagnostic.
- **z_st and z_mean are circular** on H3 metrics when gpath2vec is on the ST input (R1/R2/R3/R4, B1/B2/B3 on v3). H3 z_he is the only clean view.
- **mc_megacluster is also partially circular on z_st** (ST input encodes the data the NMF clustered). z_he mc test is the citable cross-modal version.
- **R4 KMeans low MC ARI is metric-induced** (manifold geometry vs KMeans globular assumption), not biology absence. must pair with patient-stratified linear probe for a metric-independent claim. <!-- TODO: rerun on v3 -->
- **gpath2vec_niche on ST input is sha-locked to v3 build** `13985cbd...` per `data/embeddings/niches_v3/manifest.json`. do not silently swap.
- **TGF-β R-HSA-170834 is structurally underpowered** on the v3 build (set_size=2) - exclude from H3 FDR family, report separately.
- **R6 is the no-gpath2vec ablation** - if R6 H3 z_A >> 0, the gpath2vec ST anchor is not load-bearing; v3 confirms R6 0/4 testable, so the anchor IS load-bearing for cross-patient pathway transfer.

---

## pointers

| asset | path |
|---|---|
| v1 routing matrix | [`../v1/tnbc92_routing_matrix.md`](../v1/tnbc92_routing_matrix.md) |
| v3 results summary | [`tnbc92_results_summary.md`](tnbc92_results_summary.md) |
| MCP / LangGraph agent design (encodes these guards) | `docs/mcp_agent_design.md` |
| evaluation-question audit (Cameron's questions) | `projects/tnbc-92/evaluation_question_audit.md` |
| proposal deviations log | `projects/tnbc-92/proposal_deviations.md` |
| v3 retrain plan + memory | `projects/tnbc-92/v3_phase2_plan.md`, [[project_v3_retrain_result]] |
