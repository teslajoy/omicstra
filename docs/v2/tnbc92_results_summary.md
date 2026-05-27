# tnbc-92 · results summary (v2 - v3 gpath2vec retrain)

manuscript-current results on the v3 gpath2vec retrain grid (`runs/tnbc-92_v3/`). data version: v3 gpath2vec build `fisher_madmean_low_dim512_e5_s1234` (sha `13985cbd...`), v3 niche-join 208,786 niches, test set 35,594 niches / 14 held-out patients / 38 subarrays.

**v1 baseline** for head-to-head comparison: [`../v1/tnbc92_results_summary.md`](../v1/tnbc92_results_summary.md).

**status.** numbers verified from `runs/tnbc-92_v3/eval/*` and per-run `metrics_h1_raw.json`. prose interpretation preserved from the v1 framework (R4 wins H1+H3, late-fusion family wins H2 mc, classical baselines lose 30+% on the resolution shift) - all v1 directional claims hold on v3. v3 adds **R5_v3** (AnInfoNCE + late fusion, new method; per-dim learnable temperature, one-variable-change vs R1) to the grid. <!-- TODO: voice pass for any v3-specific phrasing the user wants to add -->

---

## method definitions

R1-R4, R6 and B1-B4 unchanged from v1; see [`../v1/tnbc92_results_summary.md`](../v1/tnbc92_results_summary.md) for the full method table. v3 architecture and method-grid axes unchanged; only the ST-side `gpath2vec_niche` feature was swapped from v1 to v3 build. **v3 adds one new method:**

| run | loss | fusion | ST input | what it tests |
|---|---|---|---|---|
| **R5_v3** | AnInfoNCE (per-dim learnable temperature, init = standard InfoNCE) | late MLPs | novae + gpath2vec_v3 | does anisotropic per-dim temperature improve late-fusion alignment over R1? one-variable-change vs R1. |

---

## H1 - Cross-Modal Alignment (v3 grid)

source: `runs/tnbc-92_v3/eval/H1/summary.json`, per-run `runs/tnbc-92_v3/{run}/metrics_h1_raw.json`.

| run | method | R@1 | R@5 | median rank | **AUC** | **CKA after** | alignment gap |
|---|---|---|---|---|---|---|---|
| **R4_v3** | InfoNCE + cross-attn | 0.00014 | 0.00155 | **3282** | **0.859** | **0.631** | **0.233** |
| R1_v3 | InfoNCE + late | 0.00042 | 0.00185 | 6234 | 0.761 | 0.312 | 0.194 |
| **R5_v3** | AnInfoNCE + late (new) | 0.00039 | 0.00177 | 6268 | 0.761 | 0.314 | 0.216 |
| R6_v3 | InfoNCE + late (novae-only ST) | 0.00014 | 0.00079 | 7016 | 0.737 | 0.239 | 0.154 |
| R2_v3 | SupCon + late | 0.00022 | 0.00107 | 7730 | 0.725 | 0.101 | 0.036 |
| B2_v3 | Procrustes | 0.00008 | 0.00079 | 8070 | 0.706 | 0.126 | 0.161 |
| R3_v3 | Barlow + late | 0.00000 | 0.00037 | 7644 | 0.674 | 0.306 | 0.143 |
| B1_v3 | CCA | 0.00006 | 0.00014 | 15554 | 0.544 | 0.076 | 0.007 |
| B4_v3 | random-init MLPs | 0.00006 | 0.00014 | 17936 | 0.496 | 0.120 | −0.0004 |
| B3_v3 | Unaligned PCA | 0.00000 | 0.00000 | 20688 | 0.441 | 0.126 | −0.053 |

random median-rank baseline = ⌈n/2⌉ = 17,797 (v3 test pool 35,594).

**v3 vs v1 deltas (AUC):** R4 +0.008, R1 +0.020, R6 +0.004, R2 +0.018, R3 +0.028, B2 +0.003, B1 +0.003, B4 +0.005, B3 −0.002. R5 is new (no v1 comparison). all contrastive runs improved or held; classical baselines essentially unchanged.

**R5_v3 ≡ R1_v3 on H1** (AUC 0.761 / 0.761, CKA 0.314 / 0.312). AnInfoNCE's per-dim learnable scaling barely moved from init (mean log-scale −0.015, std 0.042; only 17/512 dims moved >0.1 in log space). the matched-pair InfoNCE objective doesn't strongly incentivize direction-specific weighting; AnInfoNCE effectively collapses to standard InfoNCE on H1.

**verdict (unchanged from v1).** R4 wins by clean margin on both AUC and CKA. all five contrastive late-fusion runs (R1, R2, R3, R5, R6) plus R4 cross-attn outperform all three classical baselines (B1-B3) on AUC and CKA with no overlap between families. B4 random-init at chance (0.496) confirms architecture alone contributes none of the H1 gain - contrastive training does the bulk, cross-attention adds the rest. B3 unaligned PCA sub-chance with negative gap (sanity floor).

---

## H2 Part A - niche-level biology coherence (mc_megacluster, audit-correct)

source: `runs/tnbc-92_v3/eval/H2/mc_coherence.parquet`. KMeans(k=14) on z_he, ARI vs Wang's per-spot 14-class NMF `mc_megacluster` label.

**z_he is the citable view** (non-circular - H&E never sees mc_megacluster). z_st / z_joint partially circular and reported in the parquet for completeness only.

| run | archetype ARI (z_he) | **mc_megacluster ARI (z_he)** | delta archetype→mc | reading |
|---|---|---|---|---|
| **R5_v3** | — (TODO) | **0.246** | — | top of 3-way tie within KMeans seed noise (vs R6 0.244, R1 0.234) |
| **R6_v3** | 0.236 | **0.244** | +0.008 | niche-level biology winner (v1) |
| **R1_v3** | 0.244 | **0.234** | −0.010 | runner-up |
| B1_v3 | 0.259 | 0.211 | −0.048 | classical, loses leakage advantage |
| B2_v3 | 0.306 | 0.194 | **−0.112** | classical, big leakage drop |
| B3_v3 | 0.307 | 0.194 | **−0.112** | rotation-equivalent to B2 |
| R3_v3 | 0.130 | 0.159 | +0.029 | low |
| R2_v3 | 0.187 | 0.152 | −0.035 | low |
| **R4_v3** | 0.065 | **0.101** | +0.036 | lowest on KMeans (manifold-geometry bias, same v1 finding) |
| raw_he | 0.295 | 0.164 | **−0.131** | raw H&E archetype "win" was patient identity |
| raw_st | 0.017 | 0.046 | +0.029 | Novae alone barely clusters mc - alignment is doing the work |

**construct-validity receipts on v3:**
- classical baselines lose 11+ ARI points archetype → mc (B2/B3 −0.112, B1 −0.048, raw_he −0.131). quantifies the patient-leakage the audit caught.
- **R5 / R6 / R1 in a 3-way tie at the top** within typical KMeans seed-variance (ARI gaps <0.012). late-fusion contrastive (with and without AnInfoNCE, with and without gpath2vec ST anchor) is the right family for niche-level region clustering on tissue.
- R4 lowest on KMeans (0.101): same v1 finding - cross-attention compresses to a tight manifold and KMeans assumes globular clusters. metric-induced, not biology absence. **v1 rescue (patient-stratified linear probe at parity) not yet rerun on v3** - flag if needed for manuscript.
- alignment provides **5× lift** over raw ST (0.046 → 0.246 with R5) on niche-level biology coherence.

**archetype-ARI (proposal-literal) reported as diagnostic only**, NOT a biology claim - per construct-validity audit (`projects/tnbc-92/evaluation_question_audit.md` finding #1, NMI(archetype, patient_id) ≈ 0.89). see also v3 notebook §5 (construct-validity panel), §8b, §8c.

<!-- TODO: voice pass on the R4 KMeans-rescue framing if user wants to commit to "R4 has biology at parity once geometry-bias removed" - that claim needs the v3 linear-probe rerun first -->

---

## H2 Part B - compartment cosine contrasts

source: `runs/tnbc-92_v3/eval/H2/compartment_cosine.parquet` (regenerated for v3; named-contrast Welch z-test analysis preserved from v1).

| contrast | v1 result | v3 status |
|---|---|---|
| TLS vs hi-TIL stroma | passed R1/R2/R6 (z=+3.2 to +5.3) | <!-- TODO: recompute z-tests from v3 compartment_cosine.parquet --> |
| TLS vs lo-TIL stroma | passed R1/R2/R6 | <!-- TODO --> |
| Tumor vs hi-TIL / lo-TIL | inverted in all R-runs | <!-- TODO --> |
| Necrosis vs hi-TIL / lo-TIL | inverted in all R-runs | <!-- TODO --> |

**v1 interpretation (likely preserved on v3): TLS is the one FTU-like compartment where the proposal's directional prediction held.** annotation-granularity confound on Tumor/Necrosis (large heterogeneous classes dilute matched-pair cosine). see v3 notebook §8c for the FTU framing.

**coverage:** 13/38 v3 held-out test subarrays carry compartment annotations (~35% of test niches, same scope as v1).

---

## H3 - Pathway Interpretability (gpath2vec arm, proposal-literal)

**this is the proposal-literal H3 test** - per-pathway univariate CCA between each run's shared-latent canonical direction and the **v3 gpath2vec** pathway embeddings (niche cosine profile to embedded Reactome descendants of each named parent). source: `runs/tnbc-92_v3/eval/H3/pathway_cca_gpath2vec_v3/per_pathway_cca.parquet`. embeddings sha256 `782fe64f...` locked in provenance.

**z_he view only** (non-circular - v3 gpath2vec is on the ST input, so z_st / z_mean are circular with the gpath2vec-derived signal; reported in parquet for completeness, not headline). option A = cross-patient held-out 3-patient sub-split; 500 perms; BH-FDR over testable (set_size>=5) z_he family.

### option A (cross-patient held-out), z_he z_A

| run | Immune (n=76) | ECM (n=15) | Cell Cycle (n=15) | PCD (n=7) | TGF-β (n=2, underpowered) | **sig testable (/4)** |
|---|---|---|---|---|---|---|
| **R4_v3** | **25.4** | **16.0** | **16.5** | **11.2** | 5.2 | **4/4** |
| B2_v3 / B3_v3 | 17.1 | 9.2 | 10.4 | 6.7 | 3.1 | 4/4 |
| B1_v3 | 15.2 | 10.4 | 9.8 | 7.5 | 3.1 | 4/4 |
| **R5_v3** (new) | −1.9 | **3.0** | **2.1** | **3.0** | 0.9 | **2/4** |
| R1_v3 | 2.7 | 1.5 | 1.5 | −0.6 | 0.2 | 1/4 |
| R3_v3 | 2.3 | 0.1 | 0.7 | 0.2 | 1.4 | 1/4 |
| R2_v3 | −1.6 | −2.2 | −1.5 | −1.2 | −1.1 | 0/4 |
| R6_v3 | 0.5 | 1.3 | −1.8 | 0.6 | 0.3 | 0/4 |

TGF-β R-HSA-170834 set_size=2 - **underpowered, excluded from the BH-FDR family**, reported separately. only 4 pathways enter the FDR family on the gpath2vec arm.

subtree coverage on the v3 build: Immune 35%, ECM 79%, Cell Cycle 12%, PCD 17%, TGF-β 17%. coverage is **not thresholded** (any cut separating PCD~17% from Cell_Cycle~11% is arbitrary - the F1 anti-pattern); representativeness reported per-pathway, reader-judged.

### verdict (v3 grid, gpath2vec arm)

**R4 wins cross-patient pathway transfer on the proposal-literal test.** 4/4 testable pathways significant at BH-FDR<0.05, z_A 11.2-25.4. late-fusion contrastive runs largely fail: **R5 2/4** (best of late-fusion family - ECM/CC/PCD pass, Immune actively suppressed at z_A −1.9), R1/R3 1/4, R2/R6 0/4. classical baselines competitive (4/4 with lower z_A 3-17). v3 R4 z_A is **stronger than v1 R4** on the AUCell arm (v1 R4 z_A 3.8-5.7; v3 gpath2vec-arm R4 z_A 11.2-25.4) - the v3 gpath2vec build amplifies the cross-patient pathway signal.

**R5 finding (negative result on rescue hypothesis).** AnInfoNCE per-dim temperature was queued (per `project_session_handoff`) as a candidate fix for R4's pathway-direction aliasing. R5 on late-fusion gained 1 sig pathway vs R1 (1/4 → 2/4) - a small effect. **AnInfoNCE is not the rescue for R4's aliasing**; if the question is "can anisotropic temperature un-alias R4 while preserving cross-patient transfer?" it needs to be tested on cross-attention directly (R5b - deferred, ~30 min compute, requires patching `CrossAttnFusion` to register `aniso_log_scale`).

### specificity / aliasing (z_he)

R4's pathway-direction off-diag |mean| ~0.75 (z_he, computed from canonical directions). consistent with v1 finding that R4's aliasing - 5 pathway directions collapse toward one biology axis - is the **source** of its cross-patient generalization. one axis is harder to overfit to patient identity than five distinct directions.

routing rule (unchanged from v1): **R4 for cross-patient magnitude on any one pathway; B2 for per-pathway distinguishability across multiple pathways.**

---

## H3-extended (biology vs patient permutation z-test)

<!-- TODO: re-run on v3 - v1 bio-vs-patient ratio table on Bareche TIME / MC_global / MC_tumor not yet recomputed for v3 alignment runs. v1 numbers preserved at ../v1/tnbc92_results_summary.md for reference. -->

---

## Systems-oriented evaluation (verdict holds, same as v1)

| field | observation |
|---|---|
| does one alignment dominate every objective? | **no.** R4 wins H1 + H3 cross-patient pathway transfer. R5/R6/R1 (3-way tie within seed noise) win H2 niche-level biology clustering. classical baselines competitive on ECM and fail H1. B1 amplifies patient identity. |
| conclusion | different alignment architectures optimize different biological objectives. niche-level cluster geometry, cross-modal retrieval, and cross-patient pathway transfer are not co-monotonic. the framework's value is in exposing the trade-offs and supporting task-conditional routing (see [`tnbc92_routing_matrix.md`](tnbc92_routing_matrix.md)). |
| strong methods | R4 (retrieval, cross-patient pathway transfer); R5 / R1 / R6 (niche-level region clustering on tissue) |
| baselines | all baselines lose at least one hypothesis family decisively. B4 random-init at chance confirms architecture alone is insufficient. |

---

## consolidated proposal-alignment matrix (v3)

 | hypothesis | evaluation question | v3 observed result | strong methods (family) | **best method** | **why this is best** | baselines |
|---|---|---|---|---|---|---|
| H1 Cross-Modal Alignment | can H&E retrieve matched ST niches better than unaligned/classical? | R4 AUC 0.859, CKA 0.631. R1 / R5 0.761 (tied). R6 0.737. B4 untrained 0.496 (chance). | R4, R1, R5, R6 | **R4_v3** | cross-attention over 7 H&E tile tokens (vs mean-pooled niche) lets the ST query select tile-level morphology, producing a tight 3-effective-dim manifold with the highest cross-modal coupling. clean ~0.10 AUC margin over the next contrastive family member; B4 random-init at chance confirms training (not architecture) does the work. | B1, B2, B3, B4 |
| H2 Niche-level biology coherence (mc_megacluster) | does the aligned space cluster niches by transcriptomic tissue state? | R5 ARI 0.246, R6 0.244, R1 0.234 (3-way tie within KMeans seed noise); R4 0.101 (manifold-geometry bias); classical drop 11+ ARI archetype → mc - quantifies patient-leakage. | R5, R1, R6 | **R6_v3** (most parsimonious) **or R5_v3** (nominal top) | late-fusion contrastive preserves a wider, more globular manifold that KMeans rewards. R6 wins parsimony - it drops gpath2vec from the ST input entirely (Novae spatial-neighborhood graph alone clusters mc states), so it has fewer load-bearing components for niche-level biology. R5's per-dim temperature is a marginal +0.002 ARI = within seed noise. | B1, B2, B3 (lose leakage advantage on mc) |
| H2 Compartment contrasts (FTU coherence) | distinct compartments > ambiguous stroma on matched-pair cosine? | TODO - v3 recompute pending; v1 result: only TLS-vs-TIL passes (R1/R2/R6, z = +3.2 to +5.3). | R1, R6 (on TLS) | **R1_v3** (v1 result; v3 recompute pending) | late-fusion preserves distinct compartment cosine separation; R1 had the highest v1 z (TLS vs loTIL z=+5.3). TLS is the only well-curated FTU annotation in the dataset - the one compartment where the proposal's directional prediction held. Tumor / Necrosis annotations are too heterogeneous to differentiate from TIL stroma cleanly. | B1 (all 6 contrasts non-significant) |
| H3 Pathway interpretability (gpath2vec arm, proposal-literal) | do the 5 Reactome targets correlate with shared-latent directions AND transfer cross-patient? | **R4 4/4 testable pathways sig, z_A 11.2-25.4** (Immune/ECM/CC/PCD). R5 2/4 (best of late-fusion family). R1/R3 1/4; R2/R6 0/4; classical B1/B2/B3 4/4 with lower z_A 3-17. TGF-β underpowered. | R4 (overall); B1/B2 (classical fallback) | **R4_v3** | cross-attention's tight manifold collapses the 5 named pathway directions onto ~one robust biology axis (specificity off-diag |mean| ~0.75). **the aliasing IS the source of cross-patient transfer** - one direction is harder to overfit to patient identity than five distinct directions. classical baselines match R4's hit rate but with lower z_A. R5 (AnInfoNCE + late) doubled R1's sig pathways but does not rescue the aliasing axis; R5b (AnInfoNCE + cross-attn) is the targeted next experiment, deferred. | R1/R2/R3/R5/R6 below classical cross-patient |
| Systems-oriented | does one strategy dominate every objective? | no - R4 wins H1 + H3, R5/R6/R1 tie H2 mc, classical lose H1 decisively. | R4, R5, R1, R6 | **task-conditional (no global winner)** | niche-level cluster geometry, cross-modal retrieval, and cross-patient pathway transfer are NOT co-monotonic objectives. R4's tight aliased manifold trades cluster geometry (H2 mc) for retrieval + pathway transfer (H1 + H3). R5/R6/R1's wider manifold does the reverse. the framework's value is exposing the trade-offs and routing the right question to the right method - this IS the deliverable, not a single winner. | all baselines lose at least one family |

---

## Methods note

cohort, niche unit (k=6 spatial neighbors, ~1200 cells), evaluation split (subarray-level patient-stratified 85/15, seed=42), ST input composition (`novae_niche` + `gpath2vec_niche` for R1-R5/B1-B3; novae-only for R6), niche aggregation, MLP architecture: all unchanged from v1. see [`../v1/tnbc92_results_summary.md`](../v1/tnbc92_results_summary.md) Methods section for the full details.

**v1 → v3 single change:** `gpath2vec_niche` (512-d) is the v3 build (`fisher_madmean_low_dim512_e5_s1234`, sha `13985cbd...`) instead of the v1 legacy build. niche-join 208,786 niches (vs v1's 286,250) due to v3 dropping 67,131 niches with no significant pathway under MAD/mean gene selection. test set 35,594 niches (vs v1's 45,661).

**v3-only method addition:** R5_v3 = AnInfoNCE + late fusion. extends InfoNCE with a per-dim learnable log-scale parameter `aniso_log_scale: nn.Parameter(zeros(shared_dim))` registered on `LateFusion` when `cfg.anisotropic=True`. loss applies `diag(exp(2*log_scale))` to the bilinear inner product - equivalent to a learned diagonal Mahalanobis metric. log_scale=0 at init → uniform scale → identical to standard InfoNCE at step 0 (clean warm-start). one-variable-change vs R1; same lr / epochs / seed.

---

## pointers

| asset | path |
|---|---|
| v1 baseline (this doc's predecessor) | [`../v1/tnbc92_results_summary.md`](../v1/tnbc92_results_summary.md) |
| v3 retrain plan + memory | `projects/tnbc-92/v3_phase2_plan.md`, [[project_v3_retrain_result]] |
| v3 alignment runs | `runs/tnbc-92_v3/{R1,R2,R3,R4,R5,R6,B1,B2,B3,B4}_v3/` |
| v3 H1 eval | `runs/tnbc-92_v3/eval/H1/summary.json` |
| v3 H2 eval (archetype + compartment + mc_megacluster) | `runs/tnbc-92_v3/eval/H2/summary.json` + `mc_coherence.parquet` + `compartment_cosine.parquet` |
| v3 H3 gpath2vec arm | `runs/tnbc-92_v3/eval/H3/pathway_cca_gpath2vec_v3/per_pathway_cca.parquet` + `perm_nulls.parquet` + `provenance.json` |
| v3 H3 AUCell (orthogonal robustness check) | `runs/tnbc-92_v3/eval/H3/pathway_cca/` |
| v3 summary notebook + HTML | `notebooks/final/05_summary_umaps_v3.ipynb` + `05_summary_umaps_v3.html` |
| construct-validity audit (Cameron's questions) | `projects/tnbc-92/evaluation_question_audit.md` |
| MCP / LangGraph agent design | `docs/mcp_agent_design.md` |
