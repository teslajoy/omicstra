# tnbc-92 MCP agent routing matrix

**purpose**: map user questions to alignment methods, observed results, and routing decisions for the OMICSTRA MCP agent. this is not a proposal-alignment summary. it is a dispatch matrix for choosing the right method based on the scientific task.

audience: research software engineers and computational biologists implementing or consuming the MCP agent. for the proposal-aligned scientific verdict see `tnbc92_three_hypotheses.md` and `current_report.md`.

---

## method definitions

| method | definition | routing role |
|---|---|---|
| R4 | InfoNCE cross-attention. ST query attends over 7 H&E tile tokens (center spot + 6 spatial neighbors) before projection. | best for cross-modal retrieval, H1 alignment, cross-patient pathway transfer (both 5-parent and 395-DAG-node tests), and patient leakage suppression. |
| R1 | InfoNCE late-fusion MLPs with independent modality projections and contrastive alignment. | best for niche-level biology and per-pathway interpretability at coarse 5-parent resolution when specificity matters. |
| R6 | R1 ablation using Novae-only ST embeddings without gpath2vec pathway embeddings. | best for niche-level MC clustering and biology-over-patient ratio diagnostics. |
| R2 | Supervised contrastive late-fusion alignment using soft mc_weights similarity targets. | secondary option for TLS-vs-TIL compartment contrasts. |
| R3 | Barlow Twins late-fusion redundancy-reduction alignment without explicit negatives. | useful for distinct pathway directions at 5-parent resolution. weaker for cross-patient transfer. |
| B1 | Canonical Correlation Analysis baseline maximizing linear cross-modal correlation. | avoid for interpretation. weak H1 (AUC 0.541) and anti-helpful for biology-over-patient ratio (0.064 below raw_he floor 0.129). |
| B2 | Procrustes alignment baseline using orthogonal rotation between modality spaces. | useful classical comparator for pathway direction specificity. passes H3-extended despite being linear. |
| B3 | Unaligned PCA baseline with independent modality embeddings and no alignment objective. | sanity floor for no cross-modal coupling (gap -0.049 by design). |
| B4 | Random-initialized untrained MLP baseline with no training applied. | strict test of training contribution versus architecture alone (AUC 0.491, gap -0.001). |

---

## integrated routing rule (per user intent)

| user asks | route to | evidence basis | agent action |
|---|---|---|---|
| find the matching niche across modalities | R4 | Q1.1, Q1.2, Q1.3. R4 AUC 0.851, CKA 0.564, gap 0.195. | use R4 embeddings for retrieval and cross-modal matching. |
| isolate contrastive training contribution | R1 vs B4 | Q1.4. B4 chance (AUC 0.491, gap -0.001). | use B4 as strict training-control evidence. |
| cluster niches by biology | R1 or R6 | Q2.1, Q2.2. R6 MC ARI 0.251, R1 MC ARI 0.231, R1 LP 0.260. | use R1/R6. Prefer LP-aware interpretation when R4 is also being compared. |
| amplify biology and suppress patient signal | R6 or R1; R4 for leakage suppression | Q3.4 and Q3.5. R6 ratio 0.548, R1 ratio 0.517, R4 patient z lowest at 22.97. | use R6/R1 for bio-over-patient ratio. use R4 when leakage minimization is primary. |
| find pathway-similar niches across held-out patients (5-parent test) | R4 | Q3.2. R4 15/15 significant cross-patient pathway transfers. | use R4 for cross-patient pathway routing and spatial pathway maps. |
| find pathway-similar niches across held-out patients (DAG resolution) | R4 primary; B2/B1/B3 for ECM specifically | Q3.7. DAG-level: R4 78 percent sig, B2 74, B1 70, B3 64, R6 37, R1 27. R4 wins or ties every parent. | use R4 as primary route for cross-patient pathway transfer at DAG resolution. fall back to classical for ECM detection. |
| show distinct per-pathway activity (5 named parents, coarse) | R1, R3, or B2 | Q3.3. R4 off-diagonal mean 0.819, highly aliased at coarse resolution. R3 0.390, B2/B3 0.376. | avoid R4 if 5-parent pathway specificity is the primary user intent. |
| identify named clinically interpretable sub-pathway driving a niche | R4 (DAG) | Q3.8. R4 wins or ties every parent at 395-DAG-node resolution. named hits: BTLA co-inhibition, IL receptor SHC signaling, IFN-α/β regulation, integrin cell surface, ECM proteoglycans, elastic fibre, collagen biosynthesis/degradation. | use R4 z_he projection onto each DAG-node canonical direction. annotate with Reactome leaf names. |
| minimize patient leakage in interpretation | R4 | Q3.5. R4 z_he patient z 22.97, lowest in grid. B1 122.13. | use R4 and flag B1 as anti-helpful. |
| recover 9 archetype labels | diagnostic only | Q2.5. archetype ARI is patient-leakage proxy, not biology. | use for leakage diagnostics. avoid as biology routing target. |

each row maps a scientific user question to the best method route, the observed result, and how the MCP agent should respond.

---

## question-to-result routing matrix

| ID | hypothesis | user / agent question | route | observed result | interpretation | MCP routing decision | confidence |
|---|---|---|---|---|---|---|---|
| Q1.1 | H1 | retrieve matched ST niche from H&E across patients | R4 | R4 AUC 0.851. R1 0.741. B1 0.541. B3 0.443. B4 0.491. | R4 is best for cross-modal matching. | route retrieval and matching queries to R4. | high |
| Q1.2 | H1 | align H&E and ST manifolds structurally using cross-modal CKA | R4 | R4 CKA 0.564. R1 0.242. R1 rank-3 control 0.234 (gap -0.331 vs R4 full). | R4 advantage is architectural, not collapse-driven compression. | route manifold-alignment questions to R4. | high |
| Q1.3 | H1 | separate matched versus mismatched cross-modal pairs | R4 | gap: R4 0.195, R1 0.159, B2 0.156, B3 -0.049. | R4 has strongest cohort-level matched-pair separation. | use R4 for pair-separation and similarity tasks. | high |
| Q1.4 | H1 | isolate contrastive training contribution versus architecture alone | R1 vs B4 | B4 chance baseline: AUC 0.491, gap -0.001. R1 to B4 contribution: +0.25 AUC from contrastive loss. | training contributes the bulk of R1 advantage. | use B4 as strict training-control evidence. | high |
| Q2.1 | H2 | cluster niches by Wang 14-class MC megacluster using KMeans | R6 or R1 | R6 ARI 0.251. R1 0.231. R4 0.098. | late-fusion methods preserve globular niche-level MC clusters best. | route clustering-by-biology queries to R6 or R1. | high |
| Q2.2 | H2 | linearly decode MC identity from embeddings | R1; R4 acceptable | LP accuracy: R1 0.260, raw_he 0.239, B3 0.237, R6 0.235, R4 0.230. | R4 encodes biology at parity but not in KMeans-friendly geometry. | use R1 for best LP. do not call R4 biology-less. | high |
| Q2.3 | H2 | distinguish TLS from TIL-stroma compartments by matched-pair cosine | R1, R2, R6 | R1/R2/R6 support TLS vs hiTIL and TLS vs loTIL. R1 z values 3.2 and 5.3. | TLS is the compartment contrast that behaves as predicted. | route TLS-vs-TIL questions to R1/R2/R6. | mid |
| Q2.4 | H2 | distinguish tumor from TIL-stroma compartments by matched-pair cosine | no reliable route | all tested methods failed or inverted tumor contrasts. | annotation-purity and tumor-heterogeneity confound. | return caveat. do not route as supported result. | high |
| Q2.5 | H2 | recover 9 archetype labels (Wang spatial archetypes) | diagnostic only | B1 ARI 0.307 wins by patient leakage. R4 ARI 0.059 with strongest patient suppression. | archetype ARI is patient-leakage proxy, not primary biology metric (archetype is patient-level pseudobulk label, verified 30/30 subarrays). | use for leakage diagnostics. avoid as biology routing target. | high |
| Q3.1 | H3 | encode 5 named Reactome pathways above null using fit-on-all CCA (proposal-spec literal test) | all pass | all 120 cells BH-FDR less than 0.001. cumulative correlation: B2/B3 4.242, R1 4.099, R4 3.943. | fit-on-all test is saturated by 512-d projection capacity onto 1-d target. not discriminating. | do not use this alone for method selection. report as proposal-compliance check. | high |
| Q3.2 | H3 | generalize 5-pathway encoding to held-out patients (cross-patient sub-split, 11 train / 3 test) | R4 | R4 15/15 cells significant (z=3.8-5.7). B1/B2 14/15. B3 12/15. R6 11/15. R1/R2/R3 5/15. | R4 dominates cross-patient pathway transfer at coarse resolution. | route cross-patient 5-pathway queries to R4. | high |
| Q3.3 | H3 | encode 5 pathways as distinct directions rather than aliased signal (coarse 5-parent test) | R1, R3, B2 | R4 off-diagonal mean 0.819, max 0.970 highly aliased. R3 0.390, B2/B3 0.376, R1 0.481. | R4 generalizes by compressing correlated parent unions into one robust axis at coarse resolution. | use R1/R3/B2 for 5-parent pathway specificity. avoid R4 when coarse specificity is required. | high |
| Q3.4 | H3 | amplify TIME subtype biology over patient signal (H3-extended Bareche test, 1e5 perms) | R6 or R1; R4 also passes | bio/patient ratio: R6 0.548, R1 0.517, B2 0.318, R4 0.438. B1 0.064 below raw_he floor 0.129 (anti-helpful). raw_st 0.225. | B1 is anti-helpful. Procrustes passes despite linear, falsifying "linear methods fail H3". | route biology-over-patient queries to R6/R1. flag B1 as avoid. | high |
| Q3.5 | H3 | minimize patient leakage in z_he specifically | R4 | R4 patient z 22.97, lowest in grid. B1 122.13, anti-helpful (3.3x amplification). | R4 suppresses patient signal best across all runs. | route leakage-minimization questions to R4. | mid |
| Q3.6 | H3 | predict pathway activity from H&E alone, spatially within new subarrays | R4 tested | R4 spatial Spearman 0.34-0.56 across 5 pathways (ECM 0.564, Immune 0.509, Cell Cycle 0.447, TGF 0.355, PCD 0.341). other routes pending. caveats: spatial autocorrelation and cell-composition confound not controlled for. | promising but only R4 tested. partial-out queued as priority 2. | route to R4 but label as partial evidence. | mid |
| Q3.7 | H3 | resolve pathway interpretability at sub-parent DAG depth (commit 3.6, 395 DAG nodes) | R4 primary; B2/B1/B3 secondary at fine resolution | 9480-cell grid (8 runs x 3 views x 395 DAG nodes). option-A sig cells: R4 934 (78 percent), B2 879 (74), B1 832 (70), B3 763 (64), R6 437 (37), R1 321 (27), R2 273 (23), R3 218 (18). R4 wins or ties every parent. percent-sig per parent for R4: Cell_Cycle 69, ECM 93, Immune 83, PCD 75, TGF 81. | R4 still wins at DAG resolution but classical baselines competitive. R4 aliasing is a 5-parent coarse-aggregation artifact, not fundamental. | route DAG-level pathway queries to R4 primary; fall back to B1/B2/B3 for ECM detection specifically (95-98 percent sig). | high |
| Q3.8 | H3 | identify named clinically interpretable sub-pathway driving a niche (commit 3.6 named hits) | R4 (DAG canonical directions) | named top-14 cross-patient hits at z_he: BTLA co-inhibition (z=6.92), IL receptor SHC signaling (6.80), integrin cell surface (6.21), ECM proteoglycans (5.73), IFN-α/β regulation (5.71), ECM degradation (5.47), elastic fibre formation (5.33), collagen chain trimerization (5.30), collagen biosynthesis (5.28), collagen degradation (5.22), FLT3 signaling (5.15), TNFs bind receptors (5.13), calcineurin activates NFAT (5.10). | R4 distinguishes immune checkpoint and cytokine signaling from ECM structural and remodeling biology at DAG depth. clinically meaningful TNBC dimensions. | route sub-pathway naming queries to R4. project z_he onto each DAG-node canonical direction; annotate top hits with Reactome leaf names from `dag_metadata.parquet`. | high |

each row maps an operational question to the method route, observed numerical evidence, and an MCP agent action. confidence levels reflect statistical strength (high = BH-FDR less than 0.05 across multiple metrics; mid = single-metric or partial evidence; pending = not yet run).

---

## hypothesis matrix (proposal-anchored, for reference)

this matrix is included for cross-reference. the proposal-aligned scientific verdict is documented in detail in `tnbc92_three_hypotheses.md`.

| hypothesis | evaluation question | yes/no outcome definition | scientific claim | primary metrics | strong methods | baselines |
|---|---|---|---|---|---|---|
| H1 cross-modal alignment | can H&E embeddings retrieve matched ST niches more accurately than unaligned or classical alignment methods? | yes if aligned methods outperform unaligned and classical baselines across retrieval metrics. | contrastive learning improves cross-modal retrieval and representation alignment relative to unaligned and classical baselines. | Recall@K, MRR, median rank | R4, R1 | B1, B2, B3, B4 |
| H1 cross-modal alignment | do aligned embeddings exhibit stronger cross-modal structural agreement than baseline methods? | yes if aligned embeddings produce stronger manifold agreement and matched-vs-mismatched separation. | alignment produces stronger shared latent geometry across modalities. | CKA, alignment gap, AUC, cosine separation | R4 | B1, B3 |
| H1 cross-modal alignment | does training improve alignment relative to untrained or purely linear projections? | yes if trained models outperform random-init and linear baselines. | contrastive objectives contribute beyond architecture alone. | delta Recall@K, delta CKA | R1, R4 | B4 |
| H2 structural coherence | do niches cluster by biology at niche-level (Wang per-spot MC labels)? | yes if niche-level MC ARI clusters track biology, not patient. | the aligned latent space preserves niche-level biological organization. | KMeans MC ARI, linear probe MC accuracy | R1, R6 | raw_he reference |
| H2 structural coherence | do morphologically distinct compartments exhibit higher matched-pair cosine similarity than ambiguous stromal compartments? | yes if tumor, TLS, and necrosis regions show stronger matched-pair similarity than ambiguous stromal regions. | alignment strength varies across TME compartments according to morphological distinctness. | matched-pair cosine similarity by compartment (Welch z) | R1, R2, R6 (TLS only) | B1 |
| H2 structural coherence | does the aligned latent space encode biology more strongly than patient identity? | yes if biological structure is retained while patient leakage is reduced. | shared embeddings preferentially preserve biological organization over patient-specific signal. | patient-ID decoding accuracy, bio/patient ratio | R6, R4 | B1 (anti-helpful) |
| H3 pathway interpretability | do Reactome pathway embeddings correlate with latent-space directions? | yes if pathway embeddings align with distinct latent-space directions above null expectation. | shared latent geometry preserves pathway-associated biological signal. | univariate CCA (per-pathway), permutation null + BH-FDR | all 8 runs pass at fit-on-all | n/a (saturated) |
| H3 pathway interpretability | can pathway-associated latent structure transfer to held-out patients? | yes if pathway-associated structure generalizes across patient splits. | pathway-associated structure generalizes across patients. | held-out canonical correlation, cross-patient z-score | R4 (15/15 at 5-parent, 78 percent at 395-DAG) | B1, B3 (5/15 to 14/15 at 5-parent) |
| H3 pathway interpretability | are pathway-associated latent directions separable and pathway-specific? | yes if pathway-associated directions remain distinct rather than aliased. | pathway-associated directions remain biologically distinct at appropriate resolution. | directional orthogonality (5x5 specificity matrix), DAG-level named sub-pathway specificity | R1, R3, B2 (at 5-parent coarse); R4 (at 395-DAG fine resolution) | B1 |

---

## artifact locations referenced

| artifact | location | role |
|---|---|---|
| H1 results | `runs/tnbc-92/eval/H1/summary.json` | Q1.1-Q1.4 |
| H1 rank-matched CKA control | `runs/tnbc-92/eval/compare/rank_matched_cka.parquet` | Q1.2 |
| H2 MC coherence | `runs/tnbc-92/eval/H2/mc_coherence.parquet` and `mc_diagnostics/` | Q2.1, Q2.2 |
| H2 named compartment contrasts | `runs/tnbc-92/eval/H2/named_contrasts.parquet` | Q2.3, Q2.4 |
| H2 archetype confound diagnostic | `runs/tnbc-92/eval/H2/summary.json` | Q2.5 |
| H3 5-pathway CCA | `runs/tnbc-92/eval/H3/pathway_cca/per_pathway_cca.parquet` and `perm_nulls.parquet` | Q3.1, Q3.2, Q3.3 |
| H3 specificity matrix | `runs/tnbc-92/eval/H3/pathway_cca/specificity_matrix.parquet` | Q3.3 |
| H3-extended Bareche labels | `runs/tnbc-92/{run}/eval/biology.parquet` | Q3.4, Q3.5 |
| H3 spatial pathway maps | `runs/tnbc-92/eval/figures/pathway_spatial_maps/` and `pathway_spatial_correlations.parquet` | Q3.6 |
| H3 DAG decomposition | `runs/tnbc-92/eval/H3/pathway_cca/dag_full/` (`dag_metadata.parquet`, `per_node_aucell.parquet`, `per_node_cca.parquet`) | Q3.7, Q3.8 |
| B4 random-init baseline | `runs/tnbc-92/B4/metrics_h1_raw.json` | Q1.4 |