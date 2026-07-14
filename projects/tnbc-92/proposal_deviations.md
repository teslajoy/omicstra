# proposal deviations and extensions

written 2026-05-11. defines what was executed against the original ResearchHub proposal (`docs/papers/sanati_2025_researchhub.pdf`) vs what diverged, was added, or is still pending. locks the framing before any further compute or writeup.

source for all proposal wording: pp.5 (H1/H2/H3) + pp.9 (planned analyses) of the proposal PDF.

---

## per-hypothesis cross-reference

### H1 — cross-modal alignment

> proposal: "contrastive learning will improve cross-modal retrieval and representation alignment relative to unaligned baselines (CCA, late fusion concatenation, and raw unaligned concatenation), as measured by retrieval metrics (Recall@K, MRR, median rank), alignment separability (alignment gap, AUC), and representation structure (CKA). cross-modal similarity improvements are expected within an approximate range of 0.65-0.75."

| proposal commitment | what was executed | status |
|---|---|---|
| baseline 1: CCA | B1 (closed-form CCA on train, applied cohort-wide) | done, terminology matches |
| baseline 2: late fusion concatenation | NOT implemented as a non-contrastive concat baseline. brief uses "late fusion" to mean *contrastive* late fusion (R1, R2, R3) | terminology slippage — rename in brief: "late-fusion contrastive" for R1/R2/R3; "late-fusion concat (non-contrastive)" gap noted |
| baseline 3: raw unaligned concatenation | B3 (per-modality PCA to 512, L2-norm, no alignment) is the closest analog | acceptable substitute; document the L2-norm + PCA-512 pre-step in brief |
| metric: R@K | done for R@1, R@5, R@10 (in `runs/tnbc-92/eval/H1/summary.json`) | done, in brief |
| metric: MRR | done | done, in brief |
| metric: **median rank** | computed and stored in `runs/tnbc-92/eval/H1/summary.json` (R1=8827, R2=10153, R3=10267, R4=4244, R6, B1, B2, B3) | **not in brief — add to H1 table (text-only fix, no recompute)** |
| metric: alignment gap | done | done, in brief |
| metric: AUC | done | done, in brief |
| metric: CKA before/after | done | done, in brief |
| prediction: 0.65-0.75 similarity range | never reported as a range comparison; R4 AUC=0.851 exceeds upper bound, R1=0.741 in middle | **add explicit range comparison to brief (text-only)** |
| encoder sensitivity: UNI2 vs Virchow2 | Virchow2 chosen as primary via encoder_qc_comparison_2026-04-09; UNI2 swap path documented but not run on H1 grid | acceptable — proposal allowed pluggable encoder; document the decision rationale |
| additional: 7-run grid (R1-R4, R6, B1-B3) with patient-held-out cross-subarray retrieval | extends proposal (proposal asked for contrastive vs 3 baselines; we ran 4 contrastive losses + 3 classical + 1 ablation) | extension, not deviation |
| additional: step 3 stress tests (rank-matched CKA, paired-perm asym, per-run-vs-raw) | extension after R4 H1 advantage emerged; tests collapse hypothesis and disentanglement claim | extension, not deviation |

### H2 — structural coherence

> proposal part A: "the nine spatial archetypes will cluster more coherently in the aligned latent space compared to individual modality spaces, as measured by adjusted Rand index and silhouette score."
> proposal part B: "alignment quality will vary across TME compartments: morphologically distinct compartments (ex. tumor, TLS, necrosis) will show higher matched-pair cosine similarity than morphologically ambiguous compartments (ex. high-TIL vs. low-TIL stroma)."

| proposal commitment | what was executed | status |
|---|---|---|
| part A: ARI on 9 archetypes, aligned vs single-modality spaces | done across R1/R4/baselines; ARI flagged confounded by patient (CCA wins ARI but amplifies patient z 3.3×). **2026-05-11 audit: deeper finding — `archetype` is sourced from `Spatial archetypes_defined_on_ST_global_pseudobulk` in Wang's Clinical.RDS, which is a PATIENT-level pseudobulk label (verified: 30/30 sampled subarrays have `archetype_unique_within_subarray = 1`). The ARI test mechanically tests patient-clustering at niche resolution, not biological-coherence.** The proper niche-level biological label is `mc_labels.megacluster` (Wang's per-spot 14-class NMF discrete labels), present in `data/embeddings/biological_signals/mc_labels.tsv` but not joined into niche-join parquets and not consumed by `eval.py`. | **commit 2.5: rebuild niche-join parquets with mc_megacluster joined, add `eval_h2_mc_coherence(run)` to eval.py, report side-by-side with archetype.** part A's "ARI confound" framing in the brief is correct but understates the resolution-level deviation; the niche-level MC test is the proposal's intent. |
| part A: **silhouette score** | dropped from brief | **add silhouette alongside ARI (text-only; report B2≡B3 rotation-invariance footnote)** |
| part B: per-compartment cosine, directional prediction tumor/TLS/necrosis > high/low-TIL stroma | parquet computed at `runs/tnbc-92/eval/H2/compartment_cosine.parquet` (96 rows = 12 compartments × 8 runs; Tumor, Necrosis, Lymphoid nodule, High TIL stroma, Low TIL stroma all present); named-contrast writeup pending | **pending writeup — compute 6 named contrasts (tumor vs hiTIL, tumor vs loTIL, Lymphoid nodule vs hiTIL, Lymphoid nodule vs loTIL, necrosis vs hiTIL, necrosis vs loTIL) with **Welch z-test** (unpooled variance; compartments have different cell counts and signal regimes, pooled would underestimate SE). pre-register direction = delta > 0. report per-run.** |
| additional: bio-vs-patient permutation z-test (Bareche TIME / MC_global / MC_tumor labels, 1e5 perms, matched-null) | done for R1/R4/R6/B1; framework ported from gpath2vec | extension — frame as H2 robustness analysis, NOT as replacement for ARI |

### H3 — pathway interpretability

> proposal: "Reactome pathway embeddings computed via gpath2vec for five cancer-relevant pathways (TGF-β Signaling, Immune System, Extracellular Matrix Organization, Cell Cycle, Programmed Cell Death) will correlate with specific directions in the shared latent space via canonical correlation analysis, indicating preservation of interpretable biological signal."

| proposal commitment | what was executed | status |
|---|---|---|
| **per-pathway CCA between gpath2vec embeddings and shared latent, 5 named pathways** | **PARTIAL.** `eval.py:eval_h3_per_run` does run CCA via `cca_top_k(z_mean, gpath2vec_512d)` — a global CCA between full shared and full gpath2vec spaces. saturated >0.77 across runs (both sides are 512d dense). does NOT isolate the 5 named Reactome pathways. brief's "H3" prose currently emphasizes bio-vs-patient z-test on Bareche labels rather than this CCA. | **refinement gap — per-pathway CCA needed. existing global CCA stays as "global pathway-space coherence" sub-readout (different question, kept honest).** |
| 5 named pathway IDs (verified in `projects/tnbc-92/program.md` lines 91-96) | TGF-β R-HSA-170834, Immune R-HSA-168256, ECM R-HSA-1474244, Cell Cycle R-HSA-1640170, PCD R-HSA-5357801 | locked |
| spatial maps of pathway-morphology associations | not done | follows from CCA work — flag for after H3 CCA lands |
| additional: bio-vs-patient z-test as currently labeled "H3" | done at 1e5 perms for R1/R4/R6/B1 | extension — relabel as "H3-extended: biology vs patient disentanglement" or move into H2 part A robustness; the proposal commitment H3 (5-pathway CCA) is unmet |

---

## scope rules (locked before H3 compute)

**run set for H3 CCA**: all 8 runs that have `runs/tnbc-92/{run}/embeddings_test.parquet` — R1, R2, R3, R4, R6, B1, B2, B3. Classical baselines (B1, B2, B3) included as the falsifier floor: the proposal's interpretability claim is that *alignment* produces directions correlated with pathways, so the unaligned-baseline comparison is required to make that falsifiable. Brief must mark classical (B1/B2/B3) vs contrastive (R1/R2/R3/R4/R6) explicitly in the H3 table.

**view set per run**: z_he, z_st, z_mean. all three because (a) the proposal does not specify which side of the shared space the CCA targets and (b) prior H3-extended analyses used z_he as primary, z_mean as joint readout — keep the convention.

**pathway-signal source**: AUCell scores per niche, restricted to the 5 named pathways' Reactome gene sets, as the per-niche per-pathway signal. AUCell scores already exist for these 5 pathways from the niche pathway pipeline. Use these as the "pathway embedding" matrix (n_niches × 5) for CCA. Falls back to gpath2vec subtree-restricted embeddings if AUCell coverage is insufficient — but AUCell is cleaner and matches the niche-level pathway scoring already in pipeline.

**CCA dimensionality**: shared latent is 512-d per run; pathway side is 1-d (single AUCell score per niche per pathway, computed once over the 5 named pathways). because the pathway side is 1-d, univariate CCA reduces to: fit a single canonical direction on shared (left singular vector of `Cov(z, score)`), project, then `pearsonr(z_proj, score)`. `scipy.stats.pearsonr` is the cleanest implementation — the existing `cca_top_k(... k=1)` machinery is overkill for the per-pathway case. Use pearsonr directly.

**null**: permutation null on pathway scores across niches (preserves niche structure, shuffles which niche gets which AUCell score). 1000 perms per (run, pathway, view) — gives p-resolution to 1e-3 which is enough for the 120-cell test grid with BH-FDR correction. Cheap since the per-perm computation is one Pearson correlation, not a 512×512 CCA refit.

**test grid size**: 8 runs × 5 pathways × 3 views = 120 tests. BH-FDR across the 120 tests.

**pre-registered prediction grid** (locked before running, makes the test falsifiable):
- R4 (cross-attention, tight manifold): expected to lead on **Immune** and **TGF-β** — immune-rich and stroma-rich niches both have strong H&E morphology correlates that cross-attention can latch onto.
- B1 (CCA closed-form): expected to lead on **saturated cumulative correlation across all 5 pathways** — it literally optimizes correlation between the two spaces. The question is whether that saturated correlation is *meaningful* per-pathway or just dense linear projection of a high-d space.
- R3 (Barlow Twins, redundancy reduction): expected to be **middling** on all 5 — redundancy reduction does not selectively encode any particular pathway direction.
- R1 (InfoNCE late-fusion): expected to be **competitive across all 5 with no single dominant pathway** — InfoNCE preserves dimensional richness without bias toward specific biological programs.
- B3 (unaligned PCA): expected to be the **floor on contrastive pathways but possibly competitive on TGF-β** if stromal morphology survives the PCA truncation.

if results diverge from these predictions, that itself is mechanism information worth capturing.

---

## H3 gpath2vec arm — closing the documented AUCell-substitution gap (added 2026-05-17)

the scope rules above chose AUCell as the per-niche pathway signal and explicitly named
"gpath2vec subtree-restricted embeddings" as the documented fallback. gpath2vec arm A has
since been produced (`data/embeddings/biological_signals/gpath2vec_output/full_cohort_tf_low_arm_A/`,
286,233 niche cluster vectors + 680 R-HSA pathway vectors, 512-d, metapath2vec 5 epochs).
this arm runs the proposal-literal method — gpath2vec pathway embeddings × shared latent
via CCA — that AUCell substituted for. it is therefore the **first actual test of the
H3 hypothesis as written**, not a refinement of the AUCell readout.

**honest framing**: a null result ("gpath2vec rollup does not beat AUCell") is a live,
publishable outcome the proposal already anticipates ("negative or partial results
establish the noise floor and signal limits"). the comparison is the deliverable; the
direction of the result is not pre-committed.

**method (frozen)**: per-niche gpath2vec cluster vector · 5 parent vectors (cosine) →
niche × 5 score matrix, run through the *identical* option-A protocol (patient-held-out
split + permutation null + BH-FDR) AUCell's `per_node_cca` used. each parent vector =
L2-normalized mean of its embedded descendants.

**TGF-β stId — pinned, not chosen**: `program.md` (locked, do-not-edit) commits
TGF-β = `R-HSA-170834` ("Signaling by TGF-beta Receptor Complex"). the proposal abstract
says only "TGF-β Signaling", which reads family-level (`R-HSA-9006936`, "Signaling by
TGFB family members"). H3 is scoped to `R-HSA-170834` per the locked spec and for
cross-method comparability — the existing AUCell `per_node_cca` rooted TGF-β at
`R-HSA-170834`, so the gpath2vec arm must use the same root or the head-to-head confounds
method with pathway scope. **TGF-β rollup = parent + 1 embedded descendant = 2 vectors**;
flagged underpowered. its CCA is reported for completeness but **excluded from any
headline H3 claim**. a family-level (`R-HSA-9006936`) supplementary may be added later,
explicitly marked non-comparable / descriptive only.

**395-vs-396 reconciliation (written down so it stops recurring)**: `dag_metadata.parquet`
has 396 nodes under the 5 parents; AUCell's `per_node_cca.parquet` tested 395. the single
dropped node is **`R-HSA-844623`** (an Immune_System node decoupler dropped on gene-set
size, not the root). the frozen hierarchy for both methods is
`dag_metadata.parquet ∩ per_node_cca.node_id` = **395 nodes**. embedded-node counts on
this frozen set: Immune 74/210, ECM 14/20, Cell Cycle 14/120, PCD 7/38, TGF-β 2/7.

**path note**: the pointer table below lists `full_cohort_tf_low/`; this arm uses
`full_cohort_tf_low_arm_A/`. do not conflate the two gpath2vec outputs.

---

## priorities (post-deviations-log)

| priority | scope | new compute | est. time |
|---|---|---|---|
| **0** | **this file (deviations log)** | none | **done** |
| 1 | H1 cleanup in brief: median rank table, AUC 0.65-0.75 range comparison, baseline terminology rename | none (data in `runs/tnbc-92/eval/H1/summary.json`) | 30 min |
| 2 | H2 part B writeup: 6 named-contrast cosines (Welch z-test) + silhouette + directional prediction | none (data in `runs/tnbc-92/eval/H2/compartment_cosine.parquet`) | 1-2 hr |
| **3** | **H3 5-pathway CCA: the proposal commitment** — 120 univariate-CCA-via-pearsonr tests, permutation null, BH-FDR, result table + spatial maps for top pairs | **yes (pearsonr + perms, light compute)** | **3-5 hr** |
| 4 | fold all of the above into the brief (rename current H3 to "H3-extended: bio-vs-patient disentanglement"; keep `cca_shared_vs_gpath2vec` as "global pathway-space coherence" sub-readout; add new primary H3 section for per-pathway CCA) | none | 1 hr |
| 5 | regenerate PDFs | none | 15 min |

priority 3 is the load-bearing item. priorities 1, 2, 4, 5 are all text or composition; no further compute beyond existing artifacts.

### seven-commit execution cadence (locked, includes 2.5)

each commit independently revertable, each scientifically meaningful standalone:

1. **commit 1**: deviations log + H1 text fixes (median_rank, AUC range, late-fusion terminology rename) ✅ landed
2. **commit 2**: H2 part B writeup with 6 named contrasts, Welch z-test, falsifier grid + silhouette addition ✅ landed (prediction failed broadly — only TLS-vs-TIL contrasts pass in R1/R2/R6)
3. **commit 2.5** (inserted 2026-05-11 after audit): H2 niche-level MC coherence. rebuild niche-join parquets with `mc_megacluster` column joined, add `eval_h2_mc_coherence(run)` to `eval.py` (KMeans k=14 → ARI + silhouette vs mc_labels.megacluster), 8 runs × 3 views = 24 cells, side-by-side comparison with archetype, integrate into report. **WHY HERE (not deferred)**: result changes how H2 reads, which changes H3 pre-registration; parquet rebuild path is shared with commit 3's AUCell join; mechanical test, low risk, ~1-2hr.
4. **commit 3**: `scripts/eval_h3_pathway_cca.py` + `runs/tnbc-92/eval/H3/pathway_cca/per_pathway_cca.parquet` (observed correlations only, no perms yet)
5. **commit 4**: perm nulls + BH-FDR table (`fdr_table.parquet`); separate commit so re-running perms doesn't rebuild observed
6. **commit 5**: spatial maps for top-canonical-direction projections (`runs/tnbc-92/eval/figures/pathway_spatial/`)
7. **commit 6**: H3 brief integration (rename current section, add new primary section, integrate cross-references)

---

## preservation rules (do not modify in next-session work)

- `projects/tnbc-92/current_report.md` step 3 section — keep additive; new sections append below
- `docs/tnbc92_three_hypotheses.pdf` (27pp) — regenerate after priorities 1-4 land
- `docs/tnbc92_summary.pdf` (12pp) — regenerate after priorities 1-4 land
- all existing `runs/tnbc-92/{run}/eval/` parquets and figures — read-only
- `program.md` — locks the 5 pathway IDs; do not edit

new artifacts go to:
- `runs/tnbc-92/eval/H3/pathway_cca/` — new CCA result parquets, perm nulls, FDR-corrected table
- `runs/tnbc-92/eval/figures/` — pathway-morphology spatial maps
- new memory files for findings
- additive sections in `current_report.md` and the brief

---

## pointers

| asset | path |
|---|---|
| proposal PDF | `docs/papers/sanati_2025_researchhub.pdf` |
| H1 metrics (median rank lives here) | `runs/tnbc-92/eval/H1/summary.json` |
| H2 part B raw | `runs/tnbc-92/eval/H2/compartment_cosine.parquet` |
| H3 gpath2vec niche embeddings | `data/embeddings/biological_signals/gpath2vec_output/full_cohort_tf_low/cluster_embeddings.parquet` |
| H3 AUCell pathway scores | `notebooks/embeddings/niche_pathway_analysis_full.ipynb` outputs (verify path on use) |
| Reactome hierarchy | `knowledge/reactome/events_hierarchy_9606.json`, `ReactomePathwaysRelation.txt`, `ReactomePathways.gmt.zip` |
| 5 pathway IDs | TGF-β R-HSA-170834, Immune R-HSA-168256, ECM R-HSA-1474244, Cell Cycle R-HSA-1640170, PCD R-HSA-5357801 (program.md lines 91-96) |
| run embeddings (all 8) | `runs/tnbc-92/{R1,R2,R3,R4,R6,B1,B2,B3}/embeddings_test.parquet` |

---

## framing note for downstream writeup

the underlying experimental arc is sound. the deviation is narrow and bounded: H1/H2 are mostly executed correctly with text-level fixes; H3 has a substantive gap because the original CCA-on-5-pathways operationalization was replaced (intentionally or not) with a different and also-valid bio-vs-patient disentanglement test. closing the H3 gap with the proposal-correct CCA is ~3-5 hours of focused work. once it lands the brief becomes honest end-to-end against the original commitments.