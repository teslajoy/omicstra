# omicstra MCP / LangGraph agent design

written 2026-05-26. audience: AI/ML systems + biology validation. scope: the LangGraph + MCP layer that wraps the existing tnbc-92 scripts (`scripts/build_niche_join.py`, `align.py`, `align_classical.py`, `eval.py`, `eval_h3_pathway_cca_gpath2vec_v2.py`) into agentic tools, grounded in the v3 evidence base. companion docs: `projects/tnbc-92/evaluation_question_audit.md` (load-bearing), `docs/tnbc92_routing_matrix.md`, `docs/tnbc92_results_summary_v2.md`, `projects/tnbc-92/program.md` (hard constraints).

`src/` is empty at time of writing (verified). this doc is the spec the first commit lands against.

---

## A. node spec

per `CLAUDE.md` agent constraints. each node is a LangGraph StateGraph node; tools are MCP-exposed. node-to-tool wraps are 1:1 unless noted.

### `orchestrator` (Sonnet 4.6)
- **purpose**: route a user / caller question to the right hypothesis family + method route via the navigation table (§D), then dispatch.
- **inputs**: natural-language question, optional `project_id` (`tnbc-92` default), optional `view` override (`z_he|z_st|z_mean`).
- **outputs**: routing decision JSON `{hypothesis, method_route, view, construct_guards[], next_node}`.
- **constraints**: cannot execute modality-specific steps (`CLAUDE.md`); cannot modify eval metrics or baselines; must surface `construct_guards` to downstream nodes verbatim. no within-subarray retrieval routing (`program.md` rule 4).

### `he_agent`
- **purpose**: produce H&E niche embeddings (Virchow2 1280-d primary; per-tile `virchow2_cell_tokens` for cross-attention).
- **inputs**: subarray list, niche-key list.
- **outputs**: `virchow2_niche` (1280-d) + `virchow2_cell_tokens` (7, 1280) parquet rows.
- **constraints**: virchow2 stays RAW - no Reinhard/Macenko/stain norm, no z-score (`program.md` hard constraint 2); cannot perform alignment or eval (`CLAUDE.md`).

### `st_agent`
- **purpose**: produce ST niche features (`novae_niche` + `gpath2vec_niche`) and the supervision-only `mc_weights_niche` bundle slot.
- **inputs**: subarray list, niche-key list, gpath2vec build path (defaults to v3 sha-locked).
- **outputs**: niche-join parquet rows; `mc_weights_niche` returned in a separate bundle slot, NEVER in `st_features`.
- **constraints**: novae z-score is per-subarray AFTER niche aggregation; `mc_weights` stays supervision-only (`program.md` hard constraint 3 - circular dep ban); v3 gpath2vec build is sha-locked to `13985cbd...` per `data/embeddings/niches_v3/manifest.json`, do not silently swap.

### `alignment_agent`
- **purpose**: fit / load one of the 10 alignment routes on the v3 niche-join: 6 contrastive (R1 InfoNCE+late, R2 SupCon+late, R3 Barlow+late, R4 InfoNCE+cross-attn, **R5 AnInfoNCE+late**, R6 InfoNCE+late+novae-only) + 3 classical (B1 CCA, B2 Procrustes, B3 unaligned PCA) + 1 random-init strict control (B4).
- **inputs**: `run_id`, `niches_dir` (`data/embeddings/niches_v3/`), `loss`, `fusion`, `st_features`, `anisotropic` (R5 only), `seed=42`.
- **outputs**: `runs/tnbc-92_v3/{run_id}/{embeddings_test.parquet, run_config.json, split.json, run.log, checkpoint.pt}`.
- **constraints**: consumes embeddings only - cannot modify raw data (`CLAUDE.md`); split.json is shared across all 10 runs from one source; R4 must persist `attention_weights` column; R5 must persist learned `aniso_log_scale` in checkpoint for downstream interpretability; hard-asserts `mc_weights ∉ st_features` (`program.md` 3); determinism re-run check required.

### `eval_agent`
- **purpose**: compute H1 / H2 / H3 metrics with paired construct-validity diagnostics.
- **inputs**: `runs_dir`, `runs[]`, `hypothesis ∈ {H1, H2, H3}`, optional `pathway_arm ∈ {aucell, gpath2vec_v3}`.
- **outputs**: `runs/tnbc-92_v3/eval/{H1,H2,H3}/...` parquets + `summary.json`; every numeric paired with 95% CI from patient-level bootstrap.
- **constraints**: computes metrics only - cannot propose configs (`CLAUDE.md`); H2 ARI on `mc_megacluster` not `archetype` (audit finding 1); H1 cross-subarray patient-held-out only (`program.md` 4); H3 `view_clean=False` for `z_st`/`z_mean` on v3-trained-on-v3 runs (circularity, see §E); BH-FDR per declared family.

### `karpathy_loop` (Opus 4.6)
- **purpose**: outer optimization. propose `alignment_config.json` edits from the search space, run the pipeline, score on the fixed scalar metric (cross-subarray R@1 per `program.md`), keep or discard.
- **inputs**: current `winner.json`, `program.md` search space, eval summary from last run.
- **outputs**: `runs/tnbc-92_v3/{run_id}/proposal.json`, decision log entry, updated `winner.json` if all 3 baselines beaten.
- **constraints**: cannot modify eval metrics or baselines mid-loop (`CLAUDE.md`); cannot promote a run to `winner.json` that does not beat random + spatial-NN + unaligned concat on R@1 (`CLAUDE.md` eval baselines); debug runs (`config/debug.json`) never candidates.

### synthesis layer (Opus 4.6, lives inside `karpathy_loop` budget)
- **purpose**: read LangSmith trace, write the manuscript-target table cell (the cells produced for `docs/tnbc92_results_summary_v2.md`), draft a notebook (`notebooks/final/`).
- **constraints**: any prose number must be derivable from a parquet cell above it (per v3 retrain plan code-review gate).

---

## B. run order (cleanup -> EDA gate -> modality -> alignment -> eval -> synthesis)

```
                                  +--------------------+
caller question -> orchestrator ->| routing decision   |
                                  +--------------------+
                                            |
                  (gate)  cleanup/EDA  <----+
                          - eda_summary.json exists + passes
                          - v3 niche-join manifest.json present + sha-matches
                          - 5 EDA chains all PROCEED (data quality / niche unit / Virchow2 / Novae / gpath2vec)
                                            |
              +-----------------------+-----+-----+----------------------+
              v                                                          v
        he_agent (Virchow2 niche)                              st_agent (Novae + gpath2vec_v3)
              \                                                          /
               +----------------> niche_join writer ---------------------+
                                  (build_niche_join.py)
                                            |
                                  +--------------------+
                                  | alignment_agent    |  (one route per call)
                                  | R1/R2/R3/R4/R5/R6  |  -> align.py
                                  | B1/B2/B3           |  -> align_classical.py
                                  | B4 strict control  |  -> align_b4.py
                                  +--------------------+
                                            |
                                  +--------------------+
                                  | eval_agent         |
                                  | H1: eval.py        |
                                  | H2: eval.py        |
                                  | H3: eval.py +      |
                                  |     eval_h3_pathway_cca_gpath2vec_v2.py
                                  +--------------------+
                                            |
                          karpathy_loop  <--+  proposes next config OR
                          synthesis layer <-+  writes verdict + notebook
```

ordering rules:
1. **EDA gate is blocking**. no modality / alignment / eval call proceeds while `projects/{project_id}/eda_summary.json` is absent or stale (`CLAUDE.md` critical constraint 1, 2).
2. **modality agents fan out in parallel** but write to disjoint paths; the niche-join writer is the join point.
3. **alignment is one route per invocation**. orchestrator chooses the route via §D; karpathy_loop iterates routes.
4. **eval is hypothesis-scoped**. each H1/H2/H3 call writes its own parquet + bootstraps. v3 pathway H3 has two arms (AUCell + gpath2vec_v3); both run, results live side-by-side under `runs/tnbc-92_v3/eval/H3/{pathway_cca, pathway_cca_gpath2vec_v3}/`.

---

## D. routing rule (centerpiece) - which method the agent picks per question

mirrors the user-facing routing matrix in `docs/v2/tnbc92_routing_matrix.md` (style adapted from `docs/v1/tnbc92_results_summary.md` "task-conditional routing rule" table). every row is a routing question the orchestrator dispatches on. construct-validity guards the agent must enforce live in §E.

| if the question is ... | route to ... | v3 evidence |
|---|---|---|
| cross-modal retrieval (find matched ST niche from H&E) | **R4_v3** | H1 AUC **0.859**, CKA 0.631, alignment gap 0.233; B4 random-init = 0.496 (chance) confirms training does the work; rank-matched control rejects collapse-driven explanation |
| group niches by recurring tumor-microenvironment state (immune-rich / stromal / necrotic / TLS ...) | **R5_v3 / R6_v3 / R1_v3** (3-way tie within KMeans seed noise) | H2 KMeans on `mc_megacluster` (Wang per-spot 14-class NMF, niche-level): R5 ARI **0.246**, R6 **0.244**, R1 **0.234** (gaps within ~0.01 = seed noise). alignment provides 5x lift over raw ST (0.046) |
| find pathway-similar niches across new patients (proposal-literal cross-patient H3) | **R4_v3** | H3 gpath2vec arm option A (cross-patient, BH-FDR<0.05), z_he: R4 **4/4 testable pathways sig**, z_A 11.2-25.4 (Immune 25.4, ECM 16.0, CC 16.5, PCD 11.2); R5_v3 **2/4** (best of late-fusion family); R1/R3 1/4; R2/R6 0/4 |
| detect fine-grained sub-pathway biology, esp. dense linear ECM | **B1_v3 / B2_v3** (preferred), **R4_v3** (competitive) | H3 gpath2vec arm: B1/B2 4/4 with ECM z_A 9.2-10.4; R4 wins overall but classical is the right tool for dense linear gene-coexpression structure |
| FTU-coherence test (does the aligned space organize by functional tissue units?) | **R1_v3 / R6_v3** on H2 Part B (TLS-vs-TIL contrasts) | TLS is the one FTU-like compartment where the proposal's directional prediction held; tumor/necrosis contrasts invert due to annotation-vocabulary confound (large heterogeneous classes), not biology absence |
| amplify biology, suppress patient identity | **R4_v3** (lowest patient z) for leakage minimization; **R6_v3 / R1_v3** for biology/patient ratio | (v3 ratios pending recompute; v1: R4 patient z 22.97 lowest in grid, R6 ratio 0.548, R1 0.517) |
| avoid for cohort-level biology interpretation | **B1_v3 (CCA)** | CCA amplifies patient identity by construction (v1: patient z 122.13 vs raw 37.25, bio/patient ratio 0.064 below raw-modality floor 0.129 - actively anti-helpful) |
| isolate alignment from architecture (control) | **B4_v3** (random-init untrained MLP, R4-architecture-equivalent shape) | v3 AUC **0.496** ≈ chance; CKA-after 0.120 ≈ CKA-before 0.115. confirms architecture alone contributes none of H1 gain |
| novelty floor / no-alignment sanity (any aligned run must beat this) | **B3_v3** (unaligned PCA, no cross-modal step) | v3 AUC **0.441** (sub-chance), alignment gap **-0.053** (correct negative sanity behavior) |
| 9 spatial archetypes recovery (proposal-literal H2 Part A) | **return as diagnostic only** - never as biology | `archetype` is patient-level pseudobulk; NMI(archetype, patient_id) ≈ 0.89; classical baselines "win" by amplifying patient identity. use `mc_megacluster` (row 2) for the audit-correct biology test |
| classical alignment family with rotation-equivalence sanity | **B2_v3** (Procrustes) vs **B3_v3** (unaligned PCA) | identical on H2 within-space metrics (rotation-equivalent), diverge on H1 cross-modal cosines (B2 AUC 0.706 vs B3 0.441). use the divergence to verify cross-modal test scoring is correct |

every routing decision carries its **construct-validity guards** (§E) into the response payload. for the full per-question evidence with file paths see `docs/v2/tnbc92_routing_matrix.md`.

---

## E. construct-validity guards (load-bearing)

these are the audit-derived rules `eval_agent` and `orchestrator` MUST encode. each cites either `program.md` (hard constraint, never violate) or a numbered audit finding from `projects/tnbc-92/evaluation_question_audit.md`.

- **archetype-ARI is patient-leakage proxy** (audit 1). never report as biology coherence without paired `mc_megacluster` ARI + LP + patient-z diagnostic. if a caller asks for "9-archetype coherence", return it labeled `diagnostic only`.
- **z_st / z_mean on v3-trained-on-v3 runs is circular** (audit gpath2vec arm note). gpath2vec_v3 is on the ST input AND the H3 test target; only `z_he` is independent. eval_agent sets `view_clean=False` for those view-run pairs; orchestrator must not surface them as headline H3.
- **within-subarray retrieval is patient-leakage shortcut** (audit 4; `program.md` 4). CCA cross-subarray R@1 = 0.000 vs within-subarray 0.667. never return within-subarray as primary H1.
- **mc_weights are supervision-only** (`program.md` 3). hard-assert `mc_weights ∉ st_features` in alignment_agent. R2 supcon consumes them via the bundle dict, never the input vector.
- **v3 gpath2vec build is sha-locked** to `13985cbd18fb72f67d41193cc0b7f1b3186b720c1e83da9403963085c7583490` per `data/embeddings/niches_v3/manifest.json`. st_agent verifies sha on every read; eval_agent records sha in every output `provenance.json`.
- **noise-floor gate before any new pathway build** (audit 5). MC z + patient z permutation test on the embedding directly. ratio < 1.0 + same/diff cos within 0.01 = degenerate (the v2 dim-collapse signature). do not promote to canonical.
- **z_A magnitude requires paired specificity matrix** (audit 2). every H3 cross-patient claim publishes the 5x5 off-diagonal cos alongside z_A. R4's z_A wins are real AND aliased at 5-parent; both numbers ship together.
- **stain normalization is banned before FM inference** (`program.md` 2). Virchow2 / UNI2 are stain-invariant (PathoROB RI=0.88 empirically reproduced); he_agent uses only `timm.create_transform()` from the model's `pretrained_cfg`.
- **ComBat is banned** (`program.md` 1). single-institution data; inter-patient variation is biological, not batch.

---

## F. open items (deferred to implementation)

- **HITL trigger thresholds**: low confidence or conflict triggers human review per the page-2 architecture diagram in `docs/tnbc92_three_hypotheses.md`. concrete thresholds (e.g. "BH-FDR borderline 0.04 < q < 0.06 -> HITL", "two routes tie on R@K within 0.01 -> HITL") undefined.
- **vector store choice**: Qdrant per `CLAUDE.md` stack table, but FAISS local-only is acceptable for the 35k-niche test grid. defer until cross-cohort retrieval lands (HEST, other tumor types per `tnbc92_results_summary_v2.md` closing paragraph).
- **Neo4j schema**: graph store listed in `CLAUDE.md` stack but no schema defined. likely lives downstream of H3 DAG decomposition (`per_node_cca.parquet`); defer until clinical-hit annotation is a user request.
- **LangSmith tracing config**: observability-only per `CLAUDE.md`; project-id, run-tagging convention, and trace-budget per node not yet specified. burn rate must be tracked from experiment 1 (`CLAUDE.md` token budget).
- **per-node prompt budget**: $850 LLM cap (`CLAUDE.md`); per-experiment ~$1-2; ~400-850 experiments total. orchestrator (Sonnet 4.6) routing prompts vs synthesis (Opus 4.6) writeup prompts need a budget split. defer to first integration test.
- **encoder swap path**: UNI2 (1536-d) -> Virchow2 (1280-d) is "pluggable" per `MODELS.md`; the swap on H1 grid has not been run. defer until a non-TNBC cohort lands and forces the question.
- **R4 attention_weights downstream use**: `attention_weights` is persisted (v3 retrain plan step 5) but not yet consumed by any eval. interpretability tool (per-niche tile-weight heatmap) is a likely a biologist-facing demo asset; defer until biology_demo.ipynb sees use.
- **R5b: AnInfoNCE + cross-attention** (deferred). R5 (AnInfoNCE + late) showed AnInfoNCE collapses mostly to standard InfoNCE on late fusion (17/512 dims moved >0.1 log-scale; H1 unchanged from R1). the targeted "does anisotropic temperature rescue R4's pathway-direction aliasing?" question requires AnInfoNCE on cross-attention - one-variable-change vs R4. needs patching `CrossAttnFusion` to register `aniso_log_scale` (`build_model` raises by design today). ~30 min compute if greenlit.

---

## pointers

| asset | path |
|---|---|
| audit (load-bearing) | `projects/tnbc-92/evaluation_question_audit.md` |
| proposal deviations | `projects/tnbc-92/proposal_deviations.md` |
| v3 retrain plan | `projects/tnbc-92/v3_phase2_plan.md` |
| program / hard constraints | `projects/tnbc-92/program.md` |
| routing matrix (Q1.1-Q3.8) | `docs/tnbc92_routing_matrix.md` |
| manuscript-target table | `docs/tnbc92_results_summary_v2.md` |
| hypothesis brief + arch figure | `docs/tnbc92_three_hypotheses.md` |
| v3 niche manifest | `data/embeddings/niches_v3/manifest.json` |
| v3 H3 gpath2vec arm | `runs/tnbc-92_v3/eval/H3/pathway_cca_gpath2vec_v3/per_pathway_cca.parquet` |
| v3 H3 provenance | `runs/tnbc-92_v3/eval/H3/pathway_cca_gpath2vec_v3/provenance.json` |
| existing tool scripts | `scripts/build_niche_join.py`, `scripts/align.py`, `scripts/align_classical.py`, `scripts/eval.py`, `scripts/eval_h3_pathway_cca_gpath2vec_v2.py` |