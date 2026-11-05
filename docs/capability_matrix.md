# capability matrix

what omicstra can answer, what data it needs, what it refuses.
single source of truth for external claims (conference talks, README, evidence pack routing decisions).

sources: `src/omicstra/configs/routing_contract.json` (task taxonomy + guards),
`projects/tnbc-92/project.json` + `cohort.json` (declaration schema),
`design/completion_roadmap.md` (known gaps), `projects/tnbc-92/proposal_deviations.md` (H1/H2/H3 tracing).

audited 2026-11-05 against main @ 2de0755.

---

## required data (all tasks)

| item | format | where |
|---|---|---|
| H&E tiles or niche embeddings | virchow2 1280d per-niche `.npy`, per-unit `.npy`; or the niche-join parquets already built | `data/embeddings/virchow2_niche/`, `niches_v3/` |
| ST embeddings | novae 64d per-subarray `.parquet`, gpath2vec 512d one-file `.parquet` | `data/embeddings/novae_niche_full/`, `gpath2vec/` |
| niche-join parquets | one `.parquet` per subarray, columns declared in `project.json#join_inputs` | `niches_dir` declared in `project.json` |
| run embeddings | `embeddings_test.parquet` per arm | `runs_dir/{run_id}/` |
| EDA verdict | `eda_summary.json` with `verdict ∈ {proceed, proceed_with_caution}` | `projects/{id}/eda_summary.json` |
| evidence pack | `routing_evidence.json` with `tasks` entries for each family asked | `projects/{id}/routing_evidence.json` |

a cohort with no `routing_evidence.json` cannot be routed. the system says so and does not reuse another cohort's pack.

---

## required declarations (project.json + cohort.json)

| field | file | who needs it |
|---|---|---|
| `encoders[].name/dim/role` | project.json | every task (encoder registry resolution) |
| `niches_dir`, `runs_dir`, `seed` | project.json | every compute task |
| `h2_label` | project.json | tissue_state_grouping, subject_identity_suppression |
| `supervision` | project.json | alignment training (join + train steps) |
| `headline_view` | project.json | circularity guard (G6) in pathway tasks |
| `pathway_scores` | project.json | pathway_transfer, pathway_discrimination |
| `biology_labels` | project.json | subject_identity_suppression (H2-C confounder labels) |
| `subject_id_column` | cohort.json | all split and confounder checks |
| `evaluate_params` per task | cohort.json | each compute step reads its own sub-block |

absent means refuse with the missing field named. `declared()` in `adapters/evaluation.py` enforces this.

---

## supported task families

### 1. `cross_modal_retrieval`

| | |
|---|---|
| **asks** | given one modality's niche representation, retrieve the matching niche in the other modality |
| **hypothesis** | H1 |
| **required data** | niche-join parquets, run embeddings (`embeddings_test.parquet`) |
| **required declarations** | `niches_dir`, `runs_dir`, `seed`, `encoders`; `evaluate_params.paired_identifiability` + `held_out_comparison` |
| **guards** | G2 `cross_subject_eval`: held-out subjects only. within-subject retrieval leaks patient identity as method quality |
| **method** | `protocols/evaluate.py` `paired_identifiability` + `held_out_comparison` steps → `measures/retrieval.py`. route resolves from `routing_evidence.json#tasks.cross_modal_retrieval` via interval overlap rule |
| **output** | AUC with 95% CI (200x niche-level bootstrap), R@1/R@5/R@10, MRR, median rank per arm. verdict: recommend / tie / refuse |
| **proposal requirement** | H1: "contrastive vs CCA + late-fusion concat + raw unaligned, Recall@K/MRR/AUC, 0.65-0.75 range." executed. range exceeded (R4 AUC 0.859). R3 Barlow (0.674) sits below B2 Procrustes (0.706) — not every contrastive arm wins |
| **limitations** | (P1) conditioning control pending: R4's H&E embedding uses its own ST query at test time; rescoring with a fixed/uniform query is a required publication check before this result can claim it is not query-driven. (P2) current CI is niche-level bootstrap, not patient-level; patient-level intervals are needed before the routing rule can recommend on a properly independent interval. (#3) determinism check: re-running R1_v3 twice with same seed to 4 decimal places — open_items #3, `v3_phase2_plan.md:183`, status open |

---

### 2. `structural_agreement`

| | |
|---|---|
| **asks** | do the two modalities' niche representations agree geometrically, beyond pairwise retrieval |
| **hypothesis** | H1 |
| **required data** | same as cross_modal_retrieval |
| **required declarations** | same as cross_modal_retrieval; `evaluate_params.structure_and_its_null` |
| **guards** | G2 `cross_subject_eval` |
| **method** | `protocols/evaluate.py` `structure_and_its_null` step → `measures/geometry.py` (CKA, alignment gap). route resolves from evidence pack |
| **output** | CKA before/after alignment, alignment gap (cosine between cross-modal centroids) per arm. verdict: recommend / tie / refuse |
| **proposal requirement** | H1: CKA listed as a registered metric. executed |
| **limitations** | same interval and determinism limitations as cross_modal_retrieval (P1, P2, #3). reference is a point estimate; tie rule cannot adjudicate it against interval-carrying candidates |

---

### 3. `tissue_state_grouping`

| | |
|---|---|
| **asks** | group niche units into recurring tissue states; do aligned representations cluster more coherently than unaligned |
| **hypothesis** | H2-A |
| **required data** | niche-join parquets with `h2_label` column, run embeddings |
| **required declarations** | `h2_label`, `supervision`, `niches_dir`, `runs_dir`, `seed`; `evaluate_params.structure_and_its_null`, `label_granularity` |
| **guards** | G1 `label_granularity`: NMI(label, subject_id) computed before any ARI. if NMI > τ (default 0.5) the row is classed diagnostic and cannot rank methods; G3 `paired_subject_z` |
| **method** | `protocols/evaluate.py` `label_granularity` guard then `structure_and_its_null` (KMeans k=14, ARI + silhouette vs declared label). route resolves from evidence pack |
| **output** | NMI(label, patient_id) guard result; ARI + silhouette per arm per view; linear probe accuracy; verdict: biological / diagnostic |
| **proposal requirement** | H2-A: "9 spatial archetypes cluster more coherently, ARI + silhouette." executed on mc_megacluster (14-class NMF per-spot label). `archetype` dropped — it is a patient-level pseudobulk label (NMI 0.89 vs patient_id); see `proposal_deviations.md` |
| **limitations** | tnbc-92: mc_megacluster carries NMI 0.539 with patient_id — above τ 0.5 → verdict is diagnostic, not biological. the 8-arm ARI intervals overlap; routing returns a tie. R4 ARI 0.246 (best) is 5x raw ST (0.046) but the label granularity guard prevents a biological claim. silhouette reported but not in the evidence pack. a label with NMI < τ is required for a biological verdict |

---

### 4. `subject_identity_suppression`

| | |
|---|---|
| **asks** | does the aligned space preserve biology signals while suppressing subject identity, enabling pooled analysis |
| **hypothesis** | H2-C |
| **required data** | run embeddings, biology labels (`biology_labels.labels_path`), subject labels |
| **required declarations** | `h2_label`, `supervision`, `headline_view`, `biology_labels`, `niches_dir`, `runs_dir`, `seed`; `evaluate_params.confounder_suppression` |
| **guards** | G3 `paired_subject_z`: biology z and patient z both computed; G1 by design (cross-patient split removes subject-level leakage) |
| **method** | `protocols/evaluate.py` `confounder_suppression` step → `measures/stats.py` (Welch z-test, BH-FDR). biology labels: MC_global, MC_tumor, TIME (Bareche subtyping). route resolves from evidence pack |
| **output** | per-view (z_he, z_st, z_mean): biology z + p + q per label; patient z + p; ratio biology-z / patient-z with 95% CI and G3 guard. verdict: biological / diagnostic |
| **proposal requirement** | extension, not in original H1/H2/H3. executed. R6 ratio 0.547; B1 disqualified (0.071, below floor); floor = 0.137 |
| **limitations** | (P5) three open items on this row before it can support the paper's Fig 1C showcase: #7 (-1 sentinel counted as a class in one label), #8 (niche-level shuffle of a patient-level label conflates leakage with signal), #26 (floor artifact). `raw_floor` guard (G7) not implemented as a function — floor comes from declared `floor` field in routing_evidence; a guard that re-derived it would be a second answer. `untrained_control` guard (B4) is an arm, not a function call |

---

### 5. `pathway_transfer`

| | |
|---|---|
| **asks** | carry a pathway-level signal (gpath2vec cosines or AUCell enrichment) onto subjects never seen in training |
| **hypothesis** | H3 |
| **required data** | run embeddings (H&E side `z_he`), pathway scores (`pathway_scores.path`), 5 Reactome pathway IDs |
| **required declarations** | `pathway_scores`, `headline_view`, `niches_dir`, `runs_dir`, `seed`; `evaluate_params.pathway_transfer_cca`, `association_and_specificity` |
| **guards** | G2 `cross_subject_eval`; G6 `circularity_view`: the z_he view does not carry gpath2vec (which gpath2vec encodes); views that do (z_st, z_mean when ST features include gpath2vec_niche) are marked circular and excluded from headline claims |
| **method** | `protocols/evaluate.py` `association_and_specificity` + `pathway_transfer_cca` steps. univariate CCA via pearsonr: per (run, pathway, view) → one canonical direction on shared latent, projected, then pearsonr(z_proj, AUCell_score). permutation null 1000 perms (pathway scores shuffled across niches). BH-FDR over 120 tests (8 runs × 5 pathways × 3 views). route resolves from evidence pack |
| **output** | per-run per-pathway per-view: pearsonr r, permutation p, BH-FDR q, n_significant per run. verdict: recommend / tie / refuse |
| **proposal requirement** | H3: "5 Reactome pathways (TGF-β R-HSA-170834, Immune R-HSA-168256, ECM R-HSA-1474244, Cell Cycle R-HSA-1640170, PCD R-HSA-5357801) correlated with directions in shared latent via CCA." executed. cross-patient sub-split (H&E side): 4/4 testable pathways significant in R4, B1, B2, B3 (TGF-β excluded: 2 embedded nodes, underpowered). rule returns tie |
| **limitations** | (P7) gpath2vec arm (proposal-literal method) uses v1 grid artifacts (`runs/tnbc-92/eval/H3/pathway_cca/`), not v3 grid. those results are in supplement labelled legacy until regenerated on v3. (P8) B3 is not in the evidence pack for this task family (pack lists R1-R6 + B1-B2); running `promote` once with 3 human notes adds it. TGF-β excluded from headline: only 2 embedded DAG nodes, underpowered. R4 aliasing explanation (high specificity = coarse aggregation artifact) is proposed, not measured |

---

### 6. `pathway_discrimination`

| | |
|---|---|
| **asks** | distinguish which of several pathways drives a specific niche unit, not just detect pathway signal at all |
| **hypothesis** | H3-DAG |
| **required data** | run embeddings, DAG-level pathway scores (395 nodes under 5 Reactome parents), `pathway_cca/dag_full/` artifacts |
| **required declarations** | `pathway_scores`, `headline_view`, `niches_dir`, `runs_dir`, `seed`; DAG artifact path |
| **guards** | G6 `circularity_view`; G4 `off_diagonal_specificity`: high off-diagonal cosine similarity between pathway directions means the method detects signal but cannot attribute it |
| **method** | same pearsonr + permutation null at DAG node resolution (395 nodes × 8 runs × 3 views). off-diagonal specificity computed as mean cosine between pathway canonical directions |
| **output** | per-node per-run per-view: r, p, q. off-diagonal specificity matrix. interpretable hits (BTLA checkpoint, IL signaling, IFN regulation, ECM collagen). verdict: recommend / tie / refuse |
| **proposal requirement** | H3 DAG extension, not in original proposal. executed on v1 grid. R4 wins (317/395 one-sided, z=3.8-5.7). at DAG resolution R4 distinguishes BTLA / integrin / collagen from IFN — clinically interpretable |
| **limitations** | (P7) artifact is v1 grid, not v3 (`runs/tnbc-92/eval/H3/pathway_cca/dag_full/`). cannot be regenerated on v3 until pathway persistence is built (WS 6). headline tables must reference this as v1; supplement only until v3 DAG is run. `off_diagonal_specificity` guard (G4) implemented; `untrained_control` guard not wired to this task. frozen hierarchy: 395 nodes (`dag_metadata.parquet ∩ per_node_cca.node_id`; R-HSA-844623 dropped on gene-set size) |

---

### 7. `morphology_decode`

| | |
|---|---|
| **asks** | predict a molecular signature (TLS score) from H&E imaging alone, with no molecular assay at prediction time |
| **hypothesis** | H2-B.2 |
| **required data** | run embeddings (H&E side), TLS score labels per niche (`tls_scores.tsv`) |
| **required declarations** | `niches_dir`, `runs_dir`, `seed`; TLS label path; `evaluate_params.held_out_comparison` |
| **guards** | G2 `cross_subject_eval`; `untrained_control` (B4 arm is the control — an arm, not a function call) |
| **method** | `protocols/evaluate.py` `held_out_comparison` step. pearsonr(z_he, tls_score) per arm, held-out patients only. LOPO (leave-one-patient-out) over 5 lymphoid patients |
| **output** | pearsonr r per arm per held-out fold; LOPO median r; point estimate r on full test set. verdict: recommend / tie / refuse |
| **proposal requirement** | H2-B: "alignment quality varies across TME compartments." B.2 is the decode variant. executed. LOPO median r 0.83 (5 lymphoid patients); registered r 0.493 (driven by one patient). B4 (untrained) wins H2-B.2 → contrastive does not outperform an untrained projection on this decode |
| **limitations** | (P2) no patient-level CI on the point estimate. B4 wins means contrastive alignment does not buy morphology-decode performance beyond what the encoder already provides. 10 patients fit / 3 score in the held-out split. `untrained_control` guard not wired as a check — B4 coverage is verified by `coverage()` on the rollup |

---

## not supported

| question | why refused | track |
|---|---|---|
| cross-modal retrieval on a second cohort | no `routing_evidence.json` for a second cohort; routing_evidence is per-cohort, never inherited. the system declines to route | WS 10 (second cohort evidence pack) |
| produce a second cohort's evidence pack | alignment through the package (not scripts) is not built; `align_run(compute=true)` on a new cohort would need the full chain through the package | WS 7 (alignment through package, open_items #33) |
| de-novo niche-type labels (no supplied annotation) | evaluation chain requires `h2_label` declared in project.json. a cohort without supplied annotations cannot run tissue_state_grouping or subject_identity_suppression | WS 7 (de-novo labeling protocol, open_items #48). method: leiden on niche embeddings, resolution as the only declared parameter |
| SLURM dispatch | not implemented. compute_backend is `in_process` or `pool: mac`. SLURM is listed in cohort.json `compute_backend` vocabulary but wired to no executor | not on the roadmap; can be added without changing the agent contract |
| multi-cohort routing (pool across cohorts) | route is per-cohort by contract. `routing_evidence.json` is cohort-specific and the tie/recommend rule operates on one cohort's intervals at a time | not in roadmap |
| H2-B.1 compartment cosine routing | `compartment_cosine.parquet` exists but no task family declared in routing_contract.json for this row. current verdict from manual analysis only | could be declared as a task family; would require a floor definition and a scoring metric added to routing_evidence |
| registration across sections | declared in platform.json as an option; not built. both tnbc-92 subarrays are same-section (H&E and ST from the same 16 µm slice), so no cross-section registration is needed for the seed cohort | not in roadmap; needed only for serial-section designs |

---

## routing rule summary

the route tool resolves deterministically from recorded evidence:

| outcome | meaning |
|---|---|
| `recommend` | one candidate's interval does not overlap the runner-up's; the system names it |
| `tie` | two or more candidates' intervals overlap; all named, none picked |
| `escalate` | question maps onto more than one task family; the caller disambiguates before asking again |
| `refuse` | every candidate falls below the cohort's declared floor, or evidence is missing for this task |
| `override_ack` | a human overrode a tie or refusal; the ledger records it |

the system never guesses between tied methods. a tie is a measurement result, not an ambiguity to resolve.

---

## hypothesis-to-task-family map

| hypothesis | task families | routing outcome (tnbc-92 v3) | proposal match |
|---|---|---|---|
| H1 | cross_modal_retrieval, structural_agreement | R4 leads R@K + AUC (tie on intervals; conditioning control P1 pending); CKA confirms geometry | H1 exceeded. 10-arm grid vs proposed ~4. R3 below B2 on AUC |
| H2-A | tissue_state_grouping | tie (8-arm ARI intervals overlap); verdict diagnostic (NMI 0.539 > 0.5) | corrected: archetype → mc_megacluster. biological verdict requires label NMI < 0.5 |
| H2-B.2 | morphology_decode | B4 (untrained) wins; contrastive does not add decode performance | partially supported. TLS direction holds; contrastive does not outperform encoder alone |
| H2-C | subject_identity_suppression | R6 leads (ratio 0.547); B1 refused (below floor 0.071) | extension. 3 open items (P5: #7, #8, #26) before Fig 1C |
| H3 | pathway_transfer, pathway_discrimination | 4/4 pathways significant in R4, B1, B2, B3 (z_he view); tie. DAG: R4 leads (317/395) but artifact is v1 | H3 met (tie); DAG exceeded but legacy grid (P7) |
| H3-DAG | pathway_discrimination | see above | v1 grid only; v3 pending WS 6 |

---

## publication criteria notes (from artifact review 2026-11-05)

the following items in the routing_evidence pack or the draft require resolution before PLOS CB submission. they are documentation tasks (no new compute, except P1):

| id | task family affected | what | local? |
|---|---|---|---|
| P1 | cross_modal_retrieval, pathway_transfer | R4 conditioning control (fixed query, uniform attn, shuffled partner) | yes, minutes |
| P2 | cross_modal_retrieval, morphology_decode | patient-level intervals (LOPO exists; niche bootstrap is not patient-level) | yes, numbers exist |
| P4 | cross_modal_retrieval | close open_items #3: re-run R1_v3 twice, AUCs must agree to 4 decimal places | yes, local compute |
| P5 | subject_identity_suppression | settle open_items #7, #8, #26 on H2-C | yes |
| P7 | pathway_discrimination | move v1-grid DAG numbers to supplement labelled legacy | writing only |
| P8 | pathway_transfer | run `promote` once to add B3 to pathway_transfer in evidence pack | local, 3 human notes |