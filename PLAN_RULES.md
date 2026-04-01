# PLAN_RULES.md

planning and execution rules for omicstra. governs agent routing, experiment
sequencing, alignment strategy selection, and failure handling.

all rules are conditioned on eda_summary.json being present and verdict != "stop".
if eda_summary.json is missing or verdict is "stop", halt and surface the reason.

project-specific phase sequencing, search spaces, and stopping criteria belong
in `projects/{project_id}/program.md`, not here.

---

## 1. agent routing

### 1.1 encoder dispatch

route to encoders based on modality of the input unit:

| input modality | primary encoder | fallback | condition |
|---|---|---|---|
| H&E patch | defined in project.json | ablation encoder in project.json | always |
| ST expression (spatial) | spatial encoder (ex. Novae GNN) | - | spatial_autocorrelation_pass: true |
| ST expression (no spatial signal) | non-spatial encoder (ex. scVI, PCA on HVGs) | - | spatial_autocorrelation_pass: false |

if `model_tissue_fit[encoder].fit == "uncertain"`, route to that encoder but
flag outputs as provisional. require embedding UMAP validation before alignment.

if `counts_are_raw: false` and `encoder_input_decision` is unresolved, do not
dispatch to any ST encoder. surface the decision to the operator and halt embedding.

### 1.2 orchestration model routing

| task | model |
|---|---|
| tool selection, state transitions | Claude Sonnet |
| hypothesis evaluation, result interpretation | Claude Opus |
| structured data extraction, JSON ops | Claude Sonnet |

do not call Opus for routing decisions. do not call Sonnet for hypothesis ranking
or biological interpretation of embedding results.

### 1.3 vector store routing

route queries by embedding type:

- modality A embeddings -> Qdrant collection: `{modality_a}_embeddings`
- modality B embeddings -> Qdrant collection: `{modality_b}_embeddings`
- aligned cross-modal embeddings -> Qdrant collection: `aligned_embeddings`

do not query `aligned_embeddings` until alignment has been validated (see section 3).

### 1.4 graph routing

Neo4j is the source of truth for relational structure:
patient -> sample -> section -> measurement unit -> embedding -> cluster

route to Neo4j for: patient-level queries, sample grouping, cluster membership,
cross-modal link traversal.
route to Qdrant for: nearest-neighbor search, similarity queries, embedding retrieval.
do not use Qdrant for relational joins - always resolve structure in Neo4j first.

---

## 2. experiment sequencing

phases must execute in order. do not skip phases. do not begin phase N+1 if
phase N produced a stop condition.

project-specific phases (encoder choices, dataset details, ablation designs)
are defined in `projects/{project_id}/program.md`.

### phase 0 - EDA gate
- eda_summary.json must exist with verdict: "proceed" or "proceed with caution"
- all risks[] items must be reviewed and either resolved or explicitly accepted
- encoder_input_decision must be set (not empty string or "unresolved")
- output: eda_summary.json committed to projects/{project_id}/

### phase 1 - embedding
- run primary encoders for each modality as defined in project.json
- run ablation encoders in parallel if specified
- validate each embedding independently before proceeding:
  - UMAP colored by known biological labels (subtype, condition, etc.)
  - flag embedding collapse (all points within 2 std of centroid)
  - flag batch artifacts (silhouette by technical variable > 0.1 in embedding space)
- output: embeddings stored in Qdrant, validation UMAPs saved to
  projects/{project_id}/validation/

### phase 2 - alignment
- run integration strategies defined in program.md
- at minimum: one baseline + one experimental strategy
- alignment requires matched cross-modal pairs. confirm pair correspondence is
  verified (cross_modal_registration_pass: true or manually confirmed) before training
- do not begin alignment if phase 1 validation flagged embedding collapse or
  batch artifacts in embedding space
- output: aligned embeddings in Qdrant, alignment metrics logged

### phase 3 - evaluation
- evaluate on task defined in program.md
- compare integration strategies on same eval set
- compare encoder ablations if specified in program.md
- all eval baselines (defined in CLAUDE.md) must be computed
- a run must beat all baselines before writing winner.json
- output: projects/{project_id}/results/eval_summary.json

---

## 3. integration strategy decisions

integration mode is a configurable experimental dimension, not a fixed pipeline
decision. program.md defines which strategies to run for each project.

### 3.1 early fusion (encoder-native)

- modalities integrated within a single encoder (ex. Novae natively incorporates
  H&E foundation model features as node-level attributes during graph construction)
- produces a single joint embedding per measurement unit
- advantage: captures cross-modal interactions during encoding
- constraint: requires encoder that supports multimodal input

### 3.2 post-encoding contrastive alignment (late interaction)

- each modality encoded independently, then projected into shared latent space
- 2-layer MLPs per modality with InfoNCE contrastive loss
- default choice when encoders are modality-specific
- use when n_matched_pairs < 5000 (lower capacity reduces overfitting risk)
- faster to train, easier to debug

### 3.3 cross-attention bridge (early interaction, post-encoding)

- cross-attention between modality token sequences before projection
- higher capacity than contrastive MLP
- use when post-encoding contrastive produces low retrieval recall (< 0.3 at k=10)
- use when modality gap is large (centroid cosine similarity < 0.3)
- requires more matched pairs to train stably - do not use if n < 2000

### 3.4 baselines

every alignment evaluation must include these unaligned baselines:
- CCA projection (canonical correlation analysis)
- late fusion concatenation (L2-normalized, no learned projection)
- unaligned raw concatenation

### 3.5 ablation requirement

if program.md specifies an ablation comparison, all specified strategies must be
run and reported. do not drop a strategy because early results look unpromising.

integration mode is an experimental variable, not an assumption.

### 3.6 alignment training constraints

- hard negatives: mine within-slide from different tissue compartments to control
  for slide-level batch effects
- cross-patient negatives: exclude from hard negative pool to avoid confounding
  biological variation with alignment failure
- train/test split: stratify by patient (not by spot or slide)
- early stopping: on validation cosine similarity, not training loss

### 3.7 alignment validation gate

before writing results:
- retrieve nearest neighbors cross-modally for held-out pairs
- compute recall@k (k=1, 5, 10)
- if recall@1 < 0.1 for all strategies: stop, alignment is not working,
  investigate embedding quality and pair correspondence
- if one strategy clearly dominates: document but still report all

---

## 4. failure handling and fallback behavior

### 4.1 embedding failures

| failure mode | action |
|---|---|
| encoder OOM | reduce batch size; process at lower resolution |
| spatial encoder embedding collapse | check counts_are_raw; try non-spatial fallback (scVI/PCA) |
| batch artifacts in embedding UMAP | apply harmony post-hoc; re-validate before alignment |
| >10% units missing embeddings | investigate coordinate mismatch; do not impute |

### 4.2 alignment failures

| failure mode | action |
|---|---|
| contrastive loss not decreasing | check pair correspondence; verify no label leakage |
| recall@1 < 0.05 after 50 epochs | stop training; check embedding spaces independently |
| cross-attention loss unstable | reduce learning rate; if persistent, fall back to contrastive MLP |
| matched pairs < 1000 | do not run cross-attention; contrastive MLP only |

### 4.3 data failures

| failure mode | action |
|---|---|
| encoder_input_decision unresolved | halt all embedding; surface to operator |
| cross-modal pair correspondence unverified | halt alignment; do not proceed on assumption |
| new sample fails QC | add to excluded_samples in eda_summary.json; re-run |
| batch artifacts emerge in phase 1 | halt phase 2; address batch in embedding space first |

### 4.4 agent failures

| failure mode | action |
|---|---|
| tool call returns null or empty | retry once; if still empty, surface to operator |
| Qdrant collection not found | do not create implicitly; halt and report |
| Neo4j traversal returns no nodes | verify ID format before assuming data absent |
| Opus call exceeds context | chunk input; do not silently truncate |

### 4.5 general fallback rule

if a failure mode is not listed above and the agent cannot resolve it with one
retry: stop, write the failure state to projects/{project_id}/failures.log,
and surface to operator. do not silently continue with degraded state.