# omicstra MCP - engineering plan

## 1. objective

Build the MCP server and orchestration layer specified in the ResearchHub proposal. The scientific
evaluation is complete; the coordination layer that was specified to produce it is not built.

The system exposes cross-modal reasoning over a cohort through the Model Context Protocol:
modality-specialised agents produce frozen-encoder embeddings, an alignment component maps them
into a shared space, an evaluation component tests hypotheses under construct-validity guards, and
an orchestrator dispatches questions to methods.

**The deliverable is a generalisable package, not a cohort pipeline.** The value is that the
analytical decisions are encoded once and not re-derived per dataset - which is what makes the
work both faster and deterministic. TNBC-92 is not the product; it is the **fixture** that proves
each component works, because it is the one cohort whose correct answers are already known and
published. HEST is a future external validation cohort, out of scope for the initial build.

Two users, one package, different surfaces:

| user | surface | needs | never touches |
|---|---|---|---|
| **biologist** | MCP client, natural language | evidence, routing decisions, guards, an answer they can act on | code, raw data, accelerated compute |
| **computational biologist** | CLI and library | the gate chain, stage interfaces, decision records, determinism across re-runs | re-deriving decisions already settled on a prior cohort |

Both consume the same components and differ only in which storage tier they read (§10) and whether
they may trigger computation.

## 2. specification and oracle

`docs/reports/internal/tnbc-92/index.html` is both the specification and the test oracle. It already
contains the correct answer to every question the server must handle. The completion test for the
interface is therefore not "does it execute" but **does the server return the value the report
states, with the qualifying guard attached**.

This mirrors the pipeline layer: `runs/tnbc-92_v3/` is the oracle for computation, the report is
the oracle for the interface. Expected values do not need to be authored at either level.

**The report is immutable for the duration of the build.** It is the fixed target the
implementation is measured against; editing it while building against it would invalidate every
test that references it.

| rule | |
|---|---|
| `docs/reports/internal/tnbc-92/index.html` | **read-only.** No edits, no regeneration, no reformatting |
| a value in the report that appears wrong | record it as a finding; do not correct the file |
| the report needs to change | that is a separate, later task, performed after the build and re-baselined deliberately |
| derived copies | permitted and expected — the evidence bundle (§10) is generated *from* it and from `runs/`, never back into it |

## 3. architecture

| component | role | seed-cohort instantiation |
|---|---|---|
| client surface | accept a natural-language question, return a routed answer | MCP over stdio |
| orchestrator | intent classification, dispatch | table lookup; the model classifies intent only |
| modality agent A | frozen encoder over imaging | Virchow2, 1280-d per tile |
| modality agent B | frozen encoders over molecular data | Novae 64-d ⊕ gpath2vec 512-d |
| aggregation | reduce per-unit encodings to the atomic unit | niche = centre spot + 6 neighbours = 7 spots |
| alignment | project both modalities into a shared space | shared 512-d; 6 contrastive + 4 classical routes |
| evaluation | hypothesis tests with paired validity guards | H1 retrieval, H2 structure, H3 pathway transfer |
| routing table | question class -> method, accumulating across cohorts | 7 rows, one cohort |
| trace | decision record per run | not built |
| HITL | escalation at declared boundaries | not built |

### corrections to the published figure

`docs/images/architecture_diagram.png` documents the funded design. Six aspects changed during
execution. Four are substitutions; two alter topology, and a system built from the figure alone
would be wrong in those two respects. These are recorded in `docs/reports/internal/tnbc-92/` but **not** in
`proposal_deviations.md`, which is where reviewers look for scope changes.

| # | figure | system | basis |
|---|---|---|---|
| 1 | imaging encoder UNI2 1536-d | Virchow2 1280-d, frozen | selected empirically on a 14-class niche-level linear probe; only encoder with positive cross-subarray morphology silhouette |
| 2 | molecular encoder 256-d | Novae 64-d, frozen | actual latent width; effective rank 11 of 64 is the constraining bottleneck |
| 3 | multi-scale (cell · niche · region), cell-scale for this study | niche only, k=6 | spot-level cross-patient retrieval collapsed to chance. The niche is now the atomic unit; changing it requires a new experiment series and human approval |
| 4 | two ablation slots | 10-run grid, 6 contrastive + 4 classical | the ablation became the experiment; its output is the routing table |
| **5** | one imaging path into alignment | **two imaging representations** - 7 tiles mean-pooled to 1280-d (9 routes), or un-pooled as (7, 1280) (1 route) | the un-pooled route wins retrieval and pathway transfer. "Early interaction" denotes exactly this |
| **6** | molecular side branches to a pathway-token stage | **two molecular encoders concatenated** into one 576-d vector | the pathway embedding is a second encoder, not post-processing |

Concatenation joins the two molecular encoders into a single input to one projection tower. The
imaging and molecular vectors are never concatenated - they are the two sides of the contrastive
pair, each with an independent tower.

### data flow and artifact dependencies

The project is an hourglass. Eleven input sources converge on a single per-niche table, which fans
out to ten alignment routes, which converge again on the hypothesis artifacts. The waist is
`data/embeddings/niches_v3/{subarray}.parquet` - 12 columns, 208,786 rows across 259 subarrays.
Every claim in the report traces through it.

```
 ── committed ──┬── git-ignored ──────────────────────────────────────────────────────

 knowledge/                     data/inputs/                        [git-ignored]
  reactome/                      byArray/{slide}/{subarray}/
   ReactomePathways.gmt           selection.RData  tissue-filtered raw counts  ◄─┐
   ReactomePathwaysRelation       all.RData        full grid (unused)            │
  pathway_commons/                Images/          H&E whole-slide               │
   tf_genes...json  1,658 TF      Clinical/        Clinical.RDS · ids.RDS        │
                                  Robjects/        annotsBySpot · clustering/    │
                                                     km14.RDS · MC deconv        │
                                                     clustPrototypes             │
        │                              │                                         │
        │        ┌─────────────────────┴──────────────────────┐                  │
        │        ▼                                            ▼                  │
        │  ╔═══════════════════════════════════╗    label extraction             │
        │  ║ GATE 0 · EDA                      ║    data/embeddings/             │
        │  ║ notebooks/exploration/            ║      biological_signals/         │
        │  ║   eda_data_inventory.ipynb        ║        mc_labels.tsv      14-cls │
        │  ║   eda_biological_signal.ipynb     ║        mc_weights.tsv     14-d   │
        │  ╠═══════════════════════════════════╣        tls_scores.tsv     cont.  │
        │  ║ → projects/tnbc-92/               ║        morphology_labels.tsv     │
        │  ║     eda_summary.json  [committed] ║        clinical_labels.tsv       │
        │  ║   verdict: proceed                ║              │                   │
        │  ║   encoder_input_decision ─────────╫──────────────┼───────────────────┘
        │  ║     "use selection.RData"         ║              │
        │  ║   spatial_autocorr_pass: true ────╫──┐           │
        │  ╚═══════════════════════════════════╝  │ selects   │
        │              BLOCKING                    │ ST branch │
        │                                          ▼           │
        │     ┌────────────────────────────────────────────┐   │
        │     ▼                                            ▼   │
        │  IMAGING ENCODER                        MOLECULAR ENCODER
        │  scripts/scale_embeddings.py            notebooks/embeddings/
        │   [git-ignored, NO notebook]              novae_st_embeddings.ipynb
        │        │                                       │
        │        ▼                                       ▼
        │  data/embeddings/                       data/embeddings/
        │    virchow2_niche/   1280-d pooled        novae_niche_full/  64-d
        │    virchow2_cell/    (7,1280) tokens      pca_hvg/  fallback, pilot only
        │                                                │
        └────────────┐                                   │
                     ▼                                   │
              PATHWAY ENCODER                            │
              notebooks/embeddings/                      │
                niche_pathway_analysis_full.ipynb        │
                  → niche_fisher_full_*.parquet          │
                       │                                 │
                       ▼                                 │
              data/embeddings/gpath2vec/                 │
                fisher_madmean_low_dim512_e5_s1234/      │
                  ◆ sha256 13985cbd…  LOCKED             │
                       │                                 │
    ┌──────────────────┴─────────────────────────────────┘
    ▼
 ╔══════════════════════════════════════════════════════════════════╗
 ║  WAIST · scripts/build_niche_join.py                             ║
 ║  → data/embeddings/niches_v3/{subarray}.parquet  + manifest.json ║
 ║                                                                  ║
 ║   subarray · patient_id          keys                            ║
 ║   virchow2_niche      1280       imaging, pooled     ─┐          ║
 ║   virchow2_cell_tokens (7,1280)  imaging, un-pooled   ├ features ║
 ║   novae_niche           64       molecular, z-scored  │          ║
 ║   gpath2vec_niche      512       pathway             ─┘          ║
 ║   mc_weights_niche      14       SUPERVISION ONLY  ◄── never a   ║
 ║   mc_megacluster · archetype · compartment · tls   labels  feature║
 ║   neighbor_spot_ids              niche provenance                ║
 ║                                                                  ║
 ║   208,786 niches · 259 subarrays · intersection of all sources   ║
 ╚═══════════════════════════════┬══════════════════════════════════╝
                                 │  split.json  85/15 patient-stratified, seed 42
                                 │  shared across all 10 routes
       ┌─────────────────────────┼─────────────────────────┐
       ▼                         ▼                         ▼
  configs/v3/*.json         scripts/                  scripts/
  [committed]                 align.py                  align_classical.py
   R1 R2 R3 R4 R5 R6          R1–R6                     B1 B2 B3
                              align_b4.py → B4
       └─────────────────────────┼─────────────────────────┘
                                 ▼
                      runs/tnbc-92_v3/{run}/        [git-ignored]
                        embeddings_test.parquet     z_he · z_st  35,594 niches
                        metrics_h1_raw.json         run_config.json  split.json
                        metrics_h1_ci.json          checkpoint.pt
                                 │
                                 ▼
                      scripts/eval.py + eval_h1_bootstrap_ci.py
                                   + _scratch/eval_h2_*.py
                                   + eval_h3_pathway_cca*.py
                                 │
                                 ▼
                      runs/tnbc-92_v3/eval/{H1,H2,H3}/
                                 │
                                 ▼
                      notebooks/final/05_summary_umaps_v3.ipynb  [committed]
                                 │
                                 ▼
                      docs/reports/internal/tnbc-92/index.html  ◄── SPEC + ORACLE
                                 │
                                 ▼
                          routing table, 7 rows
```

### hypothesis to artifact

Each hypothesis is answered by named artifacts, not by the run directory as a whole. These are the
fixtures the fast test suite asserts against.

| hypothesis | question | artifact | guard |
|---|---|---|---|
| H1 | does alignment order matched pairs above mismatched? | `{run}/metrics_h1_raw.json`, `metrics_h1_ci.json` | cross-subarray only; within-subarray is a leakage shortcut |
| H2-A | do units group by tissue state? | `eval/H2/mc_coherence.parquet`, `mc_diagnostics/linear_probe.parquet` | label confounder check - the patient-level label is diagnostic only |
| H2-B.1 | do distinct compartments cohere more than ambiguous ones? | `eval/H2/compartment_welch_contrasts.parquet` | annotation coverage: 13 of 38 test subarrays |
| H2-B.2 | is a signature decodable from imaging alone? | `eval/H2/tls_signature_cca.parquet` | cross-patient split; permutation null |
| H2-C | biology over subject identity? | `{run}/eval/biology.parquet` | ratio against the raw-modality floor |
| H3 | does pathway signal transfer to unseen subjects? | `eval/H3/pathway_cca_gpath2vec_v3/per_pathway_cca.parquet` | circular views refused; specificity matrix published alongside magnitude |

### governance to artifact

Four committed documents constrain the flow rather than participating in it.

| document | constrains | mechanism |
|---|---|---|
| `EDA.md` | gate 0 | defines the pass criteria that produce `eda_summary.json` |
| `program.md` | encoders, aggregation, alignment inputs | hard constraints: no batch correction, no stain normalisation before encoding, supervision never enters the feature vector, cross-subarray evaluation only, niche is atomic |
| `PLAN_RULES.md` | branch selection and phase order | encoder dispatch by autocorrelation result; label-selection validity rule |
| `evaluation_question_audit.md` | evaluation | the five construct-validity findings; the specification for the guard library |

### the twelve roles

Twelve columns, each with exactly one role. The roles are what enforce the project's hard
constraints, and the roles - not the column names or dimensions - are what a new cohort must
satisfy.

| role | column | dim | notes |
|---|---|---|---|
| key | `subarray`, `spot_id` | — | |
| confounder axis | `patient_id` | — | every validity guard tests against this |
| provenance | `neighbor_spot_ids` | 7 | traceability to source units |
| imaging feature, pooled | `virchow2_niche` | 1280 | 9 of 10 routes |
| imaging feature, un-pooled | `virchow2_cell_tokens` | (7, 1280) | 1 route |
| molecular feature | `novae_niche` | 64 | all routes |
| pathway feature | `gpath2vec_niche` | 512 | 9 of 10; one route ablates it |
| **supervision** | `mc_weights_niche` | 14 | one route's loss only. **Never a feature** |
| label, unit-level | `mc_megacluster` | 1 | the audit-correct evaluation target |
| label, subject-level | `archetype` | 1 | demoted to diagnostic by guard |
| label, annotation | `compartment` | 1 | |
| label, continuous | `tls` | 1 | |

### measured storage

| measurement | value |
|---|---|
| `niches_v3/` total | **13 GB** |
| `virchow2_cell_tokens` share of each file | **81%** |
| routes reading that column | **1 of 10** |
| all keys, labels and supervision combined | **~0.1%** |

### four data-structure issues

| # | in plain words | blocks the demo | fix when | what is actually done |
|---|---|---|---|---|
| 1 | the table is mostly one fat column that nine of ten routes never open | no | RERUN | move the un-pooled tokens to their own file; the one route that needs them joins to it |
| 2 | nothing in the data stops a new consumer training on the supervision column - the rule is an `assert` inside `align.py` | no, but the read tools **are** the new consumer | before the read tools | add a role map in `knowledge/` |
| 3 | labels are welded to 13 GB of blobs, so adding one small label means rewriting all of it | no | RERUN | labels as their own small table on the same key |
| 4 | the aggregate summary files disagree with the per-run files | possibly - the read path may reach them | now | delete them; the artifact store already resolves per run |

Issue 1 is co-location, not necessity: the un-pooled column is required by the route that wins two
of three hypotheses. It is in the wrong file, not superfluous.

**On issue 2, the fix is a role map, not a rename.** Prefixing columns with role markers would
require rewriting the 13 GB table, would break every configuration and script referencing the
current names, and would invalidate the frozen artifacts that serve as the regression oracle. A
JSON role map in `knowledge/` that the read tools consult gives the same guarantee for one file
and no data movement. The rename, if wanted, belongs to RERUN when the table is rewritten anyway.

### two hash locks

| lock | covers | recorded in | prevents |
|---|---|---|---|
| pathway build `sha256` | the 512-d pathway embedding | every `run_config.json`, the join manifest | a silent encoder swap between runs, which would make them incomparable |
| join `manifest.json` | the waist itself | every `run_config.json` | a run being attributed to the wrong input table |

A prior pathway build failed a degeneracy check and was retired; the lock is what prevents it
reappearing unnoticed.

### what is tracked

The policy is consistent - anything regenerable is excluded, anything recording a decision is
committed. One path violates it.

| path | tracked | regenerable |
|---|---|---|
| `data/inputs/`, `data/embeddings/`, `runs/` | no | yes |
| `notebooks/`, `configs/`, `knowledge/`, `docs/`, `projects/*/eda_summary.json` | yes | no |
| **`scripts/`** | **no** | **no - local disk only** |

## 4. the primary artifact is a routing table

No single route wins every objective. The evaluation produces task-conditional dispatch, and that
table is the transferable result. Each additional cohort appends rows.

| # | question class | route | basis |
|---|---|---|---|
| 1 | retrieve the matched molecular unit from an imaging unit | cross-attention | AUC 0.859 [0.857, 0.862]; CKA 0.631 |
| 2 | group units by tissue state within a section | late-fusion contrastive (3-way tie) | KMeans ARI 0.234-0.246 on the niche-level label |
| 3 | amplify biology, suppress subject identity | late-fusion contrastive | ratio 0.547 [0.476, 0.617] |
| 4 | transfer pathway signal to unseen subjects | cross-attention | 4/4 pathways cross-patient; 80% of 395 sub-pathways |
| 5 | distinguish which pathway drives a unit | classical orthogonal rotation | off-diagonal cosine 0.38 vs 0.75 |
| 6 | decode a signature from imaging alone | untrained projection | z = 25.5 cross-patient; contrastive training degrades it |
| 7 | **contraindicated** for cohort-level interpretation | CCA | ratio 0.071 [0.053, 0.088], below the raw-modality floor 0.137 |

Two implementation consequences. Dispatch is a lookup, not a model judgement. And a route that
wins one row is contraindicated in another, so every returned metric carries the guard that
qualifies it.

## 5. tool surface

Four read tools. No accelerated compute, no network, one model call for intent classification.
Each maps to a section of the report.

```
route_question(question: str, project_id: str = "tnbc-92")
    -> { question_class: int | null,
         route: str, view: str,
         evidence: [{metric, value, ci, source_artifact}],
         guards:   [{id, verdict, note}],
         refused:  bool, reason: str | null }

get_result(hypothesis: "H1"|"H2"|"H3", route: str, view: str = "z_he")
    -> { metric, value, ci_low, ci_high, ci_method, n,
         guards: [...], provenance: {git_commit, seed, sha256, artifact_path} }

explain_metric(name: str)
    -> { measures, computed_as, good_value, chance_level, caveats: [...] }

list_answerable(project_id: str)
    -> { question_classes: [...], out_of_scope: [...], cohort_manifest }
```

**Unmatched questions.** Anything outside the seven classes returns `refused: true` with the
output of `list_answerable`. This is specified behaviour, not an error path.

**Contraindicated routes.** Row 7 refuses the *routing*: the server declines to dispatch CCA for
cohort-level biological interpretation and returns the ratio and the raw-modality floor as the
reason. `get_result` still returns the CCA value if requested directly. The distinction is that
the system declines to recommend a method it can compute.

## 6. acceptance tests

| tier | asserts | speed | fixtures |
|---|---|---|---|
| **fast** (whole suite) | routing decisions, guard activation, gate verdicts, label-validity outcomes | milliseconds, no accelerated compute, no model calls | frozen subset of committed run outputs |
| **trajectory** | control flow, not values: the confounder guard fired before any clustering metric was reported; circular views refused for the relevant hypothesis | milliseconds | recorded decision ledgers |
| **contract** (compute stages only) | shape, dtype, determinism under seed 42, serialisability at activity boundaries. Values are never asserted for accelerated-compute components | seconds | small synthetic |

The seven routing rows are the fast suite's fixture set: each row asserts route, value, CI and
guard against the report. The fast suite is written first and becomes the regression net for the
pipeline port.

## 7. implementation order

| milestone | components | completion criterion |
|---|---|---|
| **ASK** | client surface, orchestrator, evaluation read path | an MCP client submits a natural-language question and receives a routed answer with metrics, 95% CIs and guard verdicts, resolved from committed artifacts without accelerated compute or network |
| **EXPLAIN** | trace, HITL | the same call emits a run record, decision ledger and notebook that re-derives its own reported values; non-interactive notebook execution exits zero |
| **RERUN** | modality agents, aggregation, pathway encoder, alignment, evaluation compute path | the emitted workflow definition reproduces the run's headline value from a clean checkout |
| **EXTEND** | modality slots, optimisation loop | a new cohort profile passes the gate chain and yields an encoder decision without modification to package code |

ASK is bounded to **one routing row end to end**; the remaining six are table entries returning
`not_wired` until after. The seven rows share a code path, so one row demonstrates the
architecture.

### prerequisites

- **Dependency verification.** Install orchestration, durability and protocol packages into a
  disposable environment on the target interpreter. Declared ranges admit Python 3.14; transitive
  extension modules are unverified. Two open decisions depend on the outcome.
- **`configs/v3/R1_v3.json`** contains invalid JSON and blocks configuration loading.
- **One displayed value is wrong.** `demo_2026_06.md:168` reports a linear-probe accuracy of
  `0.235`; the artifact states `0.2757`. The stated figure belongs to a different run.

### RERUN prerequisite, not an ASK prerequisite

`scripts/` is git-ignored and untracked, and there is no extraction notebook for the primary
imaging encoder. **The primary encoder's extraction code exists only on local disk.** Recovering
and committing it precedes RERUN; it does not block ASK. Cleanup of superseded notebooks and
partially updated planning documents is likewise a RERUN prerequisite - no demonstration path
displays any of it.

## 8. durability

Graph nodes are wrapped as durable activities by `temporalio[langgraph]`, which supports both
authoring styles, interrupts and continue-as-new. The graph is not authored twice.

The separation is required because workflow code replays and must be deterministic, whereas a
graph invocation containing a model call is not:

```python
@workflow.defn                    # deterministic skeleton; replays exactly
class CohortWorkflow:
    async def run(self, profile):
        gate = await workflow.execute_activity(run_gate, profile, ...)
        if gate.verdict == "stop":
            return gate
        pick = await workflow.execute_activity(decide_encoder, gate, ...)

@activity.defn                    # non-deterministic inside, recorded outside
async def decide_encoder(gate) -> EncoderDecision:
    return encoder_selection_chain.invoke(gate)
```

The model call executes once. Its output is persisted to the event history and replayed on
recovery rather than re-derived. Determinism over a non-deterministic component is obtained by
recording it, not by removing it. The same seam allows a stage body to move from a local call to a
workflow-manager submission without changing the graph.

## 9. run outputs

Three renderings of one typed record, never authored independently:

| artifact | audience | contents |
|---|---|---|
| `run_record.json` | machines | the source: question, answer, provenance, decisions, stages, metrics, caveats |
| `report.ipynb` | the reader | compact, re-derives every reported value. Non-interactive execution is the correctness test |
| `decisions.jsonl` | the reviewer | one line per decision, each classified `deterministic \| model \| human` |
| `trace.otlp.json` | tooling | OpenTelemetry spans, local file primary, hosted tracing an optional consumer |
| `pipeline.nf` | the replicator | the deterministic subset with parameters resolved, agent removed |

The actor classification yields a per-run ratio - for example *24 decisions, 21 deterministic, 2
model, 1 human* - which converts the reproducibility claim into a checkable number. Traces are
written locally because a reviewer cannot open a trace held in a vendor account.

### what hosted tracing adds beyond the local file

The two are not alternatives serving one purpose. The local file is the **reproducibility
artifact**; hosted tracing is a **development and operations surface**. Conflating them is what
produces a reproducibility claim that depends on a vendor account.

| capability | what it provides | needed here | confidence |
|---|---|---|---|
| nested, replayable span tree | step-through of every model call, tool call and node transition | yes - the proposal names decision-path inspection as a deliverable | documented |
| unified cost attribution | cost per step spanning model calls, tool execution and external calls, not tokens alone | yes - a fixed LLM budget must be tracked from the first experiment | documented |
| datasets built from traces | the seven routing rows as a versioned dataset, runs compared side by side with regression flags | probably - this is the acceptance suite in another form | **spike** |
| judge-style evaluators | automates the standing review rule that every reported value be derivable from an artifact | probably | **spike** |
| judge calibration against human preference | makes the above trustworthy | only if the judge proves useful | not evaluated |
| annotation interface | captures biological validation as structured feedback rather than prose in markdown | plausible - the outstanding biology questions currently live in documents | not evaluated |
| test-framework integration | traces attached to test runs | useful for the trajectory tier | **spike** |

### constraints on adoption

| constraint | reason |
|---|---|
| tracing must not be required for ASK | ASK is specified to run without network access |
| the shipped trace remains a local file | a reviewer must be able to open it without an account |
| the optimisation loop must not trace every experiment by default | pricing is per trace and the loop is high-volume by construction |
| no reported value may be recoverable only from a hosted trace | that would make the vendor a dependency of the scientific record |

### unknowns requiring implementation before commitment

| question | why it determines the design |
|---|---|
| can one instrumentation pass feed both sinks | the exporter honours standard environment configuration, so a single emission may serve both. If it does, there is no duplicated code; if not, the local file is authoritative and hosted becomes a second sink |
| what the cost view actually attributes | determines whether budget tracking is obtained free or must be built |
| trace volume and cost under the loop | determines the sampling policy - trace loop *decisions* rather than every experiment |
| can datasets be constructed from committed artifacts rather than live traces | if so, the acceptance suite and the hosted dataset are one object rather than two |

Sequencing: adopt after EXPLAIN emits local traces. Instrumenting a second consumer is cheap once
the first exists; doing both simultaneously couples an unbuilt component to a vendor and makes
the local-file guarantee harder to hold.

## 10. storage tiers and the evidence bundle

The operative principle: **a query must never open a tier deeper than it needs.** This is
assertable in a test - which tier did this call touch - so a tool reaching too deep is a defect,
not a performance characteristic.

| tier | contents | size | required for |
|---|---|---|---|
| **0 · index** | what exists, what is answerable, hashes, schema version | KB | every call |
| **1 · answers** | metrics, CIs, guard verdicts, routing table, metric definitions, role map | ~MB | **the entire ASK path** |
| **2 · per-unit derived** | shared-space vectors, cached projections, per-unit pathway scores | ~100s MB | spatial maps, retrieval demonstrations |
| **3 · features** | the waist | GB | re-running alignment |
| **4 · raw** | source inputs | 10s GB | rebuilding from source |

### the evidence bundle

Tiers 0 and 1 are generated from `runs/` and the report by a build-time script, into a
self-contained `evidence/` directory. Generation is one-directional; nothing is written back.

| property | consequence |
|---|---|
| ASK reads a purpose-built bundle, not a research directory | fast by construction, in the low milliseconds |
| the bundle is self-contained | ASK does not require the 13 GB waist to be present at all |
| the bundle is generated, never hand-edited | the aggregate-summary divergence (§3, issue 4) cannot recur |
| the bundle has no tensor-library dependency | it is what ships to a non-technical user as a portable bundle |

This single artifact resolves storage, latency, distribution and the drift defect together. It is
the input to the ASK milestone and is built first.

## 11. change scopes

Three scopes of change with different blast radii. The distinction determines what is stored,
what is invalidated, and which comparison is required before a result is admissible.

| scope | example | what changes | what re-runs | routing table |
|---|---|---|---|---|
| **within-method** | retune an existing route's hyperparameters | run configuration only | that route | a value moves inside an existing row |
| **new-method** | add an alignment route | a configuration and a grid position | that route, then the full comparison | **a new row**, requiring evidence against all existing routes |
| **new-modality** | add an encoder | the waist's schema and the role map | everything downstream | possibly new rows and new question classes |

The optimisation loop operates at the **first** scope only. That is what makes its constraint -
the scalar metric cannot change within an instance - coherent: at within-method scope the metric
is fixed by definition. At new-method scope the activity is not optimisation but admission, and
the full grid is the test. Conflating the two is what makes the loop appear underspecified.

Storage follows the same split: within-method variants are siblings under the same route; a
new-method candidate gets its own route directory plus a routing-table entry that cannot be
written until the comparison exists.

## 12. filesystem and distribution

MCP servers are not distributed by `pip install`. The package is published to a package index and
launched ephemerally; a self-contained bundle serves non-technical users.

| root | contents | notes |
|---|---|---|
| package | code, `knowledge/*.json`, templates | ships in the wheel; the knowledge tables are the transferable asset |
| project | cohort config, inputs, embeddings, runs, decisions | selected by `--project-dir`, which is both the security boundary and the cohort selector |
| cache | encoder weights, hashed UMAP caches | user-level, shared across cohorts |

Installation creates nothing and downloads nothing; scaffolding is an explicit command. Two
consumption profiles: an evidence-bundle profile requiring no accelerated compute and no raw data,
and a full-cohort profile. The bundle profile requires a dependency split isolating the read path
from tensor libraries.

## 13. new-cohort admission

### the contract specifies roles, not a schema

The modality contract requires a cohort to satisfy the **twelve roles** in §3 - key, confounder
axis, provenance, imaging feature pooled and un-pooled, molecular feature, pathway feature,
supervision, and four label kinds. It does not require the current column names, dimensions, or
file layout.

| if the contract specifies | consequence |
|---|---|
| the table as it exists today | every future cohort inherits a layout that is 81% a column most consumers do not read, and inherits it permanently |
| the twelve roles | each cohort satisfies them however its encoders produce them; the co-location defect stops propagating |

This is the difference between a contract and a snapshot, and it is the one long-term decision in
this document. Everything else in §3 is housekeeping.

### the gate chain

The chain the project already executed once, made explicit. Consumes a cohort profile and emits a
decision record.

```
cohort profile
  -> encoder-tissue compatibility lookup      { validated | constrained | uncertain }
  -> data gate (blocking)                     { proceed | caution | stop }
       spatial autocorrelation result selects the molecular encoder branch
  -> platform floor checks                    encoder-specific minimum unit counts
  -> encoder QC probe                         linear probe against a systematically derived label
  -> preprocessing ablation                   per encoder, empirically validated, never inherited
  -> aggregation unit                         determines admissible fusion strategies
  -> label construct validity                 determines which hypotheses are answerable
  -> decision record
```

Escalation attaches at three boundaries: uncertain compatibility, a cautionary gate verdict, and a
borderline guard result.

## 14. open decisions

| decision | blocks | position |
|---|---|---|
| which routing row leads the demonstration | ASK scope | undetermined. Row 7 demonstrates principled refusal; row 1 is the conventional opener |
| graph shape - stateful graph throughout, or stateful graph for gates with deterministic lookups behind it | ASK | mixed |
| knowledge source - versioned JSON as single source, documents rendering from it | ASK | adopt; the routing table currently exists in five locations and has diverged |
| durability timing | RERUN | defer; the integration plugin means the graph is not authored twice under either choice |
| modality contract scope - minimal data contract or full gate chain | EXTEND | full |

## 15. deferred

Twelve evidence-integrity items block publication rather than demonstration: spatial
autocorrelation computed on a single subarray; an unreconciled cohort count appearing as three
values; two null gate fields under a passing verdict; an unrecomputed coverage figure; a
resolution mismatch between two extraction notebooks; and documentation drift. The applicable
filter is whether a demonstration path displays the value.

Also deferred: vector and graph stores, the alternative imaging encoder swap, one queued alignment
variant, DAG-resolution pathway analysis on the current grid, and HEST as external validation.

Two items require closing before the corresponding claim is made: the optimisation loop's cycle
length is undefined, which by its own definition leaves no instance well-formed; and the two
topology corrections in §3 are absent from `proposal_deviations.md`.