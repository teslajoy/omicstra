# omic<span style="color:#5DCAA5">stra</span>

an MCP server for cross-modal reasoning in spatial biology. it exposes fixed scientific protocols as tools and returns evidence; the client holds the model. foundation model encoders are pluggable; integration strategy is a configurable experimental dimension, not a fixed pipeline choice.

the seed implementation evaluates 92 TNBC patients from [Wang et al. 2024](https://www.nature.com/articles/s41467-024-54145-w) using H&E morphology (Virchow2 primary, UNI2 swap) and spatial transcriptomics (Novae GNN).

`v0.1` `ResearchHub Foundation grant` `Brenden-Colson Center / Sears Lab, OHSU` `MIT`

**v0.1 is:** the inventory and EDA protocols, the admissibility gate with its
human-in-the-loop interrupt, stdio and streamable-http transport, and a record
contract - tested on two cohorts with different platforms, formats and
declarations, and no code changes between them. **Alignment and evaluation run as
scripts and are not yet protocols.** The v3 results below were produced by those
scripts.

---

## hypotheses

| | | metrics |
|---|---|---|
| H1 | aligned embeddings retrieve better cross-modal matches than any unaligned baseline | Recall@K - MRR - CKA |
| H2 | manifold alignment preserves biological structure better than either modality alone | ARI - silhouette - UMAP |
| H3 | shared space encodes interpretable biological pathway signals | CCA - spatial maps |

early vs late fusion is an experimental variable, not an assumption. 

---

## seed cohort results (v3, 10-run grid)

live report: **[teslajoy.github.io/omicstra/reports/internal/tnbc-92/](https://teslajoy.github.io/omicstra/reports/internal/tnbc-92/)**

evaluated on **35,594 niches across 14 held-out TNBC patients** (zero patient leakage across train / val / test). 6 contrastive runs (R1-R6) + 4 classical baselines (B1-B4):

- **H1 (cross-modal retrieval).** R4 wins. AUC 0.859 (95% CI 0.855-0.862). R4 is the only run using cross-attention - the other 5 contrastive runs use late fusion (mean-pooled H&E + independent MLPs).
- **H2 (biology preservation).** Split outcome. Late-fusion contrastive (R1, R6) wins K-means MC ARI; R4 still encodes MC biology at linear-probe parity. KMeans-ARI is geometry-biased - load-bearing diagnostic is the linear probe.
- **H3 (pathway transfer).** R4 wins again. 15/15 cells BH-FDR < 0.05 cross-patient (Welch z 3.8-5.7). At DAG resolution, R4 stays best (78% sig nodes) on clinically interpretable hits: BTLA checkpoint, IL signaling, IFN regulation, ECM collagen biology.

**R4 = the only non-late-fusion run** (cross-attention between the niche's 7 Virchow2 tile tokens and the 576-d ST vector). H1 + H3 both reward keeping local morphological heterogeneity un-pooled. See report for the full proposal-alignment matrix and 95% CIs.

---

## architecture

three layers:

```
layer 0: EDA (gate)
  is data usable? quality? signal? modalities alignable?
  -> proceed / proceed with caution / stop

layer 1: pipeline (execution)
  embed -> align -> eval
  fixed protocol, deterministic with seed

layer 2: karpathy loop (outer optimization)
  editable asset: alignment_config.json
  scalar metric: Recall@K or MRR (fixed)
  proposes config -> runs pipeline -> scores -> keeps or discards
  output: git log of validated decisions + winner.json
```

program.md defines search space, constraints, stopping criteria.

embedding dimensions: Virchow2 1280d (primary) / UNI2 1536d (alternative) - Novae GNN 64d (novae_latent) -> shared 512d

alignment strategies (experimental variable):
```
  late interaction  - tiles aggregated per spot (mean/max),
                      then independent MLPs -> shared 512d (InfoNCE)
  early interaction - tile-level cross-attention between H&E tokens
                      and ST spot before projection (~4M params, local K tiles)
  comparison: does tile-level morphological detail lost at aggregation
              matter for cross-modal alignment quality?
```

---

## repo structure

audited against the filesystem 2026-08-20. `·` is built, `○` is named in the design and **not yet built** - kept here because the name is referenced elsewhere, not because it exists.

```
omicstra/
│
· CLAUDE.md                          # north star for Claude Code
· EDA.md                             # gate - data quality checks before pipeline
· PLAN_RULES.md                      # constraints - no alignment until EDA passes
· MODELS.md                          # training provenance + tissue compatibility per model
○ MODALITY_TEMPLATE.md               # how to add a new modality - cited by README + CLAUDE.md
· .mcp.json                          # MCP server entry
· .env.example                       # LANGSMITH_* only - backends are cohort declarations
│
· design/                            # the planning tree - tracked, this is where "what next" lives
│   · mcp_plan.md                    # the engineering plan
│   · workflow.md                    # the executable graph - what runs, in what order
│   · decisions.md                   # the decision graph - what is decided and by whom
│   · progress.md  status.md         # build log - repo snapshot
│   · _scratch/                      # BACKLOG.md ("this is the plan") - ROADMAP - doc_drift_audit
│
· src/omicstra/                      # the pip-installable package - src layout
│   · graph.py                       # LEVEL 0 - the only entry, the only checkpointer
│   ○ graphs/                        # LEVEL 1 - one file per gate, each can interrupt()
│   │   ○ eda.py  encode.py          # <- eda_graph.py  agents/graph.py - not moved yet
│   │   ○ route.py  promote.py       # named in the design - not split out yet
│   · protocols/                     # LEVEL 2 - ORDER, applicability, authority. no maths
│   │   · inventory.py               # 8 steps: files .. bind - "what is this data"
│   │   · eda.py                     # 10 checks registered - "is it usable"
│   │   ○ align.py  evaluate.py      # <- stages.py / guards.py - signatures only
│   · measures/                      # the MATHS. pure functions, no ctx, no order,
│   │                                #   no authority. callable from a notebook.
│   │                                #   organised by what is computed, never by
│   │                                #   which protocol calls it
│   · contracts/                     # READS a declaration, decides nothing
│   │   · project.py                 # <- config.py - a cohort's project.json
│   │   · routing.py  eda.py         # contract + evidence · contract + calibration
│   · adapters/                      # a cohort's files -> AnnData conforming to raw_counts
│   │   · wang_st.py  ○ hest.py
│   · agents/modality.py             # modality agents - encoder capability metadata
│   · mcp/server.py                  # MCP server - read path, zero model calls
│   · records.py  artifacts.py       # the record contract - inventory/eda_summary io
│   · settings.py  cli.py            # omicstra inventory | eda | serve
│   · eda_graph.py  agents/graph.py  # LEVEL 1 today, pending the move into graphs/
│   · routing.py  eda.py  eda_steps.py  guards.py  stages.py  config.py  figures.py
│
│   the rule: a thing gets its own graph only if it can ask a human;
│   everything else is a protocol. a new .py never lands in src/omicstra/ directly.
│
· configs/                           # contracts are generalizable, evidence is per-cohort
│   · eda_contract.json              # which EDA steps, which criteria
│   · routing_contract.json          # task taxonomy, outcome vocabulary, rules
│   · data_contract.json             # 8 declared roles - bind checks conformance
│   · v3/                            # R1_v3 .. R6_v3 run configs
│   ○ qc_params.json                 # K1 editable asset - not built
│   ○ alignment_config.json          # K2 editable asset - not built
│
· projects/                          # committed
│   · registry.json
│   · {project_id}/
│       · project.json  program.md   # karpathy loop - search space + constraints + metric
│       · cohort.json                # DECLARED: subject_id_column, compute_backend,
│       │                            #   model_backend, data_classification - fail closed
│       · platform.json              # DECLARED per sample: platform, position_columns, pitch
│       · inventory.json             # inventory protocol output - conformance report
│       · eda_summary.json           # EDA gate output
│       · routing_evidence.json      # this cohort's measured evaluation - never inherited
│
· site/{institution}/  [ignored]     # compute.md governance.md - question -> answer,
│                                    #   served as MCP resources, never parsed
· scripts/                           # the chain behind the published results
· tests/                             # test_eda_authority.py - test_routing.py
· knowledge/                         # shared domain assets - committed
│   · reactome/  pathway_commons/
│
· docs/                              # -> teslajoy.github.io/omicstra
│   · index.html  timeline.html  images/  reports/internal/{project_id}/
│   · TODO_*.md                      # parked work - untracked
│
· notebooks/                         # exploration - embeddings - experiments - final
· runs/            [ignored]         # {project_id}/{run_id}/ - run_config, metrics, embeddings
│   ○ winner.json                    # best validated run record - not written yet
· data/            [ignored]         # inputs/ - embeddings/
· demo/            [ignored]         # talk assets, figure scripts
· logs/  venv/  venv_test/
│
○ .claude/skills/  .claude/hooks/    # named in the design - not built (.claude/ holds settings only)
○ compute/slurm/                     # optional array jobs - not built
```

---

## extending to a new modality

`MODALITY_TEMPLATE.md` (**planned, not yet written**) is the contract a new modality agent (single-cell, protein, electron microscopy, etc.) must satisfy: a frozen foundation-model encoder, a niche-aggregation step, a clean ST/H&E-equivalent feature parquet shape, and the construct-validity guards the eval agent enforces. each modality adds one row to the alignment grid; the orchestrator's routing table and the karpathy-loop search space pick it up automatically.

---

## seed cohort

[Wang et al. 2024](https://www.nature.com/articles/s41467-024-54145-w) - 92 TNBC patients, co-registered ST + H&E WSI. platform: original Spatial Transcriptomics (Stahl et al. 2016, KTH/Spatial Transcriptomics AB, acquired by 10x Genomics 2018) - 1934 spots/array, 100um diameter, 150um center-to-center. not 10x Visium.

> corrected 2026-08-12: previously read 200um center-to-center. Wang 2024 Methods states 150um and the lattice measures to it - nearest neighbour 161.6 px with the (+2,0)/(0,+2) offsets at 216/240 px, i.e. NN x sqrt(2), so NN is centre-to-centre. at 150um that is 0.93 um/px and the array spans 6.5 x 6.9 mm, matching the stated capture area; at 200um it would span 8.7 x 9.2 mm, which no ST array is.

| resource | url |
|---|---|
| processed data | [doi:10.5281/zenodo.14204217](https://doi.org/10.5281/zenodo.14204217) (mc-labels release also at [doi:10.5281/zenodo.8135721](https://doi.org/10.5281/zenodo.8135721)) |
| original analysis code | [BCTL-Bordet/ST](https://github.com/BCTL-Bordet/ST) |

---

## quick start

---

## stack

| layer | tool |
|---|---|
| orchestration | LangGraph StateGraph |
| routing | Claude Sonnet 4.6 |
| synthesis | Claude Opus 4.6 |
| tracing | LangSmith - observability only |
| H&E foundation model | Virchow2 (primary, 1280-d) - UNI2 as swap (1536-d) |
| ST foundation model | Novae GNN |
| pathway | gpath2vec + biological pathway embeddings |
| alignment | contrastive MLP + cross-attention bridge |
| compute | laptop / cloud GPU / SLURM (ARC HPC at OHSU) |
| vector retrieval | Qdrant *(planned - retrieval is currently exact cosine over the run's parquet)* |
| graph store | Neo4j *(planned - not yet used)* |
| interface | Model Context Protocol |

---

## reproducibility

every run writes `runs/{project_id}/{run_id}/` - config, QC, embeddings, alignment, evaluation, synthesis. `notebooks/final/` contains the best run in notebook which is Opus-generated from the LangSmith trace, reruns can be done top-to-bottom with fixed seed.

---

## deliverables

- [x] v3 10-run alignment grid on TNBC-92 ([live report](https://teslajoy.github.io/omicstra/reports/internal/tnbc-92/))
- [ ] pip-installable MCP server (MIT)
- [ ] second-modality plug-in via `MODALITY_TEMPLATE.md` (target: CODA or CyCIF)

---

## how the evidence pack is produced

The router reads `projects/{id}/routing_evidence.json`. **No script writes it.**

```
eval summaries  ──▶  proposed evidence   deterministic: winners, margins, refusals
                ──▶  a person reviews, edits the notes, approves
                ──▶  routing_evidence.json
```

Deciding which metric answers which task family, what the floors are, and which
caveat attaches to which candidate is editorial work - the file's own notes
argue with the report in places. Generating it would invent the judgements it
records, so `run_eval` scores the grid and **resolves** the pack; it never
produces one.

**In v0.1 that promotion step is manual and recorded in `docs/`. In v0.2 it is
`omicstra promote`** - a gate, not a protocol, because it asks a person.

The acceptance test for the port is pinned in `tests/fixtures/`: a full-grid
rerun with one retrained baseline reproduced every numeric field of the H1
summary, and the fixture is checked against the evidence pack so the router can
never quote a number nothing produced.

---

## declared deviation from the proposal

The registered report put a model **inside** the orchestration: one model for
intent classification and routing, another for hypothesis evaluation, with
conditional routing in the graph.

**The build moved the model to the client and made routing a deterministic rule
over evidence.** The server holds no model, generates no code, and executes
nothing a model wrote. A model has three narrow jobs, all on the client side:
classify a question into a closed task set, choose among enumerated options at a
gate, and write the "why". Everything downstream resolves from lookup tables.

The reason is reviewability. A server with no model is portable across clients,
has no second router to reason about, and can be reviewed for what it does rather
than what it might decide. It also means swapping the model cannot change a route
- the same question reaches the same method, because the method came from
recorded evidence rather than from inference.

**Deliverables, metrics and hypotheses are unchanged.**

---

## team

**Nasim Sanati, M.S.** - AI/ML systems - MCP server - agent orchestration - embedding pipelines  

Brenden-Colson Center for Pancreatic Care / Sears Lab, OHSU

---

## citation

```bibtex
@software{sanati2026omicstra,
  author  = {Sanati, Nasim},
  title   = {omicstra: cross-modal biomedical embedding alignment + reasoning via MCP orchestration},
  year    = {2026},
  url     = {https://github.com/teslajoy/omicstra},
  license = {MIT}
}
```

*funded by ResearchHub Foundation ([doi:10.55277/researchhub.6ou1w3h3](https://doi.org/10.55277/researchhub.6ou1w3h3))*  
*nonprofit recipient: OHSU Foundation - EIN 23-7083114*  
*seed cohort: Wang et al. 2024, Nat. Commun. 15:10232*
