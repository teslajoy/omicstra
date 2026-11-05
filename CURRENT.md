# CURRENT

the one task in flight. read this at the start of every session, before any
design doc and before any code.

this file holds ONE task. when it is done, it is replaced, not appended to.
everything that is not the one task goes to the parking lot at the bottom and
is not worked on, discussed, or measured until the task card is closed.

---

## the task

**capability matrix: supported questions × required data × workflow × limitations.**

parent goal: a researcher picks up omicstra and knows immediately what it can
and cannot answer, what data it needs, and what it refuses. the matrix is the
single source of truth for external claims (conference talks, the README's
"what it cannot do yet" section, the evidence pack's routing decisions).

previous task card (checkpoint-1 schema + normalization, 5 workstreams done
2026-11-04) preserved at `design/_scratch/CURRENT_checkpoint1.md`.

this is completion_roadmap WS 1.

### what to build

a document at `docs/capability_matrix.md` with one row per supported question
type, columns:

| question type | task family (routing_contract.json id) | required data | required declarations | method | output | limitations | proposal requirement |

populated from:

1. `src/omicstra/configs/routing_contract.json` - 7 task families, their
   required steps and accepted params
2. `projects/tnbc-92/project.json` + `cohort.json` - what a cohort must declare
3. `design/completion_roadmap.md` - 11 workstreams, known gaps
4. `docs/open_items.md` - 39+ items with status
5. `docs/papers/Sanati_2025_proposal.pdf` or the proposal text - H1/H2/H3
   original requirements
6. `projects/tnbc-92/proposal_deviations.md` - what changed from the proposal

### acceptance

- one row per task_family in `routing_contract.json` (7 rows minimum)
- every row has: required data format, required declarations, the graph nodes
  it invokes, the output it returns, its limitations
- proposal requirements traced: H1/H2/H3 mapped to task families, deviations
  noted where present
- "not supported" rows: at least cross-modal retrieval on a second cohort
  (WS 10), de-novo labeling (WS 7), SLURM (not built)
- the README "what it cannot do yet" section updated to reference the matrix

### stopping condition

`docs/capability_matrix.md` written and committed. README updated. no code
changes required or in scope.

### not in scope

LinkML schema (WS 2), code changes, new adapters, test runs, any paid calls.

---

## checkpoint 1 summary (closed 2026-11-04)

**branch**: `checkpoint-1-schema-normalization`

five workstreams delivered:

- **WS 3 source normalization**: `adapters/alignment_data.py` AlignmentContract
  + `from_project_dir()`; `protocols/join.py` supervision_set declaration-driven.
  hardcoded column names and dimensions replaced by declared roles.
- **WS 4 analytical bindings**: `contracts/project.py` h2_label/supervision/
  headline_view defaults removed (None). `protocols/evaluate.py` gate_h3_pathway
  and pathway_transfer_cca resolve clean view from declaration, not "z_he" literal.
- **WS 5 question contract**: `adapters/evaluation.py` missing_declarations()
  soft check; `question.py` validate() calls it and appends to AnalysisRequest.missing.
  same REQUIRED contract in both pipeline and question paths.
- **design/schema_standards.md**: standards decision written and completed.
- **40/40 alignment tests + 19/19 contract tests + 82/82 join tests** pass.
  no paid calls.

stopping condition met: declaration -> source mapping -> loaded arrays with
correct shapes on both tnbc-92 and alternate fixture.

not delivered in checkpoint 1 (tracked in completion_roadmap):
- full LinkML schema (WS 2)
- question.execute() entering through discover() (WS 5 remainder)
- pathway artifact persistence (WS 6)
- second cohort evidence pack (WS 10)

---

## parking lot

- cohort two evidence pack (the v1.3 release claim, WS 10)
- publish 1.3.0 to PyPI
- gpath2vec fix/v3.0.1 (verified, not merged)
- .zenodo.json description (still describes v1.1 features)
- CITATION.cff date-released (still 2026-10-02)
- LinkML schema (WS 2)
- question.execute() -> discover() wiring (WS 5 remainder)