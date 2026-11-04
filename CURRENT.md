# CURRENT

the one task in flight. read this at the start of every session, before any
design doc and before any code.

this file holds ONE task. when it is done, it is replaced, not appended to.
everything that is not the one task goes to the parking lot at the bottom and
is not worked on, discussed, or measured until the task card is closed.

---

## the task

**one schema-driven input contract consumed by one real adapter.**

parent goal: a supported question -> required tasks -> grounded result,
notebook and provenance. the schema contract is the first slice of that path.

previous task card (v1.3.0 build, 5 items done 2026-10-20) preserved at
`design/_scratch/CURRENT_v1.3.0.md`.

three layers, kept separate:

1. **biomedical schema** - declares Representation (encoder, modality, dim,
   derivation lineage), observation identities and annotations. no column
   names, no cohort specifics
2. **source mapping** - binds actual source columns and files to schema
   entities. one mapping per cohort. no analytical decisions
3. **analytical binding** - selects roles for the requested method.
   supervision, split grouping, evaluation target are bindings, not data
   properties

tnbc-92 and an alternate fixture (different encoder, dimensions, column names)
are test instances only. no cohort-name branches, encoder-name special cases,
inferred column-name conventions or silent tnbc defaults in production code.

### progress

**first adapter slice landed** (2026-11-03, branch `checkpoint-1-schema-normalization`).

committed files:
- `src/omicstra/adapters/alignment_data.py` - AlignmentContract, from_project_dir,
  dimension validation, refusal on missing declarations
- `src/omicstra/protocols/align.py` - project_dir passthrough to both chains
- `tests/test_classical_chain.py` - fixture tokens corrected to 7*HE_DIM,
  explicit project declarations, project_dir on all chain calls
- `tests/test_alignment_contract.py` - 19 tests: seed+alt contracts, refusal,
  dimension mismatch, load with both fixtures
- `design/schema_standards.md` - standards decision (pre-step)

40/40 focused tests pass. no paid calls.

staged but not committed: `docs/open_items.md` (de-novo labeling design, separate scope).

**next action (2026-11-04):** `protocols/join.py` SUPERVISION_ONLY / FEATURE_COLUMNS
(WS 3, same order) - done. `assert_no_circular_supervision` takes declared
`supervision_set`; `join_funnel` reads and passes it; params_space updated.
82/82 join tests pass.

remaining for WS 3: none - both alignment_data.py and join.py hardcoded columns
are now declaration-driven.

WS 3 complete.

**WS 4 (2026-11-04):** analytical bindings done.

- `contracts/project.py`: h2_label, supervision, headline_view defaults removed
  (`str | None = None`). tnbc-92 project.json already declares all three - no
  data file change needed. a second cohort that omits any of them no longer
  silently inherits seed-cohort values.
- `protocols/evaluate.py:gate_h3_pathway`: `view` parameter no longer defaults
  to "z_he". added `clean_view` parameter; `_clean` local resolves which view is
  non-circular. the two `r["view"] != "z_he"` / `view != "z_he"` checks now use
  `_clean`. `_h3_pathway` reads view/clean_view from task + sources declarations.
- `protocols/evaluate.py:pathway_transfer_cca`: `_cln` resolves from declared
  `headline_view` via `_edeclared(ctx, "headline_view")`. the loop circularity
  check and guard note strings use `_cln`, not the hardcoded literal.
- `tests/test_evaluate_reachable.py`: assertion updated - `cfg.h2_label is None`
  when not declared (the trap the test named is now fixed, not just documented).

WS 4 complete. next: WS 5 - question.validate() contract consumption.

what is NOT done by this commit:
- full LinkML schema (WS 2)
- HTAN2/schema_crush integration (WS 2-3)
- question-to-answer flow using the contract
- evaluation adapter normalization
- second cohort evidence pack (WS 10)

### pre-step: standards decision

written: `design/schema_standards.md`, bounded to first adapter slice.

two existing foundations:

- **schema_crush** (`teslajoy/schema_crush` @ ad04ded) - curated
  HTAN/GDC-to-FHIR mappings, field and content mapping loaders, cached
  mapping database, matchers. the mapping foundation for omicstra. reuse
  applicable mappings deterministically at runtime; do not introduce
  mandatory model calls or rebuild the mapping framework
- **HTAN2** (`ncihtan/htan2-data-model` @ 0f13d60) - LinkML modules for
  clinical data, biospecimens, imaging, digital pathology and Spatial Omics.
  the SpatialOmics module covers Visium, Xenium, CosMx and other platforms,
  processed bundles, interoperable h5ad/RDS/Zarr, panel info, QC and
  processing metadata. spatial transcriptomics metadata is not absent from
  the standard

the standards decision must identify:

1. the exact reusable HTAN2 definitions for the first adapter slice, with
   pinned versions. distinguish older HTAN mappings (bts: terms, incomplete
   entries) from current HTAN2 coverage
2. which schema_crush mappings apply to this slice, and whether they cover
   current HTAN2 fields or only older HTAN
3. what omicstra must define itself: the analytical contracts not covered by
   either foundation - observation alignment, neighborhood membership,
   embedding dimensions, analytical lineage (Representation with derivation),
   task-specific role bindings
4. schema_crush's runtime role: reuse reviewed mappings deterministically.
   optional mapping assistance does not run during every analysis. FHIR
   remains an interoperability mapping where appropriate, not the storage
   format for expression matrices or embeddings

direction: HTAN2 definitions + applicable schema_crush mappings -> validated
omicstra inputs -> task-specific analytical bindings.

do not invent a parallel biomedical standard. do not assume LinkML itself
supplies the domain model - it is a schema language, not a vocabulary.

bounded to the first adapter slice.

### acceptance

- one real adapter (alignment_data or evaluation) loads both tnbc-92 and an
  alternate fixture (different encoder, dimensions, column names) using their
  declarations
- returns correctly aligned arrays with declared dimensions
- missing or incompatible inputs reported precisely, every problem named
- `question.validate()` uses the same contract for parameter resolution
- generic column access and supported encoder dispatch are fine. no
  cohort-specific special cases, no encoder-name branches, no inferred
  column-name conventions

### stopping condition

actual adapter consumption on both fixtures: declaration -> source mapping ->
loaded arrays with correct shapes. not validation alone.

### not in scope

full LinkML entity coverage, every adapter, upstream discovery, artifact
persistence, second-cohort evidence.

---

## what shipped in v1.3.0

tag on main (af19798), GitHub release. not yet on PyPI (latest there: 1.2.0).

build steps A-E from `design/v1_3_scope.md`, all done:

- `omicstra worker` - encode queue process
- `evaluate_run` behind run handle; verdicts per hypothesis
- one record store, records reach the report
- level 0 wires align and evaluate
- Bedrock end-to-end test (7 tool calls, 0 mismatches)

also:

- H2-C biology scoring ported to measures/
- H3 pathway CCA through graph with cached permutations
- question-to-execution path restored
- notebook generation for all 7 units
- pathway_assay bridge node with flag-driven artifact selection
- pytest markers (paid_api, resource_intensive)
- version consistency enforced across 4 files

## what did NOT ship

- **second cohort evidence pack** - the v1.3 defining claim, not delivered.
  tracked: completion_roadmap WS 10
- **pathway artifact persistence** - bridge in `src/omicstra/graph.py` is
  wired, AUCell chain demonstrated (flag-driven selection, refusal on missing
  inputs, per-sample records, notebook generation). computed scores exist only
  in graph state (explicit TODO in code). downstream consumption by the
  selected analysis not demonstrated. tracked: completion_roadmap WS 6
- **generalizable schema** - the three-layer architecture is the
  completion_roadmap's subject. existing adapters and declarations provide
  partial implementations; the shared schema and consistent integration remain
  unfinished. see implementation status in completion_roadmap.md

---

## forward docs

| doc | role |
|-----|------|
| `docs/capability_report.html` | 10-area capability-to-deliverable table, MCP surface, proposal comparison |
| `design/completion_roadmap.md` | 11 workstreams, approved for checkpoint 1 |
| `docs/open_items.md` | 39+ items with status, the full gap ledger |
| `design/graph.md` | which file owns which step |
| `design/v1_3_scope.md` | what shipped (build A-E), historical record |

---

## parking lot

- cohort two evidence pack (the release claim)
- publish 1.3.0 to PyPI
- gpath2vec fix/v3.0.1 (verified, not merged)
- .zenodo.json description (still describes v1.1 features)
- CITATION.cff date-released (still 2026-10-02)
