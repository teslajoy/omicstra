# CURRENT

the one task in flight. read this at the start of every session, before any
design doc and before any code.

this file holds ONE task. when it is done, it is replaced, not appended to.
everything that is not the one task goes to the parking lot at the bottom and
is not worked on, discussed, or measured until the task card is closed.

---

## the task

**one schema-driven input contract consumed by one real adapter.**

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

### pre-step: standards decision

before writing schema code, document in `design/schema_standards.md`:

- what we reuse from HTAN (entity definitions, controlled vocabularies, field
  conventions)
- what maps to FHIR (observation model, specimen hierarchy, if anything)
- what omicstra must define itself (Representation with derivation lineage,
  CoordinateFrame, analytical role bindings - things no existing standard
  covers)
- whether linkml-schemauto / schemasheets / schema_crush is tooling (a build
  step that generates validators) or a model dependency (imported at runtime,
  required by the wheel)

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
- no code path inspects cohort name, encoder name or column name to decide
  behavior

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
- **pathway artifact persistence** - bridge is wired, AUCell chain
  demonstrated (flag-driven selection, refusal on missing inputs, per-sample
  records, notebook generation). computed scores exist only in graph state
  (explicit TODO in code). downstream consumption by align not demonstrated.
  tracked: completion_roadmap WS 6
- **generalizable schema** - the three-layer architecture is the
  completion_roadmap's subject. existing adapters and declarations provide
  partial implementations; the shared schema and consistent integration remain
  unfinished. see implementation status in completion_roadmap.md

---

## forward docs

| doc | role |
|-----|------|
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
