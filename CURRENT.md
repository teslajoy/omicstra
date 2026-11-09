# CURRENT

the one task in flight. read this at the start of every session, before any
design doc and before any code.

this file holds ONE task. when it is done, it is replaced, not appended to.
everything that is not the one task goes to the parking lot at the bottom and
is not worked on, discussed, or measured until the task card is closed.

---

## the task

**linkml schema v0.1.1: biomedical entities for cross-modal spatial biology.**

parent goal: a cohort's data has a declared schema, not an implicit one baked
into column names. Representation + DerivationRecord carry the meaning that
was hidden in naming conventions. circularity is a lineage traversal, not a
string comparison on view names.

previous task card (WS 1 capability matrix, done 2026-11-06) preserved at
`design/_scratch/CURRENT_ws1.md`.

this is completion_roadmap WS 2.

### what was built

`src/omicstra/schema/omicstra.yaml` — version 0.1.1.

key entities: Participant, Biospecimen, TissueSection, Assay, CoordinateFrame,
SpatialUnit, DerivationRecord, Representation, Annotation, AnnotationProvenance,
DataFile. container: CohortPack (tree_root).

fixes applied from linkml-lint + gen-pydantic review of 0.1.0 draft:
- B1: block-scalar all descriptions (parses cleanly)
- B2: CohortPack tree_root for FK integrity
- B3: typed FK ranges
- G2: supervision_annotation_ids on DerivationRecord
- G3-lite: UnitLevel enum + unit_level required on Representation
- enforcement: minimum_value, pattern, rules block for aligned -> run_id

### acceptance criteria

- [x] valid seed-cohort example: `tests/fixtures/schema/tnbc92_minimal.yaml`
- [x] valid alternate-cohort example: `tests/fixtures/schema/alternate_cohort_minimal.yaml`
  (UNI2 1536d, donor_id, visium, k=4 — no code changes)
- [x] 36/36 structural tests pass with PyYAML only (no linkml install)
- [ ] `linkml-lint` passes (requires `pip install linkml>=1.8`)
- [ ] 10 must-reject probes pass: `python -m pytest tests/test_schema_validation.py`
- [ ] invalid examples rejected with every problem named

### stopping condition

`linkml-lint` clean + `test_schema_validation.py` all pass + branch merged.

### not in scope

source mapping changes (WS 3, done in checkpoint 1), code path changes,
second cohort evidence pack (WS 10), evidence schema (omicstra_evidence.yaml,
deferred to 0.2.0).

---

## checkpoint 1 summary (closed 2026-11-04)

**branch**: `checkpoint-1-schema-normalization`

five workstreams delivered:

- **WS 1 capability matrix**: `docs/capability_matrix.md` — 7 task-family rows,
  not-supported table, H1/H2/H3 map, P-items noted. README updated. committed
  to main 2026-11-06 (554bb44).
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

---

## parking lot

- cohort two evidence pack (the v1.3 release claim, WS 10)
- publish 1.3.0 to PyPI
- gpath2vec fix/v3.0.1 (verified, not merged)
- .zenodo.json description (still describes v1.1 features)
- CITATION.cff date-released (still 2026-10-02)
- question.execute() -> discover() wiring (WS 5 remainder)
- omicstra_evidence.yaml (evidence/routing schema layer, WS 2 deferred)