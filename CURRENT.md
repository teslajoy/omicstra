# CURRENT

the one task in flight. read this at the start of every session, before any
design doc and before any code.

this file holds ONE task. when it is done, it is replaced, not appended to.
everything that is not the one task goes to the parking lot at the bottom and
is not worked on, discussed, or measured until the task card is closed.

---

## the task

**analytical bindings: encoder selection and pathway targets from declarations.**

parent goal: a cohort with a different primary encoder or a different set of
pathway targets requires a declaration entry, not a code change.

previous task card (WS 3 source normalization, 2026-11-09) preserved at
`design/_scratch/CURRENT_ws3.md`.

this is completion_roadmap WS 4 (remaining).

### what was done in checkpoint 1 (partial WS 4)

- `contracts/project.py`: h2_label, supervision, headline_view are `None` —
  refused if absent, no silent defaults.
- `protocols/evaluate.py`: circularity checks read declared headline_view
  (not hardcoded `z_he`). gate_h3_pathway resolves clean view from declaration.
- `adapters/evaluation.py`: REQUIRED + `declared()` pattern — refuses with
  every missing field named.
- `question.validate()`: same REQUIRED contract as the evaluation pipeline.

### what remains

two items. both verified in code 2026-11-09:

**item 1: `graphs/encode.py` encoder defaults.**
5 places with `.get("encoder", "virchow2")`. when the state carries no encoder,
the graph silently uses virchow2. should read from the cohort's declared primary
encoder instead (project.json#encoders[role=primary|morphology].name).

files to update:
- `src/omicstra/graphs/encode.py` — replace `.get("encoder", "virchow2")` with
  `_resolve_encoder(state, project_dir)` that reads the declared primary encoder
  and refuses when none is declared with role primary or morphology.

**item 2: `NAMED_PATHWAY_TARGETS` in `adapters/evaluation.py`.**
5 Reactome pathway IDs hardcoded at line 457. a cohort using different pathways
requires a code change. should come from `project.json#pathway_targets`.

files to update:
- `src/omicstra/adapters/evaluation.py` — `pathway_cosines()` reads
  `project.json#pathway_targets` (list of {name, reactome_id}). absent =
  refused with field named. `NAMED_PATHWAY_TARGETS` becomes the seed cohort's
  declared value, not a package constant.
- `projects/tnbc-92/project.json` — add `pathway_targets` list (5 entries,
  same IDs as current constant).

### acceptance criteria

- [ ] `graphs/encode.py` selects encoder from declaration, not `.get(..., "virchow2")`.
  an alternate cohort with uni2 as primary encoder uses uni2 without code changes.
- [ ] `pathway_cosines()` reads pathway targets from `project.json#pathway_targets`.
  missing declaration refused with field named.
- [ ] seed cohort: same 5 pathways as today (IDs identical).
- [ ] alternate fixture (no pathway_targets declared) is refused, not silently empty.
- [ ] tests: `tests/test_analytical_bindings.py`. no paid calls.

### stopping condition

both encoder selection and pathway targets are declaration-driven. tests pass
locally before any commit. branch merged.

### not in scope

question.execute() -> discover() wiring (WS 5 remainder), pathway artifact
persistence (WS 6), evidence schema deferred to 0.2.0.

---

## checkpoint 3 summary (closed 2026-11-09)

**branch**: `checkpoint-3-source-normalization` (pending merge)

one workstream delivered:

- **WS 3 source normalization (remaining)**: `SourceMapping` dataclass +
  `load_from_yaml()` in `adapters/source_mapping.py`. `source_mapping.yaml`
  per cohort replaces the `{encoder_name}_niche` naming convention in
  `AlignmentContract.from_project_dir()`. seed cohort yaml declares same names
  convention would derive; alternate fixture (uni2_embeddings, uni2_tile_tokens,
  slide_id) proves non-conventional naming works without code changes.
  convention fallback kept for existing tests. 12 new tests, 1346 total pass.

---

## checkpoint 2 summary (closed 2026-11-09)

**branch**: `checkpoint-2-linkml-schema` (be7d0fd)

one workstream delivered:

- **WS 2 linkml schema**: `src/omicstra/schema/omicstra.yaml` v0.1.1.
  36/36 structural tests (PyYAML only) + 13/13 linkml probe tests. two cohort
  fixtures: `tnbc92_minimal.yaml` (original_st, virchow2 1280d) and
  `alternate_cohort_minimal.yaml` (visium, uni2 1536d, donor_id). CohortPack
  tree_root, typed FK ranges, supervision_annotation_ids on DerivationRecord,
  UnitLevel enum, enforcement (minimum_value, pattern, rules). CI green on
  3.11 / 3.12 / 3.14.

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
- linkml schema 0.2.0 entities: Cohort, Split, PlatformDescriptor (deferred from 0.1.1)