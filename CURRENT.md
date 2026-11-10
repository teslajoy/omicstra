# CURRENT

the one task in flight. read this at the start of every session, before any
design doc and before any code.

this file holds ONE task. when it is done, it is replaced, not appended to.
everything that is not the one task goes to the parking lot at the bottom and
is not worked on, discussed, or measured until the task card is closed.

---

## the task

**pathway artifact persistence: scores reach disk, not just graph state.**

parent goal: pathway scores computed by the pathway_assay bridge currently
live only in graph state. a second run cannot reuse them, notebooks cannot
load them, and the evidence pack cannot cite them. this card persists them.

previous task card (WS 5 execute wiring, 2026-11-10) preserved at
`design/_scratch/CURRENT_ws5.md`.

this is completion_roadmap WS 6.

### what remains

**item: persist pathway scores from graph state to disk.**

the pathway_assay node computes AUCell scores and gpath2vec cosines and puts
them in state["pathway_scores"]. nothing writes them to a file afterward.

files to investigate:
- `src/omicstra/graphs/align.py` — the pathway_assay node
- `src/omicstra/graph.py` — where graph state is written at end of run
- `projects/tnbc-92/project.json` — where the output path should be declared

### acceptance criteria

- [ ] pathway scores written to `data/embeddings/pathway_scores/{run_id}.parquet`
  (or the path declared in project.json) after the pathway_assay node.
- [ ] the written file is cited in the evidence pack / run record.
- [ ] a second invoke with the same run reuses the file, not recomputes.
- [ ] tests: scores absent → computed and written; scores present → reused.

### stopping condition

pathway_cosines() in adapters/evaluation.py can load the persisted file.
tests pass locally.

### not in scope

second-cohort evidence (WS 10), downstream consumption by align (not this card).

---

## checkpoint 5 summary (closed 2026-11-10)

**branch**: main (pending push)

two items delivered:

- **WS 5 execute wiring**: `question.execute()` now calls `discover()` before
  `build_evaluate_graph()`. `Route.COMPUTE` returns a 4-step compute plan
  (`he_encode → st_encode → align_run → evaluate_run`). `Route.REFUSE` returns
  refused status. `Route.ASK` proceeds as before, taking `cohort` from the
  discover result. 5 new tests in `test_ws5_question_execute.py`.

- **TF-IDF question router**: `src/omicstra/question_router.py` — `Router`
  protocol (runtime-checkable) + `TFIDFRouter` (bigram TF-IDF, fit on
  FAMILY_BLOCKS vocabulary). 91.67% top-1 on 24 held-out + near-miss questions
  at threshold=0.15, 100% OOS refusal rate. 0 silent misroutes above threshold.
  labeled eval set at `tests/fixtures/router_questions.json` (pre-registered bar
  ≥90% top-1, ≥75% OOS refusal). swap rule: if bar missed, add
  `omicstra[router-embed]` with `EmbeddingRouter` behind same seam.
  7 new tests in `test_question_router.py`. 1369 total pass.

---

## checkpoint 4 summary (closed 2026-11-09)

**branch**: main (7b42664)

one workstream delivered:

- **WS 4 analytical bindings (remaining)**: `graphs/encode.py` —
  `_resolve_encoder(state)` helper replaces 6× `.get("encoder", "virchow2")`.
  reads declared primary/morphology encoder from `project.json#encoders`.
  refuses with `ValueError` when no primary encoder declared. `adapters/evaluation.py`
  — `pathway_cosines()` reads `project.json#pathway_targets` (list of
  `{name, reactome_id}`). `NAMED_PATHWAY_TARGETS` constant removed. `projects/tnbc-92/
  project.json` — 5 pathway targets added (TGF-beta R-HSA-170834, Immune_System
  R-HSA-168256, ECM_Organization R-HSA-1474244, Cell_Cycle R-HSA-1640170,
  Programmed_Cell_Death R-HSA-5357801). 9 new tests in `test_analytical_bindings.py`.
  1355 total pass.

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