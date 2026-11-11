# CURRENT

the one task in flight. read this at the start of every session, before any
design doc and before any code.

this file holds ONE task. when it is done, it is replaced, not appended to.
everything that is not the one task goes to the parking lot at the bottom and
is not worked on, discussed, or measured until the task card is closed.

---

## the task

**WS 9: provenance wiring — run records carry what each step was given.**

parent goal: `_link` in `protocols/__init__.py` records params per step, but
chain invocations in `graph.py` pass params as a flat dict. the record for
each step carries only what it was given, not what the full cohort params file
declared. a second run cannot reproduce which cohort parameters drove each
decision from the record alone.

this is completion_roadmap WS 9.

### what remains

**item: each step's record carries the params it was actually given.**

the `_link` function already stamps `supplied` onto the record's `params`
field (line 185 in `protocols/__init__.py`). the gap is that `pathway_assay`
in `graph.py` passes `pathway_params` as the top-level dict keyed by step_id
— verify this is correct and that records written to `all_records` carry the
per-step params, not the whole cohort dict.

files to check:
- `src/omicstra/protocols/__init__.py` — the `_link` and `_supplied` functions
- `src/omicstra/graph.py` — `pathway_assay` chain invocations
- `tests/test_pathway_chain.py` — existing params recording tests

### acceptance criteria

- [ ] a record written by any step in the AUCell chain carries its declared
  params (e.g. `target_sum`, `group_by`) in its `params` field.
- [ ] an end-to-end invocation through `pathway_assay` in graph.py produces
  records whose `params` match the cohort declaration, not the whole dict.
- [ ] existing pathway chain tests continue to pass.

### stopping condition

records written to disk carry their step-level params. tests pass locally.

### not in scope

second-cohort evidence (WS 10), downstream schema integration.

---

## checkpoint 7 summary (closed 2026-11-11)

**branch**: `ws7-8-graph-completion` (pending merge)

two workstreams delivered:

- **WS 7 analytical completeness**: `tests/test_ws7_8_integration.py` —
  `test_execute_returns_execution_result` proves the full chain:
  `execute()` → `discover()` → `Route.ASK` (tnbc-92 has routing evidence for
  `cross_modal_retrieval`) → `build_evaluate_graph(compute=True)` → `measure`
  → `ExecutionResult`. `test_execute_route_is_ask_not_compute` asserts
  discover fires the ask arm, not compute. `test_execute_no_plan_on_ask_route`
  confirms no spurious plan in the report.

- **WS 8 results + delivery**: `test_evaluate_graph_interrupt_and_resume` drives
  the pack arm (`compute=False`) through `InMemorySaver` — `gates()` finds 3
  open gates (hypothesis_scope, artifact_pinning, guard_failure), `ask()` pauses
  returning `{"__interrupt__": [...]}`, then `Command(resume={...})` resumes to
  `evaluate` with status in {"evaluated", "halted", "refused", "not_run"}.
  `test_evaluate_graph_halt_on_no_answer` proves `Command(resume=False)` halts
  cleanly. `test_provenance_has_required_keys`, `test_summary_has_required_keys`,
  `test_findings_is_dict` prove the delivery bundle shape. 8 new tests,
  1354 total pass.

---

## checkpoint 6 summary (closed 2026-11-11)

**branch**: main (pending commit)

one workstream delivered:

- **WS 6 pathway artifact persistence**: `protocols/pathway.py` — `save_scores`
  step added to `AUCELL_STEPS` with `applicable_when=lambda ctx: bool(ctx.get("_output_path"))`.
  writes per-sample parquet at `_output_path`; `not_applicable` in notebook/test
  contexts. `graph.py` — `_persist_pathway_scores` helper concatenates per-sample
  parquets to `data/embeddings/pathway_scores/aucell_v1.parquet` and updates
  `project.json#pathway_scores` so the reuse check finds it on the next run.
  `pathway_assay` node passes `_output_path` per sample and injects
  `"save_scores": {}` into params. TODO comment removed. 5 new tests in
  `test_pathway_persistence.py`; `test_pathway_chain.py` updated to allow
  `not_applicable` for `save_scores`. 1374 total pass.

---

## checkpoint 5 summary (closed 2026-11-10)

**branch**: main (pushed)

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