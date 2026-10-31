# omicstra completion roadmap

status: approved for checkpoint 1 (workstreams 1-3). 2026-10-30.
written: 2026-10-30. baseline: main @ a095d6a.

---

## finish line

a researcher asks a supported biological question about supported data.
omicstra resolves the question into tasks, validates inputs and methodological
requirements, reuses compatible artifacts, executes missing work through its
existing graphs and chains, and returns computed findings, limitations,
notebook and provenance. cohort differences require declarations and mappings,
not changes to analytical code.

---

## layer ownership

| layer | owns | example |
|---|---|---|
| biomedical schema (linkml) | what entities, measurements and relationships mean | a pathologist's annotation is an Annotation with provenance |
| source mapping | how source files and fields populate that model | `.obs["donor_id"]` -> Participant.id |
| analytical binding | how a particular task uses those entities | that annotation supervises alignment training; in a different task it evaluates H2 independently |
| method configuration | scientific settings and genuine algorithm constraints | shared_dim=512 is a method requirement; k=6 is a cohort's geometry |
| orchestration | dependencies, execution order, reuse and researcher decisions | question -> tasks -> missing work -> gate -> resume -> results |
| result interpretation | what the computed evidence supports | verdict is biological/diagnostic/unguarded/not_run |

---

## workstream 1: scope and supported capabilities

**required outcome**: supported questions, assays, formats, methods and
outputs defined. release requirements separated from future extensions.

**current evidence**: routing_contract.json defines 7 task families. proposal
H1/H2/H3. v1.3 scope doc.

**gap**: no capability matrix linking question x required data x workflow x
limitations.

**checkpoint**: capability matrix written, linked to proposal requirements and
acceptance tests.

**effort**: 4-6h. **depends on**: nothing.

---

## workstream 2: biomedical schema

**required outcome**: linkml schema describes the research entities
independently of any hypothesis or workflow.

**entities**:

- Participant, ClinicalObservation, Treatment, Outcome
- Biospecimen, TissueSection, Assay
- SpatialUnit, Region, Neighborhood, CoordinateFrame
- PathologyAssessment, Annotation, AnnotationProvenance
- MeasurementMatrix, Representation (encoder, modality, dim, derivation
  lineage), DataFile

large matrices remain in anndata/parquet. the schema describes and references
them.

**what this replaces**: the implicit encoding where column name = encoder
name = dimension = modality. a Representation declares its observation IDs,
modality, encoder, dimensions and derivation. it doesn't need a column named
`virchow2_niche`.

**critical design point - derivation lineage**: a Representation carries how
it was derived (encoder, inputs, alignment run). circularity is determined by
tracing lineage - "does this view's derivation chain include the signal being
tested?" - not by comparing view names.

**checkpoint**: valid seed-cohort and alternate-cohort examples. invalid
examples rejected with every problem named. linkml generates the validation.

**effort**: 8-10h. **depends on**: nothing.

---

## workstream 3: source normalization

**required outcome**: existing adapters map supported source data into the
shared model. new column names require a mapping entry, not edits to
analytical functions.

**mapping examples**:

| source | canonical meaning |
|---|---|
| `.obs["donor_id"]` | Participant identifier |
| `.obs["section"]` | TissueSection identifier |
| `.obsm["spatial"]` | Coordinates in a declared CoordinateFrame |
| `.obsm["UNI"]` | Representation (modality=morphology, encoder=uni, dim=1024) |
| clinical table `response` column | ClinicalObservation with subject and timepoint |
| pathologist polygon file | Region geometry plus attributed Annotations |

the adapter validates identifiers, joins, shapes, units and missingness. it
preserves original files and records the mapping. ambiguous meanings (pixels
vs micrometers) require clarification, not guessing.

**what this replaces in code**: `adapters/alignment_data.py` hardcoding
`virchow2_niche`, `.reshape(7, 1280)`, `patient_id`. `protocols/join.py`
hardcoding `SUPERVISION_ONLY` and `FEATURE_COLUMNS` frozensets.

**checkpoint**: `load_subarray()` reads a source mapping. seed cohort's
mapping loads identically to today. alternate fixture (uni/1024d, k=4,
donor_id) loads without code changes.

**effort**: 8-10h. **depends on**: workstream 2.

---

## workstream 4: analytical bindings

**required outcome**: each task binds canonical entities to its required
analytical roles. bindings are validated per-task, not all-at-once at project
load.

**role bindings per task**:

| analytical role | bound entity | example |
|---|---|---|
| morphology input | selected Representation (modality=morphology) | Virchow2 niche embedding |
| molecular input | selected Representation(s) (modality=molecular) | Novae + gpath2vec |
| split grouping | Participant identity | patient-stratified holdout |
| biological target | selected Annotation | pathologist's mc_megacluster |
| supervision | selected Annotation or Measurement | mc_weights for alignment training |
| confounder | Participant, batch, or another declared variable | patient identity suppression |
| evaluation population | explicit observation/subject selection | held-out test set |

**design rules**:

- supervision is HERE (layer 3), not in source mapping. the same annotation
  supervises one task and evaluates another
- no `archetype` assumed universal - stratification variable is a binding if
  one exists
- validation is per-task: a cohort running EDA doesn't need pathway targets.
  a cohort running H3 does
- encoder resolution is by modality + role with explicit disambiguation when
  multiple match, not first-match selection
- circularity is lineage-based: the binding traces whether the pathway
  embedding appears in the derivation chain of the view being tested

**what this replaces in code**: `protocols/evaluate.py` hardcoding `z_he` in
6 circularity checks. `graphs/encode.py` defaulting to `virchow2`.
`agents/modality.py` resolving by encoder name. `contracts/project.py` silent
defaults for h2_label/supervision/headline_view. `NAMED_PATHWAY_TARGETS`
hardcoded in evaluation adapter.

**checkpoint**: alignment and evaluation chains validate bindings before
execution. unsupported combinations refused with the specific conflict named.
same task on two different binding sets (seed + alternate cohort) produces
correct, distinct results.

**effort**: 6-8h. **depends on**: workstreams 2-3.

---

## workstream 5: question interpretation and task planning

**required outcome**: question -> validated request -> dependency plan.
evidence lookup, recomputation and new analysis remain distinct intents.

**gap**: `question.execute()` calls `build_evaluate_graph()` directly,
bypassing `discover()` and the COMPUTE_CHAIN. cannot trigger upstream work
when artifacts are missing.

**checkpoint**: execute() enters through discover(). if upstream work needed,
COMPUTE_CHAIN determines stages. binding validation before execution.
ambiguity produces focused clarification, not a crash.

**acceptance**: representative questions select correct tasks.
missing-artifact question pauses at appropriate gate with clarification
request. complete-artifact question executes and returns results.

**effort**: 6-8h. **depends on**: workstreams 3-4.

---

## workstream 6: graph execution

**required outcome**: existing LangGraph orchestration executes only required
stages, supports decisions/resume and invokes existing LCEL workflows.

**gap**: entry-point inconsistency (MCP tools build subgraphs directly).
resume-to-results not demonstrated.

**partial checkpoint (0bd0516 + current)**: pathway_assay bridge node wired
into the compute chain (st -> pathway_assay -> align). demonstrated:

- artifact reuse: declared artifact on disk -> resolved, chain continues
- failure propagation: missing unproducible artifact -> refused, chain stops
  before align (keeper + conditional after pathway_assay)
- flag-driven selection: include_pathway_scores / include_pathway_cosines
  on state determine required artifacts, not task-family lookup
- per-sample records and notebook generation from stage provenance
- computation: AUCell chain runs on synthetic data, all 6 steps pass,
  records carry sample identity and step params, propagate through graph
  state to the align node, notebook generates with provenance (bugfix:
  pathway_sets injection was broken by reversed tuple unpacking)

**not yet demonstrated**:

- artifact persistence: computed scores exist only as records in graph
  state, not as a saved artifact on disk
- actual downstream consumption: the align stub captures state but does
  not read or use the pathway records as analytical input
- question interpretation -> flag selection (workstream 5)
- entry-point consistency (MCP tools)
- gate interrupt + resume to completed results
- encode and alignment notebook coverage

**remaining effort**: 4-6h. **depends on**: workstream 5.

---

## workstream 7: analytical completeness

**required outcome**: every release-required H1/H2/H3 subanalysis and
baseline is callable and connected through the graph.

**current state**: v3 analysis complete. H3v2 ported. evaluate chain works.
alignment LCEL chain exists but entry-point wiring incomplete.

**checkpoint**: method-by-method status. every release-required analysis
callable through the graph. verification records. unresolved discrepancies
explicit.

**effort**: 8-12h. **depends on**: workstreams 3-4.

---

## workstream 8: results and delivery

**required outcome**: computed findings, uncertainty, guards and limitations
reach the answer, report and rerunnable notebook.

**checkpoint**: one linked delivery bundle (answer + report + notebook +
provenance) for a demonstrated analysis.

**effort**: 4-6h. **depends on**: workstreams 5-7.

---

## workstream 9: provenance and data restrictions

**required outcome**: question, interpretation, bindings, decisions, actual
parameters, inputs and outputs remain linked. external tracing respects
cohort restrictions.

**checkpoint**: local provenance tests end-to-end. tracing to external
services refused for restricted cohorts.

**effort**: 3-4h. **depends on**: workstream 6.

---

## workstream 10: generalizability demonstration

**required outcome**: a genuinely different supported cohort runs through the
normal user-facing path.

**fixture**: different encoder (uni, 1024d), k=4, different
subject/annotation columns, different pathway targets. small synthetic data.

**acceptance**: same supported question and workflow on both datasets without
code changes. at least one meaningful computed result. unavailable analyses
explained without fabricated inputs.

**effort**: 4-6h. **depends on**: workstreams 2-6.

---

## workstream 11: packaging and release

**required outcome**: installed package, CLI and MCP expose the same
supported behavior with accurate documentation.

**checkpoint**: clean-environment install, wheel checks, CI, MCP integration,
release checklist.

**effort**: 4-6h. **depends on**: workstreams 1-10.

---

## execution checkpoints

```
checkpoint 1: schema + normalization proven (ws 1-3)
  capability matrix. linkml schema with seed + alternate examples.
  source mapping loads both without code changes.
  ~20-26h

checkpoint 2: bindings + question flow (ws 4-5)
  per-task validation. lineage-based circularity.
  question enters discover(), resolves dependencies.
  ~12-16h

checkpoint 3: execution through results (ws 6-8)
  artifact reuse. gate interrupt + resume to completed results.
  delivery bundle.
  ~18-26h

checkpoint 4: generalizability + release (ws 9-11)
  provenance. alternate cohort end-to-end. package ships.
  ~11-16h

total: 61-84h
```

---

## release scope

generalizability is a requirement of the current release. if the total
exceeds available hours, propose a scope reduction for the human's decision -
do not silently defer workstreams.

---

## authoritative documents

| document | purpose |
|---|---|
| this roadmap (`design/completion_roadmap.md`) | workstreams, dependencies, milestones, effort |
| linkml schema + examples | authoritative biomedical contract |
| capability matrix | supported question x data x workflow x limitations |
| decision record | schema ownership, normalization boundary, consequential choices |
| verification ledger | what was tested, on which inputs and commit, results, unresolved |

---

## stopping rule

the release is complete when the agreed supported capabilities pass their
acceptance checks - not when every schema concept is modeled, every
historical experiment is rerun, or every optional feature is added.

---

## context for a new session

this roadmap was written after a generalizability audit of `src/omicstra/` on
main (a095d6a). the audit found 14 hardcoded seed-cohort assumptions (column
names, dimensions, encoder names, pathway IDs, view names) concentrated in
three files:

- `adapters/alignment_data.py` - every column name hardcoded, `.reshape(7, 1280)`
- `protocols/evaluate.py` - `z_he` hardcoded in 6 circularity checks
- `adapters/evaluation.py` - `NAMED_PATHWAY_TARGETS`, subject fallback,
  `cluster_` prefix, reactome path

the fix is not parameter-threading. it is a three-layer architecture:

1. **biomedical schema** (linkml) - what the data means, independent of any
   workflow. a Representation carries derivation lineage so circularity is
   traced, not name-compared.
2. **source mapping** - how source files populate that model. adapter boundary.
   no analytical code sees cohort-specific column names.
3. **analytical binding** - how a task uses canonical entities. supervision is
   a role binding per task, not a data property. validation is per-task.

the pattern already exists in the codebase:
`adapters/evaluation.py:REQUIRED` + `declared()` reads required fields,
refuses with every missing one named, never defaults. that pattern needs to
be applied through the linkml-generated models rather than extended ad hoc.

key design decisions (agreed, not yet implemented):

- linkml is the authority from the start, not deferred
- supervision is layer 3 (binding), not layer 2 (mapping)
- circularity traces derivation lineage, not view names
- validation is per-task, not all-at-once at project load
- encoder resolution is by modality + role, not by name or first-match
- generalizability is current-release, not deferred to a future version
- resume-to-results is required, not just stopping at a gate
- `archetype` is not universal - stratification variable is a binding

read `design/v1_3_scope.md` for what shipped. read `design/graph.md` for
which file owns which step. read this file for what comes next.