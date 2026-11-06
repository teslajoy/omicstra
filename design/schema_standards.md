# schema standards decision

bounded to the first adapter slice: `adapters/alignment_data.py`.

written: 2026-11-02. completed: 2026-11-04. branch: checkpoint-1-schema-normalization.

---

## what this slice resolves

`alignment_data.load_subarray()` hardcodes 8 source column names and 2 tensor
dimensions. the join adapter upstream is already declaration-driven. the
evaluation adapter downstream is partially declaration-driven. this slice
closes the gap between them.

## what we reuse

### from project.json (already declared, already loaded by ProjectConfig)

| field | used by this slice | currently hardcoded instead |
|-------|--------------------|-----------------------------|
| `encoders[].name` | morphology column name | `"virchow2_niche"` |
| `encoders[].dim` | reshape width | `1280` |
| `encoders[].role` | which encoder is morphology vs molecular | `encoders[0]` by position |
| `k_neighbors` | token block height = k+1 | `7` |
| `supervision` | supervision column name | `"mc_weights_niche"` |
| `h2_label` | (not this slice - evaluation) | - |
| `headline_view` | (not this slice - evaluation) | - |

### from cohort.json (declared, not yet on ProjectConfig)

| field | used by this slice | currently hardcoded |
|-------|--------------------|--------------------|
| `subject_id_column` | confounder/split axis | `"patient_id"` |

### from join_inputs (declared, not read by alignment_data)

the join's label declarations already name every non-feature column
(`compartment`, `mc_megacluster`, `mc_weights_niche`, `tls`) with their roles
(`centre` or `pooled`). the alignment adapter's metadata columns are exactly
the join's `centre` labels plus the subject resolution.

## what we resolve by role, not by name or position

| role | resolution rule | seed cohort value |
|------|----------------|-------------------|
| morphology niche | `encoders` where `role` in `("primary", "morphology")` | virchow2, 1280d |
| morphology tokens | same encoder, column = `{name}_cell_tokens` | virchow2_cell_tokens |
| token geometry | `k_neighbors + 1` x `encoder.dim` | 7 x 1280 |
| molecular features | `st_features` tuple (already passed in, unchanged) | ("novae_niche", "gpath2vec_niche") |
| supervision | `project.supervision` | mc_weights_niche |
| subject | `cohort.subject_id_column` | patient_id |
| stratification | first subject label if declared, else None | archetype (from spatial_archetype) |
| unit labels | join's `centre` labels present in the parquet | compartment |
| sample id | convention: `"subarray"` column in niche parquet | subarray |

## what omicstra defines (not in any standard)

- **representation resolution**: resolve encoder by modality + role, read dim
  from the declaration, refuse if the array width does not match
- **token geometry**: k_neighbors + 1 from platform/project declaration, not a
  literal. validated against the array's actual shape
- **supervision exclusion**: the supervision column is resolved by name from
  the declaration and is never in the feature set. this is an invariant, not a
  convention

## what this does NOT change

- join adapter (already declaration-driven)
- evaluation adapter (separate slice)
- historical scripts in scripts/
- niche parquet format (oracle cache)
- alignment training code (LCEL chains, loss functions)
- build_matrices() output dict keys (x_he_niche, x_he_tokens, x_st, mc_w)
- split_by_patient() logic
- save_test_artifacts() logic

## HTAN and schema_crush applicability to this slice

this slice operates on niche-join parquets that are already downstream of
ingest. the HTAN2 participant-biospecimen-section chain and schema_crush
clinical mappings apply to the biomedical schema layer (WS 2), not to this
adapter slice.

the one HTAN concept relevant here is `subject_id_column` - the participant
identity that defines the confounder axis. HTAN calls it
`HTAN_PARTICIPANT_ID`. the schema should map it; this slice reads whatever
column name the cohort declares.

## acceptance (all met, 2026-11-04)

1. `load_subarray()` takes a project config (or its relevant fields) instead
   of hardcoded names — done: `AlignmentContract`, `from_project_dir()`
2. seed cohort loads identically (array values unchanged) — done: 40/40 tests pass
3. alternate fixture (uni2/1024d, k=4, donor_id, different supervision column)
   loads without code changes — done: `ALT_CONTRACT` fixture in
   `tests/test_alignment_contract.py`
4. missing or mismatched declarations refused with every problem named — done:
   `AlignmentDeclarationMissing` with collected problem list
5. no cohort-name or encoder-name branches in the code — done: resolution is
   by declared role, not by name inspection

---

## workstream 2: biomedical schema (linkml)

written: 2026-11-06. schema version: 0.1.1.
file: `src/omicstra/schema/omicstra.yaml`.

### what we reuse

**from HTAN** (vocabulary only, not tooling or JSON Schema):
- `HTAN_PARTICIPANT_ID` -> `Participant.participant_id`
- biospecimen hierarchy: Participant -> Biospecimen -> TissueSection
- field conventions: snake_case identifiers, stable IDs never inferred from filenames

HTAN covers the clinical/specimen side. we borrow the vocabulary, not the tooling.
no HTAN JSON Schema is imported; linkml validates against our own definitions.

**from FHIR** (hierarchy concept only):
- Participant ~ Patient, Biospecimen ~ Specimen, TissueSection ~ Specimen.container
  the three-level hierarchy maps. FHIR is not imported; it is too heavy for this
  use case and does not cover spatial units or embeddings.

**from W3C PROV** (slot_uri mapping, future):
- DerivationRecord maps to prov:wasGeneratedBy + prov:used. a future 0.2.x patch
  can add slot_uri annotations to make the claim defensible:
  "no existing standard covers it" is fair for HTAN/FHIR/GEO; PROV covers generic
  derivation. what omicstra adds is embedding semantics: modality, unit level,
  supervision annotation edges. those are not in PROV.

### what omicstra defines (not in any existing standard at this specificity)

- **Representation**: encoder, modality, unit_level, dim, derivation lineage.
  the entity that replaces implicit column-name encoding.
- **DerivationRecord**: input_repr_ids (what was used as input), 
  supervision_annotation_ids (what supervised training). together these are the
  graph the circularity check traverses.
- **UnitLevel enum**: spot / cell / tile / niche / subject. niche is the primary
  delivered unit (k=6 aggregation). no existing standard distinguishes these.
- **CoordinateFrame.scale_to_microns**: a GAT input, not a display label.

### linkml as build-time only (not a runtime dependency)

linkml is in `[dev]` extras. the wheel does not import `linkml_runtime`.

generated artifacts (not committed, regenerated by `make schema`):
- `gen-pydantic omicstra.yaml` -> `src/omicstra/schema/_generated.py`
- `gen-jsonschema omicstra.yaml` -> `src/omicstra/schema/omicstra.schema.json`

validation in tests: `pytest.importorskip("linkml.validator")` - skipped in CI
when linkml is not installed, so the CI gate is not broken for users without it.

### enforcement: what linkml validates vs what needs load-time checks

| rule | mechanism |
|---|---|
| `dim >= 1` | `minimum_value: 1` on dim slot |
| `scale_to_microns >= 0.0001` | `minimum_value: 0.0001` |
| `sha256` is 64 lowercase hex chars | `pattern: "^[0-9a-f]{64}$"` |
| `path` is relative (no leading /) | `pattern: "^(?!/)"` |
| `aligned` -> `run_id` required | `rules` block with precondition on modality |
| FK integrity (section_id exists) | typed ranges + `tree_root: true` CohortPack |
| `derivation` required | `required: true` on derivation slot |
| closed enums (platform, modality) | linkml enum validation |
| `encoder == derivation.encoder_name` | load-time check (cross-object; linkml cannot reach across slots) |
| `dim == array width in file` | load-time check (linkml cannot read the parquet) |

### versioning plan

- 0.1.1: core entities, B1-B3 fixes, enforcement patterns, supervision lineage
- 0.2.0: Cohort entity (from cohort.json), Split entity (from runs/{id}/split.json),
  PlatformDescriptor class (from platform.json), Aggregation class for niche geometry,
  affine transform on CoordinateFrame (replaces rotation_deg for anisotropy)
- 0.3.0: analytical role bindings per task (WS 4)
- 1.0.0: second cohort evidence pack produced end-to-end

### deferred (0.2.0)

- G1 Cohort entity: aligns with cohort.json fields (subject_id_column,
  data_classification, compute_backend). wait until confirmed against code.
- G5 Split entity: aligns with runs/{id}/split.json written by the alignment chain.
  field names and schema must match the actual file.
- G6 PlatformDescriptor class: aligns with platform.json (spot_diameter_um,
  pitch_um, lattice). original_st hardcoding in the AssayPlatform enum
  description is a known limitation; different cohorts may have measured different
  pitch values.
- G7 affine transform: rotation_deg alone cannot express 11% anisotropy. a 2x3
  or 3x3 affine (6 floats) on CoordinateFrame.parent_frame_id replaces it.
- omicstra_evidence.yaml: separate schema for routing/evidence layer (task
  families, guard records, routing decisions). imports omicstra.yaml.

### review notes (2026-11-06, from linkml-lint + gen-pydantic run on 0.1.0 draft)

- B1 parse error (unquoted scalars with colons): fixed by using block scalars
  everywhere
- B2 no tree_root: fixed by CohortPack with tree_root: true
- B3 FKs as bare strings: fixed by typed ranges + inlined: false
- 8 of 10 enforcement probes accepted in 0.1.0: 7 fixed by minimum_value,
  pattern, rules, required. the 8th (encoder == derivation.encoder_name) is
  cross-object and stays a load-time check.
- `omicstra_evidence.yaml` separation: deferred, design confirmed correct.
- W3C PROV mapping for defensible novelty claim: noted above, deferred to 0.2.x.
