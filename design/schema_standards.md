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
