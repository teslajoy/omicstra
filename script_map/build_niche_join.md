# script map: `scripts/build_niche_join.py`

489 lines, read whole. this is the specification for `JOIN_STEPS`. every step in
the translation cites the rows here that it mirrors; everything no step covers is
listed at the bottom with the reason.

the script is the truth. its own module docstring is already wrong in two places
(noted in rows M-1 and X-3), so a row here cites a LINE, never a comment.

---

## 1 · module level

| line | operation | inputs | outputs | constants | filters / drops | column lists |
|:--|:--|:--|:--|:--|:--|:--|
| 1-31 | docstring: declares the output row contract | - | - | dims 1280, 7*1280=8960, 64, 512, 14, 6 | - | the output column list, M-1 |
| 46 | `warnings.filterwarnings('ignore')` | - | - | - | silences every warning for the whole process | - |
| 47 | `np.random.seed(42)` | - | - | `42` | - | - |
| 49-52 | path roots | `__file__` | `ROOT`, `EMB`, `BIO`, `CLIN` | - | - | - |
| 54-56 | assay input dirs | - | `VIRCHOW2_NICHE_DIR`, `VIRCHOW2_CELL_DIR`, `NOVAE_DIR` | - | - | - |
| 58 | pathway input, v1 default | - | `DEFAULT_GPATH_PARQ` | - | - | - |
| 59-63 | label tables + subject table | - | `MC_LABELS_TSV`, `MC_WEIGHTS_TSV`, `TLS_TSV`, `MORPH_TSV`, `CLINICAL_RDS` | - | - | - |
| 66 | output dir, v1 default | - | `DEFAULT_OUT_DIR` | - | - | - |
| **68** | **niche width** | - | `K_NEIGHBORS` | **`6`** (self + 6 = 7) | - | - |
| **71** | **the pathway parquet's non-data columns** | - | `GPATH_META_COLS` | - | - | **`('subarray','patient','spot_id','pixel_x','pixel_y')`** |

row 71 is the schema. a script that declares its own non-data columns IS the
schema for that file; inferring it by dtype pulls coordinates into the embedding
slice.

---

## 2 · helpers

| line | operation | inputs | outputs | constants | filters / drops | column lists |
|:--|:--|:--|:--|:--|:--|:--|
| 74-85 | `parse_niche_id`: split `{subarray}::{spot_id}` on first `::` | unit key str | `(subarray, spot_id)` | split maxsplit `1` | asserts on 2 parts and on both non-empty: a malformed key raises | - |
| 90-91 | import-time asserts on two sample keys | - | - | two cohort ids | trips on import, not at run time | - |
| 94-100 | `_sha256_of_file` | path | hex digest | chunk `1 << 20` | - | - |
| 102-111 | `_git_commit_sha` | `ROOT` | 12-char sha or `'unknown'` | `12`, timeout `5` | any exception -> `'unknown'` | - |
| 114-129 | `load_archetypes`: per-subject label via `Rscript` on an `.RDS` | `CLINICAL_RDS` | `{subject_id: archetype}` | stderr slice `200` | `assert returncode == 0`; **`df.dropna()` at 128 silently drops subjects with no archetype** | `tnbc_id`, `Spatial archetypes_defined_on_ST_global_pseudobulk` |
| 132-145 | `discover_matched_subarrays`: three-way key intersection | novae dir glob, virchow2 dir glob, pathway parquet `subarray` column | sorted sample keys | - | **a sample missing from ANY of the three never enters the loop and is never counted** | reads `['subarray']` only |
| 147-150 | `slide_pos`: `TNBC1_CN1_C1` -> `CN1_C1` | sample key | label-table key | split maxsplit `1` | - | - |
| 153-166 | `niche_mean_pool`: reindex `[center] + neighbors`, `mean(axis=0, skipna=True)` | per-unit frame, center, neighbours, columns | `float32` vector | - | missing keys become NaN rows and are skipped by `skipna`, so the mean is over a VARYING denominator | caller supplies `columns` |

`niche_mean_pool` is the pooling contract: reindex, then skipna-mean. a unit at a
boundary is averaged over fewer members, not over zeros.

---

## 3 · `process_subarray` — per sample, lines 168-332

### 3a · load and establish the unit set

| line | operation | inputs | outputs | constants | filters / drops | column lists |
|:--|:--|:--|:--|:--|:--|:--|
| 179-181 | subject + subject label resolved by `slide_pos` key | `patient_by_sub`, `archetype_map` | `patient_id`, `archetype` | - | `.get` -> `None` when absent; **`None` propagates into the rows, it does not refuse** | - |
| 184-189 | load pre-pooled morphology niche vectors + their meta | `{key}.npy`, `{key}_meta.tsv` (`index_col=0`) | `vn`, `meta` | - | - | meta index = unit ids |
| 190-195 | load per-unit morphology vectors; assert row counts agree | `{key}.npy` | `vc` | - | `assert vn.rows == vc.rows == len(meta)` | - |
| 195 | `spot_to_row`: unit id -> row index | `meta.index` | dict | - | - | - |
| 198-199 | load the molecular parquet; select the embedding columns | `{key}_novae_embeddings.parquet` | `nv`, `novae_cols` | - | - | **`startswith('novae_')`** - a PREFIX rule, not a declared list |
| **201** | **the canonical unit set** | `meta.index`, `nv.index` | `shared` | - | **`[s for s in meta.index if s in nv.index]` - units in one assay only are dropped here, SILENTLY and UNCOUNTED. the only drop in the script with no counter** | - |
| 204-208 | per-sample lookup tables | the five by-sample dicts | `gp_sub`, `mcw_sub`, `mclab_sub`, `tls_sub`, `morph_sub` | - | `.get` -> `None`; each absence is handled per unit below | two key spaces: `subarray_key` for 204/207/208, `sp` for 205/206 |
| 211 | molecular values frame for reindex pooling | `nv` | `nv_vals` | - | - | `novae_cols` |
| 214-215 | counters | - | `n_missing_gp`, `n_missing_tls`, `n_missing_morph`, `n_missing_neighbor`, `n_missing_mc` | - | - | - |

### 3b · per unit, lines 216-292

| line | operation | inputs | outputs | constants | filters / drops | column lists |
|:--|:--|:--|:--|:--|:--|:--|
| 218 | read the declared neighbour list | `nv.loc[sid,'neighbor_spot_ids']` | `neighbors_raw` | - | the neighbourhood is READ from the molecular assay, not recomputed from coordinates | `neighbor_spot_ids` |
| **220** | keep only neighbours present in the morphology assay | `spot_to_row` | `neighbors` | - | **a neighbour absent from the other assay is dropped** | - |
| **221-224** | **pad to fixed width by repeating self** | - | `neighbors` len `K` | `K_NEIGHBORS` | **`n_missing_neighbor` counted. a boundary unit's own vector is weighted (1 + pad)/7 in every pooled field** | - |
| 227-228 | stack self + K neighbours into the per-unit token block | `vc` | `(7, 1280)` `float32` | - | - | - |
| 231 | pool the molecular vector: reindex `[sid]+neighbors`, `mean(axis=0)` | `nv_vals` | 64d `float32` | - | default `skipna=True`; every key exists after 220, so the denominator is 7 with padding | `novae_cols` |
| 234-238 | pathway vector, already per unit, looked up not pooled | `gp_sub` | 512d or NaN | **`512` HARDCODED at 238** | absent -> all-NaN row, `n_missing_gp` counted. **238 contradicts 390/461, which take the real width as `len(gp_cols)`** | - |
| 241-251 | pool the scalar signal | `tls_sub` | float or NaN | - | table absent, or all 7 keys NaN -> NaN, `n_missing_tls` counted | `tls_score` |
| 253-258 | pool the 14-way weight vector | `mcw_sub`, `niche_mean_pool` | 14d `float32` | **`14` via `range(1, 15)` at 255, `14` again at 258** | table absent -> all-NaN | **`['mc1'..'mc14']` generated, not declared** |
| 260-265 | categorical label, **center unit only, not pooled** | `morph_sub` | str or `None` | - | absent -> `None`, `n_missing_morph` counted | `dominant_18class` |
| 267-276 | discrete class label, **center unit only, not pooled** | `mclab_sub` | int or `-1` | **`-1` sentinel at 276** | absent -> `-1`, `n_missing_mc` counted. the sentinel is a value in the column; downstream filters `< 0` | `megacluster` |
| 278-292 | assemble the row | all of the above | one dict | - | **291 stores `neighbors_raw[:K]` - the RAW list, before the 220 filter and before the 224 padding. the provenance column and the pooling do not agree by construction** | the 13-key output row |
| 294 | frame, indexed on unit id | `rows` | `df` | - | - | - |

discrete labels are taken from the center unit and continuous ones are pooled
over 7. that is a decision, stated at 270-271, and it is the reason a label and
an embedding in the same row describe different footprints.

### 3c · the drop, then the transform

| line | operation | inputs | outputs | constants | filters / drops | column lists |
|:--|:--|:--|:--|:--|:--|:--|
| **301-305** | **the intersection drop: any unit whose pathway vector is NaN** | `df['gpath2vec_niche']` | `df`, `n_intersect_dropped` | - | **`keep = ~np.isnan(gp_mat).any(axis=1)`. this is the big one: a unit with no significant pathway is removed from the cohort entirely** | `gpath2vec_niche` |
| **307-315** | **per-sample z-score of the molecular vector, AFTER the drop** | `df['novae_niche_raw']` | `novae_niche` | **`1e-8` floor at 312** | mean and std are computed over the SURVIVORS, so the drop at 301 changes every retained value. `np.std` -> **`ddof=0`** | `novae_niche_raw` dropped at 315 |
| 317 | coverage of the discrete label after the drop | `df` | `n_mc_post_intersect` | `>= 0` | - | `mc_megacluster` |
| 318-331 | the per-sample funnel | counters | `stats` | - | - | 11 keys: `subarray`, `patient_id`, `archetype`, `n_niches`, `n_niches_pre_intersection`, `n_intersection_dropped_no_gpath2vec`, `n_boundary_niches_padded`, `n_missing_gpath2vec`, `n_missing_mc_megacluster_pre_intersection`, `n_with_mc_megacluster_post_intersection`, `n_missing_tls`, `n_missing_morphology` |

**301 before 307 is the order, and it is load-bearing.** the comment at 307-308
says why: the std is meant to reflect the units that feed alignment. reversing
the two produces different numbers for every surviving unit. this is tier-1 case 6.

the funnel at 318-331 counts nine things and misses one: the drop at 201.
`n_niches_pre_intersection` is already post-201, so `len(shared)` is nowhere in
the record.

---

## 4 · `main` — lines 335-486

| line | operation | inputs | outputs | constants | filters / drops | column lists |
|:--|:--|:--|:--|:--|:--|:--|
| 336-353 | CLI | argv | `args` | `--progress-every` default `10` at 341 | - | - |
| 354-356 | create out dir; assert the pathway parquet exists | - | - | - | missing input -> assert | - |
| 359 | subject labels | `load_archetypes` | `archetype_map` | - | see 114-129 | - |
| 362-370 | the subject map and the 14-way weight tables | `MC_WEIGHTS_TSV` | `patient_by_sub`, `mcw_by_sub` | - | **`drop_duplicates('subarray')` at 364 - first row per sample wins; a sample with two subject ids is not detected** | `subarray`, `patient_id`, `spot_id` |
| 372-374 | discrete labels by sample | `MC_LABELS_TSV` | `mclab_by_sub` | - | - | `subarray`, `spot_id` |
| 376-378 | scalar signal by sample | `TLS_TSV` | `tls_by_sub` | - | - | **`subarray_id`**, `spot_id` |
| 380-382 | categorical labels by sample | `MORPH_TSV` | `morph_by_sub` | - | - | **`subarray_id`**, `spot_id` |
| 386-387 | hash the pathway input | `_sha256_of_file` | `gpath_sha256` | - | - | - |
| **390** | **the embedding columns = everything NOT in the declared non-data list** | `gp.columns`, `GPATH_META_COLS` | `gp_cols` | - | - | see row 71 |
| 391-394 | v1 compatibility: derive the unit id from the index when absent | `parse_niche_id` | `gp['spot_id']` | - | - | `spot_id` |
| 395-398 | group by sample, index by unit | `gp` | `gpath_by_sub` | - | - | `gp_cols` |
| 402-405 | enumerate samples; optional subset | `discover_matched_subarrays`, `--subarrays` | `matched` | - | `--subarrays` narrows for smoke tests only | - |
| 410-412 | **idempotency on the output file** | `out_path.exists()` | status `exists` | - | present and no `--force` -> skipped | - |
| 413-426 | run one sample; write; record | `process_subarray` | parquet + `stats` | error slice `120` at 423 | **any exception -> status `failed`, loop CONTINUES. a sample that fails is one row in the manifest, not a stop** | - |
| 432-444 | aggregate the funnel across samples | `summary` | totals, `mc_coverage` | - | `mc_coverage` is `0.0` when the total is 0 | - |
| 446-480 | manifest | all of the above | `manifest.json` | **dims `1280`, `[7,1280]`, `64`, `14` hardcoded at 458-462; `gpath2vec_niche_dim` is `len(gp_cols)`, computed at 461** | - | `run_type`, `git_commit_sha`, 4 counts, 2 niche totals, 2 coverage keys, `columns`, `sources`, `out_dir`, `per_subarray` |
| 481-486 | write and report | - | - | - | - | - |

the manifest is the provenance record: the input hash at 386, the commit at 448,
the per-sample funnel at 479, the normalisation named as a string at 464.

---

## 5 · every drop in the script, in execution order

the funnel a translation has to reproduce. the counted column is what the script
records today.

| # | line | what is removed | counted as |
|:--|:--|:--|:--|
| 1 | 128 | subjects with no subject-level label | **no** |
| 2 | 145 | samples missing from any of the three assays | **no** (only `n_subarrays_enumerated` survives) |
| 3 | 201 | units not present in BOTH embedding assays | **no** |
| 4 | 220 | neighbours absent from the morphology assay | via #5 |
| 5 | 221-224 | nothing removed; width restored by repeating self | `n_boundary_niches_padded` |
| 6 | 234-238 | nothing removed; pathway absence becomes NaN | `n_missing_gpath2vec` |
| 7 | 241-251 | nothing removed; scalar absence becomes NaN | `n_missing_tls` |
| 8 | 260-265 | nothing removed; categorical absence becomes `None` | `n_missing_morphology` |
| 9 | 272-276 | nothing removed; class absence becomes `-1` | `n_missing_mc_megacluster_pre_intersection` |
| 10 | **301-305** | **units with no pathway vector** | `n_intersection_dropped_no_gpath2vec` |
| 11 | 410-412 | samples already on disk | status `exists` |
| 12 | 413-426 | samples that raised | status `failed` |

three of the twelve are uncounted (#1, #2, #3). each one is an unexplained drop
under the funnel rule, so each needs a stage with a reason in the translation.

---

## 6 · every numeric literal, and what it is

| line | literal | kind | in the translation |
|:--|:--|:--|:--|
| 68 | `6` | the niche width | a declaration |
| 238 | `512` | the pathway encoder's width | resolved from the encoder spec, never written |
| 255, 258 | `14`, `range(1,15)` | a label vocabulary's size | read from the table's declared columns |
| 276 | `-1` | an absence sentinel | a declaration, or absence encoded as a column |
| 312 | `1e-8` | a degenerate-variance floor | a declaration |
| 458-462 | `1280`, `[7,1280]`, `64`, `14` | recorded dims | computed, as 461 already does for the pathway width |
| 47 | `42` | a seed | a declaration |
| 100, 110, 125, 341, 423 | `1<<20`, `12`, `200`, `10`, `120` | plumbing: chunk, sha length, slices, progress | not translated |

---

## 7 · what no `JOIN_STEPS` step will cover, and why

| line | thing | why not |
|:--|:--|:--|
| 46-47 | global warning filter and seed | process setup. the package configures neither globally |
| 90-91 | import-time asserts on two cohort keys | cohort constants. these become fixtures in a test |
| 94-111 | file hashing and the git sha | provenance, already owned by the run record |
| 114-129 | `Rscript` on an `.RDS` | a cohort's native format is converted in `scripts/`, once, never inside the package |
| 147-150 | `slide_pos` | a key-space adapter for this cohort's two naming conventions. the package reads a declared key, it does not strip prefixes |
| 336-356 | the CLI | `scripts/` keeps its CLI |
| 408-426 | the sample loop, idempotency, the per-sample try | `dispatch/` already owns shards, `is_done` on the output file, retries, and recorded failure |
| 445-486 | the manifest | the run record is the manifest |
| 199 | `startswith('novae_')` | a prefix rule. the package resolves an assay's columns from its declaration |

---

## 8 · two places the script's own prose is wrong

- **M-1** · the docstring at 14 says `novae_niche` is `len=64 (mean-pool self +
  6 neighbors, z-score per sub)`, which is right, but 20-21 says the scope is
  "expected 260 subarrays" - a count, in a docstring, that nothing enforces.
- **X-3** · the docstring at 24 names the output as
  `data/embeddings/niches/{key}.parquet`. that is the v1 default at 66; the
  published v3 build was written elsewhere via `--out-dir`. the path in the
  docstring is not the path of the artifact anything downstream reads.
