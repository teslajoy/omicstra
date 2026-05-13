# tnbc-92 · data schema for archetypes, TME signatures, and embeddings

reference for where every biological label and embedding lives, and how each layer is derived from the one above. start here when you need to know "is `archetype` per-patient or per-spot?" or "where do I get TLS for a niche?".

last verified: 2026-05-13.

---

## layered overview

```
layer 0: upstream raw       (Wang 2024 RDS + Reactome)
   |
   v  build scripts: scripts/extract_biological_signals.py, scripts/build_niche_join.py
   |
layer 1: per-spot / per-patient flat files   (data/embeddings/biological_signals/)
   |
   v  scripts/build_niche_join.py
   |
layer 2: niche-joined per-subarray parquets   (data/embeddings/niches/)
   |
   v  alignment training (src/, scripts/align.py, etc.)
   |
layer 3: run-time per-run caches              (runs/tnbc-92/{run}/embeddings_test.parquet)
   |
   v  eval (scripts/eval.py, eval_h3_pathway_cca*.py)
   |
layer 4: hypothesis eval outputs              (runs/tnbc-92/eval/{H1,H2,H3}/)
```

---

## layer 0 - upstream sources

| upstream | what's in it | extracted to |
|---|---|---|
| `data/inputs/clinical/Clinical.RDS` (Wang) | per-patient 9-archetype, TIME labels, molecular subtypes | `clinical_labels.tsv` via `Rscript` in `scripts/build_niche_join.py:67-82` (`load_archetypes()`) |
| Wang NMF outputs (sup. methods) | per-spot 14-MC hard label + 14-MC soft weights | `mc_labels.tsv`, `mc_weights.tsv` |
| Wang H&E pathologist annotation (5 + 18 class) | per-spot compartment fractions | `morphology_labels.tsv` |
| Cabrita 2020 TLS gene set | TLS gene signature score per spot | `tls_scores.tsv` (built by `scripts/extract_biological_signals.py --step tls`) |
| Reactome + gpath2vec + decoupler AUCell | per-niche pathway activity | `niche_aucell_*.parquet` |

key construct-validity caveat (locked in `proposal_deviations.md`): the 9-archetype label is `Spatial archetypes_defined_on_ST_global_pseudobulk` from Clinical.RDS, which is a PATIENT-level pseudobulk label. 30/30 sampled subarrays have unique archetype values per patient. ARI on archetype is mechanically a 9-class patient classification at niche resolution, not a spatial-coherence test. niche-level biology rescue is `mc_megacluster` (per-spot 14-class NMF).

---

## layer 1 - flat per-spot / per-patient files

location: `data/embeddings/biological_signals/`

| file | granularity | rows | key columns |
|---|---|---|---|
| `clinical_labels.tsv` | per patient (95) | 95 | `patient_id, spatial_archetype (1-9), molecular_subtype_{global,tumor,stroma}, TIME_pathologist, TIME_expression_{global,bulk}` |
| `mc_labels.tsv` | per spot | 270,137 | `subarray (CN*_*), spot_id, patient_id, intra_cluster, megacluster (1-14)` |
| `mc_weights.tsv` | per spot | 270,311 | `subarray, spot_id, patient_id, mc1..mc14` (14-d soft NMF weights) |
| `morphology_labels.tsv` | per spot | 95,162 | `subarray_id (TNBC*_CN*_*), spot_id, dominant_5class, dominant_18class, dominant_18class_fraction, frac_<each of 17 categories>` |
| `tls_scores.tsv` | per spot | 101,955 | `subarray_id, spot_id, tls_score, n_tls_genes_present` |
| `niche_aucell_5targets.parquet` | per niche | 286,250 | `niche_id (index), subarray, patient, spot_id, pixel_x, pixel_y, <pathway>, <pathway>_z` for 5 named Reactome parents (raw AUCell + per-subarray z-score) |
| `niche_aucell_low_level.parquet` | per niche | 286,250 x 395 DAG-node columns | full-DAG expansion (commit 3.6) |

5 named Reactome parents in `niche_aucell_5targets.parquet`:
- TGF-β Signaling (R-HSA-170834)
- Immune System (R-HSA-168256)
- ECM Organization (R-HSA-1474244)
- Cell Cycle (R-HSA-1640170)
- Programmed Cell Death (R-HSA-5357801)

`subarray` key formats across files (this trips people up):
- `mc_labels.tsv` / `mc_weights.tsv`: `CN1_C1` (no TNBC prefix)
- `morphology_labels.tsv` / `tls_scores.tsv` / `niche_aucell*.parquet`: `TNBC1_CN1_C1` (full)
- the join in `build_niche_join.py:100-103` strips `TNBC{id}_` prefix for the mc_* lookups

---

## layer 2 - niche-joined per-subarray parquets

location: `data/embeddings/niches/{TNBC*_CN*_*}.parquet` (260 files, one per subarray)

built by: `scripts/build_niche_join.py`. manifest at `data/embeddings/niches/manifest.json`.

each row = one niche (7-spot footprint, indexed by center `spot_id`):

```
spot_id              str   (pandas index)
subarray             str         TNBC{id}_CN{x}_{pos}
patient_id           int64
archetype            int64       Wang 9-archetype, PATIENT-level (every niche in a patient = one value)
compartment          str         Wang dominant_18class, CENTER-spot only (not pooled across neighbors)
virchow2_niche       list[1280]  Virchow2 pre-pooled tile embedding
virchow2_cell_tokens list[8960]  7 x 1280 = self + 6 neighbors stacked (for cross-attention)
novae_niche          list[64]    mean-pooled, per-subarray z-scored
gpath2vec_niche      list[512]   pathway embedding
tls                  float32     mean-pooled across self + 6 neighbors (nan if missing)
mc_weights_niche     list[14]    mean-pooled soft NMF over 7 spots
mc_megacluster       int64       Wang 14-class hard NMF on CENTER spot
neighbor_spot_ids    list[str]   provenance, len=6
```

pooling discipline (from `build_niche_join.py`):
- `archetype` -> per-patient lookup, propagated to every niche of that patient
- `compartment` -> center-only from `dominant_18class` (NOT pooled across the 7-spot niche)
- `tls` -> mean-pooled, nan-safe
- `mc_weights_niche` -> mean-pooled (14-d soft NMF over 7 spots)
- `mc_megacluster` -> center-only hard label (NOT pooled - pooling a hard label is undefined)
- `virchow2_niche` -> Virchow2's own self-attention over 7 tiles -> 1280-d
- `novae_niche` -> mean-pool then per-subarray z-score (Novae GNN output)

coverage:
- 260 subarrays = intersection of `novae_niche_full/` + `virchow2_niche/` + gpath2vec full-cohort coverage
- archetype: 259/260 subarrays populated (1 patient missing from clinical)
- compartment: 90/260 subarrays populated (15 distinct compartment classes in the cohort; proposal cited 17 from Wang's annotation set)
- mc_megacluster: ~277,695 / 286,250 niches non-null

---

## layer 3 - run-time per-run caches

location: `runs/tnbc-92/{R1,R2,R3,R4,R6,B1,B2,B3}/embeddings_test.parquet`

45,661 rows = held-out test niches (14 test patients × 38 subarrays). all 8 runs share the same test split.

```
niche_id          synthesized   subarray::spot_id
subarray          str
patient_id        int
archetype         int           (propagated from layer 2)
compartment       str           (propagated from layer 2)
z_he              list[512]     post-projection H&E view
z_st              list[512]     post-projection ST view
attention_weights list          cross-attention runs only (R4 etc.)
```

z_mean (joint view) is not stored - compute as `(z_he + z_st) / 2` then L2-normalize.

run labels:
- R1 = late-fusion contrastive, deterministic
- R2-R3 = late-fusion contrastive, hyperparameter variants
- R4 = cross-attention bridge (H1 + H3 winner)
- R6 = late-fusion contrastive variant
- B1 = CCA classical baseline (patient-collapsed)
- B2 = late fusion concatenation (CONCH-style)
- B3 = unaligned concatenation (raw L2-norm)

---

## layer 4 - hypothesis eval outputs

location: `runs/tnbc-92/eval/`

### H1 - retrieval

`runs/tnbc-92/eval/H1/summary.json`
- per-run Recall@K, MRR, median rank, AUC, alignment gap, CKA before / after

### H2 - structural coherence

`runs/tnbc-92/eval/H2/`
- archetype part A (proposal-literal, patient-confounded) - in `current_report.md`, see proposal_deviations.md for the audit
- mc_megacluster part A (niche-level rescue) - `eval_h2_mc_coherence` results in summary.json
- compartment part B - `compartment_cosine.parquet` (per-niche cosine vs nearest-MC centroid by compartment)

### H3 - pathway interpretability

`runs/tnbc-92/eval/H3/pathway_cca/`
- 5-pathway CCA (commit 3 / 3.5):
  - `per_pathway_cca.parquet` (observed + null + BH-FDR)
  - `canonical_directions.parquet` (120 × 512-d canonical direction vectors)
  - `specificity_matrix.parquet`
  - `perm_nulls.parquet`
- DAG expansion (commit 3.6):
  - `dag_full/dag_metadata.parquet` (395 DAG nodes: node_id, name, parent_name, depth, n_genes, n_leaves, is_leaf)
  - `dag_full/dag_gene_sets.json` (descendant-leaves gene union per node)
  - `dag_full/per_node_aucell.parquet` (286k niches × 395 DAG nodes)
  - `dag_full/per_node_cca.parquet` (9,480 rows = run × view × node × ~8 cell-types; 19 cols: obs_A/B, null mean/std, p-values, BH-FDR, sig flags)

---

## quick lookup recipes

**9-archetype for a patient or niche:**
```python
import pandas as pd
clin = pd.read_csv('data/embeddings/biological_signals/clinical_labels.tsv', sep='\t')
# or per-niche (already joined):
niche = pd.read_parquet('data/embeddings/niches/TNBC10_CN5_D2.parquet')
niche['archetype']  # all same value within a patient
```

**TME signature (Wang 14-class megacluster + soft weights):**
```python
niche['mc_megacluster']      # hard 14-class on center spot
niche['mc_weights_niche']    # 14-d soft weights, mean-pooled over 7 spots
```

**pathway AUCell (5 named Reactome parents):**
```python
auc = pd.read_parquet('data/embeddings/biological_signals/niche_aucell_5targets.parquet')
auc.loc['TNBC10_CN5_D2::2x12', 'Immune_System_z']  # niche_id is index
```

**pathway AUCell (DAG nodes, 395):**
```python
dag_auc = pd.read_parquet('runs/tnbc-92/eval/H3/pathway_cca/dag_full/per_node_aucell.parquet')
dag_meta = pd.read_parquet('runs/tnbc-92/eval/H3/pathway_cca/dag_full/dag_metadata.parquet')
# find a node by name
btla = dag_meta[dag_meta['node_name'] == 'Co-inhibition by BTLA']['node_id'].iloc[0]
dag_auc.loc['TNBC10_CN5_D2::2x12', btla]
```

**compartment + TLS:**
```python
niche['compartment']  # str dominant_18class, CENTER spot only
niche['tls']          # float, mean-pooled over niche
```

**TIME / molecular subtypes (Bareche):**
```python
clin = pd.read_csv('data/embeddings/biological_signals/clinical_labels.tsv', sep='\t')
clin[['patient_id', 'molecular_subtype_global', 'TIME_pathologist',
      'TIME_expression_global', 'TIME_expression_bulk']]
```

**aligned latent (per-run):**
```python
emb = pd.read_parquet('runs/tnbc-92/R4/embeddings_test.parquet')
# z_he / z_st are list-of-float; stack for matrix ops
import numpy as np
Z = np.stack([np.asarray(v, dtype=np.float32) for v in emb['z_he']])
```

---

## granularity cheat sheet

| label | granularity | propagation |
|---|---|---|
| `spatial_archetype` (1-9) | **per patient** | every niche in a patient inherits one value |
| `molecular_subtype_*` | per patient | per-patient |
| `TIME_pathologist`, `TIME_expression_*` | per patient | per-patient |
| `megacluster` (1-14) | per spot | center-only in niche-join |
| `mc_weights` (14-d) | per spot | mean-pooled to niche |
| `dominant_18class` (compartment) | per spot | center-only in niche-join |
| `tls_score` | per spot | mean-pooled to niche |
| AUCell pathway scores | per niche | already niche-level |
| `z_he`, `z_st` | per niche | post-alignment |

remember: a niche is 1 center spot + 6 spatial neighbors ≈ 1200 cells ≈ 100µm footprint. one H&E section (subarray) contains ~1000-1500 niches.