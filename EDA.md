# EDA.md

exploratory data analysis gate for omicstra. every dataset must pass these checks
before embedding, alignment, or agent routing.

scope: multi-modal spatial biology projects (ST + morphology). checks are written
for spot-level platforms (Visium) but include conditional sections for single-cell
resolution (Xenium, MERFISH, CosMx) and multi-section alignment.

---

## principle

EDA is the first decision layer. the system must determine whether the data supports
meaningful cross-modal alignment before building anything. if these checks fail,
stop and fix preprocessing.

---

## required checks

### 1. cohort sanity
- count patients, samples, slides, sections, measurement units, files
- verify sample IDs and pairing structure (patient -> slide -> section -> measurement units)
- identify missing metadata, duplicated IDs, inconsistent labels
- confirm all samples belong to the intended tissue/disease context
- flag any normal tissue, controls, or out-of-scope samples
- cross-reference counts against the source publication

terminology: Visium -> spots/subarrays, Visium HD -> bins, Xenium/MERFISH/CosMx -> cells/FOVs, Slide-seq -> beads

### 2. measurement unit filtering
- distinguish pre-filter (full grid/FOV) from post-filter (tissue-selected) units
- verify filtering ratios against published QC if available
- identify dead or near-dead regions (< platform minimum unit threshold)
- confirm unit indices match between coordinate files and expression matrices
- document which unit set the pipeline should use

### 3. expression matrix integrity
- verify gene identifier format (Ensembl, symbol, or mixed) and genome reference
- determine if values are raw counts or normalized/corrected
- check sparsity, depth distributions, genes detected per unit
- flag outlier samples by sequencing/capture depth (IQR-based)
- document normalization and batch correction already applied
- record encoder compatibility decision: if counts_are_raw is false, explicitly decide
  whether to (a) seek raw data, (b) use normalized values with a compatible encoder,
  or (c) reverse-transform - document as encoder_input_decision in outputs

### 4. biological signal
- check known positive markers for the tissue/disease type (expression level + % units expressing)
- check known negative markers as controls (should be absent or near-zero)
- confirm marker patterns are consistent with the expected biology
- marker lists are project-specific - define in projects/{project_id}/project.json

### 5. spatial autocorrelation (Moran's I)
- compute Moran's I on key marker genes using k-NN spatial weights
- k is platform-dependent: k=6 for Visium hex grid, k=10-20 for irregular point
  clouds (Xenium, MERFISH) - tune to median nearest-neighbor distance
- pass threshold: I > 0.3 for at least 3 of the top 5 positive markers
- negative controls (known-absent markers) must show I ~ 0, p > 0.05
- if < 3 markers pass, flag as spatial_autocorrelation_pass: false and document
  whether normalization or tissue type is the likely cause before proceeding
- this check is the primary justification for spatial encoders (Novae) over non-spatial baselines

### 6. batch structure
- compute pseudobulk per sample (mean expression over selected units)
- PCA + UMAP colored by technical variables (plate, lane, batch, FOV) and
  biological variables (subtype, condition)
- quantify clustering by silhouette score for each variable
- pass threshold: technical silhouette <= biological silhouette, or both <= 0
- if technical silhouette > 0.1 and exceeds biological silhouette,
  batch dominates - address before alignment, set batch_correction_needed: true

### 7. cell segmentation QC (if single-cell resolution)
conditional: Xenium, MERFISH, CosMx, or any platform with cell segmentation.

- check cell area distribution - flag bimodal peaks (oversegmentation) or
  heavy right tail (undersegmentation)
- check transcript count per cell - flag cells below platform-specific minimum
- inspect segmentation boundaries on tissue image for artifact regions
- compare segmented cell count to expected cellularity for tissue type
- record pass/fail as segmentation_qc_pass in outputs

### 8. cross-modal sanity (if multi-modal)
conditional: aligning representations across modalities (H&E + ST, or similar).

- verify coordinate correspondence between modalities
- check registration/transformation assumptions (scaling, rotation, flipping)
- confirm resolution mismatch is documented (patch-level image vs spot/cell-level transcriptomics)
- overlay spatial coordinates on tissue image to visually confirm alignment
- record pass/fail as cross_modal_registration_pass in outputs

### 9. multi-section alignment (if serial sections)
conditional: serial tissue sections or 3D reconstruction.

- check inter-section registration quality (landmark overlap, coordinate drift)
- verify transformation matrices exist and produce plausible alignments
- flag sections with poor registration that may introduce spatial noise
- record pass/fail as multi_section_alignment_pass in outputs

### 10. model-tissue compatibility (if using foundation model encoders)
conditional: pretrained encoders (UNI2-h, Virchow2, Novae, etc.).

- check encoder training data coverage for the target tissue (see MODELS.md)
- flag uncertain tissue-model combinations
- if encoder was not trained on this tissue type, validate embedding structure
  (UMAP for collapse or batch artifacts) before trusting alignment results
- record per-encoder as model_tissue_fit in outputs

---

## outputs

produce projects/{project_id}/eda_summary.json:

```json
{
  "dataset": "",
  "dataset_doi": "",
  "dataset_source": "",
  "eda_date": "",
  "platform": "",
  "patients": 0,
  "samples": 0,
  "units_total": 0,
  "units_selected": 0,
  "units_post_qc": 0,
  "gene_id_format": "",
  "genome_reference": "",
  "normalization": "",
  "batch_correction": "",
  "counts_are_raw": false,
  "encoder_input_decision": "",
  "positive_markers_pass": true,
  "negative_markers_pass": true,
  "spatial_autocorrelation_pass": true,
  "morans_i_summary": {},
  "batch_silhouette": {},
  "batch_correction_needed": false,
  "excluded_samples": [],
  "segmentation_qc_pass": null,
  "cross_modal_registration_pass": null,
  "multi_section_alignment_pass": null,
  "model_tissue_fit": {},
  "verdict": "proceed | proceed with caution | stop",
  "risks": []
}
```

null values for conditional fields indicate the check was not applicable.
detailed findings remain in the EDA notebooks (notebooks/exploration/).

---

## decision rule

proceed to embeddings, alignment, or agent routing only if all applicable checks pass.

required checks (all projects):
1. cohort counts match publication - no corruption
2. positive markers detected, negative controls pass
3. Moran's I > 0.3 for >= 3 of top 5 markers; negative controls non-significant
4. technical batch silhouette <= biological silhouette, or both <= 0

conditional checks (if applicable):
5. cross-modal registration verified (if multi-modal)
6. model-tissue compatibility documented (if using foundation model encoders)
7. segmentation QC passes (if single-cell resolution platform)
8. inter-section alignment verified (if serial sections)

graded outcomes:
- if spatial_autocorrelation_pass is false but I > 0.1 for most markers:
  proceed with caution, flag as low-spatial-signal risk, document in risks[]
- if batch_correction_needed is true:
  stop. do not proceed to alignment until batch is addressed.
- if encoder_input_decision is unresolved:
  stop. encoder input format must be decided before embedding.

if a required check fails, document the reason and either fix preprocessing or stop.