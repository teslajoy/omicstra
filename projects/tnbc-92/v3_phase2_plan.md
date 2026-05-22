# v3 phase-2 retrain plan (post-2026-05-21)

written 2026-05-21 after v3 gpath2vec (`fisher_madmean_low_dim512_e5_s1234`) was validated as the canonical Fisher build for omicstra. v3 replaces v1 gpath2vec as the ST-side pathway anchor in alignment training. this document is the load-bearing handoff: scope, dependencies, time estimates, exact commands, risks.

phase 1 (this work) confirmed v3 is geometrically healthy (MC z = +30.3, MC/patient ratio 1.03, no dim-collapse) and that the v3 H3 cross-patient signal on the existing alignment runs is much stronger than v1 (R4 4/4 testable pathways sig at BH-FDR 0.004, z_A 9.6-22.3). but the 5 contrastive runs were trained against **v1** gpath2vec on the ST input. for the manuscript-canonical omicstra deliverable, alignment must be retrained with v3.

---

## why retrain

### the v1 -> v3 swap is purely an ST-side feature change

omicstra ST input = `novae(64) ⊕ gpath2vec(512)` = 576-d (verified across all 7 run_config.json files). v3 is dim **512** (same as v1) -> **no alignment architecture change**: LayerNorm(576), Linear(576, 512), ReLU, BN, Dropout, Linear(512, 512), L2-norm. config-only swap.

### what's actually changing

| axis | v1 (current alignment) | v3 (post-retrain) |
|---|---|---|
| ST feature `gpath2vec_niche` | 512-d, from `gpath2vec_output/full_cohort_tf_low/cluster_embeddings_v1_legacy.parquet` (irreproducible) | 512-d, from `gpath2vec/fisher_madmean_low_dim512_e5_s1234/fisher_madmean_low_cluster_embeddings.parquet` (reproducible, sha-locked) |
| per-niche gene selection upstream | top-100 highest-expressed | MAD/mean variable |
| niche coverage | 286,233 niches | **215,388 niches** (70,845 dropped: no significant pathway under MAD/mean) |
| H&E feature (`virchow2_niche`) | unchanged | unchanged |
| novae feature (`novae_niche`) | unchanged | unchanged |
| `mc_weights_niche` supervision | unchanged | unchanged |

---

## the 70k-niche drop is the only structural complication

v3 dropped 70,845 niches at the gpath2vec step (no significant pathway after MAD/mean variable-gene selection). this changes which niches are eligible for the niche-join.

**decision (recommended): intersection split.**
- niche-join uses only niches present in all three sources: virchow2_niche × novae_niche_full × gpath2vec_v3
- expected niche-join size: ~215,000 (limited by v3, since v1's coverage was the larger superset)
- existing patient-stratified 85/15 split (seed 42) re-applied on the new niche set
- expected test grid: ~35,594 niches (verified in phase-1 v3 H3 evaluation - this is the actual count after intersection)
- 14 held-out test patients / 38 subarrays preserved

**rejected alternatives:**
- backfill with zeros: introduces noise, distorts InfoNCE batch statistics, banned
- keep niche-join geometry, NaN-mask v3-missing rows in training loop: equivalent to intersection in practice, more code complexity

---

## prerequisites

- v3 gpath2vec build: ✅ landed at `data/embeddings/gpath2vec/fisher_madmean_low_dim512_e5_s1234/` (sha256 locked, see `run_provenance.json`)
- v3 H3 phase-1 evaluation: ✅ done (see `notebooks/final/h3_gpath2vec_v3.ipynb`)
- niche-join builder script: `scripts/build_niche_join.py` exists, currently points at v1; needs path parameterization

---

## TODO (concrete steps, ordered)

### per-step structure

each step below has four explicit gates that must pass before moving on:
- **does it (the step)**: what the work physically is
- **proposal relevance**: which proposal commitment this step delivers (none, weak, strong) and the section ref
- **code review / publishability checks**: review-defensible gates (seeds, paths, provenance, asserts, suppressed-warnings policy, etc.) that must be present in the code BEFORE we ship. these are not suggestions - they are gates.
- **acceptance**: numerical / structural outputs that confirm the step worked

publishability is bounded by the weakest reproducibility link in any step. one missing `np.random.seed(42)` upstream invalidates everything downstream.

---

### step 1 - parameterize `scripts/build_niche_join.py` (10 min)

**does it**: current code hardcodes
```python
GPATH_PARQ = BIO / 'gpath2vec_output' / 'full_cohort_tf_low' / 'cluster_embeddings.parquet'
```
add a CLI arg `--gpath2vec-parquet PATH` defaulting to the v1 location (for back-compat), `--out-dir PATH` defaulting to `data/embeddings/niches/`, and adjust per-niche lookup to handle v3's parquet format (niche_id index, 512 dim cols `0..511` + subarray, patient, spot_id, pixel_x, pixel_y).

**proposal relevance**: strong. proposal §3 Methods specifies "spatial transcriptomics modality agent (tool: Novae GNN encoder, 256-d spot embeddings)" plus gpath2vec pathway projection (proposal §3 Pathway Projection); the niche-join is the canonical ST-side input table that both agents consume. parameterizing it is the prerequisite for an MCP-server "encoder-pluggable" claim.

**code review / publishability checks**:
- [ ] CLI args have explicit defaults; no breaking back-compat (running with no args reproduces v1 exactly)
- [ ] paths use `Path(__file__).resolve().parents[N] / ...` not hardcoded absolute paths (per `feedback_no_absolute_paths.md`)
- [ ] write `data/embeddings/niches_v3/manifest.json` with: `{gpath2vec_parquet_path, gpath2vec_parquet_sha256, n_subarrays, n_niches_per_subarray, git_commit_sha, mc_label_join_coverage}`
- [ ] add `np.random.seed(42)` at top of script (even if not currently random - future-proof against numpy ops that drift)
- [ ] `warnings.filterwarnings('ignore')` only inside script entry, never at module import
- [ ] niche key format converter (the `cluster_X` / `cluster_X__X` ambiguity from earlier this session) is centralized in one function and unit-asserted against both v1 and v3 sample keys

**acceptance**:
- existing v1 niche-join still rebuilds identically when called without args (byte-equal on a sample subarray)
- passing the v3 parquet path produces the v3 niche-join without errors
- manifest.json contains sha256 of v3 input + git_commit_sha at build time

### step 2 - build the v3 niche-join (~30 min)

**does it**:
```bash
python scripts/build_niche_join.py \
  --gpath2vec-parquet data/embeddings/gpath2vec/fisher_madmean_low_dim512_e5_s1234/fisher_madmean_low_cluster_embeddings.parquet \
  --out-dir data/embeddings/niches_v3
```
writes per-subarray parquets to `data/embeddings/niches_v3/`. v1 niche-join at `data/embeddings/niches/` stays intact. expected output: ~215k rows across `virchow2_niche × novae_niche_full × v3`.

**proposal relevance**: strong. the niche-join geometry is the foundation of every H1/H2/H3 measurement. all proposal hypotheses are evaluated on the niche grid.

**code review / publishability checks**:
- [ ] manifest.json includes sha256 of the v3 parquet input + the v1 input (for the parallel build's reproducibility)
- [ ] log every dropped niche with the reason (`no_gpath2vec_coverage`, `nan_in_novae`, `boundary_padding_failed`) - reviewer can audit the drop
- [ ] `mc_label` join uses the documented subarray-prefix-stripping rule (per `project_mc_weights_column_ambiguity.md` memory)
- [ ] NO `mc_weights_niche` rolled into any input slot - it stays as supervision-only per `program.md` hard constraint #3 (circular dependency ban)
- [ ] novae per-subarray z-score happens AFTER niche aggregation (per `program.md` ALWAYS rule)
- [ ] virchow2 stays RAW (no z-score, no stain norm) per `program.md` hard constraint #2

**acceptance**:
- manifest.json reports expected n_niches per subarray (~215k total)
- `mc_megacluster` join coverage on the v3 niche set matches v1's join coverage on the same niches (sanity: same `mc_labels.tsv` lookup)
- spot-check 3 subarrays (`TNBC1_CN1_C1`, `TNBC3_CN2_C1`, `TNBC83_CN42_C1`): `gpath2vec_niche` is 512-d, finite (no NaN/Inf), L2-norm > 0
- niche key format is `{subarray}::{spot_id}` (matching the convention used in `niche_aucell_5targets.parquet` + per_pathway_cca runs)

### step 3 - parameterize `scripts/align.py` for the v3 niche dir (5 min)

**does it**: current `NICHES_DIR` is hardcoded. add `--niches-dir PATH` CLI arg (defaulting to v1 location). each run's `run_config.json` records `niches_dir` for provenance.

**proposal relevance**: weak (plumbing) but unblocks strong-relevance steps 4-5.

**code review / publishability checks**:
- [ ] every seed-controlling line present: `torch.manual_seed(seed)`, `np.random.seed(seed)`, `random.seed(seed)`, `torch.use_deterministic_algorithms(True)` if GPU, `torch.backends.cudnn.deterministic = True` + `cudnn.benchmark = False` (covers the CUDNN non-deterministic-conv bug class)
- [ ] DataLoader `worker_init_fn` seeds workers; `generator=torch.Generator().manual_seed(seed)` on the DataLoader
- [ ] split.json records seed + split logic; re-running with same seed reproduces patient assignment
- [ ] run_config.json captures every hyperparam: seed, lr, weight_decay, batch_size, tau, tau_target, dropout, shared_dim, epochs, patience, st_features, fusion type, niches_dir, git_commit_sha, gpath2vec_parquet_sha256 (from step 2 manifest)
- [ ] training_log.csv flushed per-epoch (no in-memory accumulation that's lost on crash)
- [ ] no `print(absolute_path)` anywhere in stdout/stderr (per `feedback_suppress_warnings.md`)

**acceptance**: passing `--niches-dir data/embeddings/niches_v3` runs alignment against the v3 grid, run_config.json shows the v3 path + sha256.

### step 4 - re-fit B1, B2, B3 classical baselines (5-15 min total, closed-form)

**does it**:
```bash
for run in B1 B2 B3; do
  python scripts/align_classical.py \
    --run-id ${run}_v3 \
    --niches-dir data/embeddings/niches_v3 \
    --out-dir runs/tnbc-92_v3/${run}
done
```
B1 = CCA, B2 = Procrustes, B3 = unaligned PCA. closed-form, no SGD.

**proposal relevance**: strong. proposal H1 explicitly names these as the three baselines: "CCA projection, late fusion concatenation, and raw unaligned concatenation".

**code review / publishability checks**:
- [ ] B1/B2/B3 use the SAME test split as the contrastive runs (read `runs/tnbc-92_v3/R1_v3/split.json` once step 5 lands, OR pre-create the split in step 3 and reuse from there)
- [ ] PCA / CCA / Procrustes all fit on TRAIN only; transform applied to TEST. no train-leak.
- [ ] `np.random.seed(42)` even though closed-form (PCA's `svd_solver='randomized'` is non-deterministic without it)
- [ ] every run writes `embeddings_test.parquet` + `metrics_h1_raw.json` + `run_config.json` + `run.log` - the structural artifact gap that bit B4 in the v1 grid must not recur

**acceptance**:
- B1 embeddings_test.parquet has same row count and column shape as R*-runs (test_niches × {subarray, patient_id, archetype, compartment, z_he, z_st})
- metrics_h1_raw.json values are finite and within sane band (AUC > 0.4, R@1 > 0.0001)
- numerical drift > 5% on H1 vs v1 classical baselines triggers an investigation (not a pass)

### step 5 - retrain contrastive R1, R2, R3, R4, R6 (hours each, several in parallel)

**does it**:
```bash
for run in R1 R2 R3 R4 R6; do
  python scripts/align.py \
    --run-id ${run}_v3 \
    --niches-dir data/embeddings/niches_v3 \
    --out-dir runs/tnbc-92_v3/${run} \
    --loss [infonce|supcon|barlow per run] \
    --fusion [late|cross_attn per run] \
    --st-features novae_niche gpath2vec_niche \
    --seed 42 &
done
wait
```
50 epochs each, CPU, 60-120 min per run; 2-3 parallel respecting CPU count.

**proposal relevance**: strong. these ARE the proposal's alignment-agent deliverable: "InfoNCE contrastive, 512-d shared space" per the architecture figure, with the documented experimental variants late vs cross-attn (`CLAUDE.md` "alignment strategies" section).

**code review / publishability checks**:
- [ ] every run shares the same test patients (read once from split.json, not re-generated per run)
- [ ] checkpoint.pt saves best-epoch model (best val cosine), not last epoch
- [ ] `attention_weights` column on R4's embeddings_test.parquet (cross-attention interpretability artifact, proposal §"H&E modality agent multi-scale extraction")
- [ ] training_log.csv records per-epoch: epoch, train_loss, val_loss, val_cosine, time_seconds. enables reviewer to plot training curves.
- [ ] DETERMINISM CHECK: run R1_v3 twice in a row with the same seed; resulting metrics_h1_raw.json AUCs must agree to 4 decimal places. if not, there's a hidden non-determinism source (DataLoader workers, CUDNN, dropout in eval mode, etc.). MUST FIX before publishing.
- [ ] supcon loss (R2) uses mc_weights_niche from the bundle dict for soft targets ONLY, not as input - the circular-dependency ban per `program.md` hard constraint #3
- [ ] no mc_weights anywhere in `st_features` (this was the original sin that bit the project in April; the script should hard-assert this)
- [ ] hard negatives policy documented in run_config.json: "within-slide from different tissue compartments; cross-patient negatives excluded" matches proposal §Alignment
- [ ] standard error / 95% CI via patient-level bootstrap (1,000 resamples per proposal §Dataset) recorded alongside point estimates - NOT just point estimates

**acceptance**:
- training_log.csv shows monotonic train loss decrease + monotonic val cosine increase (loss spikes > 2x rolling-mean for > 3 epochs = failure to converge)
- early stopping fires within [10, 50] epochs - earlier than 10 = under-trained; not firing by 50 = under-spec'd patience
- embeddings_test.parquet has expected row count (~35,594 niches, 14 patients, 38 subarrays)
- R4 attention_weights mean entropy across niches > 0.5 of uniform (otherwise the cross-attention degenerated to a hard-max, the dim-collapse failure mode v2 demonstrated)
- determinism re-run check passes

### step 6 - regenerate B4 strict control (5 min)

**does it**: rerun `scripts/_scratch/b4_random_late_fusion_concat.py` such that it writes the full artifact set (embeddings_test.parquet, run_config.json, split.json, run.log, eval/) - the H1 audit caught that v1 B4 was a single 442-byte JSON file. promote out of `_scratch/` since B4 is a load-bearing control, not a one-off.

**proposal relevance**: medium. the B4 → R1 → R4 decomposition ("contrastive loss does ~70% of R4's work; cross-attention adds 30%") is referenced in `tnbc92_results_summary_v2.md` Methods + relevant for proposal §Impact Statement ("noise floor and signal limits").

**code review / publishability checks**:
- [ ] writes full artifact set parallel to other runs
- [ ] xavier init explicitly seeded (torch.manual_seed(42) before model construction)
- [ ] B4 architecture is byte-identical to R1 (verified by reading R1's model code, not by remembering)
- [ ] run_config.json includes `training_steps: 0` and `loss: none` for clarity

**acceptance**: B4 appears in the same H1/H2/H3 eval parquets as other runs; AUC near 0.5 (chance, per random-projection theory).

### step 7 - re-evaluate H1, H2, H3 on the v3 grid (30 min)

**does it**:
```bash
python scripts/eval.py --runs-dir runs/tnbc-92_v3 --out-dir runs/tnbc-92_v3/eval
python scripts/eval_h3_pathway_cca_gpath2vec_v2.py \
  --embeddings-pkl data/embeddings/gpath2vec/fisher_madmean_low_dim512_e5_s1234/fisher_madmean_low_embeddings.pkl \
  --out-dir runs/tnbc-92_v3/eval/H3/pathway_cca_v3 \
  --allow-set-size-drift
python scripts/eval_h3_gpath2vec_noise_floor.py --arm v3_madmean_retrained <new_pkl_if_any>
python scripts/eval_h3_gpath2vec_direction_contribution.py \
  --embeddings-pkl <same as above> \
  --cca-dir runs/tnbc-92_v3/eval/H3/pathway_cca_v3
```

**proposal relevance**: strong. this is the proposal's H1/H2/H3 deliverable. every metric the proposal explicitly names is computed here: R@K, MRR, median rank, AUC, alignment gap, CKA (H1); ARI, silhouette (H2 part A); per-compartment cosine (H2 part B); CCA (H3).

**code review / publishability checks**:
- [ ] every metric reported with 95% CI from patient-level bootstrap (per proposal §Dataset "1,000 resamples to report 95% confidence intervals")
- [ ] BH-FDR applied within each FDR family (declared per family in eval.py)
- [ ] H2 ARI on `mc_megacluster` (niche-level, audit-correct) NOT `archetype` (patient-level pseudobulk - the construct-validity audit finding)
- [ ] H3 view_clean field set correctly for v3-trained runs (z_st + z_mean now ARE circular with v3 because v3 is on the ST input; z_he remains clean. opposite of v1-trained-with-v3-evaluated)
- [ ] every output parquet includes provenance.json next to it (input pkl sha256, git_commit_sha, n_perms, seed)
- [ ] specificity matrix off-diagonal |cos| computed per run from the v3-trained shared latents (the test #3 we did pre-retrain; rerun here on the trained runs)

**acceptance**:
- H1 summary.json includes 8 runs × {R@1, R@5, R@10, MRR, median_rank, AUC, alignment_gap, CKA_before, CKA_after} with bootstrap CIs
- H2 includes mc_megacluster ARI + silhouette per run + patient-z vs biology-z table
- H3 includes per_pathway_cca + direction_contribution + specificity matrix per run
- noise_floor.parquet shows MC z, patient z, ratio (target: ratio > 1 means biology dominates; v3-trained should exceed v3-evaluated-on-v1-trained ratio of 1.03)

### step 8 - regenerate `notebooks/final/05_summary_umaps_v3.ipynb` (1-2 hr)

**does it**: mirror `05_summary_umaps.ipynb` structure (§§3-13) on the v3 grid. this IS the proposal's deliverable: "Reproducible evaluation notebooks ... will be included as output logic" (proposal §Data Sharing).

**proposal relevance**: strong. the explain-to-people artifact; the closest thing the proposal has to a "demo." also satisfies Cameron's request.

**code review / publishability checks**:
- [ ] `sns.set_theme(style='ticks', palette='Set2', context='notebook')` at top (per `feedback_plot_theme.md` memory)
- [ ] `warnings.filterwarnings('ignore')` at top (per `feedback_suppress_warnings.md`)
- [ ] all paths relative to repo root via `Path().resolve().parents[N]` lookup (per `feedback_no_absolute_paths.md`)
- [ ] every plot has axis labels, legend, title; no raw matplotlib defaults
- [ ] UMAPs hashed by `(n_neighbors, min_dist, seed, metric)` and cached - rerunning notebook does not recompute identical UMAPs (cost saving + determinism)
- [ ] z_st and z_mean panels marked "circular with v3 on ST input" - cannot be cited as independent H3 evidence
- [ ] no hardcoded "the answer is X" prose in markdown that's not derived from a code cell above it (Reviewer-defensible: prose numbers must be readable from notebook outputs)
- [ ] include the specificity-matrix heatmap from `h3_gpath2vec_v3.ipynb` §4 in this notebook's H3 section (paired with z_A always per the manuscript-review rule)
- [ ] section structure: §0 scope (read first), §1-7 v1's H1/H2 panels regenerated on v3 grid, §9-13 H3 (AUCell + DAG + spatial + specificity), §14 routing-rule table for omicstra
- [ ] HTML export check: `jupyter nbconvert --to html` produces a working file with all plotly hovers intact

**acceptance**:
- notebook executes top-to-bottom without manual intervention (`jupyter nbconvert --to notebook --execute --inplace` exit 0)
- nbformat HTML export renders all 8 run panels
- numerical claims in markdown match the computed cell outputs (manual scan + ideally a small `assert` in the next code cell)

---

## risks + mitigations

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| 70k-niche drop reduces train-set diversity, hurts contrastive learning | medium | medium | track val cosine per epoch; compare epoch-50 val to v1; if val cosine drops > 0.1 the gap is real |
| R4 cross-attention overfits on the smaller train grid | low | medium | early stopping (patience=10) catches this; flag in training_log if it fires before epoch 20 |
| v3 retrain z_st and z_mean become circular with v3 (was clean post-hoc on v1-trained runs) | high - by design | low - documented | view_clean flag preserved in eval; H3 headline uses z_he across all comparisons; circularity is *expected* in z_st now |
| classical baselines (B1/B2/B3) numerics change beyond rounding due to the 70k drop changing covariance structure | medium | low | re-fit is cheap; numerical drift > 5% on H1 metrics is a flag for investigation |
| existing memory files reference v1 numbers that get stale | high | low | update `memory/MEMORY.md` once v3 retrain lands; add a `project_v3_retrain_result.md` |

---

## estimated total time

| step | estimate |
|---|---|
| 1. parameterize build_niche_join.py | 10 min |
| 2. build v3 niche-join | 30 min |
| 3. parameterize align.py | 5 min |
| 4. classical baselines (B1, B2, B3) | 15 min |
| 5. contrastive retrain (R1-R6, parallel) | 2-4 hr |
| 6. B4 regen | 5 min |
| 7. H1/H2/H3 eval | 30 min |
| 8. v3 summary UMAPs notebook | 1-2 hr |
| **total wall time** | **4-7 hr**, mostly contrastive retrain |

steps 1-3 can be done now (config / code changes). step 4 is fast and decouples cleanly. steps 5-8 are the heavy work but sequential after step 2.

---

## handoff notes

- `data/embeddings/niches/` (v1 niche-join) stays intact. v3 lives at `data/embeddings/niches_v3/`.
- `runs/tnbc-92/` (v1-trained alignment runs) stays intact. v3 lives at `runs/tnbc-92_v3/`.
- v3 H3 numbers from phase 1 (R4 z_A 22.3 etc.) are on v1-trained shared latents + v3 pathway features. they are **not** the same as the post-retrain numbers will be. the phase-1 v3 H3 readout is a preview, not the manuscript number.
- post-retrain, regenerate this file's headline numbers in `notebooks/final/05_summary_umaps_v3.ipynb` and the memory file.