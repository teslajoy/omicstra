# niche-level alignment grid - tnbc-92

three-axis ablation at niche resolution over the 260 matched subarrays (novae ∩ virchow2 ∩ gpath2vec). one script (`scripts/align.py`), one evaluator (`scripts/eval.py`), config-driven. MCP wrapping deferred.

## hypotheses

| H | question | primary axis | evidence |
|---|---|---|---|
| H1 | does learned contrastive alignment improve cross-modal retrieval and representation structure vs classical and unaligned baselines? | loss | R@{1,5,10}, MRR, alignment gap, AUC, CKA |
| H2 | does the shared manifold preserve 9 spatial archetypes better than single-modality spaces? | fusion | ARI, silhouette, per-compartment cosine across 17 categories |
| H3 | does the shared space carry interpretable biological signal? | - | pathway coherence (CCA to gpath2vec), spatial grounding (attention maps), reduced patient leakage (linear probe) |

H3 is not a projection competition. the question is whether biology survives alignment. CCA, probes, and spatial maps are readouts providing evidence, not alignment mechanisms.

## runs

niche-level, 260 matched subarrays, 85/15 patient-stratified split, seed 42.

| id | loss | fusion | projection | supervision | serves |
|---|---|---|---|---|---|
| R1 | infonce | late | mlp | matched-pair only | H1, H2, H3 |
| R2 | supcon | late | mlp | mc_weights cosine | H1 |
| R3 | barlow | late | mlp | none (redundancy) | H1 |
| R4 | infonce | cross_attn | mlp | matched-pair only | H2, H3 |
| B1 | classical linear alignment (CCA) | - | - | H1 baseline |
| B2 | classical linear alignment (Procrustes + PCA) | - | - | H1 baseline |
| B3 | unaligned per-modality PCA + L2-norm | - | - | H1 baseline / geometric null |

4 trained + 3 classical. R1 is shared across all three hypotheses. B1/B2 are classical linear-alignment baselines, not alternative projections alongside MLP. R5 (early fusion) is deferred - in a two-stream setup "early fusion" is architecturally ambiguous (Novae-native early fusion would require re-running Novae with H&E graph attributes, a separate experiment). Fusion axis is {late, cross_attn}; H2 uses R1 vs R4.

## supervision distinction

- **supcon** uses mc_weights cosine as soft-positive targets - cohort-specific (TNBC only), ablation signal
- **infonce, barlow** use no labels - cohort-agnostic, the default deployable inference mode
- R1 vs R2 tests whether TNBC-specific supervision outperforms portable alternatives

## fixed across runs

- H&E stream: virchow2_niche 1280d (late, early); 7 tile tokens from virchow2_cell (cross_attn)
- ST stream: `novae(64, z-score per sub) + gpath2vec(512) = 576d`
  - TLS dropped from core input (only 90/260 coverage, no imputation -> no bias)
- projection: MLP, 2 layers, ReLU + BatchNorm + Dropout(0.3), output 512d L2-normed
- loss: τ=0.07 (infonce, supcon); barlow default
- batch = one subarray, max 50 epochs, early stop on val cos-sim patience 10
- hard negatives: within-slide from different compartment label
- cross-patient negatives excluded from training pool (proposal spec)

## scope per analysis

- **primary (260 subarrays)**: H1 retrieval, H1 CKA, H2 archetype ARI + silhouette, H3 pathway CCA, H3 patient-leakage probe
- **annotated subset (90 subarrays)**: H2 per-compartment cosine across 18 categories, within-slide cross-compartment hard-negatives (fall back to random within-slide for the other 170), TLS-stratified readout as H2 side analysis

archetype labels are per-patient so propagate to all 260 via Clinical.RDS. compartment and TLS require per-spot annotation and are restricted to 90.

## consumption

- H1: R1, R2, R3, B1, B2, B3, random -> R@K curves, CKA table, alignment-gap + AUC (260)
- H2: R1, R4 -> ARI + silhouette on 9 archetypes (260); per-compartment cosine across 18 categories (90 subset); TLS-stratified cosine as side readout (90 subset)
- H3 (on H1/H2 winners):
  - **pathway coherence** - CCA(shared latent, gpath2vec niche 512d) for 5 target pathways (TGF-β, Immune System, ECM, Cell Cycle, Programmed Cell Death); canonical correlations + permutation significance
  - **spatial grounding** - R4 attention weights per niche aggregated by compartment; does ST query select morphologically-correct H&E tiles?
  - **reduced patient leakage** - linear probe for patient_id on shared latent vs raw modalities; lower probe accuracy in shared = better disentanglement. parallels the gpath2vec validation finding (biology z↑, patient z↓).

## file layout

```
scripts/align.py                           one config -> one run (checkpoint + metrics.json + run_config.json)
scripts/eval.py                            reads run outputs -> hypothesis.json
projects/tnbc-92/alignment/config/         per-run config json (R1.json ... B3.json)
runs/tnbc-92/{run_id}/                     run artifacts
notebooks/experiments/                     synthesis plots + winner.json only (not training)
```

CLI:

```
python scripts/align.py --config projects/tnbc-92/alignment/config/R1.json --run-id R1
python scripts/eval.py --hypothesis H1 --runs R1 R2 R3 B1 B2 B3
```

## data contract (blocking prereq)

one parquet per subarray at `data/embeddings/niches/{subarray}.parquet`, 260 files. columns:

- `spot_id` (index), `subarray`, `patient_id`
- `virchow2_niche` (1280d)
- `virchow2_cell_tokens` (7, 1280) - for cross_attn
- `novae_niche` (64d, z-scored per sub)
- `gpath2vec_niche` (512d)
- `tls` (1d)
- `mc_weights_niche` (14d, niche-pooled soft targets for supcon only)
- `archetype` (int, Wang 9-archetype, H2 label)
- `compartment` (str, Wang 17-category, H2 stratification)

built by a small one-shot notebook that reads the individual embedding parquets, applies niche pooling over self + 6 neighbors, joins everything on `(subarray, spot_id)`.

**neighborhood parity**: virchow2_niche and novae_niche_full agreed 1075/1075 on TNBC1_CN1_C1 (verified 2026-04-22). same KDTree(pixel_coords).query(k=7) with K_NEIGHBORS=6 on both sides. direct join on `(subarray, spot_id)` is valid.

## blocking pre-steps

1. ~~verify virchow2_niche k=6 neighborhoods match novae neighbor_spot_ids~~ DONE 2026-04-22
2. build the niche join table (section above)
3. `scripts/align.py` skeleton: config parser, data loader that reads `data/embeddings/niches/`, train loop stub, metrics writer
4. subset smoke test: 20 subarrays, 5 epochs, R1 config only. checks: loss decreasing, val not diverging, R@1 > random, no NaN

## sequence

1. niche join table (1 notebook)
2. `scripts/align.py` skeleton + R1 subset smoke test
3. R1 full run -> H1 partial
4. R2, R3, B1, B2, B3 -> H1 complete
5. R4, R5 -> H2
6. H3 on R1 + R4 + B1
7. `notebooks/experiments/synthesis.ipynb` -> winner.json

## orchestration handoff (deferred)

after H1/H2/H3 pass, `align.py` and `eval.py` become MCP agent tool calls. CLI args map 1:1 to agent config surface:

```
alignment_agent.train(loss, fusion, projection, supervision, split)
eval_agent.hypothesis(h, runs)
```

script first, wrap later. MCP work belongs in `src/mcp/`, not here.