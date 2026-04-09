# program.md - tnbc-92 alignment search space

## status

2026-04-09 update: niche-level alignment with circular dependency fix.
Block 1 (rank1a) PASSED with nuance: Virchow2 niche -> MC linear probe = 21.4% (3x chance).
rank2 architecture redesigned: mc_weights are SUPERVISION ONLY, not ST input.
NEXT: niche aggregation (Block 1 of session todo) -> rank2 update -> subset test -> full run.

## hard constraints (NEVER VIOLATE)

1. **NEVER apply ComBat** - single-institution data, inter-patient variation is biological
2. **NEVER apply Reinhard/Macenko/stain normalization before FM inference**
   - Virchow2/UNI2 are already stain-invariant (PathoROB Virchow2 RI=0.88 reproduced empirically)
   - z-score test on this data: Novae helps (+0.11 sil), Virchow2 hurts (-13% probe)
   - use only timm `create_transform()` with model's `pretrained_cfg`
3. **NEVER add mc_weights to ST input vector** - circular dependency
   - mc_weights are supervision (define positive pairs), NOT features
   - putting them in input AND using them as labels = model predicts its own input
4. **NEVER use within-subarray retrieval as the primary evaluation**
   - CCA within-subarray = 0.667 R@1 is patient leakage shortcut (CCA cross-sub = 0.000)
   - patient-held-out / cross-subarray is the proposal-correct eval
5. **Do not run alignment until subset test (5 sanity checks) passes**
6. **One variable change per experiment** - no multi-variable jumps
7. **Changing alignment_unit requires new experiment series + human approval**

## ALWAYS

- atomic unit is NICHE (k=6 spatial neighbors), not spot
- z-score novae per subarray AFTER niche aggregation (recovers +0.11 sil)
- keep mc_weights separate in bundle dict (loss + eval, never input)
- soft InfoNCE targets from mc_weight cosine similarity (not hard CLIP diagonal)
- aggregate to niche level BEFORE pathway scoring (per-spot too sparse)
- evaluate cross-subarray (patient-held-out)

## proposal scope

- alignment target: shared 512-d space, niche level
- positive pairs: mc_weight similarity (biology-driven, not co-registration)
- evaluation: beat 3 baselines from proposal:
  1. CCA projection
  2. late fusion concatenation
  3. unaligned concatenation
- H1: alignment beats baselines on cross-subarray retrieval
- H2: archetype/compartment coherence in aligned space (ARI vs Wang et al. SAs/MCs)
- H3: pathway signal in aligned space (CCA, spatial maps at niche level)

## architecture (rank2 v2)

H&E side:
  Virchow2 niche (1280-d, raw, no z-score)
  -> Linear(1280, 256) -> GELU -> Dropout(0.3) -> Linear(256, 256)
  -> L2 normalize -> 256-d shared space

ST side (NO mc_weights in input):
  rich niche ST = novae(64, z-scored) + tls(1, raw) + future pathway/cytotrace
  interim st_dim: 65 (novae 64 + tls 1)
  full st_dim: 70+d after pathway pipeline ready
  -> Linear(st_dim, 256) -> GELU -> Dropout(0.3) -> Linear(256, 256)
  -> L2 normalize -> 256-d shared space

loss (soft InfoNCE with biology supervision):
  mc_sim = niche_mc_weights @ niche_mc_weights.T   (n x n)
  target = softmax(mc_sim / tau_target)            (soft labels)
  logits = h_e_emb @ st_emb.T / temperature
  loss = cross_entropy(logits, target)             (soft, multi-positive)

config:
  shared_dim: 256 (was 512)
  dropout: 0.3 (was 0.1)
  lr: 5e-4 (was 1e-3)
  weight_decay: 1e-3 (was 1e-4)
  temperature: 0.1
  tau_target: ~1.0 (controls softness of mc_w-derived targets)
  patience: 10
  epochs: 50

## the rich niche ST representation pipeline

niche definition: central spot + k=6 spatial neighbors (~500um footprint)

per niche, aggregate from k=6 spots:
  niche_novae   = mean(novae[neighbors])      -> z-score per subarray
  niche_tls     = mean(tls[neighbors])         -> raw
  niche_mc_w    = mean(mc_weights[neighbors])  -> untouched (supervision)
  niche_counts  = sum(counts[neighbors])       -> for downstream pipeline

from niche_counts:
  A. Leiden clustering on niche_counts (reduces sparsity)
  B. pseudo-bulk per cluster (aggregate counts)
  C. GSVA on pseudo-bulk for 5 Reactome pathways:
     TGF-beta R-HSA-170834
     Immune R-HSA-168256
     ECM R-HSA-1474244
     Cell Cycle R-HSA-1640170
     PCD R-HSA-5357801
  D. map cluster pathway scores back to niches -> enrichment_scores per niche
  E. CytoTRACE on niche_counts -> trajectory score per niche [HOLD - confirm ST applicability]
  F. gpath2vec on enrichment_scores -> pathway_embed per niche [BLOCKED on Nasim API]

final rich niche ST input (after pipeline complete):
  novae(64) + pathway_embed(d) + enrichment(5) + tls(1) [+ cytotrace(1) if confirmed]
  = 70+d
  mc_weights(14) NOT in input - supervision only

## generalizability for MCP tool

new dataset workflow:
  1. compute Virchow2 niche embeddings (frozen FM)
  2. NMF(k=14) on new ST counts -> new mc_weights
  3. Hungarian-match new factors to reference Wang et al. prototypes
  4. run pathway/cytotrace pipeline (all unsupervised, Reactome-based)
  5. apply trained alignment MLP -> 512d shared space

all steps unsupervised or use known biology. fully generalizable.

## ranked experiment queue (revised)

| rank | task | status | result |
|------|------|--------|--------|
| 1a | Virchow2 niche -> MC 14-class linear probe | DONE | 21.4% (3x chance), PASS with nuance |
| 2 | niche InfoNCE alignment with biology supervision | NEXT | requires niche aggregation + rank2 v2 update |
| 3 | rank 2 + pathway/cytotrace ST input | BLOCKED | gpath2vec API + cytotrace decision |
| 4 | early interaction (cross-attention on Virchow2 patch tokens) | LATER | needs cell-scale extraction |

## stopping rules

- subset test (20 subarrays, 5 epochs) MUST pass 5 sanity checks before full run
  1. loss decreases epoch 1->5
  2. val loss not immediately diverging
  3. R@1 > random (~0.001 cross-sub) for aligned model
  4. no NaN
  5. R@1_pos > R@1_self (sanity that bio retrieval is easier)
- full run target: beat random + CCA + late fusion + unaligned cross-subarray
  even R@1 = 0.05 cross-subarray is a strong novel result
- budget: max 20 experiments, max 5 HITL escalations

## constraint graph

- if alignment_unit == region: classification head only
- if alignment_unit == niche: late_MLP or early_attention
- if alignment_unit == cell: early_attention only (cross-attention on patch tokens)
- if best_epoch == 1 for 2 consecutive runs: STOP, escalate
- if subset test fails any sanity check: do NOT launch full run

## valid experiment series

- exp01* = spot level (COMPLETED - exhausted)
- exp02* = spot level with ComBat (COMPLETED - DEPRECATED, ComBat banned)
- rank1a = Virchow2 niche linear probe (COMPLETED - PASS)
- rank2 = niche InfoNCE with mc_w supervision (NEXT)
- rank3 = niche InfoNCE with full pathway-enriched ST (after gpath2vec ready)
- rank4 = cell-scale early attention (later, needs Virchow2 cell extraction)

## metric

primary: cross-subarray R@1 (patient-held-out test, exact niche self-retrieval)
secondary:
  - R@5, R@10, MRR
  - ARI of aligned-space clusters vs MC labels per subarray (H2)
  - CCA between aligned space and pathway scores per subarray (H3)
fixed - never changed mid-loop.

## baselines (required for every experiment)

- random retrieval (~1/n_spots ≈ 0.001)
- CCA cross-subarray (~0 in our test, the proposal CCA baseline)
- late fusion concat (~random, mismatched dims)
- unaligned (raw cosine, requires same dim or projection)

## what we learned

1. mc_weights from Wang et al. NMF are interpretable cell-state mixtures, not embeddings
2. circular dependency: cannot use mc_weights as both ST input AND positive label
3. CCA within-subarray = 0.667 R@1 is patient leakage shortcut
4. CCA cross-subarray = 0.000 - all baselines collapse cross-patient
5. mc_weights drove almost all CCA signal (mc_w only 14d > rich ST 79d)
6. novae adds noise to CCA (within-sub), useful only as within-subarray context
7. Virchow2 z-score HURTS performance (-13% probe), Novae z-score HELPS (+0.11 sil)
8. PathoROB Virchow2 RI=0.88 reproduced empirically on TNBC-92
9. NMF mc_weights are generalizable: NMF on new data + Hungarian matching to ref factors
10. niche scale (k=6 neighbors) reduces sparsity for pathway scoring vs spot level