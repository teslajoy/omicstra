# analysis pipeline steps

every operation in the published chain, one line each, in order. shapes in
`(...)`, symbols generalize, the seed cohort's value follows in `[...]`.

read as: `name: output = operation(input)  # condition -> branch`.

the scripts are the method; this file is the index to their maths. where a line
and a script disagree, the script is right and this file is a bug.

## 0 · symbols

```
S            samples (subarrays)                      [280 ingested, 259 joined]
n_s          spots in sample s                        [median ~960, 1934 max]
N            niches after the join                    [208,786]
k            spatial neighbours per niche             [6 -> 7 spots]
D_he         H&E encoder width                        [1280 Virchow2 | 1536 UNI2]
D_st         ST encoder width                         [64 novae_latent]
D_pw         pathway embedding width                  [512 gpath2vec]
D            shared space width                       [512]
P            subjects                                 [92 patients]
C            supervision classes                      [14 mc_megacluster]
pitch        spot centre-to-centre, µm                [150]
nn           measured nearest-neighbour distance, px  [158.6 small / 528.7 HD]
u            nominal tile width, µm (declared)        [128 -> 95.9 actual at the true pitch]
σ            coordinate -> image scale                [3.3334 = 31744/9523]
```

## 1 · one XYZ frame (precondition)

```
case A  same physical section, both assays        -> no registration
  xy_img = xy_assay · σ                           # one scalar, σ declared     [seed cohort]
case B  serial sections / separate slides         -> registration REQUIRED
  step 1  tissue mask per modality                 M = mask(I)
  step 2  image<->image                            T_img = register(I_mov, I_fix)
  step 3  assay<->assay                            T_st  = align(xy_a, xy_b)
  step 4  apply, then verify                       resid = median |T(xy_a) - xy_b|
  gate    if resid > pitch/2 -> niche membership changes -> declare as an arm, not a fix
```

current options for step 2/3, with what each assumes:

```
image<->image   VALIS (WSI, multi-stain, rigid->affine->nonrigid) | ANTs/elastix (classical, control)
assay<->assay   PASTE / PASTE2 (OT on shared spots; PASTE2 allows partial overlap)
                moscot (OT, scales to 1e6 cells, mappings + lineage)
                GPSA (probabilistic common coordinate system, warps several slides into one frame)
                CAST (graph/feature-based, handles different technologies)
                STalign (LDDMM diffeomorphic; ST -> histology, so also covers step 2->3 in one)
choose by       do the slides share tissue? (PASTE2) · different assay generations? (CAST)
                need one common frame for many slides? (GPSA) · need histology as the target? (STalign)
record          the transform, its residual and its software version are a declaration,
                not a detail: every downstream niche depends on them
```

`PIVOT` — I could not confirm which tool this names; treat the list above as the
candidates and add it once identified.

## 2 · niche geometry

```
coords        X_s (n_s, 2)                           # measured, not idealised     [pixel_x, pixel_y]
neighbours    idx = KDTree(X_s).query(X_s, k+1)      # self included, by RANK not radius
counts        c_i = #{j : idx_ij valid}              # boundary spots keep fewer
niche         ν_i = {i} ∪ {idx_i1..idx_ik}           # 7 spots                     [~1400 cells]
property      rank selection on a square lattice takes 2 of 4 diagonals
              anisotropic -> nearer pair wins by distance  [99.8% of interior spots, 7th/6th = 1.095]
              isotropic   -> tie, argsort decides -> declare the alternative (k=4 | k=8)
footprint     report in µm, never in k alone         [~500 × 322 µm]
```

## 3 · H&E arm

```
tile size     p = round(u · nn_img / pitch)          # pitch is an INPUT to p    [338 px = 95.9 µm]
centres       c_i = round(x_i · σ)                   (n_s, 2)
boxes         b_i = clamp(c_i ± p/2, image)          # edge boxes come back small
crop          T_i = resize(pad_black(crop(I, b_i), p), 224)
encode        h_i = f_he(T_i)[CLS]                   (D_he,)                     [Virchow2 1280]
cell cache    H_cell = stack(h_i)                    (n_s, D_he)
niche pool    h^ν_i = mean_{j ∈ ν_i} h_j             (D_he,)   # unweighted, encode THEN pool
token stack   T^ν_i = [h_i, h_{idx_i1..k}]           (k+1, D_he) = (7, 1280) -> flat 8960
```

## 4 · ST arm

```
counts        C_s (n_s, G) raw integer               # from the assay's own selection
gene ids      Ensembl -> symbol, strip version, sum duplicates
graph         E_s = Delaunay(X_s)                    # topology is scale-free
edge feature  e_uv = ||x_u - x_v|| · scale_µm / 20   # scale_µm is an INPUT       [1.2195 as built]
encode        z_i = f_st(C_s, E_s)                   (D_st,)                     [novae_latent 64]
floor         if n_s < min_units -> drop sample      [512 prototypes; 19/281 dropped]
niche pool    z^ν_i = mean_{j ∈ ν_i} z_j             (D_st,)
normalise     z^ν ← (z^ν - µ_s) / σ_s                # per sample, AFTER the join filter
```

## 5 · pathway arm

per niche, from raw counts of its 7 spots:

```
normalise     A = log1p(c7 / libsize · 1e6)          (7, g)
robust CV     r_g = MAD_g / (mean_g + ε)             # MAD over the 7 spots
background    B = {g : mean_g > 0}                   # locally detected           [min 100]
foreground    F = top 25% of {g ∈ B : r_g > 0} by r  # percentile on NON-ZERO only [min 20]
qc            drop niche if Σ UMI < 1000 | <3 of 7 spots ≥ 200 UMI
universe      Reactome leaves at level `low`, TF-filtered, size 15..500
enrich        per pathway P:  Fisher(|F ∩ P|, |F|, |B ∩ P|, |B|)  # background fixed = B
correct       q = BH(p) over pathways                 # sig: q < 0.05 ∧ OR > 2
graph         nodes = niches ∪ pathways; edge (ν,P) weight = 1 - q   # significant rows only
walks         metapath2vec: 10 walks/node, length 100, window 5, 5 epochs, lr 0.005, seed 1234
embed         g_ν, g_P ∈ R^{D_pw}                    [512]
niche vector  g^ν_i = embedding of node ν_i           (D_pw,)
pathway set   for target Q: members = {Q if embedded} ∪ {embedded descendants(Q)}
              testable if |members| ≥ 5              # else excluded, reported separately
pathway view  Y_i = [cos(g^ν_i, g_m) for m ∈ members] (|members|,)
```

## 6 · the join (one row per unit of analysis)

```
spot set      U_s = spots(H&E) ∩ spots(ST)           # a unit needs both sides
neighbours    lists computed per modality; identical when U_s matches   [all 260 here]
padding       missing neighbour -> repeat centre     # keeps (7, D_he) fixed
row i         [h^ν, T^ν, z^ν, g^ν, mc_w^ν, labels, subject, sample]
coverage      drop row if g^ν is NaN                 # no significant pathway    [67,131 dropped]
funnel        every stage records n_in, n_out, reason  # unexplained drop = error
supervision   mc_w^ν = mean_{j ∈ ν} w_j              (C,) — NEVER an input feature
guard         st_features ∩ supervision = ∅          # G6, hard error
```

## 7 · fusion (the trained arms)

```
inputs        x_he = h^ν (D_he) | tokens T^ν (7, D_he);  x_st = concat(selected st features)
              [576 = 64 ⊕ 512; R6 ablation = 64]
block         MLP(d_in, D): LN -> Linear -> ReLU -> BN -> Dropout(0.3) -> Linear -> L2norm
late          z_he = MLP(x_he); z_st = MLP(x_st)                      (D,) each
cross-attn    q = W_q·LN(x_st) (1, D);  K, V = W_k·LN(T^ν), W_v·LN(T^ν) (7, D)
              α = softmax(q Kᵀ / √D)                 (1, 7)
              z_he = L2(W_o·LN(α V))                 # content is H&E ONLY; ST shapes α only
              z_st = L2(W_s·q)
batch         one sample per batch                   # the InfoNCE denominator is within-sample
split         three-way by SUBJECT, stratified where every class has ≥2   [70/15/15, seed 42]
optimise      AdamW(lr 5e-4, wd 1e-3), ≤50 epochs, early stop on val cosine, patience 10
select        checkpoint = best val cosine, reloaded before test
```

losses, pick one per arm:

```
InfoNCE       L = ½[CE(z_he z_stᵀ/τ, I) + CE(z_st z_heᵀ/τ, I)]        τ=0.07
AnInfoNCE     logits = (z_he ⊙ e^{2a}) z_stᵀ / τ                      a learnable (D,), a=0 ⇒ InfoNCE
SupCon-soft   target = softmax(cos(mc_w, mc_wᵀ)/τ_t); L = ½[CE_soft both ways]   τ_t=0.1
Barlow        C = z̃_heᵀ z̃_st / N;  L = Σ(diag(C)-1)² + λ Σ off(C)²   λ=5e-3
```

## 8 · classical baselines (no training)

```
B1 CCA        Cxx, Cyy, Cxy from train; W = Cxx^{-½} Cxy Cyy^{-½}; SVD; project; L2   reg 1e-4
B2 Procrustes PCA_D each modality on train; R = argmin ||HR - S||_F (orthogonal); apply
B3 unaligned  PCA_D each modality, L2, NO alignment                   # geometric null
B4 random     R1's architecture exactly, random Xavier init, NO training   # architecture null
rule          every baseline sees EXACTLY the split its leader saw
```

## 9 · evaluation

```
H1 retrieval  sim = Z_he Z_stᵀ over held-out SUBJECTS, pooled across samples
              rank_i = position of i in row i;  R@K, MRR, median rank
              gap = mean(diag) - mean(off);  AUC = matched vs sampled mismatched
H1 geometry   CKA_lin(X, Y) = ||Xcᵀ Yc||_F² / (||Xcᵀ Xc||_F ||Ycᵀ Yc||_F)
              ρ_geo = spearman(upper(S_x), upper(S_y))   on a 5,000-unit subsample
H2-A group    KMeans(k = #classes of the label, n_init 10) -> ARI; silhouette on ≤5,000 units
              [k=9 archetypes in the eval.py rollup; k=14 mc_megacluster in the cited table]
              linear probe = logreg accuracy vs chance   # load-bearing when ARI is geometry-biased
H2-B decode   per compartment: mean_i cos(z_he, z_st);  contrast by Welch z (unpooled)
H2-C bio/subj cross-subject pairs only; δ = mean(sim | same label) - mean(sim | diff label)
              null: permute the label ACROSS SUBJECTS, 10,000×   -> z = (δ - µ_0)/σ_0
              ratio = z_bio / z_subject      # >1 required to claim suppression
H3 pathway    Z (N, D) shared view;  Y (N, |members|) pathway cosines
              option B (in-sample):  r_B = top canonical corr(Z, Y) via QR + SVD
              option A (transfer):   fit (a, b) on train subjects    [3 held-out subjects]
                                     r_A = pearson(Z_te a, Y_te b)      # held-out subjects
              null A: permute Y rows within train, refit directions, re-score test   500×
              p = (1 + #{|null| ≥ |obs|}) / (1 + n);  q = BH over the testable family
              claim: q < 0.05 in the confounder-clean view                 [z_he]
subject probe logreg(standardised Z) -> accuracy vs 1/P chance   # confounder strength, not a verdict
```

## 10 · guards (each is a predicate, not a comment)

```
G1 label      NMI(label, subject) ≤ 0.5      else every clustering score is diagnostic  [0.645 fails]
              exempt when the test is built cross-subject with a subject-level null (H2-C)
G2 held-out   every arm scored on the SAME held-out subjects; n_test identical
G3 bio/subj   z_bio / z_subject ≥ 1 and ≥ raw-modality floor
G4 specificity max off-diagonal |cos| between axis vectors ≤ 0.5, reported beside every magnitude
G5 noise      empirical p, (1 + #{null ≥ obs}) / (n + 1) — never a normal approximation
G6 circular   a signal is input XOR supervision, never both; a view whose ST input carries
              the pathway embedding is circular for pathway claims
```

## 11 · instantiating a new cohort

```
declare   platform: pitch, spot diameter, σ, tile px, st_graph method + scale_µm
declare   encoders by name (registry resolves width and floor), supervision column, subject column
declare   read_only_inputs once results exist   # finished caches AND the finished run grid
measure   lattice: nn distance, neighbour-distance profile, footprint in µm
measure   encoder floor vs n_s per sample -> drop policy
verify    every port against the cache it replaces: drift, cosine, and top-k neighbour sets
then      steps 2-9 are unchanged code; only the declarations differ
```

## 12 · what is computed vs declared

```
computed  every number above
declared  geometry, encoders, policies, splits, read-only paths, alternatives not run
human     notes, caveats, the reason a refusal stands
rule      an alternative that changes a vector is an ARM (declared, re-run, scored),
          never a silent fix
```
