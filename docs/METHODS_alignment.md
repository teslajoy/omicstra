# Fusion and alignment: the methods, and the maths

Every method in the ten-run grid, what it does and what it optimises. Written
from the code, with line references, so it can be checked rather than trusted.

Source files:

| file | what is in it |
|:--|:--|
| `scripts/align.py` | the six trained runs - towers, fusions, losses |
| `scripts/align_classical.py` | B1 CCA, B2 Procrustes, B3 unaligned PCA |
| `scripts/align_b4.py` | B4, random-init, zero training steps |

---

## 0 · Notation and what is fixed

For each niche $i$:

- $x^{\text{tile}}_i \in \mathbb{R}^{7 \times 1280}$ - Virchow2, one vector per spot in the niche
- $x^{\text{he}}_i \in \mathbb{R}^{1280}$ - the mean of those seven
- $x^{\text{st}}_i \in \mathbb{R}^{576}$ - Novae 64 concatenated with gpath2vec 512

All three encoders are **frozen**. Nothing below fine-tunes them. The only
learned parameters are the projections into the shared space
$\mathbb{R}^{d}, d = 512$, and every method L2-normalises its outputs, so all
similarities are cosines on the unit sphere:

$$z \mapsto \frac{z}{\lVert z \rVert_2}$$

gpath2vec enters the **ST tower only**. There is no `he_features` key in any run
config, so the H&E side is Virchow2 in all ten runs.

---

## 1 · The projection block

`MLPBlock`, `align.py:224`. Both towers of the late-fusion model are one of
these.

$$\text{MLP}(x) = \text{normalize}\big(W_2 \,\sigma_{\text{drop}}(\text{BN}(\text{ReLU}(W_1 \,\text{LN}(x))))\big)$$

LayerNorm first is not cosmetic: **Novae emits raw GAT output** with mean norm
2.618 and no normalisation of its own, while Virchow2 emits raw CLS tokens. Two
inputs on very different scales enter the same architecture, and the LayerNorm is
what makes that safe.

---

## 2 · Late fusion - R1, R2, R3, R5, R6

`LateFusion`, `align.py:243`. Two independent towers. The seven H&E tiles are
**mean-pooled before the model ever runs**, so the two modalities never interact
until the loss compares them.

$$z^{\text{he}}_i = \text{MLP}_{\text{he}}(x^{\text{he}}_i), \qquad
  z^{\text{st}}_i = \text{MLP}_{\text{st}}(x^{\text{st}}_i)$$

This is the proposal's default. What differs between R1, R2, R3, R5 and R6 is
only the **loss** and the **ST input** - the architecture is identical.

R6 is the one ablation that matters: $x^{\text{st}}$ is Novae 64 **only**, no
gpath2vec. R1 versus R6 is therefore the single matched comparison in the grid
that isolates the pathway representation.

---

## 3 · Cross-attention - R4

`CrossAttnFusion`, `align.py:269`. The seven tiles are **not** pooled. The
molecular side forms a query and attends over them.

$$q_i = W_q\,\text{LN}(x^{\text{st}}_i) \in \mathbb{R}^{d}, \qquad
  K_i = W_k\,\text{LN}(x^{\text{tile}}_i), \qquad
  V_i = W_v\,\text{LN}(x^{\text{tile}}_i) \in \mathbb{R}^{7 \times d}$$

$$\alpha_i = \operatorname{softmax}\!\left(\frac{q_i K_i^{\top}}{\sqrt{d}}\right) \in \mathbb{R}^{7}, \qquad
  h_i = \alpha_i V_i$$

$$z^{\text{he}}_i = \text{normalize}(W_o\,\text{LN}(h_i)), \qquad
  z^{\text{st}}_i = \text{normalize}(W_s\, q_i)$$

**The design constraint that makes this legitimate**, and it is stated in the
code: `z_he` is derived **purely from the tiles**. The ST side supplies only the
attention *weights* $\alpha_i$, never any content. There is no residual path from
$x^{\text{st}}$ into $z^{\text{he}}$.

Without that constraint the retrieval result would be circular - the H&E
representation would contain the thing it is being asked to retrieve. With it,
what R4 learns is *which of the seven spots to look at*, given the molecular
context. Everything the H&E vector contains still came from the image.

The attention weights are returned, so which tiles a niche attended to is
inspectable.

---

## 4 · The losses

### InfoNCE - R1, R4, R6

`align.py:319`. CLIP-style, symmetric. Matched pairs sit on the diagonal.

$$\ell_{ij} = \frac{\langle z^{\text{he}}_i, z^{\text{st}}_j\rangle}{\tau},
  \qquad \tau = 0.07$$

$$\mathcal{L} = \tfrac{1}{2}\Big[
  \operatorname{CE}(\ell, \text{diag}) + \operatorname{CE}(\ell^{\top}, \text{diag})
\Big]$$

Each niche's own molecular partner must beat every other niche in the batch. The
other niches are the negatives - nothing is labelled, and this is why **H1 is
scored against nothing external**: the supervision is the pairing itself, which
is guaranteed because both assays came off the same section.

### AnInfoNCE - R5

`align.py:329`. InfoNCE with a **learned diagonal metric** on the shared space.
A per-dimension parameter $s \in \mathbb{R}^{d}$ is registered on the model.

$$\ell_{ij} = \frac{z^{\text{he}\top}_i \operatorname{diag}(e^{2s}) z^{\text{st}}_j}{\tau}$$

At $s = 0$ this is exactly InfoNCE, which is why it warm-starts cleanly. Learning
$s$ lets the loss weight some axes of the latent more heavily than others - a
diagonal Mahalanobis metric rather than plain cosine.

The motivation was specific: **R4's specificity finding.** Standard InfoNCE
treats all directions equivalently, which on correlated pathway-anchored inputs
can alias directions together. An anisotropic temperature can down-weight axes
that carry no matched-pair signal.

### SupCon - R2

`align.py:354`. The only run with **external supervision**, and the only one
where the targets are not the identity.

Wang's per-niche megacluster weight vectors $m_i$ define a soft target:

$$T = \operatorname{softmax}\!\left(\frac{\hat{m}\hat{m}^{\top}}{\tau_t}\right),
  \qquad \tau_t = 0.1, \quad \hat{m} = \text{normalize}(m)$$

$$\mathcal{L} = -\tfrac{1}{2}\Big[
  \textstyle\sum_j T_{ij}\log p^{\text{he}}_{ij} + \sum_j T_{ij}\log p^{\text{st}}_{ij}
\Big]$$

So niches with similar megacluster composition are pulled together even when
they are not the same niche. **`mc_weights` is used as supervision only.** It is
never an input feature, because using the same signal as both input and positive-
pair definition is the circularity this project explicitly guards against.

### Barlow Twins - R3

`align.py:368`. Not contrastive at all. It operates on the **cross-correlation
matrix** between the two standardised views.

$$C = \frac{\tilde{Z}^{\text{he}\top}\tilde{Z}^{\text{st}}}{N}
  \in \mathbb{R}^{d \times d}, \qquad
  \tilde{Z} = \frac{Z - \mu}{\sigma}$$

$$\mathcal{L} = \underbrace{\sum_k (C_{kk} - 1)^2}_{\text{invariance}}
  + \lambda \underbrace{\sum_{k \neq l} C_{kl}^2}_{\text{redundancy reduction}},
  \qquad \lambda = 5\times10^{-3}$$

Drive the diagonal to 1 - the same dimension should agree across modalities - and
the off-diagonal to 0, so different dimensions carry different information. No
negatives, no batch-level competition.

That difference shows up in the results: **R3 is the one aligned run that loses
to a classical baseline on retrieval**, 0.674 against Procrustes at 0.706. It is
optimising decorrelation, not ranking, and H1 measures ranking.

---

## 5 · The classical baselines

None of these train. All three are closed-form, fitted on the training patients
and applied to held-out patients unchanged.

### B1 - CCA

`align_classical.py:79`. Closed-form via SVD of the whitened cross-covariance,
rather than `sklearn.CCA`, which does not scale here.

With $C_{xx}, C_{yy}, C_{xy}$ the (regularised) covariances, whiten each side and
take the SVD of the whitened cross-covariance:

$$M = C_{xx}^{-1/2}\,C_{xy}\,C_{yy}^{-1/2} = U\Sigma V^{\top}$$

$$A = C_{xx}^{-1/2}U_{:k}, \qquad B = C_{yy}^{-1/2}V_{:k}$$

$$z^{\text{he}} = (X^{\text{he}} - \mu_{\text{he}})A, \qquad
  z^{\text{st}} = (X^{\text{st}} - \mu_{\text{st}})B$$

CCA finds the directions of **maximum correlation** between the two spaces. That
is precisely why it fails H2-C: on this cohort the most correlated thing between
morphology and expression is *which patient the tissue came from*, so CCA
amplifies it by construction. Bio/patient ratio 0.071, below the raw-H&E floor of
0.137, and **contraindicated** in the routing contract.

### B2 - Procrustes

`align_classical.py:117`. PCA each modality to $d$, then solve for the best
**orthogonal** map between them.

$$R = \arg\min_{R^{\top}R = I} \lVert H R - S \rVert_F = UV^{\top},
  \qquad H^{\top}S = U\Sigma V^{\top}$$

A rotation only - no stretching, no reweighting. B2 and B3 are therefore
rotation-equivalent on any within-space metric and must diverge on cross-modal
ones. That is used as a **scoring self-check**: if B2 ever equals B3 on a
cross-modal metric, the test is broken.

### B3 - unaligned PCA

`align_classical.py:136`. PCA each modality independently, L2-normalise, and do
nothing else. The geometric null: what does cross-modal cosine look like when the
only structure is per-modality covariance?

Answer on this cohort: **AUC 0.440, below the 0.500 chance floor**, with a
negative alignment gap. Correct sanity behaviour, and the reason it is in the
grid.

### B4 - random init

`align_b4.py`. The late-fusion architecture, Xavier-initialised, **zero training
steps**. It isolates whether the contrastive loss adds anything over a random
projection of the same features.

It does, for retrieval - B4 sits at chance. It does **not** for the morphology
decode, where B4 wins outright at z 25.5 while several trained runs actively
degrade the signal.

The mechanism is **not** a rotation. A linear probe is invariant under orthogonal
transforms - rotate the space and the probe simply rotates its weight vector to
match, scoring identically. Whatever costs the trained runs their morphology
signal has to be something a rotation cannot do. What training applies is a
learned **non-linear** map (ReLU, batch norm, L2 projection onto the sphere), and
measurably it **destroys variance directions** rather than turning them. In
participation ratio, measured in §6:

```
raw Virchow2 niche features   16.6      what the encoder hands over
B2 / B3   rigid linear        16.1      a rotation preserves it, as it must
B4        random non-linear   17.3      untrained, so nothing is destroyed
R4        trained contrastive  2.8      three usable directions out of 512
```

B4 keeps roughly seventeen effective directions and R4 keeps three. A hyperplane
can only use what is still there.

This single control is what licenses the headline: **alignment creates
correspondence, not information.**

---

## 6 · What actually differs: the shape of the space

The classical baselines and the contrastive runs differ in one structural way,
and everything else follows from it.

### The formal difference, in one line

A **linear** map's effect on a pair of points depends only on their difference
vector, never on *where in the space they sit*: $f(a) - f(b) = M(a - b)$. Two
niches that are close in Virchow2 are close after **any** linear map, up to one
global stretch factor per axis. A **non-linear** map has no such guarantee - a
ReLU boundary can fall between two neighbours and send them to opposite ends.

That is the whole difference. Non-linearity buys **location-dependence**.

| | B2 Procrustes | B1 CCA | R1-R6 contrastive |
|:--|:--|:--|:--|
| the map | orthogonal $R$, $R^{\top}R = I$ | affine, whitened | piecewise-affine MLP, then L2 |
| does it stretch | no, a rigid isometry | yes, one fixed factor per axis, the same everywhere | yes, and the factor depends on where you are |
| separate two points close in the input | never | only by inflating a low-variance axis | yes, if a ReLU boundary falls between them |
| optimum | closed form, global | closed form, global | SGD, non-convex, seed 42 |
| what it optimises | squared error of the rigid overlay | total correlation | rank of the true partner against the batch |

### What that does here, measured

Held-out test niches, `runs/tnbc-92_v3/*/embeddings_test.parquet`, n = 35,594,
H&E side. **PR** is the participation ratio $(\sum\lambda)^2 / \sum\lambda^2$ -
how many of the 512 directions carry real variance. **local** is how much closer
a niche's nearest neighbour is than a random niche, in units of the cloud's own
spread.

| run | PR | top PC | mean cos | local |
|:--|--:|--:|--:|--:|
| B1 CCA | **257.8** | 2.2% | 0.010 | **9.87** |
| B2 Procrustes | 16.1 | 17.7% | 0.024 | 3.38 |
| B3 unaligned | 16.1 | 17.7% | 0.022 | 3.41 |
| B4 random init | 17.3 | 16.7% | 0.745 | 2.45 |
| R1 InfoNCE | 6.4 | 31.4% | 0.105 | 2.16 |
| R2 SupCon | 3.6 | 41.5% | 0.036 | 1.78 |
| R3 Barlow | 2.8 | 55.7% | 0.551 | 1.37 |
| R4 cross-attn | **2.8** | 49.9% | 0.589 | 1.47 |
| R5 AnInfoNCE | 6.2 | 32.3% | 0.101 | 2.11 |
| R6 Novae only | 6.3 | 31.2% | 0.107 | 2.26 |

So the intuition that contrastive training "stretches things out" is **backwards
on this data.** It compresses, hard. CCA is the one that stretches.

**CCA's spread is manufactured, not discovered.** $C_{xx}^{-1/2}$ forces the
output covariance toward the identity by construction, so PR near 258 is what
whitening does, not evidence that CCA found 258 directions of biology. The top
component holds 2.2% of variance because whitening flattened the spectrum
deliberately.

**Contrastive drops below the input's own concentration.** Raw Virchow2 niche
features sit at PR 16.6. The rigid baselines carry that through unchanged
(16.1) - a rotation must. B4 carries it too (17.3), untrained. The trained runs
land at 2.8 to 6.4, so training is not reorganising 512 directions, it is
**discarding most of them**.

**Two different shapes, not two positions on one scale.** CCA is globally
isotropic (mean cosine 0.010, essentially orthogonal) with extremely sharp local
structure (contrast 9.87): many tiny hard islands scattered over a big sphere.
Cross-attention is the reverse - one continuous smear inside a narrow cone (mean
cosine 0.589, contrast 1.47). Being spread out is not the same as being mixed,
which is exactly why CCA can be maximally spread and still have **0.999** 30-NN
patient purity while R4 sits at 0.327.

**Barlow is structurally blind to this.** `barlow_loss` divides each dimension by
its own standard deviation (`align.py:372-373`) before forming $C$, so the loss
is invariant to per-dimension scale and **cannot see variance collapse at all**.
A direction that has nearly died gets rescaled back up to look healthy. R3 is
tied for the most collapsed run in the grid despite redundancy reduction being
its entire purpose.

### The limit of this metric

Participation ratio measures **variance concentration, not information**. A
low-variance direction can still decide a cosine ranking, which is how R4 holds
PR 2.8 and the best retrieval AUC in the grid at the same time. What PR does
predict is what a **hyperplane** can reach - which is why it explains the B4
morphology-decode result in §5 and does not explain the retrieval result.

---

## 7 · The grid

| run | fusion | loss | ST tower |
|---|---|---|---|
| R1 | late | InfoNCE | Novae 64 + gpath2vec 512 |
| R2 | late | SupCon | Novae 64 + gpath2vec 512 |
| R3 | late | Barlow | Novae 64 + gpath2vec 512 |
| R4 | **cross-attention** | InfoNCE | Novae 64 + gpath2vec 512 |
| R5 | late | **AnInfoNCE** | Novae 64 + gpath2vec 512 |
| R6 | late | InfoNCE | **Novae 64 only** |
| B1 | - | closed-form CCA | Novae 64 + gpath2vec 512 |
| B2 | - | orthogonal Procrustes | Novae 64 + gpath2vec 512 |
| B3 | - | none, independent PCA | Novae 64 + gpath2vec 512 |
| B4 | late | none, 0 steps | Novae 64 + gpath2vec 512 |

Two comparisons in that table are clean and the rest are not:

**R1 vs R6** - identical architecture, identical loss, one input removed. The
only matched ablation of gpath2vec in the grid.

**R1 vs R4** - identical loss, identical inputs, pooled versus un-pooled tiles.
The early-versus-late fusion question.

Everything else varies more than one thing at a time, which is why the grid is
read as *which method for which question* rather than as a controlled experiment.

---

## 8 · A note on probes, since the word "linear" is doing work

A **probe** is a readout head fitted on top of a frozen representation. Nothing
upstream moves. The only question it answers is *how accessible is label $y$ from
vector $z$*, and the answer depends entirely on what the head is allowed to be.

### Linear vs non-linear, compactly

| | linear probe | non-linear probe |
|:--|:--|:--|
| the model | $\hat{y} = \operatorname{softmax}(Wz + b)$ | $\hat{y} = \operatorname{softmax}(W_2\,\sigma(W_1 z))$, or kNN, or an RBF-SVM |
| decision boundary | hyperplanes between class pairs; each class region an intersection of half-spaces | any surface, curved, disconnected, arbitrarily folded |
| capacity | $O(dC)$ parameters, fixed by the input | grows with width or with $N$; kNN's capacity **is** the training set |
| what it measures | did the representation **organise** $y$ into a direction | is $y$ **present anywhere** in the numbers |
| how it fails | says no when the information is there but folded | stops separating representations, and on held-out groups overfits back down |

The one-line version: **a linear probe asks whether the geometry already sorted
the classes; a non-linear probe asks whether the information survived at all.**
Both are legitimate, they are just different questions, and only the first one
predicts anything about how the space will behave downstream.

"Deeper probe" means any of: an MLP with a hidden layer, kNN, a kernel SVM,
gradient boosting. Not more data, not more folds, not a bootstrap.

### Why the first question is the one this project wants

Everything that consumes the shared space is a **geometric** operation, and each
one is confined to flat structure for a specific reason: k-means carves Voronoi
cells, whose boundaries are hyperplanes; CCA is by definition a linear map;
cosine ranking is a projection onto one direction. None of them can follow a
curved boundary. So accuracy a non-linear head reaches by folding the space is
unreachable by any actual consumer, and quoting it would overstate what the
embedding is good for.

### Measured, rather than asserted

Three probes on the same held-out niches, pathologist categories with at least
500 niches (K = 6, chance 0.167), 4-fold **grouped by patient**, balanced
accuracy. Test score first, train score in brackets.

| run | PR | linear | MLP h=512 | kNN k=15 |
|:--|--:|:--|:--|:--|
| B1 CCA | 257.8 | 0.296 (0.764) | 0.302 (**0.999**) | **0.465** (0.755) |
| B4 random init | 17.3 | **0.526** (0.716) | 0.488 (0.812) | 0.488 (0.763) |
| R1 InfoNCE | 6.4 | **0.513** (0.724) | 0.498 (0.873) | 0.502 (0.761) |
| R4 cross-attn | 2.8 | **0.491** (0.592) | 0.429 (0.703) | 0.469 (0.665) |

Three things fall out, and none of them is "the deeper probe wins".

**Extra capacity does not buy generalisation.** The MLP reaches 0.999 train on
B1 and 0.302 test. It is not converging on a ceiling, it is overfitting, and on
three of four runs it scores **below** the linear probe. Memorisation shows up in
the train column and gets punished in the test column.

**But the comparison between runs is destroyed anyway.** Spread across the four
runs is **0.230** under the linear probe and **0.037** under kNN. The linear probe
says B4 beats B1 by 0.23; kNN says by 0.02. Whichever probe you pick you get a
number, but only one of them can still rank the representations - which is the
capacity trap in its real form. Not that the score inflates, but that it stops
depending on the thing being measured.

**The one place non-linearity genuinely helps is diagnostic.** kNN beats linear
on B1 by **+0.169**, and only on B1. That is CCA's geometry from §6 showing
through: PR 258 with local contrast 9.87 means tight islands scattered near-
orthogonally, so a local method reads them and a hyperplane cannot. The gap
between the two probes is a **measurement of how non-linearly organised a space
is** - which is worth having, and is a different quantity from how good the
representation is.

### No, this is not a bootstrap argument

Two orthogonal axes, easy to conflate:

- **probe capacity** decides *what is being measured* - organised versus merely present
- **resampling** decides *how precisely the number is known* - the error bar

Changing the head changes the estimand. Bootstrapping does not; it puts an
interval around whatever estimand you already chose.

That said, there **is** a real gap on the resampling axis, and it is worth
logging rather than leaving implicit. `encoder_qc_cell_vs_niche.json` stores bare
point estimates:

```
virchow2_cell    lp_morph 0.668   chance 0.200     lp_mc 0.190   chance 0.0714
virchow2_niche   lp_morph 0.571   chance 0.200     lp_mc 0.236   chance 0.0714
```

No interval, no per-fold spread. So `d20_fm_probe.png` currently claims 0.67
against 0.20 with nothing to say how stable the 0.67 is. The right fix is a **patient-level
cluster bootstrap** - resample whole patients with replacement, because niches
within a patient are not independent, which is the same exposure that killed the
TLS sub-type nulls. That would let the cell-scale 0.668 versus niche-scale 0.571
gap be called real or not, which at present it cannot be.

It does not change any conclusion in the deck. The morphology probe clears chance
by 3.3x and the vocabulary argument does not depend on the third decimal.

---

## 9 · What none of this does

No encoder is fine-tuned. No LLM is involved anywhere. The evaluation downstream
uses **no neural network at all** - retrieval is cosine similarity, clustering is
k-means, pathway transfer is canonical correlation, and the only fitted object is
the deliberately linear probe above.