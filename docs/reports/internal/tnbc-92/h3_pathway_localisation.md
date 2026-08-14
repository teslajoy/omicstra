# H3 · where pathways sit in tissue, and which signal can answer that

Findings from the pathway-localisation work. Two results: one about the biology,
one about which measurement can carry the question. The second is the more
consequential of the two and it constrains what the report can claim.

---

## 1. the measurement finding: the v3 Fisher table cannot answer a localisation question

The v3 gpath2vec build wrote `fisher_madmean_low_enrichment.parquet` -
189,704,360 rows, one per niche x pathway, with `oddsratio`, `pvalue`, `fdr_bh`
and `sig_pathway` over 680 pathways and 286,250 niches. It is the v3-native
pathway signal and it is the same object that produced the H3 headline, so it is
the obvious candidate for making the sub-pathway spatial claim in v3 rather than
borrowing it from v1.

**It does not work, for a structural reason.**

Restricting to the 42,694 annotated niches across seven compartments and asking,
per pathway per compartment, what fraction of niches call the pathway
significant:

| observation | value |
|---|---|
| pathway x compartment pairs tested | 4,760 |
| pairs called (>=2x lift, >=5 points, replicating on >=50% of slides) | **3** |
| median `rate_in` across all pairs | **0.000** |
| median significant pathways per niche | **4 of 680** |
| gpath2vec graph edges per niche | ~9 (1,954,581 over 215,388) |

The three that pass are Response to elevated platelet cytosolic Ca2+, Signaling
by ROBO receptors, and Ribosome-associated quality control, all in Low TIL
stroma. None is a result anyone would present.

The strongest surviving signal by raw difference is **Neutrophil degranulation
appearing in five of seven compartments simultaneously** (Lymphoid nodule 42.5%,
Lymphocyte 38.0%, High TIL stroma 39.0%, Low TIL stroma 31.5%, Stroma cell
29.8%), followed by Processing of Capped Intron-Containing Pre-mRNA at 63.3% in
Tumor. Those are 478-gene and comparable generic sets. Large sets clear Fisher
more often, everywhere, which is the opposite of localisation.

### why, precisely

`sig_pathway` is a **binary call**, and a median of 4 of 680 come back true per
niche. So per niche the table gives a top-4 selection, not a profile - 676
pathways are indistinguishable zeros. For any single pathway the niches that
"have" it are a small scattered subset, so a compartment comparison reduces to
2% against 1%, and that difference is driven by which niches happened to clear
FDR rather than by graded activity.

That sparsity is **by design**. The table is the edge list that builds the
gpath2vec heterogeneous graph. Sparse edges are what make metapath2vec walks
informative. It was never an activity matrix and it breaks when read as one.

Checked and ruled out: this is not a detection-rate artifact. The correlation
between a niche's variable-gene count and its significant-pathway count is
**-0.215** (Spearman -0.207), the wrong direction for that explanation. No niche
has zero significant pathways.

### consequence

`docs/v2/tnbc92_results_summary.md` states that DAG-resolution H3 decomposition
is deferred for v3. **That is correct and should not be treated as an
outstanding task.** The v1 AUCell arm is the appropriate measurement for a
per-niche localisation question because AUCell scores every pathway in every
niche continuously.

The fix to the report paragraph is therefore a labelling fix, not a re-run: say
that the parent-level result is gpath2vec and the DAG-level result is AUCell.
Those are different quantifications with different nulls, and z = 11-25 from one
is not comparable to z = 5-7 from the other. Reading them in consecutive
sentences invites the false inference that sub-pathways transfer more weakly.

Reproduce with `demo/pathway_v3_compartment.py` (~14 s over the full table).
Output at `runs/tnbc-92_v3/eval/H3/pathway_compartment_v3.parquet`.

---

## 2. the biology finding: three compartments carry localised, transferable pathway signal

On the AUCell arm, with the gating described below, Reactome leaf nodes localise
to tissue compartments and the localised programs transfer to held-out patients.

| pathway | stId | concentrates in | median d | sections | Reactome root | cross-patient z |
|---|---|---|---|---|---|---|
| Phosphorylation of CD3 and TCR zeta chains | R-HSA-202427 | Lymphoid nodule | 2.84 | 11 · 91% | Immune System | 5.03 |
| Interleukin receptor SHC signaling | R-HSA-912526 | Lymphoid nodule | 2.83 | 11 · 91% | Immune System | 6.80 |
| Immunoregulatory lymphoid / non-lymphoid interactions | R-HSA-198933 | Lymphoid nodule | 2.80 | 11 · 91% | Immune System | 4.64 |
| Antigen activates B cell receptor | R-HSA-983695 | Lymphoid nodule | 2.68 | 11 · 91% | Immune System | 3.90 |
| MHC class II antigen presentation | R-HSA-2132295 | High TIL stroma | 1.19 | 23 · 96% | Immune System | 4.41 |
| IRAK2 activation of TAK1, via TLR7/8/9 | R-HSA-975163 | High TIL stroma | 0.87 | 23 · 83% | Immune System | 3.10 |
| FCERI mediated NF-kB activation | R-HSA-2871837 | High TIL stroma | 0.83 | 23 · 78% | Immune System | 4.31 |
| ER-phagosome pathway | R-HSA-1236974 | High TIL stroma | 0.70 | 23 · 78% | Immune System | 4.42 |
| Separation of sister chromatids | R-HSA-2467813 | Tumor | 0.67 | 85 · 87% | Cell Cycle | 4.65 |
| Orc1 removal from chromatin | R-HSA-68949 | Tumor | 0.58 | 85 · 81% | Cell Cycle; DNA Replication | 4.48 |
| Activation of BAD, translocation to mitochondria | R-HSA-111447 | Tumor | 0.51 | 85 · 78% | Programmed Cell Death | 3.30 |

`median d` is Cohen's d of the compartment against all others, taken per section
and then medianed - not pooled over niches. `sections` is how many of the 90
annotated sections the compartment was testable on, and the percentage is how
many of those replicate the effect. `cross-patient z` is `z_A` from
`dag_full/per_node_cca.parquet` (R4, view `z_he`): held-out CCA correlation
expressed in permutation-null standard deviations.

Note the asymmetry the table exposes. The Lymphoid nodule effects are the
largest by far, d ~2.8, and rest on **11 sections**. The Tumor effects are the
smallest, d ~0.6, and rest on **85**. Effect size and evidential weight point in
opposite directions here, and any slide built from this has to show both columns.

### gating applied

An earlier version of this analysis reported a `peak` column that was an argmax
over a z-scored profile with no effect size behind it. That is not a defensible
statistic and it is gone. What replaced it:

- **effect floor** d >= 0.5 with a >= 0.2 margin over the runner-up compartment;
  anything failing both is reported as `not specific` rather than assigned
- **per-section median**, not a pooled statistic. Pooling over 42,694 niches let
  effects carried by a handful of sections look like cohort effects. Applying
  this removed Stroma cell, Necrosis and Lymphocyte entirely, including a
  collagen and syndecan result that had looked textbook-correct when pooled
- **DAG tiering**: 146 of 396 tested nodes are Reactome ancestors of another
  tested node, 985 ancestor-descendant pairs. AUCell on a parent includes its
  children's genes, so a node never appears alongside its own descendant
- **lateral dedupe** at Jaccard 0.3. Tiering handles vertical redundancy but not
  sideways overlap; Classical antibody-mediated complement activation and Role of
  LAT2/NTAL/LAB on calcium mobilization share Jaccard 0.74. This collapsed Tumor
  from 20 nodes to 3
- **gene set floor** of 15 genes. IL-2 signaling has 12, SLC15A4:TASL-dependent
  IRF5 activation has 6

### caveats that belong on any slide built from this

- **This is cell-type composition read through a pathway vocabulary.** TCR-zeta
  phosphorylation scoring high in a lymphoid nodule means T cells are present,
  not that TCR signalling is upregulated per cell. Say "where these programs
  concentrate", never "where they are activated"
- Compartment annotation covers **90 of 92 patients, one section each** out of
  259 subarrays. Lymphoid nodule appears on only 23 of those 90
- The compartment effects use every annotated niche including training patients;
  `cross-patient z` is held out. Two scopes in one table, deliberately
- AUCell's dynamic range depends on gene set size (a 258-gene set spanned 0.062,
  a 29-gene set spanned 0.275). It is sound for comparing niches within one
  pathway and weak for ranking pathways against each other. The ranking in the
  table above should be read as a grouping, not a league table
- An ARI comparison between the data-driven grouping and the Reactome hierarchy
  moved 0.326 -> 0.310 -> 0.068 across three defensible methodology variants.
  It is not stable and is not reported

---

## 3. what is still not delivered

The proposal language is *"Reactome pathway embeddings projected into the shared
space reveal where specific programs co-localize with tissue architecture."*

The transfer statistics support the second half. **No spatial artifact exists in
either `runs/tnbc-92/eval/H3/` or `runs/tnbc-92_v3/eval/H3/`** - the directories
contain per-node CCA, permutation nulls, metadata, a clustermap and linked UMAPs,
and `pathway_axes.parquet` holds pathway metadata only, with no per-niche
projections. Until a pathway axis is projected onto tissue coordinates, the
correct verb is that the signal transfers cross-patient and is localisable in
principle, not that maps were delivered.

`demo/fig_he_pathway_grid.py` renders compartment contours against AUCell
activity across four sections and is the closest existing artifact.

## reproduce

| script | output |
|---|---|
| `demo/pathway_groups.py` | gating and grouping, all decisions |
| `demo/fig_pathway_table.py` | the table above, as a slide |
| `demo/fig_pathway_compartment.py` | the same data, dot-and-interval |
| `demo/fig_he_pathway_grid.py` | compartment contours over four sections |
| `demo/pathway_v3_compartment.py` | the v3 negative result |