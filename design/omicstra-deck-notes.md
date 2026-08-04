# omicstra - speaker notes

13 slides. Every number below was checked against the run artifacts on 2026-08-04;
the source file is named so you can pull it up if challenged.

**Before you start:** the report link is `teslajoy.github.io/omicstra/reports/internal/tnbc-92/`.
The old `/reports/tnbc-92/` path 404s - it was moved deliberately.

---

## 1 - Why?

**Beat.** Three ways to look at one piece of tissue. Each is partial. The question is
whether an agent can combine them well enough to answer a biological question.

**Say.** "Omicstra started as a multi-agent MCP server for spatial biology. To make
decisions I could defend, I first had to find out what each computational method was
actually capable of answering. That turned into the project."

**Land the last line.** What we can ask is bounded by the annotations, not the data
volume. That constraint returns on slide 10 and again on slide 13.

---

## 2 - Original vision

**Beat.** Three questions a biologist would actually type. Read them aloud - they are
the spine of the talk.

**Say.** "Hold these. By slide 10 they route to three different methods, and one of
them the system refuses to answer until you disambiguate it."

Don't explain the routing yet. Just plant them.

---

## 3 - The problem

**Beat.** The pivot. An agent cannot pick a tool without evidence about what each tool
is good for.

**Say.** "Everyone is looking for the best multimodal model. But biology isn't one
question - it's many, and they have different answers."

Short slide. Don't linger.

---

## 4 - From tissue to a biological answer

**Beat.** Two measurements, three representations. gpath2vec is derived from the same
counts Novae encodes - a different level of abstraction over one measurement, not an
independent survey.

**The load-bearing distinction.** Virchow2 and Novae encode *only the measurement*.
gpath2vec additionally reads an external curated pathway database. All three are learned
encodings; only one imports outside knowledge.

**If asked "isn't Novae also an inference?"** Yes - it is a frozen model carrying priors
from other platforms, and our own EDA records it as `validated_with_constraints` because
it was trained at subcellular resolution and our spots are 100 µm. The distinction on the
slide is about what *enters* the encoding, not about which is trustworthy.

**Guardrails.** Three outcomes: recommend, tie, refuse. Refusal is the part no benchmark
leaderboard has.

---

## 5 - So I stepped back

**Beat.** The unit is the niche - one spot plus six neighbours, about 1,200 cells. Not
the spot. Spot-level cross-patient retrieval collapsed to chance, which is why the unit
moved.

**The `+` node matters.** The two molecular encoders are concatenated into one 576-d
vector (64 + 512) and enter as one side. Imaging enters as the other. Imaging and
molecular are never concatenated with each other - they are the two sides of the
contrastive pair.

**If asked about "only H&E enters at prediction time":** true for the deployed use and
for the headline `z_he` view. Retrieval is the *evaluation*, and that necessarily scores
both sides - which is what slide 6's x-axis is.

---

## 6 - What surprised me

**Beat.** I expected one winner. The run that pairs best (R4, AUC 0.859) is near-worst at
grouping (ARI 0.101). The runs that group best (R5 0.246, R6 0.244, R1 0.234) carry
almost no pathway signal to unseen patients - R6 carries **zero of four**.

**Marker size** = pathways transferring cross-patient, printed on each marker.

**The careful claim.** R4 is the only *contrastive* run carrying all four. The classical
baselines B1, B2 and B3 also reach 4 of 4 - R4 beats them on magnitude, roughly 1.5×
on every pathway (Immune 25.4 vs 17.1 best baseline). Do not say "the only run."

**Punchline.** No method occupies a corner.

*Source: `metrics_h1_ci.json`, `H2/mc_coherence.parquet`,
`H3/pathway_cca_gpath2vec_v3/per_pathway_cca.parquet`.*

---

## 7 - And one method was disqualified

**Beat.** B1 passed both axes on the previous slide and scored 4 of 4 on pathway
transfer. It was scoring by encoding *which patient the sample came from*.

**The number.** Ratio of two permutation z-scores on the same space, 10,000 label
shuffles each: how much more similar niches are when they share a tumour immune
microenvironment class, over how much more similar they are when they share a patient.

```
R6 0.547   R5 0.497   R1 0.461   R4 0.376   unaligned floor 0.137   B1 0.071
```

Below the floor means the space is reading provenance, not biology. **Without this
diagnostic B1 would have been selected.**

**The BForePC line.** LG PanIN, HG PanIN and PDAC ROIs come from the same block, same
patient, by design. This is the check to run before trusting any comparison across them.
This is structural in any study where several regions come from one patient.

*Source: `H2/metrics_h2_ci.json`, block `h2c`.*

---

## 8 - The real contribution

**Beat.** The reframe. Not "which model is best" but "which method is best for *this*
biological question."

**Say.** "It tells you which method to trust for which question, and shows the
measurement that put it there. Internally that is a routing table."

---

## 9 - What the MCP becomes

**Beat.** Four steps. The one that matters is step 3: it routes to the method the
evidence supports, and it records a tie or refuses when the evidence does not separate
candidates.

**Say.** "Instead of an AI agent guessing, it consults evidence and can explain why."

Step 4 is the reproducibility claim as a number: every decision is labelled rule, model,
or human.

---

## 10 - The three questions, resolved

**Beat.** Back to slide 2. Same two assays, same data, three different destinations.

| question | level | route |
|---|---|---|
| checkpoint signalling | program | R4 |
| find TLS-like regions | structure | B4 - an *untrained* projection; training makes it worse (z = 25.5) |
| stromal remodelling | ambiguous | R4 **or** a three-way tie - the router has to ask first |

**The point.** Nothing about the data changes between the three. What changes is the
level of biological abstraction the question lives at, and that decides which
representation can answer it.

**The B4 result is the counter-intuitive one** - the best method for that question is the
one with no alignment training at all. Expect a question there.

---

## 11 - Why this matters

**Beat.** New foundation models keep arriving. We don't rebuild - we evaluate them on the
same H1/H2/H3 tests, against the same floors, and update the routing table. A new method
is admitted only if it beats **every** existing route, not just the incumbent.

---

## 12 - The framework generalises, the evaluation doesn't

**Beat.** Two different claims, and only one is free.

**Say.** "A new cohort does not inherit a threshold. It escalates, and someone has to
derive it or adopt it deliberately."

**If asked "does R4 win in pancreas too?"** The honest answer is unknown, and the system
escalates rather than assuming. That is a stronger position than a confident yes.

**If you want the concrete example:** our spatial-signal criterion is `Moran's I > 0.3
for 3 of 5 markers`. It has no external basis - published work selects by FDR or by rank,
not by a magnitude bar. So it is marked as calibrated-here, and on a new cohort the gate
stops rather than reusing it.

---

## 13 - None of this predicts outcome

**Beat.** Say this before anyone asks. Spatially localising checkpoint and stromal
programs from archived material is a *precondition* for morphology-based stratification,
not an instance of it. That needs a cohort with outcomes attached, prospectively.

Then read the quote. It is the summary of the whole talk.

---

## questions you should expect

**"Why not just use one big multimodal model?"**
Because the evaluation says no single run wins every objective - slide 6 is the evidence.
A single model would still need this layer to know when to trust it.

**"How do you know R4 isn't overfitting?"**
Everything on slides 6, 7 and 10 is measured on 35,594 held-out niches from 14 patients
never seen in training, patient-stratified 85/15, seed 42.

**"Is 92 patients enough?"**
For the routing conclusions, the confidence intervals are on the report. For the
calibrated thresholds, the honest answer is that they are calibrated on this cohort -
which is exactly what slide 12 is about.

**"What is a niche again?"**
One spot plus its six neighbours, about 1,200 cells, roughly 200 µm across.

**"Where can I see the numbers?"**
`teslajoy.github.io/omicstra/reports/internal/tnbc-92/` - every figure in the deck traces
to an artifact named in that report.