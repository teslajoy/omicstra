# the decision graph

`mcp_plan.md` §1 states the deliverable: *"the analytical decisions are encoded once
and not re-derived per dataset."* This file is that encoding. Everything else in the
package is plumbing that executes it.

Each decision is marked by who resolves it:

| | |
|---|---|
| **D** | deterministic - a lookup or a rule, no judgement |
| **E** | empirical - measured on this cohort, never inherited |
| **H** | human - escalation, `interrupt()`, hard pause |

---

## the override rule

**Every decision below is locked. The system does not silently re-derive, substitute,
or relax one.** A decision changes only when a human asks it to change, and the change
is recorded.

| situation | behaviour |
|---|---|
| a default holds | proceed silently |
| a default fails on this cohort | **escalate**, do not substitute |
| a question resolves to several methods | surface all + evidence + consequence, do not pick |
| a method is contraindicated | refuse the routing, return the reason |
| a human overrides | record the override and the reason, then proceed |

"Escalate" means a hard HITL pause: the workflow stops, state is checkpointed, a human
answers, execution resumes from that exact point. Not a logged warning that continues.

---

## the complete flow

```
                        computational biologist has data
                                       │
╔══════════════════════════════════════▼══════════════════════════════════════╗
║ [1] does a project already exist for this data?                      D      ║
║     registry lookup + data fingerprint                                      ║
╚══════════════════════════════════════┬══════════════════════════════════════╝
              ┌───────── yes ──────────┴───────── no ─────────┐
              │                                               │
    load ProjectConfig                                        │
              │                                               ▼
              │              ╔══════════════════════════════════════════════════╗
              │              ║ [2] EDA - profile AND gate in one pass    A · E  ║
              │              ║     the eleven checks. emits plots.              ║
              │              ║     ──► eda_summary.json                         ║
              │              ║     ──► verdict  proceed | caution | stop        ║
              │              ╚═══════════════════┬══════════════════════════════╝
              │                    stop ─► HALT  │  caution ─► H
              │                                  │
              └──────────────────┬───────────────┘
                                 ▼
╔════════════════════════════════════════════════════════════════════════════╗
║ [3] ENCODERS - fan out, then compare                               B       ║
║                                                                            ║
║        ┌──────────────┬──────────────┬──────────────┐                      ║
║        ▼              ▼              ▼                                     ║
║   he_agent       st_agent      pathway_agent                               ║
║   Virchow2       Novae 64      gpath2vec 512                               ║
║   1280 RAW       z-score       Fisher/TF                                   ║
║                  AFTER pool    ──► G5 noise floor before promote           ║
║        │              │              │                                     ║
║        └──────────────┼──────────────┘                                     ║
║                       ▼                                                    ║
║          ENCODER QC COMPARISON                                    E        ║
║          common probe on a systematically derived label:                   ║
║            linear probe · cross-unit silhouette                            ║
║            morphology silhouette · batch ratio                             ║
║                       │                                                    ║
║            prior holds ──► proceed        prior fails ──► H                ║
╚════════════════════════════════════════════════════════════════════════════╝
                                 ▼
╔════════════════════════════════════════════════════════════════════════════╗
║ [4] UNIT                                                          C        ║
║     new cohort: derive from platform resolution                    D       ║
║     established cohort: changing it needs approval          HC7 ─► H       ║
║     ──► constrains admissible fusion (C2)                                  ║
╚════════════════════════════════════════════════════════════════════════════╝
                                 ▼
╔════════════════════════════════════════════════════════════════════════════╗
║ [5] QUESTION ──► METHOD                                           D        ║
║     classify the question into one of the 7 classes    MODEL (only call)   ║
║     look up the method                                             D       ║
║                                                                            ║
║       one winner      ──► dispatch + its guard                             ║
║       several         ──► ASK: answer with all + evidence                  ║
║                           COMPUTE: interrupt()                    H        ║
║       contraindicated ──► refuse the routing, return the reason            ║
║       no match        ──► refuse, return what IS answerable                ║
╚════════════════════════════════════════════════════════════════════════════╝
                                 ▼
╔════════════════════════════════════════════════════════════════════════════╗
║ [6] RUN                          LangGraph nodes / Temporal activities     ║
║     alignment constraints D1-D8 apply                              D       ║
╚════════════════════════════════════════════════════════════════════════════╝
                                 ▼
╔════════════════════════════════════════════════════════════════════════════╗
║ [7] EVALUATE - no metric leaves without its guard                 E        ║
║     G1 label granularity  G2 held-out honesty  G3 bio/subject              ║
║     G4 specificity        G5 noise floor       G6 supervision              ║
╚════════════════════════════════════════════════════════════════════════════╝
                                 ▼
╔════════════════════════════════════════════════════════════════════════════╗
║ [8] PROMOTE                                                       F        ║
║     beat all baselines, clear the floors, or it is not a winner            ║
╚════════════════════════════════════════════════════════════════════════════╝
                                 ▼
╔════════════════════════════════════════════════════════════════════════════╗
║ [9] EMIT   run_record ──► notebook · trace · evidence bundle               ║
╚════════════════════════════════════════════════════════════════════════════╝
```

**Only [2]-A6 changes the pipeline's shape.** Everything else is a threshold, a lock,
a halt, or an escalation. That single fork is why the workflow is deterministic.

---

## A · data

Gate 0. Blocking. Profiling and gating are one pass, not two.

| # | decision | | rule |
|---|---|---|---|
| A1 | are counts raw? | E | resolves `encoder_input_decision`. **unresolved ⇒ halt all embedding** |
| A2 | gene ID format | D | Ensembl vs symbol; mapping required? |
| A3 | which unit set | D | full grid vs tissue-selected |
| A4 | exclusions | E | IQR outliers · dead units · below platform floor |
| A5 | markers | E | positives present, negatives near zero |
| **A6** | **spatial autocorrelation** | **E** | **Moran's I > 0.3 for >= 3 of top 5 ══► ROUTES THE MOLECULAR ENCODER** |
| A7 | batch structure | E | technical silhouette > 0.1 and > biological ⇒ **stop**, correct first |
| A8 | cross-modal registration | E | conditional: multi-modal |
| A9 | segmentation QC | E | conditional: single-cell platform |
| A10 | multi-section alignment | E | conditional: serial sections |
| A11 | encoder x tissue fit | D | validated / constrained / uncertain. uncertain ⇒ validation protocol ⇒ H |

Verdict: `proceed` \| `proceed_with_caution` ⇒ H \| `stop`.

A statistic computed on one sample does not gate a cohort decision.

## B · encoders

Not three independent lookups. The choice is made **after** running the candidates and
comparing them on a common downstream probe.

| # | decision | | rule |
|---|---|---|---|
| B1 | imaging | D | **Virchow2 1280, raw.** Alternatives only on explicit request |
| B2 | molecular | D | A6 pass ⇒ **Novae 64**. A6 fail ⇒ non-spatial fallback |
| B3 | pathway | D | **gpath2vec 512**, hard default. Its value appears only downstream at H3 |
| B4 | preprocessing | E | **never inherited, always measured.** Virchow2 raw (z-score halves the probe). Molecular z-scored *after* unit pooling |
| B5 | platform floor | D | units per sample >= the encoder's prototype count |
| B6 | depth routing | D | median genes >= 2000 run · 500-1999 flag · < 500 fall back |
| B7 | QC comparison | E | prior holds ⇒ proceed. prior fails ⇒ **escalate, do not substitute** |

Never stain-normalise before FM inference (HC2). Never batch-correct single-institution
data (HC1). The imaging encoder is selected using molecular labels, so the imaging
decision is not independent of the molecular data being present and good.

## C · unit

| # | decision | | rule |
|---|---|---|---|
| C1 | atomic unit | D/H | new cohort: derive from platform resolution. **changing an established unit needs approval** (HC7) |
| C2 | admissible fusion | D | region ⇒ classification head · niche ⇒ late or early · cell ⇒ early only |
| C3 | pooling discipline | D | hard labels centre-only · continuous mean-pooled · supervision mean-pooled, never a feature |

## D · alignment

| # | decision | | rule |
|---|---|---|---|
| D1 | interaction mode | E | experimental variable, never assumed |
| D2 | pairs < 5000 | D | ⇒ late fusion, capacity control |
| D3 | pairs < 1000 | D | ⇒ cross-attention **forbidden** |
| D4 | recall < 0.3@k10, or modality gap < 0.3 | D | ⇒ try cross-attention |
| D5 | supervision ∉ features | D | HC3, static assert |
| D6 | one variable per experiment | H | HC6 |
| D7 | subset test, 5 checks | E | HC5. fail ⇒ do not launch the full run |
| D8 | best_epoch = 1 twice running | D | ⇒ **stop, escalate** |

## E · evaluation - the guard library

Dataset-agnostic. Each came from a failure on the seed cohort; none names it.

| # | guard | | trips when |
|---|---|---|---|
| G1 | label granularity | E | NMI(label, subject) > 0.5 ⇒ demote the label to diagnostic |
| G2 | held-out honesty | D | evaluate cross-unit, subject-held-out only (HC4) |
| G3 | biology vs subject | E | pair with every clustering metric. below the raw-modality floor ⇒ anti-helpful |
| G4 | magnitude + specificity | E | every direction-magnitude claim ships with its off-diagonal cosine |
| G5 | noise floor | E | permutation test before promoting any new embedding build |
| G6 | supervision leakage | D | the supervision signal must not appear in the input vector |
| E7 | metric at scale | D | rank-based metrics floor above ~35k units; use AUC |
| E8 | view circularity | D | when a signal feeds both input and test target, only the opposite view is clean |
| E9 | multiple testing | D | BH-FDR per declared family |

## F · promotion

| # | decision | | rule |
|---|---|---|---|
| F1 | AUC <= 0.50 for all strategies | D | ⇒ stop, alignment is not working |
| F2 | baselines | D | must beat random + spatial-NN + unaligned concat |
| F3 | below the raw-modality floor | D | ⇒ contraindicated for that question |
| F4 | admitting a new method | H | evidence against **all** existing routes, not just the incumbent |

---

## the routing table

Ten runs produced **four winners and one contraindication**. The rest are evidence that
the winners won, not product.

| # | question | method | evidence | resolution |
|---|---|---|---|---|
| 1 | retrieve the matched molecular unit from an imaging unit | cross-attention, un-pooled tiles | AUC 0.859 [0.857, 0.862] · CKA 0.631 | single |
| 2 | group units by tissue state within a section | late-fusion contrastive, molecular-only | ARI 0.244 | **TIE** |
| 3 | amplify biology, suppress subject identity | late-fusion contrastive, molecular-only | ratio 0.547 [0.476, 0.617] vs floor 0.137 | **TIE** |
| 4 | transfer signal to unseen subjects | cross-attention | 4/4 pathways · 80% of 395 DAG nodes | single |
| 5 | attribute which signal drives a unit | classical orthogonal rotation | off-diagonal cosine 0.306 vs 0.75 | **TIE** |
| 6 | decode a signature from one modality alone | untrained projection | z 25.5 · contrastive training degrades it | single |
| 7 | cohort-level biological interpretation | **contraindicated** | ratio 0.071, below the raw-modality floor | refuse |

### the three ties, and what the human is being asked

| row | why tied | the trade-off |
|---|---|---|
| 2 | CIs overlap; effective dimensionality also ties at 19/20/20 | one candidate drops the pathway encoder and still ties, **but scores 0/4 on row 4.** Parsimony now forecloses pathway transfer later |
| 3 | second candidate falls inside the first's CI | cross-attention has the lowest subject z in the grid but a worse ratio. Minimise leakage, or maximise the ratio? |
| 5 | the two candidates are **rotation-equivalent** - identical off-diagonal cosine, 0.306 both | the metric that defines this row cannot separate them. More runs will not help. They diverge only on retrieval, which is irrelevant here |

Row 5 is the honest case: not close, **mathematically indistinguishable** on the
defining metric.

### escalation payload

```
question_class      2
resolution          TIE
candidates          [ {method, metric, value, ci, effective_dim, evidence} ]
why_tied            CIs overlap; separation is within seed noise
tiebreak_arguments  parsimony: one needs no pathway encoder
                    capability: that one scores 0/4 on cross-subject transfer
consequence         choosing parsimony forecloses row 4
no_default          the system does not pick
```

`consequence` is the load-bearing field. Numbers alone do not tell a human what the
choice forecloses.

### delivery differs by path

| path | on a tie |
|---|---|
| **ASK** - answering about committed results | return all candidates + evidence. **no interrupt** - the human is already in the loop |
| **COMPUTE** - choosing what to run | `interrupt()`. A machine is about to spend accelerated compute on a values call |

---

## state

| category | encoded as | built |
|---|---|---|
| A · data | `configs/eda_contract.json` + `check_eda_gate` | **validator only** - it checks a summary, it cannot produce one |
| B · encoders | prose in `MODELS.md`, defaults in `project.json` | no |
| C · unit | prose in `program.md` | no |
| D · alignment | prose in `program.md`, `PLAN_RULES.md` | no |
| E · guards | `guards.py` | six stubs |
| F · promotion | prose in `CLAUDE.md` | no |
| routing table | prose in the report + five diverged copies | no |

One of seven categories is partially executable. The rest is correct, written down, and
not yet code - which is the actual state of the build, and the reason the decision graph
had to come before more tools.
