# the workflow

The executable graph. `decisions.md` says *what* is decided and by whom; this says
*what runs*, in what order, what is parallel, and where it pauses.

Every node is a LangChain step with a declared input / output / params - a nextflow
process in shape. Every conditional edge is a LangGraph branch. The body of a node is
swappable (python call -> workflow submission) without changing the graph.

Two paths share one node and diverge after it. **ASK** reads committed results and
touches no accelerated compute. **COMPUTE** produces them.

---

## ASK - the read path

```
                            question (natural language)
                                       │
                          ┌────────────▼────────────┐
                          │  discover               │  D
                          │  registry + fingerprint │
                          └────────────┬────────────┘
                              no project │ project found
                          ┌─────────────┴──────┐
                          ▼                    ▼
                    refuse: "no cohort   ┌──────────────┐
                     for this data"      │ load_config  │
                          │              └──────┬───────┘
                          │                     ▼
                          │         ┌────────────────────────┐
                          │         │  route_question        │  MODEL - the only
                          │         │  classify -> 1 of N    │  model call in ASK
                          │         └───────────┬────────────┘
                          │     ┌───────────────┼──────────────┬──────────────┐
                          │  no match       contraindicated   tie          single
                          │     │               │              │              │
                          │     ▼               ▼              ▼              ▼
                          │  refuse +      refuse the      answer with   read_result
                          │  what IS       routing,        ALL + evidence     │
                          │  answerable    return reason   + consequence      │
                          │     │               │           (no interrupt)    │
                          └─────┴───────────────┴──────────────┴──────────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │  attach_guards        │  D
                                    │  no metric leaves     │
                                    │  without its guard    │
                                    └───────────┬───────────┘
                                                ▼
                                       answer + provenance
```

No accelerated compute, no network, no encoder. Reads the evidence bundle only.

---

## COMPUTE - the full graph

```
                                     cohort directory
                                            │
                               ┌────────────▼────────────┐
                               │  discover               │  D
                               └────────────┬────────────┘
                            exists │                     │ new
                    ┌──────────────┘                     ▼
                    │                    ┌───────────────────────────┐
                    │                    │  eda_inventory            │  E
                    │                    │  counts · pairing · files │
                    │                    └─────────────┬─────────────┘
                    │                                  ▼
                    │                    ┌───────────────────────────┐
                    │                    │  eda_signal               │  E
                    │                    │  markers · Moran's I      │
                    │                    │  batch · registration     │
                    │                    │  emits PLOTS              │
                    │                    └─────────────┬─────────────┘
                    │                                  ▼
                    │                    ╔═══════════════════════════╗
                    │                    ║  eda_gate         GATE 0  ║
                    │                    ║  ──► eda_summary.json     ║
                    │                    ╚═══════════════════════════╝
                    │                     stop │  caution │  proceed
                    │                          ▼          ▼         │
                    │                        HALT    ┌────────┐     │
                    │                                │INTERRUPT│────┤
                    │                                └────────┘     │
                    └──────────────────────┬──────────────────────  ┘
                                           ▼
                        ┌──────────────────────────────────────┐
                        │            FAN OUT  (parallel)       │
                        └──┬──────────────┬──────────────┬─────┘
                           ▼              ▼              ▼
                    ┌───────────┐  ┌───────────┐  ┌─────────────┐
                    │ he_encode │  │ st_encode │  │pathway_encode│
                    │ Virchow2  │  │ Novae 64  │  │ gpath2vec   │
                    │ 1280 RAW  │  │ z AFTER   │  │ 512         │
                    │           │  │ pooling   │  │ ──► G5 gate │
                    └─────┬─────┘  └─────┬─────┘  └──────┬──────┘
                          │              │               │
                          └──────────────┼───────────────┘
                                         ▼
                        ╔════════════════════════════════════╗
                        ║  encoder_qc            JOIN   E    ║
                        ║  common probe, one label           ║
                        ╚════════════════════════════════════╝
                          prior holds │        │ prior fails
                                      │        ▼
                                      │   ┌─────────┐
                                      │   │INTERRUPT│  escalate,
                                      │   └────┬────┘  do not substitute
                                      └────────┤
                                               ▼
                                 ┌──────────────────────────┐
                                 │  unit_join    THE WAIST  │  D
                                 │  one row per unit        │
                                 │  supervision in its own  │
                                 │  slot, never a feature   │
                                 └────────────┬─────────────┘
                                              ▼
                                 ┌──────────────────────────┐
                                 │  route_question          │  MODEL + D
                                 └────────────┬─────────────┘
                              tie │                        │ single
                                  ▼                        │
                            ┌─────────┐                    │
                            │INTERRUPT│  values call,       │
                            └────┬────┘  compute is costly  │
                                 └────────────┬─────────────┘
                                              ▼
                                 ┌──────────────────────────┐
                                 │  align                   │  D1-D8 apply
                                 │  D7 subset test first    │
                                 └────────────┬─────────────┘
                                    fail │              │ pass
                                         ▼              ▼
                                       HALT   ┌──────────────────────┐
                                              │  evaluate            │  E
                                              │  G1-G6 · every       │
                                              │  metric + its guard  │
                                              └──────────┬───────────┘
                                                         ▼
                                              ┌──────────────────────┐
                                              │  promote             │  D
                                              │  beat all baselines  │
                                              └──────────┬───────────┘
                                            not a winner │  │ winner
                                                         ▼  ▼
                                              ┌──────────────────────┐
                                              │  emit                │
                                              │  run_record          │
                                              │   ├─► notebook       │
                                              │   ├─► trace          │
                                              │   └─► evidence bundle│
                                              └──────────────────────┘
```

---

## nodes

| node | in | out | resolves | |
|---|---|---|---|---|
| `discover` | cohort dir | ProjectConfig or null | [1] does a project exist | D |
| `eda_inventory` | raw inputs | inventory + plots | A1-A4 | E |
| `eda_signal` | inventory, raw | signal + plots | A5-A10 | E |
| `eda_gate` | both | `eda_summary.json` + verdict | A11, the verdict | D |
| `he_encode` | units, images | imaging features | B1, B4 | E |
| `st_encode` | units, counts | molecular features | B2, B4, B5, B6 | E |
| `pathway_encode` | units, counts, pathway DB | pathway features | B3 + G5 | E |
| `encoder_qc` | all three | comparison + decision | B7 | E |
| `unit_join` | all features + labels | the waist + manifest | C3, G6 | D |
| `route_question` | question, routing table | method + guards | [5] | MODEL + D |
| `align` | waist, config | shared space | D1-D8 | D |
| `evaluate` | shared space, labels | metrics + guard verdicts | E/G1-G9 | E |
| `promote` | metrics | winner or not | F1-F4 | D |
| `emit` | run record | notebook · trace · evidence | - | D |

## state

Carried through the graph, checkpointed at every node:

```
project        ProjectConfig | null
profile        cohort_profile          from eda_inventory + eda_signal
verdict        proceed | caution | stop
encoder_decision   which encoder per modality + the evidence
unit           atomic unit + admissible fusion modes
question_class int | null
method         one | [tied] | contraindicated | null
guards         [{id, verdict, note}]
decisions      [{id, actor: deterministic|model|human, value, reason}]
```

`decisions` is the ledger. Its actor breakdown is the reproducibility claim as a number:
*N decisions, X deterministic, Y model, Z human.*

## the four interrupts

| where | fires when | the human is asked |
|---|---|---|
| `eda_gate` | verdict = caution | accept the risks, or stop |
| `encoder_qc` | the locked prior loses on this cohort | override the default, or halt |
| `route_question` | several methods tie | pick, given evidence + what the choice forecloses |
| `promote` | a new method is a candidate | admit it against **all** existing routes |

Each pauses hard, checkpoints state, and resumes at that exact point. None has a default;
the system does not pick.

## what is parallel

Only the encoder fan-out. Three agents, disjoint outputs, joined at `encoder_qc`. Every
other edge is sequential because each consumes the previous node's decision.

## state of the build

| node | built |
|---|---|
| `eda_gate` | **partially** - validates a summary, cannot produce one |
| `route_question` | routing table exists as prose, not data |
| `align`, `evaluate` | run as scripts, unported, no graph |
| everything else | no |

The graph is the specification. One node is partially executable.