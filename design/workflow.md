# the workflow

The executable graph. `decisions.md` says *what* is decided and by whom; this says
*what runs*, in what order, what is parallel, where it pauses, and what each piece is
called.

Every node is a LangChain step with a declared input / output / params - a nextflow
process in shape. Every conditional edge is a LangGraph branch. A node's body is
swappable (python call -> workflow submission) without changing the graph.

Two paths share one node and diverge after it. **ASK** reads committed results and
touches no accelerated compute. **COMPUTE** produces them.

---

## package layout

```
src/omicstra/
  records.py     Record · TransformRecord · DiagnosticRecord
                 GateRecord · SelectionRecord
  state.py       WorkflowState          ← module-level TypedDict, see note below
  knowledge.py   EDAContract · RoutingTable
  guards.py      GuardResult · G1-G6
  artifacts.py   ArtifactStore · RecordStore
  nodes.py       the graph functions
  graph.py       build_graph() -> StateGraph
  agents/        HEAgent · STAgent · PathwayAgent
                 AlignmentAgent · EvalAgent · Orchestrator
  mcp/server.py  the tools
```

> `WorkflowState` must live at module level. Under `from __future__ import annotations`
> (which every file in this package uses), a function-local `TypedDict` raises
> `NameError` when langgraph resolves node annotations against module globals - only
> when a conditional edge is present, so it surfaces late. Verified on 3.11 / 3.13 /
> 3.14; not a version issue and not a langgraph bug.

---

## the record contract

Every node emits **exactly one** Record. Four kinds, discriminated on `kind`.

```
Record                                     the base, on every kind
  step_id · kind · params
  inputs[]   {path, sha256}
  outputs[]  {path, sha256}
  status · actor · caveats[] · timing · code_version
```

| kind | payload | emitted by |
|---|---|---|
| **transform** | `produced: [{path, shape, dtype, sha256}]` | encode_he · encode_st · encode_pathway · join_units · align · emit |
| **diagnostic** | `method · scope · observed · criterion · result · guards[]` | profile_inventory · profile_signal · evaluate |
| **gate** | `verdict · edge_taken · escalated · reason` | discover_project · gate_eda · promote |
| **selection** | `candidates[] · chosen · tie · evidence · consequence` | select_encoder · route_question |

Three consequences worth naming:

- `actor` on the base means the decision ledger is `[r.actor for r in records]`. The
  reproducibility claim becomes a number with no extra bookkeeping.
- `inputs`/`outputs` with hashes on the base gives provenance uniformly, so the sha-lock
  pattern already guarding the pathway build applies everywhere for free.
- kind-specific payloads keep nodes honest: a transform cannot fake a criterion it does
  not have.

**`selection` covers two nodes that look nothing alike operationally** - choosing an
encoder and choosing a method - and are identical in record shape. Both may resolve to
no choice at all, which is what escalation is.

**`evaluation` is not a separate kind.** It folds into `diagnostic`: both observe and
compare against a criterion, and a guard verdict is a criterion with a name.

`stages.py` already returns typed artifact refs for six nodes - `HEArtifacts`,
`NicheJoin`, `AlignGrid` are `transform` payloads under an earlier name. This subsumes
them rather than competing.

---

## ASK - the read path

```
                            question (natural language)
                                       │
                          ┌────────────▼────────────┐
                          │ discover_project()      │  D
                          │ Orchestrator.discover() │
                          └────────────┬────────────┘
                              no project │ project found
                          ┌─────────────┴──────┐
                          ▼                    ▼
                    refuse: "no cohort   ProjectConfig.from_dir()
                     for this data"            │
                          │                    ▼
                          │      ┌──────────────────────────┐
                          │      │ route_question()         │  MODEL - the only
                          │      │ Orchestrator.route()     │  model call in ASK
                          │      │ RoutingTable.resolve()   │
                          │      └───────────┬──────────────┘
                          │  ┌───────────────┼──────────────┬──────────────┐
                          │ no match   contraindicated     tie          single
                          │  │               │              │              │
                          │  ▼               ▼              ▼              ▼
                          │ refuse +    refuse the     answer with    ArtifactStore
                          │ what IS     routing,       ALL + evidence  .read()
                          │ answerable  return reason  + consequence      │
                          │  │               │         (no interrupt)     │
                          └──┴───────────────┴──────────────┴─────────────┘
                                             │
                                 ┌───────────▼───────────┐
                                 │ attach_guards()       │  D
                                 │ GuardResult[]         │
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
              ┌──────────────────────▼──────────────────────┐
              │  discover_project(state)                    │
              │  owner   Orchestrator.discover()            │
              │  reads   ProjectConfig.from_dir()           │
              │  emits   GateRecord(edge = exists | new)    │
              └───────────────┬─────────────────────────────┘
                  exists ─────┤───── new
                              ▼
              ┌─────────────────────────────────────────────┐
              │  profile_inventory(state)                   │
              │  owner   EDAAgent.inventory()               │
              │  emits   DiagnosticRecord × A1-A4           │
              ├─────────────────────────────────────────────┤
              │  profile_signal(state)                      │
              │  owner   EDAAgent.signal()                  │
              │  emits   DiagnosticRecord × A5-A10 + plots  │
              └───────────────┬─────────────────────────────┘
                              ▼
              ╔═════════════════════════════════════════════╗
              ║  gate_eda(state)                     GATE 0 ║
              ║  owner   EDAAgent.gate()                    ║
              ║  reads   EDAContract.evaluate()             ║
              ║  emits   GateRecord(verdict)                ║
              ║          ──► eda_summary.json               ║
              ╚═══════════════┬═════════════════════════════╝
                stop ─► END   │   caution ─► interrupt()   │ proceed
                              └──────────────┬─────────────┘
                                             ▼
              ┌──────────────── FAN OUT ─────────────────────┐
              ▼                    ▼                    ▼
  ┌───────────────────┐ ┌──────────────────┐ ┌───────────────────┐
  │ encode_he()       │ │ encode_st()      │ │ encode_pathway()  │
  │ HEAgent.encode()  │ │ STAgent.encode() │ │ PathwayAgent      │
  │                   │ │                  │ │   .encode()       │
  │ TransformRecord   │ │ TransformRecord  │ │ TransformRecord   │
  │                   │ │                  │ │ + guards.G5       │
  └─────────┬─────────┘ └────────┬─────────┘ └─────────┬─────────┘
            └────────────────────┼─────────────────────┘
                                 ▼
              ╔═════════════════════════════════════════════╗
              ║  select_encoder(state)               JOIN   ║
              ║  owner   Orchestrator.select_encoder()      ║
              ║  emits   SelectionRecord                    ║
              ║          candidates · chosen · tie          ║
              ╚═══════════════┬═════════════════════════════╝
                 prior fails ─┴─► interrupt()      │ holds
                                                   ▼
              ┌─────────────────────────────────────────────┐
              │  join_units(state)              THE WAIST   │
              │  owner   Orchestrator.join()                │
              │  asserts guards.G6  supervision ∉ features  │
              │  emits   TransformRecord + manifest         │
              └───────────────┬─────────────────────────────┘
                              ▼
              ╔═════════════════════════════════════════════╗
              ║  route_question(state)                      ║
              ║  owner   Orchestrator.route()               ║
              ║  reads   RoutingTable.resolve(question)     ║
              ║  emits   SelectionRecord                    ║
              ║          single | tie | contraindicated     ║
              ╚═══════════════┬═════════════════════════════╝
                    tie ──────┴──► interrupt()      │ single
                                                    ▼
              ┌─────────────────────────────────────────────┐
              │  align(state)                               │
              │  owner   AlignmentAgent.fit()               │
              │  D1-D8 apply · D7 subset test first         │
              │  emits   TransformRecord                    │
              └───────────────┬─────────────────────────────┘
                              ▼
              ┌─────────────────────────────────────────────┐
              │  evaluate(state)                            │
              │  owner   EvalAgent.run()                    │
              │  emits   DiagnosticRecord × H1/H2/H3        │
              │          each carrying GuardResult[]        │
              └───────────────┬─────────────────────────────┘
                              ▼
              ╔═════════════════════════════════════════════╗
              ║  promote(state)                             ║
              ║  owner   EvalAgent.promote()                ║
              ║  emits   GateRecord(winner | not)           ║
              ╚═══════════════┬═════════════════════════════╝
                              ▼
              ┌─────────────────────────────────────────────┐
              │  emit(state)                                │
              │  owner   Orchestrator.emit()                │
              │  reads   RecordStore.all()                  │
              │  emits   run_record · notebook · trace      │
              │          · evidence bundle                  │
              └─────────────────────────────────────────────┘
```

---

## naming conventions

| thing | convention | examples |
|---|---|---|
| graph node | `verb_noun(state) -> dict` | `gate_eda` · `join_units` · `route_question` |
| agent | `<Modality>Agent`, noun class | `HEAgent` · `PathwayAgent` |
| agent method | bare verb | `.encode()` · `.gate()` · `.fit()` |
| record | `<Kind>Record` | `DiagnosticRecord` · `SelectionRecord` |
| contract | noun, loaded from `configs/` | `EDAContract` · `RoutingTable` |
| resolver | `.resolve()` / `.evaluate()` | `RoutingTable.resolve(question)` |
| guard | `G1`-`G6`, returns `GuardResult` | `guards.nmi_label_confounder()` |
| store | `<Thing>Store` | `ArtifactStore` · `RecordStore` |

Nodes are **functions** because langgraph nodes are functions. Agents are **classes**
because they hold an encoder and its config. A node delegates to an agent method and
wraps the result in a Record - the node stays thin and testable, the agent is where the
modality lives.

## state

```python
class WorkflowState(TypedDict):        # module level - see the PEP 563 note
    project           ProjectConfig | None
    profile           CohortProfile | None
    verdict           Literal["proceed", "proceed_with_caution", "stop"] | None
    encoder_decision  SelectionRecord | None
    unit              UnitSpec | None
    question_class    int | None
    method            str | list[str] | None      # single | tie | contraindicated
    records           list[Record]                # the ledger
```

`records` is the ledger. `[r.actor for r in state["records"]]` gives the
deterministic / model / human breakdown - the reproducibility claim as a number.

## the four interrupts

| node | fires when | the human is asked |
|---|---|---|
| `gate_eda` | verdict = caution | accept the risks, or stop |
| `select_encoder` | the locked prior loses on this cohort | override the default, or halt |
| `route_question` | several methods tie | pick, given evidence + what the choice forecloses |
| `promote` | a new method is a candidate | admit it against **all** existing routes |

Each pauses hard, checkpoints state, resumes at that exact point. None has a default;
the system does not pick.

## what is parallel

Only the encoder fan-out. Three agents, disjoint outputs, joined at `select_encoder`.
Every other edge is sequential because each node consumes the previous node's decision.

## state of the build

| node | built |
|---|---|
| `gate_eda` | **partially** - validates a summary, cannot produce one |
| `route_question` | routing table exists as prose, not data |
| `align`, `evaluate` | run as scripts, unported, no graph |
| everything else | no |

The graph is the specification. One node is partially executable.