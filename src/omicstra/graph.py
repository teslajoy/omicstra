"""level 0 - the only entry point, and the only place a checkpointer is passed.

three levels, and one rule decides which a thing is:

    a thing gets its own GRAPH only if it can ask a human.
    everything else is a PROTOCOL.

    level 0  this file         routes between subgraphs. owns the checkpointer.
    level 1  eda_graph,        one compiled StateGraph per subgraph, each added
             agents/graph      as ONE node. each holds exactly one interrupt.
    level 2  protocols/        langchain Runnables. fixed order, no state, no
                               interrupt, ever. a transcription of a published
                               method, not a design.

why the checkpointer lives here and nowhere else
------------------------------------------------
`interrupt()` requires one. a subgraph compiled with its own would checkpoint to
a thread the caller cannot resume, so subgraphs compile with `checkpointer=None`
and INHERIT the parent's. an interrupt raised inside `eda` pauses the parent at
the `eda` node; the caller resumes with `Command(resume=...)` against the parent
using the same `thread_id`. so the CLI and the MCP server only ever talk to one
graph, and there is exactly one place that decides durability.

state
-----
subgraphs share keys with the parent BY NAME. a key a subgraph writes that the
parent does not declare stays inside the subgraph. `records` needs a reducer
because more than one node appends to it; scalars do not.
"""
from __future__ import annotations

import logging
import operator
import os
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from omicstra.contracts.project import ProjectConfig
from omicstra.labels import (HandleArm, Resolution, Route, StageStatus,
                             UnknownLabel, parse)
from omicstra.routing import load_routing_evidence
from omicstra.settings import settings

_log = logging.getLogger(__name__)

# "refuse" is a third arm and not an error state. a question the contract does
# not declare has no method behind it, so neither resolving nor computing is
# available - and saying so is the answer rather than a failure to answer.
#
# the vocabulary is `labels.Route` rather than a Literal: a Literal is checked by
# a type checker and by nothing at runtime, so it could not be iterated for the
# exhaustiveness lint and could not refuse an unknown value at a boundary. `Arm`
# stays as an alias because the name is published in tool descriptions.
Arm = Route


# MUST be module level. under `from __future__ import annotations` a
# function-local TypedDict raises NameError when langgraph resolves node
# annotations against module globals, and only when a conditional edge is
# present - so it surfaces late. same constraint as EDAState and RouteState.
class OmicstraState(TypedDict, total=False):
    # inputs
    question: str
    task_id: str | None
    project_id: str | None
    project_dir: str | None
    # discovery
    has_evidence: bool            # does the cohort hold ANY evidence
    has_evidence_for_task: bool   # does it hold evidence for the task asked
    # ask | compute | refuse, and the field is `route` because `arm` meant this
    # AND the run handle's he | st | align | evaluate - two enums in one key,
    # shared with every subgraph BY NAME, so a subgraph wired here received
    # `compute` in the field the MCP reads as a handle arm.
    #
    # renaming THIS side is what removes the collision: the subgraphs keep `arm`
    # and it now has exactly one meaning in the codebase. renaming their side
    # too would read better and would invalidate every persisted checkpoint for
    # no additional safety, since `labels.parse` already refuses a value from
    # the wrong vocabulary at every boundary.
    #
    # `arm` is NOT declared here, and that is the fix rather than a detail.
    # langgraph filters state by the declared keys in BOTH directions, so a
    # declared `arm` would take the subgraph's handle value on the way back up -
    # measured: a child writing arm="st" lands in the parent when the parent
    # declares the key, and is dropped when it does not. the field would then be
    # typed Route and hold a HandleArm.
    #
    # no deprecated read alias either. it would have existed for one case - a
    # level-0 thread checkpointed before the rename, resumed after - and in that
    # case the subgraph has already overwritten `arm` with its own vocabulary, so
    # the fallback reads "he" and `parse` refuses it. a fallback that fails in
    # the one case it exists for is worse than its absence, because the absence
    # is a legible refusal. `_arm` names the migration instead.
    route: Route
    route_reason: str
    # --- shared with subgraphs BY NAME ------------------------------------
    # a key the parent does not declare is DROPPED at the boundary, in both
    # directions. so every key a subgraph reads from or writes to the parent
    # must appear here, or it silently vanishes and the subgraph raises a
    # KeyError that looks like a bug in the subgraph.
    params: dict             # per-step params, cohort-supplied
    # the inventory chain takes its OWN params, keyed by inventory step id, and
    # the eda subgraph reads them under this name. undeclared, it was dropped at
    # this boundary and every inventory step reported "no params for this step" -
    # which is the failure this class of key is documented above to produce. it
    # went unseen because the only cohort that existed carried an evidence pack
    # and therefore always took the ask arm.
    inventory_params: dict
    # --- the encode subgraph's half of the boundary ------------------------
    # the gate decides from DECLARATIONS plus counts the inventory already
    # measured. it re-measures nothing, which is why `unit_counts` crosses the
    # boundary rather than being recomputed one level down.
    cohort: dict             # the cohort declaration, read once by discover
    encoder: str             # which encoder the gates are being asked about
    unit_counts: dict        # sample -> n_obs, lifted from the inventory record
    min_scope: int | None    # declared minimum for an authoritative run
    device: str | None       # declared, never probed - planning must work where the
                             # encoder cannot run, so nothing here imports a tensor library
    plan: dict               # what the run will cost, or why that is not sayable yet
    compute: bool            # False resolves and reports what WOULD run
    platform: str | None     # whose tile geometry - a cohort may carry several
    samples: list[str] | None  # an explicit subset, or every ingested sample
    # --- the evaluate subgraph's compute arm -------------------------------
    # which fitted run a COMPUTED verdict measures, and which continuous signals
    # to test for association. `signals` is the caller's question rather than the
    # cohort's declaration - a cohort declares what it HAS, not what somebody
    # will ask of it. both are here because the boundary drops what it is not
    # told about, which the test above this file's own rule asserts.
    run: str                 # the fitted run the compute arm measures
    signals: list[str]       # continuous signals to associate, caller-chosen
    include_pathway_scores: bool  # True loads AUCell scores for H3
    include_pathway_cosines: bool  # True loads gpath2vec cosine profiles for H3v2
    paired_view: str | None  # the other modality's view, for identifiability
    gates: list[dict]
    answers: dict
    hypotheses: list[str]    # which hypotheses the evaluate stage returns a verdict on
    # --- the assay node's half of the boundary ---------------------------------
    # pathway protocols need declared thresholds, keyed by step id. the assay
    # node runs only when the TASK needs pathway data, not when this key is
    # populated: a cohort may declare its thresholds without every question
    # triggering the chain. which operations fire is decided by task_id.
    pathway_params: dict
    # `report` is ONE key and every stage writes it, so the last writer wins.
    # that is correct for a caller asking "where did this end" and useless for
    # "what happened" - hence the accumulator below, written by a keeper node
    # after each stage rather than by the stages themselves, which are subgraphs
    # shared with the MCP and do not know they are in a chain.
    report: dict
    stages: Annotated[list[dict], operator.add]
    # the reducer is why more than one node may append. without it the second
    # writer OVERWRITES the first - invisible until two nodes both write.
    records: Annotated[list[dict], operator.add]
    # --- the route subgraph's half of the boundary --------------------------
    # plain types, no reducer. the three modality agents ARE concurrent, but they
    # fan in inside the subgraph, which carries the reducer on its own
    # `modality`; the parent sees one completed list written once, by one node.
    # a reducer here would be cargo - the rule that bites at this boundary is
    # declaration, not merging: drop a name and the value silently vanishes.
    proposed_method: str | None
    modality: list[dict]
    resolution: str | None
    override: bool
    awaiting_human: bool
    # outcomes
    verdict: str | None
    violations: list[str]
    cautions: list[str]
    escalations: list[dict]
    approved: bool | None
    decision: dict | None
    halted: bool


def discover(state: OmicstraState) -> dict:
    """which arm answers THIS question - resolved per task, not per cohort.

    absent evidence is not an error and not a fallback: a cohort that has never
    been evaluated cannot be routed from another cohort's table, so it goes to
    COMPUTE rather than borrowing an answer.

    the arm is a property of the QUESTION crossed with the cohort, not of the
    cohort alone. this read `bool(ev.get("tasks"))`, so a cohort holding evidence
    for one task of seven sent all seven to ask - and the six without evidence
    came back "this cohort records no evidence for X", which is a refusal
    standing where a routing decision belongs. "that needs compute, and here are
    the gates" is the true answer, and the compute arm exists to give it.

    three outcomes, and the third is the only real refusal:

        ask       this task has recorded evidence -> resolve from it
        compute   the task is in the contract, this cohort has not run it
        refuse    the task is not in the contract at all - nothing to compute
    """
    ev = load_routing_evidence(state.get("project_id"))
    tasks = ev.get("tasks") or {}
    task_id = state.get("task_id")
    has = bool(tasks)

    # the cohort declaration is read HERE, once, and travels. the encode gate
    # resolves five of its questions from it, and a subgraph that loaded the
    # file itself would read it again per invocation and could disagree with
    # what the parent recorded.
    cohort = _load_cohort(state.get("project_id"))

    arm, why = _arm_for(task_id, tasks, state.get("project_id"))
    out = {"has_evidence": has, "has_evidence_for_task": bool(task_id and task_id in tasks),
           "route": arm, "route_reason": why, "cohort": cohort}
    if not state.get("encoder"):
        out["encoder"] = _primary_encoder(state.get("project_id"))
    return out


def _known_task(task_id: str, project_id: str | None) -> bool:
    """is this a task the CONTRACT declares, whatever this cohort has run?

    the distinction carries the refusal. a declared task with no evidence here is
    work not yet done; an undeclared one is a question the system has no method
    for, and no amount of compute produces one.
    """
    from omicstra.routing import list_task_families

    try:
        fams = list_task_families(project_id=project_id)
    except Exception:                       # noqa: BLE001 - unreadable contract refuses
        return False
    return any(f.get("task_id") == task_id for f in fams.get("families", []))


def _arm_for(task_id: str | None, tasks: dict,
             project_id: str | None) -> tuple[Arm, str]:
    """the arm for one task, and the reason, which travels into the record."""
    if not task_id:
        # no task named: the caller is asking about the cohort rather than a
        # question. evidence decides, as it did before.
        return (Route.ASK if tasks else Route.COMPUTE,
                "no task named - the arm follows whether the cohort has any evidence")
    if task_id in tasks:
        return Route.ASK, f"{task_id} has recorded evidence on this cohort"
    if _known_task(task_id, project_id):
        return Route.COMPUTE, (f"{task_id} is a declared task family and this cohort has "
                           "not run it - that is work outstanding, not a refusal")
    return Route.REFUSE, (f"{task_id} is not a task family the contract declares, so there "
                      "is no method to run and nothing to compute")


def _load_cohort(project_id: str | None) -> dict:
    """the cohort's declarations, or an empty dict.

    absent is not an error: a scaffolded cohort has declared nothing yet, and
    every gate then stays open, which is the correct fail-closed result rather
    than a crash.
    """
    import json

    p = settings.project_root(project_id) / "cohort.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def _primary_encoder(project_id: str | None) -> str:
    """the encoder the cohort declares as primary, by name.

    the gates are asked about ONE encoder at a time because their answers differ
    per encoder - a gated model raises the terms question and an ungated one does
    not, and only some declare a unit floor.
    """
    try:
        cfg = ProjectConfig.load(project_id)
    except Exception as e:     # noqa: BLE001 - absent or unreadable is not an error here
        # a scaffolded cohort has no usable project.json yet, and the gates
        # staying open on an unnamed encoder is the correct fail-closed answer.
        _log.debug("no project config for %r: %s", project_id, e)
        return ""
    for enc in cfg.encoders:
        if getattr(enc, "role", "") == "primary" and getattr(enc, "name", ""):
            return enc.name
    return ""


def unit_counts_from_records(records: list[dict]) -> dict[str, int]:
    """sample -> n_obs, taken from the inventory's shape record.

    the compute contract says the platform_floor gate "re-asks nothing" and
    reads counts already measured. this is that reading: the inventory recorded
    n_obs per sample when it described the data, so the gate consumes it rather
    than opening the objects again.
    """
    for r in records:
        if r.get("step_id") == "shape":
            per = (r.get("observed") or {}).get("per_sample") or []
            return {s["sample"]: int(s["n_obs"]) for s in per
                    if s.get("sample") and s.get("n_obs") is not None}
    return {}


def _arm(state: OmicstraState) -> str:
    """the branch `discover` chose, parsed.

    `state.get("arm", "compute")` defaulted a MISSING value to the compute
    chain, which is a wiring fault answered with a guess - and the one guess
    that runs the most work. `discover` always sets `route`, so absent means
    either the graph was assembled wrong or this is a level-0 thread
    checkpointed before `arm` became `route`; the refusal says both.
    """
    route = state.get("route")
    if route is None and state.get("arm") is not None:
        raise UnknownLabel(
            "this level-0 thread carries `arm` and no `route`, so it was "
            "checkpointed before the two vocabularies were separated. its arm "
            "may already hold a handle label written by a subgraph, which is "
            "why there is no fallback: start a new run rather than resume it.")
    return str(parse(Route, route, "graph._arm route"))


def carry_counts(state: OmicstraState) -> dict:
    """lift per-sample unit counts out of the inventory records, for the gate.

    it exists because the encode gate must not re-measure what the inventory
    described - the compute contract says the platform_floor gate "re-asks
    nothing" - and a subgraph cannot reach the parent's records except through a
    declared key.
    """
    return {"unit_counts": unit_counts_from_records(state.get("records", []))}


def _halted(state: OmicstraState) -> str:
    """a stopped gate stops the run. encode does not start after a halt.

    without this the eda subgraph could refuse a cohort and the parent would
    walk straight into spending accelerated compute on it, which is the one
    thing the gate exists to prevent.
    """
    return "halt" if state.get("halted") else "encode"


# --- the pathway bridge, a function node like carry_counts ------------------
# it is NOT a subgraph because pathway protocols hold no gate. it invokes
# level-2 chains via build_protocol, once per sample, and appends the records.
# which chains run is decided by the caller's explicit flags
# (include_pathway_scores, include_pathway_cosines), not by the presence of
# params: a cohort may declare pathway_params without every question
# triggering the chains.

# which project.json declaration each evaluate flag resolves to, and which
# chain produces it when the declared file is absent. the flag on state is
# the selection boundary: it says what the caller's variant needs, and the
# assay node resolves it against declarations, not task families.
_FLAG_TO_SLOT: dict[str, tuple[str, str | None]] = {
    # (project.json key that holds the path, chain that produces it)
    "include_pathway_scores": ("pathway_scores", "assay_pathway_aucell"),
    "include_pathway_cosines": ("pathway_node_embeddings", None),
}


def _task_needs_pathway(state: OmicstraState) -> bool:
    """does this walk's task need any assay-level pathway artifact."""
    return any(state.get(flag) for flag in _FLAG_TO_SLOT)


def pathway_assay(state: OmicstraState) -> dict:
    """ensure assay-level pathway artifacts exist for downstream evaluation.

    the task variant determines which artifacts are required
    (TASK_ASSAY_REQUIREMENTS). each required artifact is checked against
    the cohort's project.json declaration: if the file is on disk, it is
    reused. only missing artifacts trigger computation.

    writes to `stages` (a reducer, so no previous stage's record is lost),
    `records` (also a reducer), and `report` (scalar, read by `_continue`
    to decide whether the chain proceeds). the keeper after pathway_assay
    captures this report before align overwrites it.
    """
    if not _task_needs_pathway(state):
        return {"report": {"status": "nothing_to_run"},
                "stages": [{"stage": "pathway_assay", "status": "nothing_to_run",
                            "outcome": "nothing_to_run",
                            "report": {"why": "the current task does not require "
                                       "pathway data"}}]}

    import json

    from omicstra.contracts.project import ProjectConfig
    from omicstra.settings import settings

    pid = state.get("project_id")
    root = settings.project_root(pid)
    proj = json.loads((root / "project.json").read_text())
    cfg = ProjectConfig.from_dir(root)

    # resolve which artifacts the caller's variant needs, and which are on disk
    required: dict[str, str] = {}     # flag -> project.json slot
    reused: dict[str, str] = {}       # slot -> path on disk
    missing: dict[str, str | None] = {}  # slot -> chain that produces it
    for flag, (slot, chain_name) in _FLAG_TO_SLOT.items():
        if not state.get(flag):
            continue
        required[flag] = slot
        decl = proj.get(slot)
        # check declaration AND file compatibility (existence)
        if isinstance(decl, dict) and decl.get("path"):
            p = cfg.path(decl["path"])
            if p.is_file():
                reused[slot] = str(p)
                continue
        elif isinstance(decl, str):
            p = cfg.path(decl)
            if p.is_file():
                reused[slot] = str(p)
                continue
        missing[slot] = chain_name

    if not missing:
        return {"report": {"status": "resolved"},
                "stages": [{"stage": "pathway_assay", "status": "resolved",
                            "outcome": "resolved",
                            "report": {"why": "all required artifacts already on disk",
                                       "reused": reused,
                                       "required": sorted(required.values())}}]}

    # any missing slot with no registered producer chain is an explicit refusal
    unproducible = {s for s, c in missing.items() if c is None}
    if unproducible:
        return {"report": {"status": "refused"},
                "stages": [{"stage": "pathway_assay", "status": "refused",
                            "outcome": "refused",
                            "report": {"why": f"required artifacts absent and not "
                                       f"producible: {sorted(unproducible)}",
                                       "reused": reused,
                                       "missing": sorted(missing)}}],
                "records": [{"step_id": "assay_pathway", "status": "not_run",
                             "result": f"no producer for {sorted(unproducible)}",
                             "decision": "provide the file or use a variant "
                                          "whose inputs are available"}]}

    # resolve computation params
    pathway_params = state.get("pathway_params") or {}
    if not pathway_params:
        cohort = state.get("cohort") or {}
        pathway_params = cohort.get("pathway_params") or {}
    if not pathway_params:
        return {"report": {"status": "refused"},
                "stages": [{"stage": "pathway_assay", "status": "refused",
                            "outcome": "refused",
                            "report": {"why": f"missing {sorted(missing)} but the "
                                       "cohort declares no pathway_params",
                                       "missing": sorted(missing)}}],
                "records": [{"step_id": "assay_pathway", "status": "not_run",
                             "result": f"missing {sorted(missing)}, no pathway_params",
                             "decision": "declare pathway_params for this cohort"}]}

    from omicstra.adapters.canonical import list_samples
    from omicstra.adapters.pathway import PathwayDataMissing, arrays
    from omicstra.protocols import build_protocol
    from omicstra.protocols.pathway import AUCELL_STEPS, VARIABLE_GENE_STEPS

    # build the chains for each missing artifact
    chains: list[tuple[str, list]] = []
    for slot, chain_name in missing.items():
        if chain_name == "assay_pathway_aucell":
            niche_qc_step = VARIABLE_GENE_STEPS[0]
            assert niche_qc_step.id == "niche_qc"
            chains.append((chain_name, [niche_qc_step, *AUCELL_STEPS]))

    # load pathway_sets for the aucell chain
    pathway_sets = None
    if any(cn == "assay_pathway_aucell" for _, cn in chains):
        ps_file = pathway_params.get("pathway_sets_file")
        if ps_file:
            ps_full = cfg.path(ps_file)
            if ps_full.is_file():
                pathway_sets = json.loads(ps_full.read_text())
            else:
                return {"report": {"status": "refused"},
                        "stages": [{"stage": "pathway_assay", "status": "refused",
                                    "outcome": "refused",
                                    "report": {"why": f"pathway_sets_file declared "
                                               f"but absent: {ps_full}"}}],
                        "records": [{"step_id": "assay_pathway", "status": "not_run",
                                     "result": f"pathway_sets_file not found: {ps_file}",
                                     "decision": "provide the file or remove the "
                                                  "declaration"}]}

    all_records: list[dict] = []
    samples_run = []
    samples_skipped = []

    for smp in list_samples(root):
        sid = smp.sample_id
        if not (smp.counts and smp.spots):
            samples_skipped.append({"sample": sid, "why": "no counts or coordinates"})
            continue
        try:
            ctx = arrays(str(root), sid)
        except PathwayDataMissing as e:
            samples_skipped.append({"sample": sid, "why": str(e)})
            continue

        if pathway_sets is not None:
            ctx["pathway_sets"] = pathway_sets

        for unit_name, steps in chains:
            chain = build_protocol(steps, unit_name)
            recs = chain.invoke({**ctx, "params": pathway_params})
            for step_id, rec in recs.items():
                rec["sample"] = sid
                rec["unit"] = unit_name
                all_records.append(rec)

        samples_run.append(sid)

    status = "computed" if samples_run else "refused"
    why = (f"{len(samples_run)} sample(s) processed" if samples_run
           else "no sample carried both counts and niche data")
    stage_report = {"status": status, "why": why,
                    "samples_run": samples_run,
                    "samples_skipped": samples_skipped,
                    "reused": reused, "missing": sorted(missing)}

    return {"report": {"status": status},
            "records": all_records,
            "stages": [{"stage": "pathway_assay", "status": status,
                        "outcome": status, "report": stage_report}]}


# the stages of the compute arm, in the order their inputs allow. a stage is
# wired only if it was injected, so a host without a tensor library still builds
# the graph - the same discipline the encode slot has had since it existed.
COMPUTE_CHAIN = ("eda", "encode", "st", "align", "evaluate")

# --- what this consumer does with every label it can be handed ---------------
# TWO vocabularies arrive here and they are not interchangeable: a stage reports
# a `StageStatus`, the routing subgraph concludes a `Resolution`. `labels.py`
# records what one list for both cost.
#
# both halves are spelled out rather than one plus a fall-through, because a
# label in neither half is a label nobody decided about, and
# `test_rule_lints.py::test_every_label_a_producer_emits_is_handled_by_its_consumer`
# fails on it rather than letting it continue by default.

# the stage left work in flight, or could not run. STOP rather than walk into a
# stage whose inputs do not exist yet.
#
# `submitted` is the one that matters and the reason this is not a straight
# chain: an encode with compute=True returns at submission, so the embeddings a
# later stage reads are not on disk when the next node would run. the run handle
# is how the chain continues - the shards land, and a caller re-enters at the
# stage that needs them.
STOP_AFTER = frozenset({
    StageStatus.SUBMITTED, StageStatus.REFUSED, StageStatus.HALTED,
    StageStatus.UNAVAILABLE, StageStatus.NOT_RUN,
})

# the stage finished and produced what the next one reads.
CONTINUE_AFTER = frozenset({
    StageStatus.READY, StageStatus.RESOLVED, StageStatus.COMPUTED,
    StageStatus.EVALUATED, StageStatus.MEASURED, StageStatus.NOTHING_TO_RUN,
})

# the routing half. the three that stop are the three where the system has NOT
# answered: a refusal stands, a tie says "the system does not pick", and an
# escalation is a question for a person. letting any of them through meant a
# chain that abstained and then stated a verdict anyway.
#
# `refuse` stopping here is also what makes `ask_human` a gate rather than only
# a pause: a declined override leaves the resolution at `refuse`, and before
# this the decline flowed on exactly as an approval did.
RESOLUTION_STOPS = frozenset({
    Resolution.REFUSE, Resolution.TIE, Resolution.ESCALATE,
})

# the system answered, so the chain may use the answer. `override_ack` is a
# human's explicit yes on the record, which is a decision and not an abstention.
RESOLUTION_CONTINUES = frozenset({
    Resolution.RECOMMEND, Resolution.OVERRIDE_ACK,
})


def _keeper(name: str):
    """record a finished stage's outcome before the next one overwrites it.

    `status` and `resolution` are kept as SEPARATE fields. collapsing them with
    `rep.get("status") or state.get("resolution")` is what made the two
    vocabularies indistinguishable one field later: a reader - and `_continue` -
    could no longer tell which producer had written the value it was holding.
    the collapse existed because a chain that read only `report` logged the
    routing stage as `unknown`; recording both answers that without the merge.
    """
    def keep(state: OmicstraState) -> dict:
        rep = dict(state.get("report") or {})
        row = {"stage": name, "status": rep.get("status"),
               "resolution": state.get("resolution"), "report": rep}
        # `outcome` is what a reader wants: whichever of the two decided, with
        # `unknown` when neither did. it is for the RECORD only - nothing
        # branches on it, because branching is what needs the vocabularies apart.
        row["outcome"] = row["status"] or row["resolution"] or "unknown"
        return {"stages": [row]}

    keep.__name__ = f"after_{name}"
    return keep


def _continue(state: OmicstraState) -> str:
    """go on, or stop here and say where.

    each vocabulary is parsed against its OWN enum, and an unknown label raises
    rather than falling through to "go". that fall-through is the defect: a
    value this function did not recognise looked exactly like a value it had
    decided to continue on.
    """
    if state.get("halted"):
        return "stop"

    status = (state.get("report") or {}).get("status")
    if status is not None:
        return "stop" if parse(StageStatus, status,
                               "graph._continue report.status") in STOP_AFTER else "go"

    # no report: the routing subgraph states its outcome as a resolution. it is
    # a different vocabulary and gets its own partition.
    res = state.get("resolution")
    if res is not None:
        return "stop" if parse(Resolution, res,
                               "graph._continue resolution") in RESOLUTION_STOPS else "go"

    # neither: the stage decided nothing it wanted to report, which is not a
    # reason to stop - carry_counts is the one that does this.
    return "go"


def make_checkpointer(spec: str | None = None):
    """`memory` or `sqlite:<path>`. the http path needs the second.

    InMemorySaver dies with the process, so `resume(run_id)` only works inside
    one invocation - which quietly makes "any instance handles any request"
    false: a second instance, or the same one after a restart, has never heard
    of the thread. a durable saver is what turns that sentence from a claim
    into a fact, and it is the difference between a gate a person can come back
    to tomorrow and one they must answer before their shell exits.

    gscratch cannot host the sqlite file - it has no POSIX locking and names
    sqlite as unsupported. use node-local disk, RDS, or postgres there.
    """
    spec = spec or os.environ.get("OMICSTRA_CHECKPOINT", "memory")
    if spec == "memory":
        from langgraph.checkpoint.memory import InMemorySaver
        return InMemorySaver(), "memory"
    if spec.startswith("sqlite:"):
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver
        path = spec.split(":", 1)[1] or "omicstra_checkpoints.sqlite"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        return saver, f"sqlite:{path}"
    raise ValueError(f"OMICSTRA_CHECKPOINT must be memory or sqlite:<path>, not {spec!r}")


def build_omicstra_graph(checkpointer: Any = None,
                         eda: Any = None,
                         encode: Any = None,
                         route: Any = None,
                         st: Any = None,
                         align: Any = None,
                         evaluate: Any = None) -> Any:
    """the only graph a caller compiles.

    every subgraph is INJECTED so this file imports none of them at module
    scope - level 0 must stay importable where no encoder dependency is
    installed, which is the same reason `pyproject` keeps them in an extra.

    the two arms take different subgraphs, and they are not interchangeable.
    `route` runs the ASK arm, for a cohort that has an evidence pack. the
    COMPUTE arm is a chain, in the order each stage's inputs allow:

        eda -> carry_counts -> encode -> st -> pathway_assay -> align -> evaluate

    and it stops at the first stage that leaves work in flight or cannot run.
    `submitted` is the usual stop: an encode with compute=True returns at
    submission, so the embeddings the next stage reads are not on disk yet. the
    run handle is how the chain continues. with compute=False the whole chain
    walks and reports what WOULD run at every stage, which is what makes "ask
    the three hypotheses" answerable in one call.

    a stage absent from the arguments is absent from the graph. so this is also
    how a caller runs a prefix of the chain deliberately.

    pass `checkpointer=InMemorySaver()` for a single session, or a durable saver
    when a review must survive a restart. passing None means no interrupt in any
    subgraph can fire - a valid read-only configuration, not a default to rely
    on.
    """
    g = StateGraph(OmicstraState)
    g.add_node("discover", discover)
    g.add_edge(START, "discover")

    subs = {"eda": eda, "encode": encode, "st": st, "align": align,
            "evaluate": evaluate}
    chain = [n for n in COMPUTE_CHAIN if subs[n] is not None]
    for name in chain:
        g.add_node(name, subs[name])            # a compiled subgraph, as ONE node

    if route is not None:
        g.add_node("route", route)

    # carry_counts is a node between two stages rather than a bare edge. the
    # encode gate reads counts the inventory already measured, and something has
    # to lift them out of the records the eda subgraph appended. putting that in
    # the subgraph would make it re-read the inventory; an edge cannot hold code.
    #
    # pathway_assay is the same pattern between st and align: it resolves
    # pathway artifacts the caller's flags request. unlike carry_counts it CAN
    # refuse, so it gets its own keeper + conditional: a refusal stops the chain
    # before align runs on missing inputs.
    pre = {"encode": "carry_counts"} if {"eda", "encode"} <= set(chain) else {}
    if {"st", "align"} <= set(chain):
        pre["align"] = "pathway_assay"
    if pre.get("encode") == "carry_counts":
        g.add_node("carry_counts", carry_counts)
    if pre.get("align") == "pathway_assay":
        g.add_node("pathway_assay", pathway_assay)

    for i, name in enumerate(chain):
        nxt = chain[i + 1] if i + 1 < len(chain) else None
        if nxt is None:
            g.add_edge(name, END)
            continue
        entry = pre.get(nxt, nxt)
        # eda keeps the dedicated halt check it has always had: a refused cohort
        # must not reach the encoder. every other junction uses the general rule,
        # which also stops on a stage that only submitted.
        if name == "eda" and entry == "carry_counts":
            g.add_edge("eda", "carry_counts")
            g.add_conditional_edges("carry_counts", _halted,
                                    {"halt": END, "encode": "encode"})
            continue
        keep = f"after_{name}"
        g.add_node(keep, _keeper(name))
        g.add_edge(name, keep)
        g.add_conditional_edges(keep, _continue, {"go": entry, "stop": END})
        if entry != nxt:
            # bridge nodes that can refuse need their own keeper + conditional
            # so a refusal blocks the downstream stage. carry_counts never
            # refuses so it wires directly; pathway_assay does.
            if entry == "pathway_assay":
                keep_bridge = "after_pathway_assay"
                g.add_node(keep_bridge, _keeper("pathway_assay"))
                g.add_edge(entry, keep_bridge)
                g.add_conditional_edges(keep_bridge, _continue,
                                        {"go": nxt, "stop": END})
            else:
                g.add_edge(entry, nxt)

    targets: dict[str, str] = {}
    if chain:
        targets["compute"] = chain[0]

    # the ASK arm resolves from a lookup table, but it is still a subgraph,
    # because resolving is not the whole job: a contraindicated method stops the
    # graph and asks a person, and that interrupt needs a node to live in. an
    # arm that terminates here can report a contraindication; only one that can
    # pause enforces it.
    if route is not None:
        targets["ask"] = "route"
        # and the ask arm continues INTO evaluate, which is where the first
        # wiring of this chain was wrong: `evaluate` reads the artifacts the
        # cohort's evidence pack cites, and a cohort with a pack takes the ask
        # arm - so hanging it off the compute arm alone put the one stage that
        # needs a pack on the one arm that is taken when there is none. it stays
        # on both: terminal on the compute arm, where it refuses and names the
        # missing pack, and terminal here, where it states the verdict.
        if evaluate is not None:
            g.add_node("after_route", _keeper("route"))
            g.add_edge("route", "after_route")
            g.add_conditional_edges("after_route", _continue,
                                    {"go": "evaluate", "stop": END})
        else:
            g.add_edge("route", END)

    # refuse terminates at the router when there is one - it can report the
    # reason as a decision - and at END otherwise.
    targets.setdefault("refuse", "route" if route is not None else END)
    targets.setdefault("ask", END)
    targets.setdefault("compute", END)
    g.add_conditional_edges("discover", _arm, targets)

    return g.compile(checkpointer=checkpointer)
