"""level 0 reaches the encode gate, and carries what the gate needs.

`graphs/encode.py` was built, tested and connected to nothing: the slot existed
in `build_omicstra_graph` and no caller filled it. these assert the wiring, not
the gate - the gate's own behaviour is covered in test_encode_graph.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from omicstra.graph import (
    OmicstraState,
    build_omicstra_graph,
    carry_counts,
    unit_counts_from_records,
)

ROOT = Path(__file__).resolve().parents[1]


def _app(**subgraphs):
    from omicstra.graphs.eda import build_eda_graph
    from omicstra.graphs.encode import build_encode_graph

    kw = {"eda": build_eda_graph(checkpointer=None),
          "encode": build_encode_graph(checkpointer=None)}
    kw.update(subgraphs)
    return build_omicstra_graph(checkpointer=InMemorySaver(), **kw)


# --- the wiring exists ------------------------------------------------------
def test_the_compute_arm_reaches_encode():
    nodes = set(_app().get_graph().nodes)
    assert {"discover", "eda", "carry_counts", "encode"} <= nodes


def test_encode_is_absent_when_not_passed():
    """it stays injectable. a host with no encoder dependencies installed must
    still be able to build the graph."""
    from omicstra.graphs.eda import build_eda_graph

    nodes = set(build_omicstra_graph(checkpointer=InMemorySaver(),
                                     eda=build_eda_graph(checkpointer=None))
                .get_graph().nodes)
    assert "encode" not in nodes and "carry_counts" not in nodes


# --- the boundary carries what the gate needs -------------------------------
def test_every_key_the_encode_subgraph_reads_is_declared_by_the_parent():
    """a key the parent does not declare is DROPPED at the boundary, in both
    directions - so it vanishes silently and the subgraph raises a KeyError that
    looks like its own bug."""
    from omicstra.graphs.encode import EncodeState

    parent = set(OmicstraState.__annotations__)
    child = set(EncodeState.__annotations__)
    missing = child - parent - {"approved", "shards"}
    assert not missing, f"the encode subgraph reads {sorted(missing)}, undeclared by level 0"


def test_counts_come_from_the_inventory_not_a_re_measurement():
    """the compute contract says the platform_floor gate re-asks nothing."""
    rec = {"step_id": "shape",
           "observed": {"per_sample": [{"sample": "A", "n_obs": 900},
                                       {"sample": "B", "n_obs": 8}]}}
    assert unit_counts_from_records([rec]) == {"A": 900, "B": 8}
    assert carry_counts({"records": [rec]}) == {"unit_counts": {"A": 900, "B": 8}}


def test_absent_inventory_records_yield_no_counts_rather_than_raising():
    """a cohort that has not been inventoried leaves the floor gate with nothing
    to decide on, which keeps it open - the correct fail-closed result."""
    assert unit_counts_from_records([]) == {}
    assert carry_counts({}) == {"unit_counts": {}}


def test_a_halted_gate_stops_before_encode():
    """without this the eda subgraph could refuse a cohort and the parent would
    walk into spending accelerated compute on it."""
    from omicstra.graph import _halted

    assert _halted({"halted": True}) == "halt"
    assert _halted({"halted": False}) == "encode"
    assert _halted({}) == "encode"


# --- the declarations the gate resolves from --------------------------------
def test_discover_reads_the_cohort_declaration_once():
    from omicstra.graph import _load_cohort

    d = _load_cohort(None)
    assert isinstance(d, dict)


def test_an_unreadable_cohort_file_is_empty_not_fatal(tmp_path, monkeypatch):
    """a scaffolded cohort has declared nothing yet; every gate then stays open,
    which is correct rather than a crash."""
    from omicstra.graph import _load_cohort, _primary_encoder
    from omicstra.settings import settings

    monkeypatch.setattr(settings, "project_dir", tmp_path)
    assert _load_cohort(None) == {}
    assert _primary_encoder(None) == ""


def test_the_primary_encoder_is_taken_from_the_declaration():
    """the gates are asked about ONE encoder because their answers differ per
    encoder - a gated model raises the terms question, an ungated one does not."""
    from omicstra.graph import _primary_encoder

    if not (ROOT / "projects" / "tnbc-92" / "project.json").is_file():
        pytest.skip("fixture cohort absent")
    assert _primary_encoder("tnbc-92") == "virchow2"


def test_the_fixture_reaches_encode_with_zero_open_gates():
    """the end-to-end precondition: declarations resolve every gate, so the
    compute arm runs with nobody in the loop."""
    root = ROOT / "projects" / "tnbc-92"
    inv, coh = root / "inventory.json", root / "cohort.json"
    if not (inv.is_file() and coh.is_file()):
        pytest.skip("fixture cohort absent")

    from omicstra.graphs.encode import gates

    counts = carry_counts({"records": [json.loads(inv.read_text())["shape"]]})
    out = gates({"cohort": json.loads(coh.read_text()), "encoder": "novae",
                 **counts})
    assert not [g for g in out["gates"] if g["open"]], \
        f"open on the fixture: {[g['id'] for g in out['gates'] if g['open']]}"


def test_level_zero_declares_the_inventory_chain_params():
    """undeclared, the key is dropped at the boundary and the chain silently dies.

    graph.py's own docstring says a key the parent does not declare is discarded
    in both directions. `inventory_params` was not declared, so every inventory
    step reported "cohort supplied no params for this step" and each one after it
    reported "upstream failed" - eight cautions that all trace to one missing
    line. it survived because the only cohort that existed carried an evidence
    pack and therefore never entered the compute arm.
    """
    from omicstra.graph import OmicstraState
    assert "inventory_params" in OmicstraState.__annotations__


def test_the_compute_arm_supplies_params_for_every_inventory_step():
    """the two chains take separate params, keyed by different step ids.

    a step with no entry is skipped rather than guessed at - which is right, and
    is exactly why an empty mapping disables the whole chain instead of erroring.
    """
    import inspect

    from omicstra import cli
    from omicstra.protocols.inventory import INVENTORY_STEPS

    src = inspect.getsource(cli)
    assert "inventory_params" in src, "the compute arm never builds them"
    assert "INVENTORY_STEPS" in src, "params must cover every declared step, not a fixed list"
    # the first step needs a directory to look in; the rest default empty
    assert '"files"' in src and "inputs_dir" in src
    assert len(INVENTORY_STEPS) >= 8


# --- the compute arm is a CHAIN, and it stops where it must -----------------
def _chain(**extra):
    """every stage level 0 can hold, so the wiring is asserted on the full shape."""
    from omicstra.graphs.align import build_align_graph
    from omicstra.graphs.eda import build_eda_graph
    from omicstra.graphs.encode import build_encode_graph
    from omicstra.graphs.evaluate import build_evaluate_graph
    from omicstra.graphs.st import build_st_graph

    kw = {"eda": build_eda_graph(checkpointer=None),
          "encode": build_encode_graph(checkpointer=None),
          "st": build_st_graph(checkpointer=None),
          "align": build_align_graph(checkpointer=None),
          "evaluate": build_evaluate_graph(checkpointer=None)}
    kw.update(extra)
    return build_omicstra_graph(checkpointer=InMemorySaver(), **kw)


def test_the_compute_arm_reaches_the_verdict():
    """align and evaluate were built, tested, and connected to nothing - the same
    thing that had happened to encode, one release earlier."""
    g = _chain().get_graph()
    nodes = set(g.nodes)
    assert {"discover", "eda", "carry_counts", "encode", "st", "align",
            "evaluate"} <= nodes

    edges = {(e.source, e.target) for e in g.edges}
    assert ("eda", "carry_counts") in edges
    assert ("carry_counts", "encode") in edges
    # the keeper nodes carry each stage's report before the next overwrites it
    for a, b in (("encode", "st"), ("st", "align"), ("align", "evaluate")):
        assert (a, f"after_{a}") in edges, f"{a} does not record its stage"
        assert (f"after_{a}", b) in edges, f"{a} does not reach {b}"
        assert (f"after_{a}", "__end__") in edges, f"{a} cannot stop the chain"


def test_a_stage_that_only_submitted_stops_the_chain():
    """an encode with compute=True returns AT SUBMISSION, so the embeddings a
    later stage reads are not on disk when the next node would run. walking on
    would have align report `unavailable` on a run that is working correctly.
    """
    from omicstra.graph import STOP_AFTER, _continue

    assert _continue({"report": {"status": "submitted"}}) == "stop"
    assert _continue({"report": {"status": "ready"}}) == "go"
    assert _continue({"report": {"status": "resolved"}}) == "go"
    assert _continue({"halted": True, "report": {"status": "ready"}}) == "stop"
    assert _continue({}) == "go", "no report yet is not a stop"
    assert "submitted" in STOP_AFTER and "refused" in STOP_AFTER


def test_each_stage_is_kept_before_the_next_overwrites_the_report():
    """`report` is one key and every stage writes it, so the last writer wins.
    that answers "where did this end" and loses "what happened".
    """
    from omicstra.graph import _keeper

    out = _keeper("encode")({"report": {"status": "ready", "n_shards": 4}})
    assert out["stages"] == [{"stage": "encode", "status": "ready",
                              "report": {"status": "ready", "n_shards": 4}}]
    assert _keeper("st")({})["stages"][0]["status"] == "unknown"
    # the parent accumulates rather than overwrites, or the second stage's row
    # would replace the first
    from typing import get_type_hints
    assert "stages" in OmicstraState.__annotations__
    assert get_type_hints(OmicstraState, include_extras=True)  # resolves under future annotations


def test_a_prefix_of_the_chain_is_a_valid_graph():
    """a stage absent from the arguments is absent from the graph, which is how a
    caller runs a prefix deliberately - and how a host with no tensor library
    builds level 0 at all."""
    nodes = set(_chain(align=None, evaluate=None).get_graph().nodes)
    assert "st" in nodes and "align" not in nodes and "evaluate" not in nodes
    edges = {(e.source, e.target) for e in _chain(align=None, evaluate=None)
             .get_graph().edges}
    assert ("st", "__end__") in edges, "the last stage present must terminate"


def test_every_key_the_later_subgraphs_read_is_declared_by_the_parent():
    """the same boundary rule that cost eight inventory cautions once: a key the
    parent does not declare is dropped silently, in both directions."""
    from omicstra.graphs.align import AlignState
    from omicstra.graphs.evaluate import EvaluateState

    # the ST arm shares `EncodeState` deliberately - the gate SET differs, the
    # state does not - and the encode case is asserted above.
    parent = set(OmicstraState.__annotations__)
    for name, child in (("align", AlignState), ("evaluate", EvaluateState)):
        missing = set(child.__annotations__) - parent - {"approved", "shards", "verdict"}
        assert not missing, f"the {name} subgraph reads {sorted(missing)}, undeclared by level 0"


def test_the_arm_key_means_two_things_and_the_reader_refuses_the_wrong_one():
    """level 0's `arm` is ask | compute | refuse; a subgraph's is the run
    handle's he | st | align | evaluate. one key, two meanings, shared BY NAME -
    so a level-0 thread handed to a resume used to re-enter the H&E graph.
    """
    from omicstra.mcp.server import ARMS, NotAnArmHandle

    assert set(ARMS) == {"he", "st", "align", "evaluate"}
    assert "compute" not in ARMS and "ask" not in ARMS
    assert issubclass(NotAnArmHandle, ValueError)


def test_evaluate_is_terminal_on_both_arms():
    """the first wiring of this put evaluate on the compute arm alone, which is
    the arm taken when a cohort has NO evidence pack - and evaluate reads the
    artifacts a pack cites. so the one stage needing a pack hung off the one arm
    that means there isn't one.
    """
    from omicstra.graphs.route import build_route_graph

    g = _chain(route=build_route_graph(checkpointer=None)).get_graph()
    edges = {(e.source, e.target) for e in g.edges}
    assert ("route", "after_route") in edges
    assert ("after_route", "evaluate") in edges, "the ask arm must reach the verdict"
    assert ("after_route", "__end__") in edges, "a contraindication still stops"
    assert ("after_align", "evaluate") in edges, "and so must the compute arm"
    assert ("evaluate", "__end__") in edges


def test_the_ask_arm_still_terminates_without_the_verdict_stage():
    """evaluate stays injectable; the ask arm must not depend on it."""
    from omicstra.graphs.route import build_route_graph

    g = _chain(route=build_route_graph(checkpointer=None), evaluate=None).get_graph()
    edges = {(e.source, e.target) for e in g.edges}
    assert ("route", "__end__") in edges and "after_route" not in set(g.nodes)


def test_a_stage_that_states_its_outcome_outside_report_is_not_logged_unknown():
    """the routing subgraph writes `resolution`, not a report, so a keeper that
    read only `report` logged the one stage that had decided something as
    `unknown`."""
    from omicstra.graph import _keeper

    assert _keeper("route")({"resolution": "recommend"})["stages"][0]["status"] == \
        "recommend"
    assert _keeper("route")({"report": {"status": "refused"},
                             "resolution": "x"})["stages"][0]["status"] == "refused"


def test_the_cli_holds_the_whole_chain():
    """align and evaluate were built, tested and reachable one arm at a time
    through the MCP - the same thing that had happened to encode a release
    earlier. a chain nothing walks is a chain nobody has run.
    """
    import inspect

    from omicstra import cli

    # a click command wraps its callback; the source is the function's
    src = inspect.getsource(cli.chain.callback)
    for fn in ("build_eda_graph", "build_encode_graph", "build_st_graph",
               "build_align_graph", "build_evaluate_graph", "build_route_graph"):
        assert fn in src, f"omicstra chain does not inject {fn}"
    assert "--answer" in src and "the system does not pick" in src
