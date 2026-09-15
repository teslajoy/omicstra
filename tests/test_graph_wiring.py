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
