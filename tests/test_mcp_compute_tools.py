"""the compute path is visible to a client, and says what it decided from.

before these, a client could gate, describe and route - nothing told it a compute
path existed. there is deliberately no tool that RUNS an extraction: MCP is
request/response and a full run is hours, so a blocking tool would time out on
every real cohort. submission is the dispatcher's job and the entry point is the
CLI.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from omicstra.mcp.server import check_compute_gates, describe_compute_plan, srv

ROOT = Path(__file__).resolve().parents[1]


def test_both_tools_are_served():
    names = {t.name for t in asyncio.run(srv.list_tools())}
    assert {"check_compute_gates", "describe_compute_plan"} <= names


def test_no_tool_runs_an_extraction():
    """a blocking multi-hour tool call has no viable shape over this protocol."""
    names = {t.name for t in asyncio.run(srv.list_tools())}
    for forbidden in ("run_encode", "run_align", "encode_cohort", "run_compute"):
        assert forbidden not in names


def test_every_served_tool_documents_itself():
    for t in asyncio.run(srv.list_tools()):
        assert t.description and len(t.description) > 40, f"{t.name} is undocumented"


# --- the gates tool ---------------------------------------------------------
def test_the_gates_tool_states_what_its_counts_cover():
    """a floor gate that stays silent because it only saw two samples is worse
    than one that fires wrongly - it is indistinguishable from a pass. so the
    coverage travels with the answer.
    """
    if not (ROOT / "projects" / "tnbc-92" / "cohort.json").is_file():
        pytest.skip("fixture cohort absent")
    d = check_compute_gates("novae")
    cov = d["unit_counts_cover"]
    assert {"source", "n_samples", "note"} <= set(cov)
    assert cov["n_samples"] > 0


def test_the_fuller_count_source_wins():
    """the inventory is the declared source but records only what it was run
    over. the canonical ingest covers every converted sample."""
    root = ROOT / "projects" / "tnbc-92"
    inv, ing = root / "inventory.json", root / "data" / "canonical" / "ingest.json"
    if not (inv.is_file() and ing.is_file()):
        pytest.skip("fixture artifacts absent")

    n_inv = len(json.loads(inv.read_text())["shape"]["observed"]["per_sample"])
    n_ing = len(json.loads(ing.read_text())["sources"])
    cov = check_compute_gates("novae")["unit_counts_cover"]
    assert cov["n_samples"] == max(n_inv, n_ing)


def test_the_floor_gate_sees_the_whole_cohort():
    """19 of 280 fall below Novae's 512-spot floor. reading a partial inventory
    made this gate silent, which looked like a pass."""
    if not (ROOT / "projects" / "tnbc-92" / "data" / "canonical" / "ingest.json").is_file():
        pytest.skip("canonical ingest absent")
    g = {x["id"]: x for x in check_compute_gates("novae")["gates"]}
    assert "platform_floor" in g, "the floor gate must appear for an encoder with a floor"
    o = g["platform_floor"]["observed"]
    assert o["n_samples"] == 280 and o["n_below"] == 19


def test_a_resolved_gate_names_where_its_answer_came_from():
    if not (ROOT / "projects" / "tnbc-92" / "cohort.json").is_file():
        pytest.skip("fixture cohort absent")
    d = check_compute_gates("novae")
    assert d["unattended"] is True
    for g in d["gates"]:
        assert g["open"] is False
        assert g["answered_from"], f"{g['id']} resolved without naming its source"


# --- the plan tool ----------------------------------------------------------
def test_the_plan_reports_what_would_be_skipped():
    """resume is decided by the output file, so a plan that ignores what exists
    would overstate the work by the whole cache."""
    if not (ROOT / "projects" / "tnbc-92" / "data" / "canonical" / "ingest.json").is_file():
        pytest.skip("canonical ingest absent")
    p = describe_compute_plan("virchow2")
    assert p["runnable"] is True
    assert p["n_shards"] == p["n_with_image"]
    assert p["already_done"] + p["remaining"] == p["n_shards"]


def test_an_unconverted_cohort_is_not_runnable_and_says_why(tmp_path, monkeypatch):
    from omicstra.settings import settings

    monkeypatch.setattr(settings, "project_dir", tmp_path)
    p = describe_compute_plan("virchow2")
    assert p["runnable"] is False
    assert "ingest" in p["note"] or "ingest" in p.get("why_not", "")
