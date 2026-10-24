"""the compute arm, and the reader that makes it reachable.

tier 1: synthetic and declaration-level, so it runs anywhere. the cohort-scale
check that the computed verdict matches what the chain produces directly is in
test_evaluate_slicediff.py.

the thing under test is reachability, not the maths. the chain takes matrices
and MCP carries JSON, so nothing could hand it a 35,594 x 512 matrix over
stdio - the reader is what closes that, and its job is to refuse by FIELD NAME
rather than to default.
"""
from __future__ import annotations

import json

import pytest

from omicstra.adapters.evaluation import REQUIRED, DeclarationMissing, declared


def _project(tmp_path, **over):
    d = tmp_path / "cohort_x"
    d.mkdir()
    full = {"project_id": "cohort-x", "platform": "p",
            "niches_dir": "data/niches", "runs_dir": "runs",
            "h2_label": "state", "headline_view": "z_a",
            "supervision": "weights", "seed": 7}
    merged = {**full, **over}
    # an OMITTED key, not a null one: that is what a scaffolded cohort looks
    # like, and it is the case ProjectConfig fills in from its own defaults
    merged = {k: v for k, v in merged.items() if v is not None}
    (d / "project.json").write_text(json.dumps(merged))
    return d


# ----------------------------------------------------------- the reader ------
def test_a_complete_declaration_is_read_whole(tmp_path):
    d = declared(_project(tmp_path))
    assert set(REQUIRED) <= set(d)
    assert d["h2_label"] == "state" and d["headline_view"] == "z_a"
    assert d["project_id"] == "cohort-x"


@pytest.mark.parametrize("field", sorted(REQUIRED))
def test_an_absent_field_is_refused_by_name(tmp_path, field):
    """and NOT defaulted. three of these carry the seed cohort's values as
    package defaults on `ProjectConfig` - h2_label, supervision, headline_view -
    so a second cohort would be scored against a label it does not have. the
    numeric threshold lint cannot see them because they are strings."""
    p = _project(tmp_path, **{field: None})
    with pytest.raises(DeclarationMissing) as e:
        declared(p)
    assert field in str(e.value)
    # the reason travels with the refusal, not just the field name
    assert REQUIRED[field].split(".")[0][:20] in str(e.value)


def test_every_absent_field_is_named_at_once(tmp_path):
    """a cohort being scaffolded wants the whole list, not six round trips."""
    p = _project(tmp_path, h2_label=None, supervision=None, seed=None)
    with pytest.raises(DeclarationMissing) as e:
        declared(p)
    msg = str(e.value)
    assert all(f in msg for f in ("h2_label", "supervision", "seed"))
    assert "3 of 6" in msg or "declares 3 of 6" in msg


def test_an_empty_string_counts_as_absent(tmp_path):
    """`niches_dir: ""` is the scaffolded default, and it is not a path."""
    with pytest.raises(DeclarationMissing, match="niches_dir"):
        declared(_project(tmp_path, niches_dir=""))


def test_a_directory_with_no_project_json_says_how_to_make_one(tmp_path):
    with pytest.raises(DeclarationMissing, match="omicstra init"):
        declared(tmp_path / "nothing_here")


def test_the_reader_does_not_fall_back_to_projectconfigs_defaults(tmp_path):
    """the specific trap: ProjectConfig.model_validate fills these in."""
    from omicstra.contracts.project import ProjectConfig

    p = _project(tmp_path, h2_label=None)
    cfg = ProjectConfig.from_dir(p)
    assert cfg.h2_label == "mc_megacluster", "the default under test moved"
    with pytest.raises(DeclarationMissing, match="h2_label"):
        declared(p)


# -------------------------------------------------------- the compute arm ----
def test_the_graph_has_both_arms_and_the_pack_arm_is_the_default():
    from omicstra.graphs.evaluate import build_evaluate_graph

    nodes = set(build_evaluate_graph().get_graph().nodes)
    assert {"gates", "ask", "evaluate", "measure", "halt"} <= nodes


def test_entry_routes_on_compute_and_bypasses_the_pack_gates():
    """deliberate: `gates` refuses a cohort with no evidence pack, which is
    exactly the cohort the compute arm exists for. a freshly measured number has
    no pins to check."""
    from omicstra.graphs.evaluate import _route_entry

    assert _route_entry({"compute": True}) == "measure"
    assert _route_entry({}) == "gates"
    assert _route_entry({"compute": False}) == "gates"


def test_the_compute_arm_refuses_an_unnamed_run():
    from omicstra.graphs.evaluate import measure

    out = measure({"project_id": "x", "cohort": {"evaluate_params": {"a": {}}}})
    assert out["report"]["status"] == "refused"
    assert "none was named" in out["report"]["why"]


def test_the_compute_arm_refuses_a_cohort_that_declares_no_params():
    """every threshold is the cohort's, and a default here is a choice made on
    one tissue."""
    from omicstra.graphs.evaluate import measure

    out = measure({"project_id": "x", "run": "R1", "cohort": {}})
    assert out["report"]["status"] == "refused"
    assert "evaluate_params" in out["report"]["why"]


def test_a_project_the_resolver_refuses_becomes_a_report_not_a_traceback():
    """`project_root` refuses a project_id that does not match a bound cohort,
    and that refusal is not DeclarationMissing. uncaught it left `invoke`, so a
    caller saw a traceback where the server had an answer."""
    from omicstra.graphs.evaluate import measure

    out = measure({"project_id": "no-such-cohort", "run": "R1",
                   "cohort": {"evaluate_params": {"a": {}}}})
    assert out["report"]["status"] == "unavailable"
    assert out["report"]["stage"] == "arrays"
    assert out["report"]["why"]
    # and it is recorded, not only reported
    assert out["records"][-1]["status"] == "not_run"


# --------------------------------------------- the boundary that drops keys ---
def test_every_key_the_compute_arm_reads_is_declared_at_level_zero():
    """a key level 0 does not declare is dropped silently in both directions.
    `run` and `signals` were missing when the arm was first wired, and the
    wiring test caught it before any data moved."""
    from omicstra.graph import OmicstraState
    from omicstra.graphs.evaluate import EvaluateState

    parent = set(OmicstraState.__annotations__)
    child = set(EvaluateState.__annotations__)
    for k in ("compute", "run", "signals", "paired_view"):
        assert k in child, f"the evaluate state does not declare {k}"
        assert k in parent, f"level 0 does not declare {k} - it would be dropped"


# ------------------------------------------------------------ the MCP tool ---
def test_the_mcp_server_exposes_the_compute_arm():
    from omicstra.mcp import server

    assert hasattr(server, "evaluate_measure")
    assert hasattr(server, "evaluate_run"), "the pack arm must stay"


def test_the_tool_takes_the_run_as_a_required_argument():
    import inspect

    from omicstra.mcp.server import evaluate_measure

    sig = inspect.signature(evaluate_measure)
    assert sig.parameters["run"].default is inspect.Parameter.empty
    for optional in ("project_id", "paired_view", "signals", "run_id"):
        assert sig.parameters[optional].default is None


def test_the_two_tools_describe_different_questions():
    """one states what a curated pack may claim; the other measures a run."""
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src/omicstra/mcp/server.py"
    text = src.read_text()
    block = text[text.index("def evaluate_measure") - 2200:
                 text.index("def evaluate_measure")]
    assert "COMPUTE a verdict" in block
    assert "refuses by FIELD NAME" in block
    assert re.search(r"calls no model", block)
