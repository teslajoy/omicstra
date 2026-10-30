"""integration: pathway_assay through the graph, blocking, and notebook.

proves:
- a resolved artifact lets the chain continue to align
- a refused artifact blocks align from running
- the resolved stage produces provenance the notebook generator consumes
"""
from __future__ import annotations

import json

import pandas as pd
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from omicstra.graph import OmicstraState, build_omicstra_graph


# --- stub subgraphs --------------------------------------------------------
# each returns a minimal stage report so the chain walks to pathway_assay.


def _stub_eda(state):
    return {"report": {"status": "ready"}, "records": []}


def _stub_encode(state):
    return {"report": {"status": "resolved"}, "records": []}


def _stub_st(state):
    return {"report": {"status": "resolved"}, "records": []}


_align_invocations: list[dict] = []


def _stub_align(state):
    _align_invocations.append(dict(state))
    return {"report": {"status": "resolved"}, "records": []}


def _stub_evaluate(state):
    return {"report": {"status": "evaluated"}, "records": []}


def _build_graph():
    return build_omicstra_graph(
        checkpointer=InMemorySaver(),
        eda=_stub_eda,
        encode=_stub_encode,
        st=_stub_st,
        align=_stub_align,
        evaluate=_stub_evaluate,
    )


# --- fixtures ---------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_align_log():
    _align_invocations.clear()


@pytest.fixture
def pathway_project(tmp_path):
    """a minimal project with project.json, cohort.json, routing evidence,
    and a routing contract that declares pathway_transfer."""
    root = tmp_path / "proj"
    root.mkdir()

    (root / "project.json").write_text(json.dumps({
        "project_id": "test", "work_dir": "work",
        "platform": "synthetic",
    }))
    (root / "cohort.json").write_text(json.dumps({}))

    # routing evidence with a pathway_transfer entry so discover picks compute
    (root / "routing_evidence.json").write_text(json.dumps({"tasks": {}}))

    # routing contract declaring pathway_transfer
    import importlib.resources
    rc_path = importlib.resources.files("omicstra.configs") / "routing_contract.json"
    rc = json.loads(rc_path.read_text())
    (root / "routing_contract.json").write_text(json.dumps(rc))

    return root


@pytest.fixture
def pathway_project_with_scores(pathway_project):
    """adds a declared pathway_scores artifact on disk."""
    root = pathway_project
    scores_dir = root / "data" / "scores"
    scores_dir.mkdir(parents=True)
    scores_path = scores_dir / "aucell.parquet"
    pd.DataFrame({"pathway_A": [0.1, 0.2]}).to_parquet(scores_path)

    proj = json.loads((root / "project.json").read_text())
    proj["pathway_scores"] = {"path": "data/scores/aucell.parquet",
                               "columns": ["pathway_A"]}
    (root / "project.json").write_text(json.dumps(proj))
    return root


# --- acceptance tests -------------------------------------------------------

def test_resolved_artifact_lets_chain_reach_align(pathway_project_with_scores):
    """when include_pathway_scores is set and the artifact exists on disk,
    pathway_assay resolves and the chain continues to align."""
    from omicstra.settings import settings

    root = pathway_project_with_scores
    before = settings.project_dir
    settings.project_dir = root
    try:
        g = _build_graph()
        result = g.invoke(
            {"task_id": "pathway_transfer",
             "project_id": "test",
             "include_pathway_scores": True},
            config={"configurable": {"thread_id": "t1"}},
        )
    finally:
        settings.project_dir = before

    # pathway_assay resolved (the node's own stage + the keeper's copy)
    pa_stages = [s for s in result.get("stages", [])
                 if s.get("stage") == "pathway_assay"]
    assert len(pa_stages) >= 1
    node_stage = pa_stages[0]
    assert node_stage["status"] == "resolved"
    assert "pathway_scores" in node_stage["report"]["reused"]

    # align was reached (stub was called)
    assert len(_align_invocations) == 1, "align was not reached after resolved artifact"


def test_missing_unproducible_artifact_blocks_align(pathway_project):
    """when include_pathway_cosines is set and the artifact is absent with no
    producer, pathway_assay refuses and the chain stops before align."""
    from omicstra.settings import settings

    root = pathway_project
    before = settings.project_dir
    settings.project_dir = root
    try:
        g = _build_graph()
        result = g.invoke(
            {"task_id": "pathway_transfer",
             "project_id": "test",
             "include_pathway_cosines": True},
            config={"configurable": {"thread_id": "t2"}},
        )
    finally:
        settings.project_dir = before

    # pathway_assay refused
    pa_stages = [s for s in result.get("stages", [])
                 if s.get("stage") == "pathway_assay"]
    assert len(pa_stages) >= 1
    assert pa_stages[0]["status"] == "refused"

    # align was NOT reached - the refusal blocked the chain
    assert len(_align_invocations) == 0, \
        "align ran despite pathway_assay refusing - the chain did not block"


def test_no_pathway_flags_lets_chain_pass_through(pathway_project):
    """when neither pathway flag is set, pathway_assay is nothing_to_run
    and the chain continues to align."""
    from omicstra.settings import settings

    root = pathway_project
    before = settings.project_dir
    settings.project_dir = root
    try:
        g = _build_graph()
        result = g.invoke(
            {"task_id": "cross_modal_retrieval",
             "project_id": "test"},
            config={"configurable": {"thread_id": "t3"}},
        )
    finally:
        settings.project_dir = before

    pa_stages = [s for s in result.get("stages", [])
                 if s.get("stage") == "pathway_assay"]
    assert len(pa_stages) >= 1
    assert pa_stages[0]["status"] == "nothing_to_run"

    assert len(_align_invocations) == 1, "align was not reached for a non-pathway task"


def test_resolved_stage_generates_notebook_provenance(pathway_project_with_scores):
    """the resolved pathway_assay stage carries enough provenance for the
    notebook generator to produce a notebook with the reused artifact."""
    from omicstra.notebook import build
    from omicstra.settings import settings

    root = pathway_project_with_scores
    before = settings.project_dir
    settings.project_dir = root
    try:
        g = _build_graph()
        result = g.invoke(
            {"task_id": "pathway_transfer",
             "project_id": "test",
             "include_pathway_scores": True},
            config={"configurable": {"thread_id": "t4"}},
        )

        pa_stages = [s for s in result.get("stages", [])
                     if s.get("stage") == "pathway_assay"]
        assert len(pa_stages) >= 1

        # the stage report carries provenance: which artifacts were reused
        report = pa_stages[0]["report"]
        assert "pathway_scores" in report["reused"]
        reused_path = report["reused"]["pathway_scores"]

        # build a notebook from the stage info
        nb = build({
            "unit": "assay_pathway_aucell",
            "project_id": "test",
            "project_dir": str(root),
            "run": None,
            "run_id": "test-provenance",
            "params": {},
            "records": [],
            "sample": None,
        })
    finally:
        settings.project_dir = before

    assert nb["metadata"]["omicstra"]["unit"] == "assay_pathway_aucell"
    # the notebook structure exists (header + setup + footer cells)
    assert len(nb["cells"]) >= 3
    # the setup cell references the pathway context loader
    setup = nb["cells"][1]["source"]
    assert "from omicstra.adapters.pathway import arrays" in setup
