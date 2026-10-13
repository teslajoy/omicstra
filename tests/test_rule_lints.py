"""the project's own rules, as tests rather than as sentences.

every rule here was broken at least once in the work that produced it, and in
each case the rule already existed in prose. the lesson of that session was the
one this repo applies to its own data: make it a test or a refusal, not a
convention. a rule a reader has to remember is a rule that gets consulted after
the mistake.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GRAPHS = ROOT / "src" / "omicstra" / "graphs"

# modules in graphs/ that are NOT gates, each with the reason it is exempt.
# a new file here must be a gate or must appear below with a reason - the list is
# the declaration, so adding to it is a visible act rather than a silent one.
NOT_A_GATE = {
    "__init__.py": "the package marker",
    "submit.py": "the shared dispatch tail - chooses an executor, hands over, "
                 "records. no gate, and the docstring says so",
}


# --- the level rule --------------------------------------------------------
def test_every_graph_module_is_a_gate_or_declares_why_not():
    """LEVEL 1 is one module per gate. a module here that asks nobody is either
    mislabelled or belongs somewhere else."""
    for f in sorted(GRAPHS.glob("*.py")):
        if f.name in NOT_A_GATE:
            continue
        src = f.read_text()
        builds = "StateGraph(" in src
        asks = "interrupt(" in src or "from omicstra.graphs.encode import" in src
        assert builds or asks, (
            f"{f.name} neither builds a graph nor reaches an interrupt. if it is "
            f"shared machinery, add it to NOT_A_GATE with the reason")


@pytest.mark.parametrize("mod,fn", [
    ("encode", "build_encode_graph"), ("st", "build_st_graph"),
    ("align", "build_align_graph"), ("eda", "build_eda_graph"),
    ("route", "build_route_graph"), ("promote", "build_promote_graph"),
])
def test_every_graph_can_reach_an_interrupt(mod, fn):
    """the second half of the level rule: anything that can interrupt gets its
    own graph, so every graph must be able to.

    found by SOURCE, not by node name - the eda graph's interrupting node is
    called `escalate`, and a name-based check would have called it gateless.
    """
    import importlib

    g = getattr(importlib.import_module(f"omicstra.graphs.{mod}"), fn)()
    nodes = set(g.get_graph().nodes)
    m = importlib.import_module(f"omicstra.graphs.{mod}")
    reaches = [n for n in nodes
               if (f := getattr(m, n, None)) is not None
               and "interrupt" in (ast.unparse(ast.parse(
                   __import__("inspect").getsource(f))) if callable(f) else "")]
    # either this module interrupts, or it reuses a node from one that does
    shared = "from omicstra.graphs.encode import" in Path(m.__file__).read_text()
    assert reaches or shared, f"{mod} builds a graph that can never pause"


def test_no_graph_module_computes_a_metric():
    """LEVEL 1 routes and pauses. the maths lives in measures/ and is callable
    from a notebook; a statistic computed in a graph is one no notebook can
    check and no protocol declares."""
    banned = {"numpy", "scipy", "sklearn", "omicstra.measures"}
    for f in sorted(GRAPHS.glob("*.py")):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for n in names:
                assert not any(n == b or n.startswith(b + ".") for b in banned), \
                    f"{f.name} imports {n} - the maths belongs in measures/"


# --- the shared-helper rule ------------------------------------------------
def test_no_shared_gate_helper_hardcodes_a_step_id():
    """`_accept` and `gate_record` are shared by every arm, and both hardcoded
    `preflight_encode`. so the align stage recorded its own gates under the
    encode step and a ledger reader could not tell which protocol had asked.
    """
    for path, fns in ((GRAPHS / "encode.py", ("_accept", "halt")),
                      (ROOT / "src/omicstra/protocols/encode.py", ("gate_record",))):
        tree = ast.parse(path.read_text())
        for fn in (n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name in fns):
            args = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
            body = ast.unparse(fn)
            if "step_id" not in body:
                continue
            assert "step_id" in args, (
                f"{path.name}:{fn.name} writes a step_id it does not take as a "
                f"parameter - a shared helper cannot name one stage")


# --- the scope-doc rules ---------------------------------------------------
SCOPES = sorted((ROOT / "design").glob("v1_*_scope.md"))


def test_the_scope_docs_exist_and_the_live_one_is_named():
    assert SCOPES, "no scope doc - the plan has no home"
    assert (ROOT / "design" / "v1_3_scope.md").is_file(), "the live plan is missing"
    claude = (ROOT / "CLAUDE.md").read_text()
    assert "v1_3_scope.md" in claude, (
        "CLAUDE.md is the first thing a session reads and must point at the live "
        "plan. it listed the constraints and not where the work was")


def test_every_funded_deliverable_row_names_an_artifact_or_no_home():
    """the deliverable table's own rule. an artifact nobody can open is not a
    delivered result, and `built` with nothing cited is the shape this catches.
    """
    doc = (ROOT / "design" / "v1_3_scope.md").read_text()
    block = re.search(r"funded biological deliverables.*?\n\n(\|.*?)\n\n", doc, re.DOTALL)
    assert block, "the funded-deliverable table is gone"
    rows = [r for r in block.group(1).splitlines()
            if r.startswith("|") and not re.match(r"\|[\s:|-]+\|$", r)][1:]
    assert len(rows) == 3, f"three funded deliverables, found {len(rows)}"
    for r in rows:
        assert ".parquet" in r or ".json" in r or "no home" in r, (
            f"a deliverable row names neither an artifact nor 'no home': {r[:90]}")
        if "**built**" in r:
            assert "cited" in r, (
                f"a row marked built must say it is cited by the pack: {r[:90]}")
