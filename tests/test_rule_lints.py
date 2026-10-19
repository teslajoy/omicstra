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
    ("align", "build_align_graph"), ("evaluate", "build_evaluate_graph"),
    ("eda", "build_eda_graph"), ("route", "build_route_graph"),
    ("promote", "build_promote_graph"),
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


# --- the route is discoverable ---------------------------------------------
GRAPH_DOC = ROOT / "design" / "graph.md"


def test_the_graph_doc_exists_and_names_every_graph_module():
    """a session's first question is WHERE DOES THIS GO, and the answer was the
    tree plus three scope docs plus CLAUDE.md's directory list.

    `design/graph.md` is one page with the route on it. a new graph that forgets
    its row fails here rather than being discoverable only by reading the tree -
    which is the same rule as every other lint in this file: make it a test or a
    refusal, not a convention.
    """
    assert GRAPH_DOC.is_file(), "design/graph.md is the route; it is missing"
    doc = GRAPH_DOC.read_text()
    for f in sorted(GRAPHS.glob("*.py")):
        if f.name == "__init__.py":
            continue
        assert f"graphs/{f.name}" in doc, (
            f"{f.name} is a graph module and design/graph.md does not name it")
    assert "graph.py" in doc and "LEVEL 0" in doc


def test_the_graph_doc_names_the_level_zero_chain_in_the_order_the_code_wires_it():
    """a diagram that drifts from the wiring is worse than none: it is read as
    authoritative and is wrong."""
    from omicstra.graph import COMPUTE_CHAIN, STOP_AFTER

    doc = GRAPH_DOC.read_text()
    at = [doc.index(f"graphs/{n}.py") for n in COMPUTE_CHAIN]
    assert at == sorted(at), (
        f"design/graph.md lists the compute chain out of order; the code wires "
        f"{list(COMPUTE_CHAIN)}")
    assert "STOP_AFTER" in doc and "COMPUTE_CHAIN" in doc
    assert "submitted" in doc, "the stop that matters must be stated"
    assert all(s in doc for s in ("submitted",)) and STOP_AFTER[0] == "submitted"


# --- the queue's name rule -------------------------------------------------
def test_every_submitted_activity_name_has_a_body_a_worker_can_hold():
    """the submitter names a body; the worker holds it. different processes.

    `fn_name=` is a string crossing a scheduler, so nothing in the language
    checks it. an arm whose name is absent from `shard_bodies()` submits runs
    that sit on the queue until a worker picks one up, fails with a KeyError on
    a name it never registered, and retries twice more.
    """
    from omicstra.dispatch.bodies import shard_bodies

    served = set(shard_bodies())
    submitted = {m for f in GRAPHS.glob("*.py")
                 for m in re.findall(r'fn_name=["\']([^"\']+)["\']', f.read_text())}
    assert submitted, "no arm submits anything - the regex has drifted"
    assert submitted <= served, (
        f"{sorted(submitted - served)} is submitted by a graph and served by no "
        f"worker. add it to dispatch/bodies.py:shard_bodies")


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


# --- no cohort in the package ----------------------------------------------
SRC = ROOT / "src" / "omicstra"
COHORT_TOKENS = ("tnbc-92", "tnbc92", "hest-breast", "B1_v3", "R4_v3")


def test_no_cohort_name_or_run_id_appears_in_package_CODE():
    """a cohort name in a docstring is context; in code it is a default.

    `selftest` - the command the readme tells a new cohort owner to run FIRST -
    defaulted --project-id to the seed cohort, named `B1_v3` as the
    contraindicated method, and printed `raw_he 0.164`, that cohort's
    measurement, as a literal. on a second cohort its checks failed because the
    cohort differed rather than because a rule broke, and one PASSED on a cohort
    with no evidence at all.
    """
    offenders = []
    for f in sorted(SRC.rglob("*.py")):
        tree = ast.parse(f.read_text())
        docs = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                d = ast.get_docstring(node, clean=False)
                if d:
                    docs.add(d)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if node.value in docs:
                continue                      # a docstring is context, not a default
            for t in COHORT_TOKENS:
                if t in node.value:
                    offenders.append(f"{f.relative_to(ROOT)}:{node.lineno} {t!r} "
                                     f"in {node.value[:48]!r}")
    assert not offenders, (
        "a cohort name or run id reached package code:\n  " + "\n  ".join(offenders))


# --- no cohort MEASUREMENT reaches package code ----------------------------
PROTOCOLS = ROOT / "src" / "omicstra" / "protocols"

# a step may hold structural constants and nothing else. 0 and 1 are identities,
# 2 is a pair, an epsilon guards a division. every other number in an analysis
# step is a THRESHOLD, and a threshold is a cohort's to declare.
STRUCTURAL = {0, 1, -1, 2, 0.0, 1.0, 2.0}

# chains written to this rule. eda.py and inventory.py predate it and carry
# defaults of their own - recorded in design/v1_3_scope.md rather than changed
# here, because rewriting a passing gate to satisfy a new lint is how a rule
# becomes a refactor.
CHAIN_MODULES = ("pathway.py", "join.py", "fusion.py")


def _is_epsilon(v) -> bool:
    return isinstance(v, float) and 0 < abs(v) <= 1e-4


def _numeric_thresholds(src: str) -> list[int]:
    """numeric literals that are THRESHOLDS, not slices or formatting.

    a slice bound and a `round()` precision are structure: they shape a
    display, not a decision. a number compared against, defaulted to, or passed
    as a parameter is a decision.
    """
    tree = ast.parse(src)
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):          # x[:20], rows[:12]
            for sub in ast.walk(node.slice):
                skip.add(id(sub))
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "round":
            for a in node.args[1:]:                  # round(x, 4) - precision
                skip.add(id(a))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or id(node) in skip:
            continue
        v = node.value
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if v in STRUCTURAL or _is_epsilon(v):
            continue
        out.append(node.lineno)
    return out


def test_an_analysis_step_holds_no_threshold_of_its_own():
    """the cohort-name lint catches `tnbc-92` in a string. this catches the
    other half: 0.25, 1000, 200 - numbers measured on one tissue, which reach
    code as defaults rather than as names, and then travel to every cohort.

    a step carries order and authority and gets its numbers from `_declared`,
    which returns None when absent so the step can refuse.
    """
    offenders = []
    for f in sorted(PROTOCOLS.glob("*.py")):
        src = f.read_text()
        # named as a chain AND defining one. `join.py` holds funnel helpers
        # today and becomes subject to this the moment JOIN_STEPS lands.
        if f.name not in CHAIN_MODULES or "list[Step]" not in src:
            continue
        offenders += [f"{f.relative_to(ROOT)}:{ln}" for ln in _numeric_thresholds(src)]
    assert not offenders, (
        "a numeric threshold reached an analysis step. it belongs in the "
        "cohort's declaration, read through `_declared`:\n  "
        + "\n  ".join(offenders))


def test_the_lint_would_catch_a_threshold_if_one_appeared():
    """a lint that passes on everything is not a lint."""
    assert _numeric_thresholds("x = 0.25") == [1]
    assert _numeric_thresholds("f(min_umi=1000)") == [1]
    assert _numeric_thresholds("if n < 200: pass") == [1]
    # and does not fire on structure
    assert _numeric_thresholds("rows[:12]") == []
    assert _numeric_thresholds("round(frac, 4)") == []
    assert _numeric_thresholds("x / (y + 1e-9)") == []


# --- a new cohort gets answers, not tracebacks ------------------------------
def test_the_read_path_answers_a_cohort_whose_gate_has_never_run(tmp_path):
    """an unrun gate is an ANSWER. raising sends a client a ToolError carrying a
    good message through the wrong channel - a model driving this gets a failure
    it cannot act on instead of a verdict it can.

    `describe_data_structure` was worse: a DECLARATIONS tool that hard-failed on
    an EDA artifact it reads for two optional extras.
    """
    import json

    from omicstra.settings import settings

    (tmp_path / "project.json").write_text(json.dumps(
        {"project_id": tmp_path.name, "platform": "none", "k_neighbors": 6,
         "seed": 42, "split": "patient", "encoders": [], "supervision": ""}))
    before = settings.project_dir
    settings.project_dir = tmp_path
    try:
        from omicstra.mcp import server as S

        d = S.describe_data_structure()
        assert d["project_id"] and "platform" in d, "declarations must still answer"

        g = S.check_eda_gate()
        assert g["verdict"] == "not_run"
        assert g["why"] and g["next"], "a refusal must say what is missing and what next"
    finally:
        settings.project_dir = before
