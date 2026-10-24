"""a run as rerunnable cells, keyed on the ANALYSIS UNIT.

tier 1: the generator emits a notebook, so what is tested is its structure and
its refusals. that the generated notebook EXECUTES and reproduces is tier 2,
where the cohort's arrays exist.
"""
from __future__ import annotations

import importlib

import pytest

from omicstra.notebook import PLACES, UNITS, _numbers, build, filenames


def _rec(step, status="pass", **kw):
    return {"step_id": step, "kind": "diagnostic", "status": status, **kw}


def _bundle(**over):
    b = {"unit": "evaluate", "project_id": "c", "project_dir": "/tmp/c",
         "run": "R1", "run_id": "eval-a3f2",
         "params": {"label_granularity": {"labels": ["x"]}},
         "records": [_rec("label_granularity", observed={"n_in": 3, "n_out": 3})]}
    return {**b, **over}


# ---------------------------------------------------- the unit is the key ----
def test_the_file_is_named_for_the_unit_not_the_run():
    """a run id is an execution artifact - nobody opens `eval-a3f2.ipynb`. this
    cohort's own committed notebooks are already named for topics."""
    assert filenames("evaluate") == ("evaluate.ipynb", "evaluate.trace.md")
    assert filenames("eda") == ("eda.ipynb", "eda.trace.md")


def test_the_run_that_produced_the_numbers_is_provenance_inside():
    nb = build(_bundle())
    assert nb["metadata"]["omicstra"]["produced_by_run"] == "eval-a3f2"
    assert nb["metadata"]["omicstra"]["unit"] == "evaluate"
    assert "produced by run `eval-a3f2`" in nb["cells"][0]["source"]


def test_the_title_says_what_the_unit_answers():
    assert "what each hypothesis may claim" in build(_bundle())["cells"][0]["source"]
    assert "is this cohort admissible" in build(
        _bundle(unit="eda", params={}, records=[]))["cells"][0]["source"]


@pytest.mark.parametrize("unit", sorted(UNITS))
def test_every_unit_resolves_a_real_step_chain(unit):
    mod, steps_name, _, asks = UNITS[unit]
    steps = getattr(importlib.import_module(mod), steps_name)
    assert steps and all(hasattr(s, "id") for s in steps)
    assert asks and asks[0].islower(), "the unit says what it answers, in prose"


def test_an_unknown_unit_is_refused_with_the_list_of_real_ones():
    with pytest.raises(ValueError, match="unknown unit"):
        build(_bundle(unit="fusion"))


def test_the_two_pathway_chains_are_separate_units():
    """they fork at AGGREGATION - one collapses a niche's spots before doing
    anything, the other never collapses them. one notebook holding both would
    imply a single method where there are two."""
    assert "assay_pathway_genesets" in UNITS
    assert "assay_pathway_aucell" in UNITS
    assert UNITS["assay_pathway_genesets"][1] != UNITS["assay_pathway_aucell"][1]


# ------------------------------------------------------------ the cells ------
def test_one_markdown_and_one_code_cell_per_step():
    recs = [_rec("a"), _rec("b"), _rec("c")]
    nb = build(_bundle(records=recs))
    kinds = [c["cell_type"] for c in nb["cells"]]
    # 2 header + 2 per step + 3 footer
    assert kinds.count("code") == 2 + len(recs)


def test_a_step_cell_carries_its_method_and_criterion():
    nb = build(_bundle(records=[_rec("a", method="KMeans ARI",
                                     criterion="G5 runs", result="0.2339")]))
    md = [c["source"] for c in nb["cells"] if c["cell_type"] == "markdown"]
    assert any("KMeans ARI" in m and "G5 runs" in m and "0.2339" in m for m in md)


def test_the_cells_call_the_steps_not_the_measures():
    """a notebook calling measures/ directly would need a step-to-measure map
    maintained beside the chain, and a drifted map reproduces the wrong thing
    convincingly. the step is already the reproducible unit."""
    src = "\n".join(c["source"] for c in build(_bundle())["cells"])
    assert "STEP[sid].fn(" in src
    assert "from omicstra.measures" not in src


def test_the_dependency_map_is_resolved_at_generation_time():
    """the first version worked it out inside the emitted cell and the
    comprehension was unreadable. a notebook a reader cannot follow is not a
    record."""
    src = "\n".join(c["source"] for c in build(_bundle())["cells"])
    assert "WAITS_ON" in src
    assert "produces" not in src, "dependency resolution leaked into a cell"


# --------------------------------------------------------- the precision ----
def test_numbers_are_compared_at_the_reported_precision():
    """one digit beyond the .3f the steps report at. float32 kernels differ by
    ~1e-6 between backends, so asserting past that compares rounding."""
    assert PLACES == 4
    assert _numbers({"ari": 0.23392871181}) == {"ari": 0.2339}
    src = "\n".join(c["source"] for c in build(_bundle())["cells"])
    assert "round(float(v), 4)" in src
    assert "4 decimals" in build(_bundle())["cells"][0]["source"]


def test_only_scalar_numbers_are_compared():
    """a nested table is checked by the step that built it; walking into it
    would assert on a shape the record does not promise."""
    got = _numbers({"ari": 0.5, "stages": [{"n_in": 1}], "name": "x",
                    "ok": True, "n": 3})
    assert got == {"ari": 0.5, "n": 3.0}


# ------------------------------------------------------------- failure ------
def test_a_failed_cell_does_not_halt_and_its_dependents_say_not_run():
    """the chain's own semantics: an upstream failure is not_run, and the chain
    runs to the end so the record is complete."""
    src = "\n".join(c["source"] for c in build(_bundle())["cells"])
    assert "status[sid] = 'not_run'" in src
    assert "status[sid] = 'fail' if off else rec.status" in src
    assert "raise" not in src and "assert " not in src, "a cell must not halt"


def test_a_step_the_cohort_declared_no_params_for_is_not_run():
    src = "\n".join(c["source"] for c in build(_bundle())["cells"])
    assert "sid not in PARAMS" in src
    assert "declared no params" in src


def test_the_footer_states_what_a_clean_rerun_proves():
    nb = build(_bundle())
    tail = "\n".join(c["source"] for c in nb["cells"][-3:])
    assert "reproduction proof" in tail
    assert "not a rounding artifact" in tail


# ------------------------------------------------------- the contract -------
def test_the_notebook_is_valid_nbformat():
    nbformat = pytest.importorskip("nbformat")

    nb = nbformat.from_dict(build(_bundle()))
    nbformat.validate(nb)
    assert nb.nbformat == 4


def test_the_generator_knows_no_cohort():
    import inspect

    from omicstra import notebook

    src = inspect.getsource(notebook)
    for name in ("tnbc", "virchow", "novae", "mc_megacluster"):
        assert name not in src.lower(), f"{name} reached the generator"


# ------------------------------------------- where a result is written ------
@pytest.fixture
def cohort(tmp_path, monkeypatch):
    """a cohort with a declared `work_dir`, so nothing lands in the seed's ledger."""
    import json

    from omicstra.settings import settings

    root = tmp_path / "acohort"
    (root / "data").mkdir(parents=True)
    (root / "project.json").write_text(json.dumps(
        {"project_id": "acohort", "platform": "synthetic", "work_dir": "work"}))
    monkeypatch.setattr(settings, "project_dir", root)
    return root


def test_a_generated_notebook_is_a_result_under_the_cohorts_own_work_root(cohort):
    """NOT the repo's `notebooks/` - that directory is a person's committed
    decision record. a generated pair is a result, so it lands beside the
    ledger and the figures, under the root `omicstra init` created."""
    from omicstra.notebook import results_dir

    assert results_dir(None) == (cohort / "work" / "results").resolve()


def test_the_pair_is_written_side_by_side_and_both_paths_are_returned(cohort):
    from omicstra.notebook import write

    nb, tr = write(_bundle())
    assert nb.parent == tr.parent == (cohort / "work" / "results").resolve()
    assert (nb.name, tr.name) == filenames("evaluate")
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.reads(nb.read_text(), as_version=4))
    assert "what each hypothesis may claim" in tr.read_text()


def test_the_trace_points_at_the_notebook_beside_it(cohort):
    """the pair is one record read two ways, so each has to name the other."""
    from omicstra.notebook import write

    _, tr = write(_bundle())
    assert "evaluate.ipynb" in tr.read_text()


def test_a_cohort_that_declares_its_work_root_read_only_is_refused(cohort):
    """the work root is diffed against the published grid. writing a result
    into it would make the next port pass against its own output."""
    import json

    from omicstra.dispatch import WouldOverwriteOracle
    from omicstra.notebook import results_dir

    d = json.loads((cohort / "project.json").read_text())
    d["read_only_inputs"] = {"policy": "refuse_writes", "paths": ["work"]}
    (cohort / "project.json").write_text(json.dumps(d))
    with pytest.raises(WouldOverwriteOracle, match="read-only"):
        results_dir(None)


# ------------------------------------------------- the context loader -------
GENERABLE = sorted(u for u, v in UNITS.items() if v.context)


@pytest.mark.parametrize("unit", GENERABLE)
def test_every_module_a_setup_cell_imports_can_actually_be_imported(unit):
    """the setup cell used to read `from omicstra.adapters.None import arrays`
    for five of the seven units: an adapter name of None formatted into a module
    path. a notebook whose first cell cannot import is not a record."""
    setup = build(_bundle(unit=unit, params={}, records=[]))["cells"][1]["source"]
    for line in setup.splitlines():
        if line.startswith("from ") and " import " in line:
            mod, names = line[5:].split(" import ")
            m = importlib.import_module(mod)
            for n in names.split(", "):
                assert hasattr(m, n), f"{mod} has no {n}"


@pytest.mark.parametrize("unit", GENERABLE)
def test_every_setup_cell_binds_ctx(unit):
    setup = build(_bundle(unit=unit, params={}, records=[]))["cells"][1]["source"]
    assert "\nctx = " in setup


def test_a_cohort_backed_unit_is_handed_the_cohort_not_an_adapter(unit="eda"):
    """eda and inventory read their own declared inputs - there is no `arrays`
    for them, and inventing one would be a second reader of the same files."""
    setup = build(_bundle(unit=unit, params={}, records=[]))["cells"][1]["source"]
    assert "arrays" not in setup
    assert "'project_dir': PROJECT" in setup


@pytest.mark.parametrize("unit", sorted(u for u, v in UNITS.items() if not v.context))
def test_a_unit_with_no_declared_context_loader_is_refused_by_name(unit):
    """declared, not guessed. the three pathway units read `niches`,
    `gene_symbols` and `pathway_sets`, and nothing in `adapters/` builds those
    from a cohort - so say that, rather than emit a cell that cannot run."""
    with pytest.raises(ValueError, match="declares no context loader"):
        build(_bundle(unit=unit, params={}, records=[]))


# ------------------------------------------- the ledger is the source -------
def test_the_bundle_is_the_last_record_each_of_the_units_steps_wrote(
        bound_synthetic_cohort, tmp_path):
    """a `Record` carries no run id - the run identity is the graph's thread id,
    which lives in the checkpointer - so a unit's current state is the last
    record each of its steps wrote, and the CHAIN is what filters the cohort's
    ledger down to one unit."""
    from omicstra.contracts.routing import ledger
    from omicstra.notebook import from_ledger

    ledger(None).append_new([
        {"step_id": "route::cross_modal_retrieval", "kind": "gate",
         "actor": "deterministic", "status": "pass"},
        {"step_id": "label_granularity", "kind": "diagnostic", "status": "pass",
         "params": {"labels": ["a"]}, "observed": {"n_in": 9, "n_out": 9}},
        {"step_id": "label_granularity", "kind": "diagnostic", "status": "pass",
         "params": {"labels": ["b"]}, "observed": {"n_in": 7, "n_out": 7}},
    ])
    b = from_ledger("evaluate", None, run="R4")
    assert [r["step_id"] for r in b["records"]] == ["label_granularity"], \
        "the route record belongs to another unit"
    assert b["params"]["label_granularity"] == {"labels": ["b"]}, \
        "append-only, so the last record is the current state"
    assert b["run"] == "R4" and b["run_id"] is None


def test_the_steps_appear_in_the_chains_order_not_the_ledgers(
        bound_synthetic_cohort):
    """a step that was re-run appears once, where the protocol puts it."""
    from omicstra.contracts.routing import ledger
    from omicstra.notebook import from_ledger
    from omicstra.protocols.evaluate import EVALUATE_STEPS

    ledger(None).append_new([
        {"step_id": "hypothesis_verdict", "kind": "gate", "status": "pass"},
        {"step_id": "label_granularity", "kind": "diagnostic", "status": "pass"},
    ])
    got = [r["step_id"] for r in from_ledger("evaluate", None)["records"]]
    order = [s.id for s in EVALUATE_STEPS]
    assert got == sorted(got, key=order.index)
    assert got[0] == "label_granularity"


def test_the_pair_lands_under_the_cohorts_work_root_from_the_ledger_alone(
        bound_synthetic_cohort):
    """the whole path, with nothing hand-assembled: records in the cohort's
    ledger, a notebook and a trace under its work root."""
    nbformat = pytest.importorskip("nbformat")

    from omicstra.contracts.routing import ledger
    from omicstra.notebook import from_ledger, results_dir, write

    ledger(None).append_new([
        {"step_id": "label_granularity", "kind": "diagnostic", "status": "pass",
         "method": "distinct labels per unit", "criterion": "> 1 label",
         "result": "9 units, 3 labels", "params": {"labels": ["a"]},
         "observed": {"n_in": 9, "n_out": 9}}])
    nb, tr = write(from_ledger("evaluate", None, run="R4"))

    assert nb.parent == results_dir(None)
    doc = nbformat.reads(nb.read_text(), as_version=4)
    nbformat.validate(doc)
    src = "\n".join(c["source"] for c in doc.cells)
    assert "label_granularity" in src and "distinct labels per unit" in src
    assert "fitted run `R4`" in src
    assert "label_granularity" in tr.read_text()


# ------------------------------------------------------- the CLI command ----
def _cli(*args):
    from click.testing import CliRunner

    from omicstra.cli import main

    return CliRunner().invoke(main, ["notebook", *args])


def test_the_command_with_no_unit_lists_what_is_generable_and_what_is_not():
    r = _cli()
    assert r.exit_code == 0, r.output
    for u in GENERABLE:
        assert u in r.output
    assert "not generable" in r.output


def test_the_command_refuses_an_empty_unit_and_names_the_ledger(
        bound_synthetic_cohort):
    r = _cli("evaluate")
    assert r.exit_code != 0
    assert "nothing to re-derive" in r.output and "decisions.jsonl" in r.output


def test_the_command_writes_the_pair_and_prints_both_paths(
        bound_synthetic_cohort):
    from omicstra.contracts.routing import ledger

    ledger(None).append_new([
        {"step_id": "label_granularity", "kind": "diagnostic", "status": "pass",
         "params": {"labels": ["a"]}, "observed": {"n_in": 9, "n_out": 9}}])
    r = _cli("evaluate", "--run", "R4")
    assert r.exit_code == 0, r.output
    assert "evaluate.ipynb" in r.output and "evaluate.trace.md" in r.output
    assert "(1 steps)" in r.output


# -------------------------------------------------------- the MCP tool ------
def test_the_mcp_server_exposes_the_unit_notebook_tool():
    from omicstra.mcp import server

    assert hasattr(server, "unit_notebook")


def test_the_tool_takes_the_unit_as_its_only_required_argument():
    import inspect

    from omicstra.mcp.server import unit_notebook

    sig = inspect.signature(unit_notebook)
    assert sig.parameters["unit"].default is inspect.Parameter.empty
    for optional in ("project_id", "run", "sample", "run_id"):
        assert sig.parameters[optional].default is None


def test_the_tool_refuses_rather_than_raising_and_lists_the_real_units():
    """a client gets a report, not a traceback - the same rule the compute arm
    follows for a project the resolver refuses."""
    from omicstra.mcp.server import unit_notebook

    out = unit_notebook(unit="fusion")
    assert out["status"] == "refused" and out["units"] == GENERABLE


def test_the_tool_returns_both_paths_and_the_steps_it_re_derives(
        bound_synthetic_cohort):
    from omicstra.contracts.routing import ledger
    from omicstra.mcp.server import unit_notebook

    ledger(None).append_new([
        {"step_id": "label_granularity", "kind": "diagnostic", "status": "pass",
         "params": {"labels": ["a"]}, "observed": {"n_in": 9, "n_out": 9}}])
    out = unit_notebook(unit="evaluate", run="R4")
    assert out["status"] == "written"
    assert out["steps"] == ["label_granularity"]
    assert out["notebook"].endswith("evaluate.ipynb")
    assert out["trace"].endswith("evaluate.trace.md")
    assert out["asks"] == UNITS["evaluate"].asks


def test_the_tool_refuses_an_empty_unit_rather_than_writing_an_empty_notebook(
        bound_synthetic_cohort):
    """a header, no step cells, and a footer claiming a clean rerun is a file
    that looks like a result."""
    from omicstra.mcp.server import unit_notebook

    out = unit_notebook(unit="evaluate")
    assert out["status"] == "refused"
    assert "nothing to re-derive" in out["why"]
    assert "evaluate.ipynb" not in str(out)


# -------------------------------------------- what the footer may claim -----
def test_no_failure_is_not_a_reproduction():
    """the footer printed `reproduced` over six not_run steps - a notebook that
    re-derived nothing and said the run reproduced. that is the one wrong answer
    the cell exists to rule out."""
    src = "\n".join(c["source"] for c in build(_bundle())["cells"])
    assert "NOTHING RE-DERIVED" in src
    assert "elif not ran:" in src
    assert "'reproduced' if not bad" not in src


def test_the_footer_counts_what_actually_ran():
    import contextlib
    import io

    class _S:
        def __init__(self, i):
            self.id = i

    code = [c["source"] for c in build(_bundle())["cells"]
            if "NOTHING RE-DERIVED" in c["source"]][0]
    step = {c: _S(c) for c in "abcdef"}

    def say(status):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(code, {"STEP": step, "status": status})
        return buf.getvalue()

    assert "NOTHING RE-DERIVED" in say({"a": "not_run", "b": "not_run"})
    assert "reproduced 2 of 6 step(s)" in say(
        {"a": "pass", "b": "not_applicable", "c": "not_run"})
    assert "DIFFERS" in say({"a": "fail", "b": "pass"})


# ----------------------------------- what a step downstream has to see ------
def test_the_cells_thread_the_records_the_way_the_chain_does():
    """`hypothesis_verdict` judges from what the steps above RECORDED rather
    than recomputing. with no `_records` it saw an empty dict and returned
    not_run for every hypothesis - a wrong answer the cell reported as `pass`,
    because the verdict lives in nested blocks the scalar comparison skips."""
    nb = build(_bundle())
    setup = nb["cells"][1]["source"]
    assert "ctx['_records'] = {}" in setup
    src = "\n".join(c["source"] for c in nb["cells"])
    assert "ctx['_records'][sid] = rec.model_dump()" in src
    assert src.count("ctx['_records'][sid] = {'step_id': sid, 'status': 'not_run'}") == 2


def test_a_step_with_no_scalar_is_still_compared_on_its_result_line():
    b = _bundle(records=[_rec("hypothesis_verdict",
                              result="H1: not_run, H2: diagnostic, H3: not_run",
                              observed={"verdicts": {"H2": {"verdict": "x"}}})],
                params={"hypothesis_verdict": {}})
    nb = build(b)
    setup = nb["cells"][1]["source"]
    assert "H2: diagnostic" in setup, "the recorded result must travel"
    src = "\n".join(c["source"] for c in nb["cells"])
    assert "off['result'] = (rec.result, said)" in src
