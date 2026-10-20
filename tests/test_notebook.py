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
    import nbformat

    nb = nbformat.from_dict(build(_bundle()))
    nbformat.validate(nb)
    assert nb.nbformat == 4


def test_the_generator_knows_no_cohort():
    import inspect

    from omicstra import notebook

    src = inspect.getsource(notebook)
    for name in ("tnbc", "virchow", "novae", "mc_megacluster"):
        assert name not in src.lower(), f"{name} reached the generator"
