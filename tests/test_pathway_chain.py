"""the molecular assay's pathway chains, as chains.

`design/analysis_pipeline_steps.md` §5 documents ONE of them. `scripts/
niche_aucell.py` is a second, with a different normalisation, and §5 does not
mention it - so porting "§5" alone would have silently dropped a chain still in
use. the two share only the niche aggregate.
"""
from __future__ import annotations

import pytest

from omicstra.protocols import build_protocol
from omicstra.protocols.pathway import NICHE_MEMBERSHIP_STEPS

CTX = {"genes": ["MT-CO1", "MTRNR2L1", "ACTB", "EPCAM"],
       "niche_members": {"n1": list(range(7)), "n2": list(range(5))}}
DECLARED = {"mito_filter": {"pattern": "^(MT-|MTRNR)"},
            "niche_membership": {"k_neighbors": 6}}


def _run(params, **ctx):
    return build_protocol(NICHE_MEMBERSHIP_STEPS, "niche_membership").invoke(
        {**CTX, **ctx, "params": params})


# --- declared, not assumed --------------------------------------------------
def test_an_undeclared_threshold_does_not_run_and_does_not_guess():
    """a number chosen on one cohort looking at one tissue must not travel to
    the next silently. EVALUATION.md section 8 calls that the third kind.

    there are TWO ways to be undeclared and they have different causes, so they
    read differently: the cohort named the step and left the field out, or it
    never named the step at all. the framework owns the second.
    """
    # the step is named, the field is missing - the step says which field
    r = _run({"mito_filter": {}, "niche_membership": {"k_neighbors": 6}})
    assert r["mito_filter"]["status"] == "not_run"
    assert "pattern is not declared" in r["mito_filter"]["result"]
    assert "pattern" in r["mito_filter"]["decision"]

    # the step is not named at all - the framework says so, before the step runs
    r = _run({"niche_membership": {"k_neighbors": 6}})
    assert r["mito_filter"]["status"] == "not_run"
    assert "no params for this step" in r["mito_filter"]["result"]


def test_a_step_cannot_read_a_parameter_it_did_not_declare():
    """params_space is enforced by the framework, so whatever else a cohort's
    file carries never reaches the measure."""
    r = _run({**DECLARED, "niche_membership": {"k_neighbors": 6, "pattern": "junk"}})
    assert r["niche_membership"]["status"] == "pass"
    assert r["niche_membership"]["observed"]["k_neighbors"] == 6


def test_an_upstream_failure_makes_the_next_step_not_run_not_error():
    r = _run({})
    assert r["niche_membership"]["status"] == "not_run"
    assert "upstream" in r["niche_membership"]["result"]


# --- the chain's own rules --------------------------------------------------
def test_mitochondrial_genes_are_dropped_before_aggregation():
    """before, not after: a niche's library size is computed on what remains,
    and summing first would change what every later normalisation divides by.
    """
    r = _run(DECLARED)
    o = r["mito_filter"]["observed"]
    assert (o["n_in"], o["n_out"]) == (4, 2) and o["reason"] == "mitochondrial"


def test_the_niche_width_comes_from_the_declaration():
    """"7 spots" is one platform's number - k=6 plus a centre. MODALITY_TEMPLATE
    says a new cohort must not have to edit code to change it."""
    for k, width in ((6, 7), (4, 5), (8, 9)):
        r = _run({**DECLARED, "niche_membership": {"k_neighbors": k}})
        assert r["niche_membership"]["observed"]["spots_per_niche"] == width


def test_membership_does_not_aggregate():
    """the two chains fork AT aggregation: one sums the spots of a niche into a
    single profile, the other keeps them apart as several observations of one
    niche. collapsing here would hand both the same row and silently make the
    variable-gene chain impossible."""
    r = _run(DECLARED)
    o = r["niche_membership"]["observed"]
    assert o["n_in"] == o["n_out"], "membership collapses nothing"
    assert "spots_per_niche" in o, "it reports the width, it does not apply it"


def test_a_partial_niche_is_counted_here_not_dropped():
    """dropping is a qc decision on declared thresholds, and it belongs to the
    step that owns those thresholds."""
    r = _run(DECLARED)
    o = r["niche_membership"]["observed"]
    assert (o["n_full"], o["n_partial"]) == (1, 1)
    assert o["n_in"] == o["n_out"], "the aggregate drops nothing"


def test_every_step_reports_a_funnel():
    """an unexplained drop is an error - section 6's rule. a step that filters
    says how many went in, how many came out, and why."""
    r = _run(DECLARED)
    for sid, rec in r.items():
        o = rec["observed"]
        assert {"n_in", "n_out"} <= set(o), f"{sid} reports no funnel"
        if o["n_in"] != o["n_out"]:
            assert o.get("reason"), f"{sid} drops rows without a reason"


# --- it is a chain, not functions that happen to use the types -------------
def test_the_order_is_asserted_at_build_time():
    """a reordered DAG is a different and wrong method carrying a confident
    trace, so it must fail before any data is touched."""
    from omicstra.protocols import ProtocolOrderError

    with pytest.raises(ProtocolOrderError):
        build_protocol(list(reversed(NICHE_MEMBERSHIP_STEPS)), "backwards")


def test_each_step_is_its_own_traced_span():
    """one span per step, named by step id, is what makes a trace readable as
    the method rather than as one opaque call."""
    from langchain_core.callbacks.base import BaseCallbackHandler

    seen: list[str] = []

    class Spy(BaseCallbackHandler):
        def on_chain_start(self, serialized, inputs, **kw):
            if kw.get("name"):
                seen.append(kw["name"])

    build_protocol(NICHE_MEMBERSHIP_STEPS, "niche_membership").invoke(
        {**CTX, "params": DECLARED}, config={"callbacks": [Spy()]})
    assert {"mito_filter", "niche_membership"} <= set(seen), seen
