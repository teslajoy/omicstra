"""external tracing is two environment variables away, and that is the problem.

`langsmith` ships transitively with langgraph and langgraph instruments itself
from the environment, so turning on a trace that sends node inputs and outputs
to a third party takes no code at all. node inputs are cohort content.

`data_contract.json#data_classification` has always declared that
`reads_this_field` includes "whether cohort content may cross to an external
model provider", and nothing read it. these are the tests for the first reader.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from omicstra.tracing import (
    TracingRefused,
    assert_may_trace,
    classification,
    enabled,
    may_leave_host,
    payloads_hidden,
    status,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tracing_on(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_not_a_real_key")
    yield


def test_absent_classification_is_restricted_not_permissive():
    """the contract's own guard: a wrong guess in the permissive direction is
    the single error this field exists to prevent."""
    assert classification(None) == "restricted"
    assert classification({}) == "restricted"
    assert may_leave_host({}) is False
    assert may_leave_host({"data_classification": "public"}) is True


def test_a_value_the_contract_does_not_declare_is_not_a_permission():
    assert may_leave_host({"data_classification": "probably_fine"}) is False


def test_may_leave_host_is_read_from_the_contract_not_from_a_literal():
    """`public` and `restricted` are the contract's vocabulary; a second copy
    here would be a second source for a governance decision."""
    c = json.loads((ROOT / "src" / "omicstra" / "configs" / "data_contract.json")
                   .read_text())["data_classification"]
    for name, v in c["values"].items():
        assert may_leave_host({"data_classification": name}) is bool(v["may_leave_host"])
    assert c["default"] == "restricted" or "restricted" in str(c["default"])


def test_nothing_is_refused_while_tracing_is_off():
    """the ledger is always on and always local. this guard is only about the
    external one."""
    assert enabled() is False
    assert_may_trace({}, "any-cohort")
    assert_may_trace({"data_classification": "restricted"}, "any-cohort")


def test_a_restricted_cohort_refuses_when_tracing_is_on(tracing_on):
    assert enabled() is True
    with pytest.raises(TracingRefused) as e:
        assert_may_trace({"data_classification": "restricted"}, "a-cohort")
    msg = str(e.value)
    assert "may not leave the host" in msg
    assert "LANGSMITH_TRACING" in msg, "the refusal must say how to proceed"
    assert "never inferred" in msg, "and that the classification is a person's"


def test_an_undeclared_cohort_refuses_too(tracing_on):
    with pytest.raises(TracingRefused):
        assert_may_trace({}, "undeclared")


def test_a_public_cohort_proceeds(tracing_on):
    assert_may_trace({"data_classification": "public"}, "published")
    st = status({"data_classification": "public"})
    assert st["external_tracing"] is True and st["may_leave_host"] is True


def test_the_sdk_cache_cannot_make_this_fail_open(monkeypatch):
    """langsmith CACHES its environment lookup, so a process that imported it
    before the variable was set reports tracing off while tracing is on. that
    is the fail-open this guard exists to prevent.
    """
    from langsmith import utils

    assert enabled() is False                      # primes the SDK's cache
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_not_a_real_key")
    assert utils.tracing_is_enabled() is False, "the SDK is still serving its cache"
    assert enabled() is True, "the guard must not inherit that answer"


def test_every_arm_checks_before_it_invokes(tracing_on, bound_synthetic_cohort,
                                            monkeypatch, tmp_path):
    """the trace captures the RUN, so a check after it would fire once the
    content had already been sent."""
    monkeypatch.setenv("OMICSTRA_CHECKPOINT", f"sqlite:{tmp_path / 'ck.sqlite'}")
    import importlib

    import omicstra.mcp.server as S
    S = importlib.reload(S)

    for call in (lambda: S.he_encode(project_id=None),
                 lambda: S.st_encode(project_id=None),
                 lambda: S.align_run(project_id=None),
                 lambda: S.evaluate_run(project_id=None)):
        with pytest.raises(TracingRefused):
            call()


def test_the_check_sits_before_the_invoke_in_source():
    """asserted against the source, because an ordering bug here sends the
    content and then complains about it."""
    import inspect

    import omicstra.mcp.server as S

    for fn in (S.he_encode, S.st_encode, S.align_run, S.evaluate_run):
        src = inspect.getsource(fn)
        assert "assert_may_trace" in src, fn.__name__
        assert src.index("assert_may_trace") < src.index("app.invoke"), fn.__name__


# --- the third state: trace the structure, mask the payloads ---------------
def test_a_restricted_cohort_may_be_traced_with_payloads_hidden(tracing_on, monkeypatch):
    """the SDK already provides this - `hide_inputs` / `hide_outputs` mask the
    payloads before anything is sent. so a restricted cohort can be traced for
    STRUCTURE - spans, timings, errors, which node stalled - without its content
    leaving. refusing that outright was the cruder answer.
    """
    r = {"data_classification": "restricted"}
    with pytest.raises(TracingRefused):
        assert_may_trace(r, "c")

    monkeypatch.setenv("LANGSMITH_HIDE_INPUTS", "true")
    monkeypatch.setenv("LANGSMITH_HIDE_OUTPUTS", "true")
    assert payloads_hidden() is True
    assert_may_trace(r, "c")                       # now permitted
    assert status(r)["payloads_hidden"] is True


def test_hiding_only_one_side_still_sends_the_other(tracing_on, monkeypatch):
    monkeypatch.setenv("LANGSMITH_HIDE_INPUTS", "true")
    assert payloads_hidden() is False
    with pytest.raises(TracingRefused):
        assert_may_trace({"data_classification": "restricted"}, "c")


def test_the_langchain_namespace_is_honoured_too(tracing_on, monkeypatch):
    """the SDK searches LANGSMITH_ then LANGCHAIN_. a second opinion about the
    variable names here could drift from the one the client acts on."""
    monkeypatch.setenv("LANGCHAIN_HIDE_INPUTS", "true")
    monkeypatch.setenv("LANGCHAIN_HIDE_OUTPUTS", "true")
    assert payloads_hidden() is True
    assert_may_trace({"data_classification": "restricted"}, "c")


def test_the_refusal_offers_the_masked_route_first():
    """the message is the documentation an operator actually reads."""
    import os
    os.environ["LANGSMITH_TRACING"] = "true"
    try:
        with pytest.raises(TracingRefused) as e:
            assert_may_trace({}, "c")
        msg = str(e.value)
        assert msg.index("HIDE_INPUTS") < msg.index("unset LANGSMITH_TRACING")
        assert "always on and always local" in msg
    finally:
        os.environ.pop("LANGSMITH_TRACING", None)
