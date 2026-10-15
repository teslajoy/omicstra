"""tracing, and the one question it has to answer first: may this leave the host.

two traces answer two questions and only one of them is free.

  the LEDGER        who decided what, on what authority, over which bytes.
                    committed, citable, and it never leaves the machine
  LANGSMITH         what a node received, how long it took, where it stalled.
                    richer, and it sends node inputs and outputs to a third party

langsmith ships transitively with langgraph and langgraph instruments itself from
the environment, so turning it on is two variables and no code. that is exactly
why this module exists: the cheapest thing to switch on is cohort content leaving
the host, and `data_contract.json#data_classification` has always declared that
`reads_this_field` includes "whether cohort content may cross to an external
model provider". it was declared and never read. this reads it.

there is a THIRD state, and the SDK already provides it: `hide_inputs` and
`hide_outputs` mask the payloads before anything is sent, settable per client or
through `LANGSMITH_HIDE_INPUTS` / `LANGSMITH_HIDE_OUTPUTS` (`LANGCHAIN_` is
honoured too - the SDK searches both namespaces). so a restricted cohort can be
traced for STRUCTURE - spans, timings, errors, which node stalled - without its
content leaving. that is the trace an operator actually wants, and refusing it
outright was the cruder answer.

what still leaves when payloads are hidden: run NAMES, timings, token counts and
error strings. step ids like `preflight_encode` are the package's vocabulary, not
a cohort's. a project id would travel in metadata, which `LANGSMITH_HIDE_METADATA`
masks in turn.

it REFUSES rather than silently disabling. an operator who set the variable meant
to set it, and a trace that quietly does not happen is as surprising as one that
quietly ships a cohort somewhere.
"""
from __future__ import annotations

from typing import Any


class TracingRefused(RuntimeError):
    """tracing is on and this cohort may not leave the host.

    not a gate. a gate is a question with a closed option set and someone to
    ask; "send this cohort's contents to a third party" is not a choice the
    server may offer, because the declaration already answered it.
    """


def enabled() -> bool:
    """is external tracing on, as the SDK itself decides it.

    asks langsmith rather than reimplementing its environment parsing - the
    variable names have changed once already, and a second opinion about whether
    tracing is on is exactly the thing that fails open.
    """
    try:
        from langsmith import utils
    except ModuleNotFoundError:          # no SDK, nothing can be sent
        return False
    try:
        # the SDK CACHES its environment lookup, so a process that imported it
        # before the variable was set reports tracing off while tracing is on.
        # that is the fail-open this guard exists to prevent, and it is the
        # reason this asks the SDK and then clears rather than trusting either
        # a cache or a reimplementation of the variable names.
        if hasattr(utils.get_env_var, "cache_clear"):
            utils.get_env_var.cache_clear()
        return bool(utils.tracing_is_enabled())
    except Exception:                    # noqa: BLE001 - unreadable means off
        return False


def payloads_hidden() -> bool:
    """are node inputs AND outputs masked before anything is sent.

    read through the SDK's own resolver, which searches `LANGSMITH_` then
    `LANGCHAIN_`, so a second opinion about the variable names cannot drift from
    the one the client will actually act on. both are required: hiding one and
    sending the other sends the content.
    """
    try:
        from langsmith import utils
    except ModuleNotFoundError:
        return False
    try:
        if hasattr(utils.get_env_var, "cache_clear"):
            utils.get_env_var.cache_clear()
        return all(utils.get_env_var(k) == "true"
                   for k in ("HIDE_INPUTS", "HIDE_OUTPUTS"))
    except Exception:                    # noqa: BLE001
        return False


def classification(cohort: dict[str, Any] | None) -> str:
    """the cohort's declared classification. absent is `restricted`.

    never inferred, which is the contract's own guard: no heuristic over paths,
    filenames or content may set it, because a wrong guess in the permissive
    direction is the single error the field exists to prevent.
    """
    return str((cohort or {}).get("data_classification") or "restricted")


def may_leave_host(cohort: dict[str, Any] | None) -> bool:
    """read from the contract, not from a literal here.

    `public` and `restricted` are the contract's vocabulary and it carries
    `may_leave_host` per value. a value the contract does not declare is treated
    as restricted - an unknown classification is not a permission.
    """
    import json

    from omicstra.settings import settings

    v = classification(cohort)
    contract = json.loads(
        (settings.resolve(settings.configs_dir) / "data_contract.json").read_text())
    values = (contract.get("data_classification") or {}).get("values") or {}
    return bool((values.get(v) or {}).get("may_leave_host", False))


def assert_may_trace(cohort: dict[str, Any] | None,
                     project_id: str | None = None) -> None:
    """refuse to run with external tracing on over a cohort that may not leave.

    called before a graph runs rather than after, because the trace captures the
    run: a check that fired at the end would fire after the content had already
    been sent.
    """
    if not enabled() or may_leave_host(cohort) or payloads_hidden():
        return
    raise TracingRefused(
        f"external tracing is enabled and {project_id or 'this cohort'} is "
        f"declared `{classification(cohort)}`, which the data contract says may "
        "not leave the host. node inputs and outputs are cohort content.\n"
        "three ways on, in the order worth trying:\n"
        "  LANGSMITH_HIDE_INPUTS=true LANGSMITH_HIDE_OUTPUTS=true   trace the "
        "STRUCTURE - spans, timings, errors - with the payloads masked before "
        "they are sent. this is usually what you wanted\n"
        "  unset LANGSMITH_TRACING                                   the "
        "committed ledger is unaffected; it is always on and always local\n"
        "  declare the cohort `public` in cohort.json, if it genuinely is - the "
        "classification is a person's statement and is never inferred"
    )


def status(cohort: dict[str, Any] | None = None) -> dict:
    """what a caller can be told without being refused."""
    return {"external_tracing": enabled(),
            "payloads_hidden": payloads_hidden(),
            "classification": classification(cohort),
            "may_leave_host": may_leave_host(cohort),
            "ledger": "always on, always local - the committed decision record"}
