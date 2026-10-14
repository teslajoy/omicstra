"""`omicstra worker` refuses legibly, and the refusals run without the extra.

the worker is the half of submit-only that is a process rather than a function,
so its failures are an operator's failures: a missing address, a missing backend,
a missing encoder module. each has to say which of the three it is at startup
rather than after the scheduler has accepted a run and spent three attempts on
an import.

the acceptance that the worker WORKS is in test_runs_kill_resume.py and needs a
scheduler. these need nothing, which is the point - they run in the clone where
the durable extra is absent, which is exactly where the first two matter.
"""
from __future__ import annotations

import sys

import pytest
from click.testing import CliRunner

from omicstra.cli import main


def _run(args, env):
    # `env=` on CliRunner only sets, so the address is passed as "" rather than
    # unset: settings reads it as falsy either way and the test does not depend
    # on the ambient environment.
    return CliRunner().invoke(main, ["worker", *args], env=env)


def test_no_address_is_a_refusal_that_names_the_variable():
    r = _run([], {"OMICSTRA_TEMPORAL_ADDRESS": ""})
    assert r.exit_code == 2
    assert "OMICSTRA_TEMPORAL_ADDRESS" in r.output
    # and says what happens WITHOUT one, because in-process is the normal mode
    # on a laptop rather than a degraded one.
    assert "in-process" in r.output


def test_no_durable_backend_names_the_extra_and_the_error(monkeypatch):
    """`None` in sys.modules is how a missing module is simulated with the extra
    installed. it raises ImportError rather than ModuleNotFoundError, which is
    why the handler catches the parent class - and why a temporalio that is
    present but broken also reaches a legible refusal instead of a traceback.
    """
    monkeypatch.setitem(sys.modules, "temporalio.client", None)
    r = _run(["--address", "localhost:7233"], {})
    assert r.exit_code == 2, r.output
    assert "omicstra[durable]" in r.output
    assert "temporalio" in r.output, "the actual import error must be reported"


def test_a_preload_that_does_not_import_fails_before_the_queue_is_joined():
    """the only one of the three that needs the extra present, because the
    backend check comes first on purpose - it is the precondition for the
    command existing at all, so a machine with neither hears about that one.
    """
    pytest.importorskip("temporalio", reason="the durable extra is not installed")
    r = _run(["--address", "localhost:7233", "--preload", "no_such_encoder_module"], {})
    assert r.exit_code == 2, r.output
    assert "no_such_encoder_module" in r.output
    assert "polling" not in r.output, "it must refuse before it starts serving"
