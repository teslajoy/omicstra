"""the dispatcher: idempotency, retries, fallthrough, and kill-and-resume.

the acceptance is the kill test at the bottom. it starts a real subprocess,
SIGKILLs it mid-run - no cleanup, no atexit, no chance to write a ledger - and
asserts the restart finishes the job without redoing what landed. anything
gentler tests the happy path with extra steps.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from omicstra.dispatch import (
    Shard,
    ShardReport,
    atomic_write,
    run_shards,
    run_shards_locally,
)

ROOT = Path(__file__).resolve().parents[1]


def _shards(d: Path, n: int) -> list[Shard]:
    return [Shard(id=f"s{i}", output=d / f"s{i}.txt") for i in range(n)]


def _write(s: Shard, hb) -> None:
    hb("working")
    atomic_write(s.output, lambda p: p.write_text(s.id))


# --- idempotency ------------------------------------------------------------
def test_a_finished_shard_is_skipped_not_redone(tmp_path):
    sh = _shards(tmp_path, 3)
    assert len(run_shards_locally(sh, _write).done) == 3
    again = run_shards_locally(sh, _write)
    assert again.skipped == ["s0", "s1", "s2"] and not again.done


def test_done_is_decided_by_the_output_file_not_a_ledger(tmp_path):
    """the file is what the next stage reads, so it is the only honest evidence.

    delete one output and that shard - and only that shard - runs again.
    """
    sh = _shards(tmp_path, 3)
    run_shards_locally(sh, _write)
    (tmp_path / "s1.txt").unlink()
    again = run_shards_locally(sh, _write)
    assert again.done == ["s1"]
    assert again.skipped == ["s0", "s2"]


def test_an_empty_output_does_not_count_as_done(tmp_path):
    """a zero-byte file is what a killed write leaves behind on a naive writer."""
    sh = _shards(tmp_path, 1)
    sh[0].output.write_text("")
    assert run_shards_locally(sh, _write).done == ["s0"]


def test_atomic_write_leaves_nothing_behind_when_the_body_raises(tmp_path):
    """without this, a crash mid-write leaves a non-empty file and the shard is
    skipped forever with corrupt contents - the failure that makes people
    distrust resume and re-run everything by hand."""
    target = tmp_path / "out.txt"

    def explode(p: Path):
        p.write_text("half")
        raise RuntimeError("died mid-write")

    with pytest.raises(RuntimeError):
        atomic_write(target, explode)
    assert not target.exists()
    assert not list(tmp_path.glob("*.partial"))


# --- retries ----------------------------------------------------------------
def test_three_attempts_then_recorded_failed(tmp_path):
    """a failing shard is retried three times, then recorded and passed over.

    paired with a shard that succeeds on purpose: a run where EVERYTHING fails
    raises instead (see the total-failure guard below), so a single always-failing
    shard would be testing that rule rather than this one.
    """
    calls = {"n": 0}

    def one_bad(s, hb):
        if s.id == "s0":
            calls["n"] += 1
            raise ValueError("nope")
        _write(s, hb)

    rep = run_shards_locally(_shards(tmp_path, 2), one_bad)
    assert calls["n"] == 3, "the retry budget is three attempts"
    assert rep.failed == ["s0"] and rep.done == ["s1"]
    assert "ValueError" in rep.results[0].error, "the error must travel on the result"


def test_a_shard_that_succeeds_on_the_second_try_is_done(tmp_path):
    state = {"n": 0}

    def flaky(s, hb):
        state["n"] += 1
        if state["n"] < 2:
            raise OSError("transient")
        _write(s, hb)

    rep = run_shards_locally(_shards(tmp_path, 1), flaky)
    assert rep.done == ["s0"] and rep.results[0].attempts == 2


def test_one_bad_shard_does_not_cost_the_others(tmp_path):
    """280 samples and one unreadable image: the run continues and says which."""
    sh = _shards(tmp_path, 4)

    def fn(s, hb):
        if s.id == "s2":
            raise RuntimeError("unreadable image")
        _write(s, hb)

    rep = run_shards_locally(sh, fn)
    assert rep.done == ["s0", "s1", "s3"] and rep.failed == ["s2"]
    assert not rep.ok


# --- fallthrough ------------------------------------------------------------
def test_unset_address_runs_in_process_and_imports_no_temporal(tmp_path, monkeypatch):
    """the default path must not depend on the durable extra at all."""
    monkeypatch.delitem(sys.modules, "temporalio", raising=False)
    monkeypatch.setattr("omicstra.settings.settings.temporal_address", None)
    rep = run_shards(_shards(tmp_path, 2), _write)
    assert rep.backend == "local" and len(rep.done) == 2
    assert "temporalio" not in sys.modules, "the local path imported the durable extra"


def test_an_explicit_address_selects_the_durable_backend(tmp_path, monkeypatch):
    """without reaching a server: the selection is what is under test.

    patching inside `dispatch.temporal` imports it, so this needs the durable
    extra even though it never connects to anything. the selection logic itself
    is covered without the extra by the fallthrough test above, which is the one
    that matters for an install that has no temporal.
    """
    pytest.importorskip("temporalio", reason="patching the durable backend imports it")
    called = {}

    def fake(shards, fn, **kw):
        called.update(kw)
        return ShardReport(backend="temporal:fake")

    monkeypatch.setattr("omicstra.dispatch.temporal.run_shards_durably", fake)
    rep = run_shards(_shards(tmp_path, 2), _write, address="localhost:7233")
    assert rep.backend == "temporal:fake"
    assert called["max_attempts"] == 3, "the retry count must reach the backend"


def test_the_durable_backend_is_not_imported_at_package_scope():
    """same lint as the encoder registry: an extra behind a core import."""
    import inspect

    from omicstra import dispatch

    src = inspect.getsource(dispatch)
    head = src.split("def run_shards(")[0]
    assert "import temporalio" not in head
    assert "from temporalio" not in head


# --- the acceptance ---------------------------------------------------------
KILL_SCRIPT = r'''
import sys, time
from pathlib import Path
from omicstra.dispatch import Shard, atomic_write, run_shards_locally

d = Path(sys.argv[1]); n = int(sys.argv[2])
shards = [Shard(id=f"s{i}", output=d / f"s{i}.txt") for i in range(n)]

def fn(s, hb):
    hb(s.id)
    def write(p):
        time.sleep(0.35)                       # wide enough to be killed inside
        p.write_text(s.id)
    atomic_write(s.output, write)
    (d / "progress.log").open("a").write(s.id + "\n")

run_shards_locally(shards, fn)
(d / "FINISHED").write_text("ok")
'''


def test_kill_and_resume(tmp_path):
    """THE acceptance. SIGKILL mid-run, restart, finish without redoing work.

    SIGKILL rather than SIGTERM on purpose: no handler runs, no buffers flush,
    nothing gets a chance to record where it stopped. what survives is the
    filesystem, which is precisely the claim being tested.
    """
    script = tmp_path / "run.py"
    script.write_text(KILL_SCRIPT)
    n = 8

    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMICSTRA_TEMPORAL_ADDRESS="")
    p = subprocess.Popen([sys.executable, str(script), str(tmp_path), str(n)], env=env)

    # wait until it has genuinely started producing, then kill mid-shard
    deadline = time.time() + 30
    while time.time() < deadline:
        if len(list(tmp_path.glob("s*.txt"))) >= 2:
            break
        time.sleep(0.05)
    else:
        p.kill(); pytest.fail("the first process produced nothing to resume from")

    p.send_signal(signal.SIGKILL)
    p.wait(timeout=10)

    assert not (tmp_path / "FINISHED").exists(), "it was not killed mid-run"
    finished_before = sorted(q.name for q in tmp_path.glob("s*.txt"))
    assert 0 < len(finished_before) < n

    # progress.log is appended by BOTH runs, so the resume's work is the tail.
    # comparing the whole file says every shard was redone even when none was.
    log_before = (tmp_path / "progress.log").read_text().split()

    # a killed write must not leave something is_done() would believe
    assert not list(tmp_path.glob("*.partial")), "a partial file survived the kill"
    for f in tmp_path.glob("s*.txt"):
        assert f.stat().st_size > 0, f"{f.name} is empty and would be skipped as done"

    r = subprocess.run([sys.executable, str(script), str(tmp_path), str(n)],
                       env=env, capture_output=True, text=True, timeout=120, check=False)
    assert r.returncode == 0, r.stderr[-800:]
    assert (tmp_path / "FINISHED").exists()
    assert len(list(tmp_path.glob("s*.txt"))) == n, "the resume did not finish the job"

    # and it did not redo what had already landed
    redone = (tmp_path / "progress.log").read_text().split()[len(log_before):]
    assert not (set(finished_before) & {f"{x}.txt" for x in redone}), \
        f"resume redid shards that were already complete: {redone}"
    assert len(redone) == n - len(finished_before), (
        f"resume ran {len(redone)} shard(s); {n - len(finished_before)} were outstanding")


def test_the_temp_path_keeps_the_real_suffix(tmp_path):
    """numpy.save appends '.npy' to a path that lacks it.

    with the temp file named `x.npy.partial`, save() wrote `x.npy.partial.npy`,
    the rename target never existed, the shard failed all three attempts and left
    a stray file behind. found by running the real encode, not by reading it.
    """
    np = pytest.importorskip("numpy")

    seen = {}

    def save(p):
        seen["tmp"] = p.name          # a lambda with `or` here short-circuits on a
        np.save(p, np.arange(4))      # truthy name and never calls save at all

    atomic_write(tmp_path / "x.npy", save)
    assert seen["tmp"] == "x.partial.npy", "the temp name must end in the real suffix"
    assert [f.name for f in tmp_path.iterdir()] == ["x.npy"], "a stray file survived"
    assert np.load(tmp_path / "x.npy").tolist() == [0, 1, 2, 3]


def test_a_writer_that_does_not_produce_the_temp_file_is_caught(tmp_path):
    """silence is the failure mode worth naming: without this the rename raises
    FileNotFoundError three times and the reason is nowhere in the message."""
    with pytest.raises(FileNotFoundError, match="did not produce"):
        atomic_write(tmp_path / "z.bin", lambda p: None)


# --- total failure is a setup problem -------------------------------------
def test_every_shard_failing_raises_rather_than_reporting_quietly(tmp_path):
    """the np.save bug produced exactly this: three retries each, a quiet
    `failed` on every row, and a report that read like a bad cohort."""
    from omicstra.dispatch import AllShardsFailed

    def always_fails(s, hb):
        raise OSError("no such model")

    with pytest.raises(AllShardsFailed, match="setup problem"):
        run_shards_locally(_shards(tmp_path, 4), always_fails)


def test_a_partial_failure_still_returns_a_report(tmp_path):
    """one bad image must not raise - that is the case the retry budget is for."""
    sh = _shards(tmp_path, 3)

    def fn(s, hb):
        if s.id == "s1":
            raise RuntimeError("unreadable")
        _write(s, hb)

    rep = run_shards_locally(sh, fn)
    assert rep.failed == ["s1"] and len(rep.done) == 2


def test_an_all_skipped_run_does_not_raise(tmp_path):
    """re-running a finished cohort attempts nothing, and that is success."""
    sh = _shards(tmp_path, 2)
    run_shards_locally(sh, _write)
    rep = run_shards_locally(sh, _write)
    assert len(rep.skipped) == 2 and rep.ok
