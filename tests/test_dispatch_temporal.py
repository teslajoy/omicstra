"""the DURABLE backend, against a real temporal server.

this is the half the local dispatcher cannot stand in for. the local path resumes
because the OUTPUT FILE survives the process; temporal resumes because the
WORKFLOW HISTORY survives the worker, and those are different claims. a run whose
worker is killed must continue on a new worker without the caller re-submitting
anything - that is the whole reason for the dependency.

the dev server is started in-process by the SDK, so this needs no brew install
and no running cluster. it skips where temporalio is absent, which is CI.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("temporalio", reason="the durable extra is not installed")

from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment

from omicstra.dispatch import Shard, atomic_write
from omicstra.dispatch.temporal import (
    EncodeWorkflow,
    ShardSpec,
    build_worker,
    register,
)
from omicstra.settings import settings

TASK_QUEUE = "omicstra-test"


def _write(s: Shard, hb) -> None:
    hb(s.id)
    atomic_write(s.output, lambda p: p.write_text(s.id))


def _specs(d: Path, n: int) -> list[ShardSpec]:
    return [ShardSpec.of(Shard(id=f"s{i}", output=d / f"s{i}.txt")) for i in range(n)]


async def _worker(client: Client):
    """built through the package's own helper, so the test exercises the same
    construction a deployment would - sandbox configuration included."""
    return build_worker(client, TASK_QUEUE)


async def _worker_on(client: Client):
    """a worker on the queue the SUBMIT path uses. `submit_shards_durably` reads
    the queue from settings rather than taking it as an argument, so a worker
    built on this test's own constant would poll a queue nothing was sent to."""
    return build_worker(client, settings.temporal_task_queue)


def test_the_durable_backend_runs_every_shard(tmp_path):
    async def main():
        register("t", _write)
        async with (await WorkflowEnvironment.start_local() as env,
                    await _worker(env.client)):
                rows = await env.client.execute_workflow(
                    EncodeWorkflow.run, args=[_specs(tmp_path, 4), "t", 3, 5],
                    id="wf-all", task_queue=TASK_QUEUE)
        return rows

    rows = asyncio.run(main())
    assert [r[1] for r in rows] == ["done"] * 4
    assert sorted(f.name for f in tmp_path.glob("s*.txt")) == [f"s{i}.txt" for i in range(4)]


def test_the_durable_backend_skips_what_is_already_written(tmp_path):
    """idempotency lives in the ACTIVITY, because it reads the filesystem and
    would break workflow replay if it sat in the workflow."""
    for i in range(4):
        (tmp_path / f"s{i}.txt").write_text("pre-existing")

    async def main():
        register("t", _write)
        async with (await WorkflowEnvironment.start_local() as env,
                    await _worker(env.client)):
                return await env.client.execute_workflow(
                    EncodeWorkflow.run, args=[_specs(tmp_path, 4), "t", 3, 5],
                    id="wf-skip", task_queue=TASK_QUEUE)

    rows = asyncio.run(main())
    assert [r[1] for r in rows] == ["skipped"] * 4
    assert (tmp_path / "s0.txt").read_text() == "pre-existing", "a skip must not rewrite"


def test_a_failing_activity_is_retried_three_times(tmp_path):
    attempts = {"n": 0}

    def flaky(s, hb):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")
        _write(s, hb)

    async def main():
        register("t", flaky)
        async with (await WorkflowEnvironment.start_local() as env,
                    await _worker(env.client)):
                return await env.client.execute_workflow(
                    EncodeWorkflow.run, args=[_specs(tmp_path, 1), "t", 3, 5],
                    id="wf-retry", task_queue=TASK_QUEUE)

    rows = asyncio.run(main())
    assert rows[0][1] == "done"
    assert attempts["n"] == 3, "the workflow's RetryPolicy must allow three attempts"


def test_kill_the_worker_and_the_run_continues_on_a_new_one(tmp_path):
    """THE acceptance for 3.6, and the claim the local dispatcher cannot make.

    the workflow is submitted once. the worker executing it is destroyed
    mid-run. a NEW worker picks the run up from history and finishes it, with no
    resubmission by the caller and no shard done twice.
    """
    done_by = {}

    def slow(s, hb):
        hb(s.id)
        import time
        time.sleep(0.4)
        atomic_write(s.output, lambda p: p.write_text(s.id))
        done_by.setdefault(s.id, current_worker["n"])

    current_worker = {"n": 1}

    async def main():
        register("t", slow)
        async with await WorkflowEnvironment.start_local() as env:
            client = env.client
            handle = await client.start_workflow(
                EncodeWorkflow.run, args=[_specs(tmp_path, 6), "t", 3, 5],
                id="wf-kill", task_queue=TASK_QUEUE)

            # worker 1 runs for a while, then is destroyed mid-run
            w1 = await _worker(client)
            t = asyncio.create_task(w1.run())
            await asyncio.sleep(1.2)
            await w1.shutdown()
            t.cancel()
            partial = sorted(f.name for f in tmp_path.glob("s*.txt"))
            assert 0 < len(partial) < 6, f"worker 1 did not stop mid-run: {partial}"

            # nothing is running now; the workflow is alive in history only
            await asyncio.sleep(0.5)
            current_worker["n"] = 2

            # worker 2 picks it up. the caller does not resubmit.
            async with await _worker(client):
                rows = await handle.result()
            return rows, partial

    rows, partial = asyncio.run(main())
    assert [r[1] for r in rows] == ["done"] * 6, rows
    assert len(list(tmp_path.glob("s*.txt"))) == 6

    # the shards worker 1 finished were not re-executed by worker 2
    redone = [sid for sid, w in done_by.items() if w == 2 and f"{sid}.txt" in partial]
    assert not redone, f"worker 2 re-ran shards worker 1 had finished: {redone}"
    assert any(w == 1 for w in done_by.values()), "worker 1 finished nothing to resume from"
    assert any(w == 2 for w in done_by.values()), "worker 2 did no work"


# --- submit-only: the handle IS the run --------------------------------------
def test_the_workflow_id_is_stable_across_processes():
    """the id is how a later process FINDS a run, so it cannot be `hash()`.

    python salts the hash of a str per process. an id built that way differs
    between the submitter and anything that comes looking - which defeats the one
    thing a derived id is for, and would let a resume start a second run over the
    same output files.
    """
    import subprocess
    import sys

    prog = ("from pathlib import Path;"
            "from omicstra.dispatch import Shard;"
            "from omicstra.dispatch.temporal import workflow_id;"
            "print(workflow_id([Shard(id='a', output=Path('a')),"
            "                   Shard(id='b', output=Path('b'))], 'encode_he'))")
    ids = {subprocess.run([sys.executable, "-c", prog], capture_output=True,
                          text=True, check=True,
                          env={"PYTHONHASHSEED": seed, "PATH": ""}).stdout.strip()
           for seed in ("0", "1", "12345")}
    assert len(ids) == 1, f"the id moved with the hash seed: {ids}"
    assert ids.pop().startswith("omicstra-encode-")


def test_a_different_activity_body_is_a_different_run():
    from omicstra.dispatch.temporal import workflow_id

    shards = [Shard(id="a", output=Path("a"))]
    assert workflow_id(shards, "encode_he") != workflow_id(shards, "encode_st")


def test_submit_returns_before_the_work_does(tmp_path):
    """the whole point of the durable path: the submitting process is not the one
    holding the run. nothing is polling the queue yet, so a submit that waited
    would never return here."""
    from omicstra.dispatch.temporal import describe_durable_run, submit_shards_durably

    shards = [Shard(id=f"s{i}", output=tmp_path / f"s{i}.txt") for i in range(4)]

    async def main():
        async with await WorkflowEnvironment.start_local() as env:
            addr = env.client.service_client.config.target_host
            # no worker. submit anyway - the scheduler holds it.
            h = await asyncio.to_thread(submit_shards_durably, shards, address=addr)
            assert not list(tmp_path.glob("s*.txt")), "nothing may have run yet"
            d = await asyncio.to_thread(describe_durable_run, h["workflow_id"],
                                        address=addr)
            # the run exists with no worker in sight, which is the durable claim.
            assert d["status"] == "RUNNING"

            register("encode_he", _write)
            async with await _worker_on(env.client):
                await env.client.get_workflow_handle(h["workflow_id"]).result()
            return h, d

    h, d = asyncio.run(main())
    assert h["workflow_id"] == d["workflow_id"] and h["run_id"] == d["run_id"]
    assert h["n_shards"] == 4 and "results" not in h
    assert sorted(f.name for f in tmp_path.glob("s*.txt")) == [f"s{i}.txt" for i in range(4)]


def test_re_submitting_the_same_shards_attaches_rather_than_starting_a_second_run(tmp_path):
    """a resume re-submits the same shard set on purpose. two runs over the same
    outputs is the failure that id is derived to prevent."""
    from omicstra.dispatch.temporal import submit_shards_durably

    shards = [Shard(id="s0", output=tmp_path / "s0.txt")]

    async def main():
        async with await WorkflowEnvironment.start_local() as env:
            addr = env.client.service_client.config.target_host
            a = await asyncio.to_thread(submit_shards_durably, shards, address=addr)
            b = await asyncio.to_thread(submit_shards_durably, shards, address=addr)
            return a, b

    a, b = asyncio.run(main())
    assert a["workflow_id"] == b["workflow_id"]
    assert a["run_id"] == b["run_id"], "a second run_id means a second run"
