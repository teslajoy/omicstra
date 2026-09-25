"""THE step-2 acceptance: kill the process, resume the run through the surface.

`server.json` claims the server is stateless because a run handle is an ordinary
parameter rather than a session. that claim is only true if the run's state is in
sqlite AND a second process can finish what the first started - with an in-memory
saver it passes in one process and fails on any restart.

what each executor promises is DIFFERENT, so both are tested:

  in_process   the OUTPUT FILES and the checkpoint survive; the thread does not.
               recovery is `runs_resume` re-dispatching what is missing.
  temporal     the HISTORY survives. recovery is a worker that never met the
               submitting process finishing the run with nobody re-submitting.

the cohort here is synthetic and the encoder is a registered stand-in, because
what is under test is the handle, not Virchow2. the real encoder is pinned by the
slice-diff against the cache.
"""
from __future__ import annotations

import functools
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

# one builder, shared with test_encode_graph.py. a second would drift, and these
# two files assert different things about the same shape.
from conftest import SECTIONS as SIDS
from conftest import build_synthetic_cohort

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r'''
import json, os, sys, time
from pathlib import Path

root = Path(sys.argv[1]); action = sys.argv[2]

# the stand-in encoder, registered before anything resolves a name. slow on
# purpose: each shard has to be wide enough to be killed inside.
import numpy as np
from omicstra.models.encoders import EncoderSpec, register

class Slow:
    """wide enough to be killed INSIDE a shard. four sections at ~2.5 s each
    leaves a window where some have landed and some have not, which is the state
    the whole test is about."""
    def embed_images(self, tiles, batch_size=32):
        time.sleep(2.5)
        return np.zeros((len(tiles), 4), dtype="float32")

register(EncoderSpec(name="slowfake", dim=4, role="primary", modality="he",
                     unit="tile", trained_on="nothing - a test stand-in",
                     extras=()), lambda: Slow())

from omicstra.mcp import server as S

if action == "start":
    rid = sys.argv[3]
    r = S.he_encode(compute=True, encoder="slowfake", platform="synthetic", run_id=rid)
    # the gates fire as an INTERRUPT, which is the designed path - a stand-in
    # encoder has no established fit and this cohort declares no fallback. the
    # answers are what the re-dispatch after the kill then has to find already
    # recorded, so the gate flow is part of the acceptance rather than noise.
    if r.get("status") == "awaiting_answer":
        print("GATE " + json.dumps([g["id"] for g in r["gate"]["gates"]]), flush=True)
        r = S.runs_resume(rid, answers={"encoder_compatibility":
                                        "proceed_anyway_recorded_as_dissent",
                                        "capacity": "continue"})
    print("START " + json.dumps(r), flush=True)
    if (r.get("report") or {}).get("executor") == "temporal":
        raise SystemExit(0)          # nothing to hold open; that is the point
    # in-process: the submit returned and the thread is still working. hold the
    # process open so the parent can kill it mid-shard.
    time.sleep(120)

elif action == "status":
    print("STATUS " + json.dumps(S.runs_status(sys.argv[3])), flush=True)

elif action == "resume":
    print("RESUME " + json.dumps(S.runs_resume(sys.argv[3])), flush=True)
    # the re-dispatch also returns at submission
    deadline = time.time() + 90
    while time.time() < deadline:
        st = S.runs_status(sys.argv[3])
        if st["status"] == "complete":
            break
        time.sleep(0.2)
    print("FINAL " + json.dumps(S.runs_status(sys.argv[3])), flush=True)

elif action == "record":
    print("RECORD " + json.dumps(S.runs_record(sys.argv[3])["records"]), flush=True)
'''


def _cohort(tmp_path: Path) -> Path:
    return build_synthetic_cohort(tmp_path)


def _env(root: Path, ck: Path, temporal: str = "") -> dict:
    return dict(os.environ, PYTHONPATH=str(ROOT / "src"),
                OMICSTRA_PROJECT_DIR=str(root),
                OMICSTRA_CHECKPOINT=f"sqlite:{ck}",
                OMICSTRA_TEMPORAL_ADDRESS=temporal)


def _run(root, ck, *args, temporal="", timeout=180):
    script = root / "srv.py"
    script.write_text(SCRIPT)
    return subprocess.run([sys.executable, str(script), str(root), *args],
                          env=_env(root, ck, temporal), capture_output=True,
                          text=True, timeout=timeout, check=False)


def _said(out: str, tag: str):
    line = next(ln for ln in out.splitlines() if ln.startswith(tag + " "))
    return json.loads(line[len(tag) + 1:])


def _outputs(root: Path) -> list[Path]:
    return sorted((root / "data" / "embeddings" / "slowfake_niche").glob("*.npy"))


# --- in_process --------------------------------------------------------------
def test_kill_the_server_and_a_second_process_finishes_the_run(tmp_path):
    """the in-process promise: the FILES and the checkpoint survive, the thread
    does not. the run is finished by a process that did not start it, addressed
    only by its run_id.
    """
    pytest.importorskip("PIL", reason="the synthetic slide needs pillow")
    root = _cohort(tmp_path)
    ck = tmp_path / "ck.sqlite"
    rid = "kill-inproc"

    script = root / "srv.py"
    script.write_text(SCRIPT)
    p = subprocess.Popen([sys.executable, str(script), str(root), "start", rid],
                         env=_env(root, ck), stdout=subprocess.PIPE, text=True)

    # wait until at least one shard has LANDED, so there is something to resume
    # from and something still missing
    deadline = time.time() + 90
    while time.time() < deadline:
        if 0 < len(_outputs(root)) < len(SIDS):
            break
        time.sleep(0.1)
    else:
        p.kill()
        pytest.fail(f"nothing to resume from: {_outputs(root)}")

    p.send_signal(signal.SIGKILL)
    p.wait(timeout=10)
    landed = {f.stem for f in _outputs(root)}
    assert 0 < len(landed) < len(SIDS), f"not killed mid-run: {landed}"

    # a SECOND process, addressed only by the run_id
    st = _said(_run(root, ck, "status", rid).stdout, "STATUS")
    assert st["executor"] == "in_process"
    assert st["status"] == "running"
    assert set(st["missing"]) == set(SIDS) - landed, "progress must be counted from the files"

    r = _run(root, ck, "resume", rid, timeout=240)
    assert r.returncode == 0, r.stderr[-1500:]
    res = _said(r.stdout, "RESUME")
    assert res["resumed_by"] == "re-dispatch"
    assert set(res["re_dispatched"]) == set(SIDS) - landed

    final = _said(r.stdout, "FINAL")
    assert final["status"] == "complete", final
    assert final["missing"] == []
    assert {f.stem for f in _outputs(root)} == set(SIDS)

    # and the LEDGER says which executor ran it and how it was recovered. read
    # from a FOURTH process that wrote none of it, because a record printed by
    # the call that made it proves only that the call returned a dict - the
    # question is whether it reached the store. `resumed_by` appears in no JSON
    # anywhere; the checkpoint is where it lives, and this is what says so.
    rec = _said(_run(root, ck, "record", rid).stdout, "RECORD")
    kinds = [(d.get("step_id"), d.get("executor"), d.get("resumed_by"))
             for d in rec if d.get("kind") == "dispatch"]
    assert ("encode_dispatch", "in_process", None) in kinds
    assert ("encode_resume", "in_process", "re-dispatch") in kinds
    # the gate answers too: the run was decided by a person, two processes ago
    human = [d for d in rec if d.get("actor") == "human"]
    assert human and human[-1]["chosen"]["encoder_compatibility"] == \
        "proceed_anyway_recorded_as_dissent"


def test_the_resume_does_not_redo_what_landed(tmp_path):
    """idempotency on the output file, through the surface rather than the
    dispatcher. a resume that recomputed a finished shard would be correct and
    still wrong - on 280 sections it is the difference between minutes and hours.
    """
    pytest.importorskip("PIL")
    root = _cohort(tmp_path)
    ck = tmp_path / "ck.sqlite"
    rid = "kill-idem"

    script = root / "srv.py"
    script.write_text(SCRIPT)
    p = subprocess.Popen([sys.executable, str(script), str(root), "start", rid],
                         env=_env(root, ck), stdout=subprocess.PIPE, text=True)
    deadline = time.time() + 90
    while time.time() < deadline and not (0 < len(_outputs(root)) < len(SIDS)):
        time.sleep(0.1)
    p.send_signal(signal.SIGKILL)
    p.wait(timeout=10)

    kept = {f.stem: f.stat().st_mtime_ns for f in _outputs(root)}
    assert kept, "nothing landed before the kill"
    time.sleep(0.05)

    r = _run(root, ck, "resume", rid, timeout=240)
    assert r.returncode == 0, r.stderr[-1500:]
    assert _said(r.stdout, "FINAL")["status"] == "complete"

    after = {f.stem: f.stat().st_mtime_ns for f in _outputs(root)}
    for sid, mt in kept.items():
        assert after[sid] == mt, f"{sid} was rewritten by the resume"


def test_a_run_id_from_another_checkpointer_is_not_silently_served(tmp_path):
    """the handle is only meaningful against the store that holds it. answering
    for an unknown id would make every claim about resume unfalsifiable.
    """
    pytest.importorskip("PIL")
    root = _cohort(tmp_path)
    st = _said(_run(root, tmp_path / "empty.sqlite", "status", "never-existed").stdout,
               "STATUS")
    assert st["status"] == "unknown" and "checkpointer" in st["why"]


# --- temporal ----------------------------------------------------------------
# a DIFFERENT promise, so a different test. in_process recovers because the files
# survive; temporal recovers because the history does. here the submitting process
# does not die mid-shard - it never ran a shard at all. it exits at submission and
# a worker it never met does the whole run.
WORKER = r'''
import sys, time
from pathlib import Path
import numpy as np

from omicstra.models.encoders import EncoderSpec, register as reg_encoder

class Slow:
    def embed_images(self, tiles, batch_size=32):
        time.sleep(2.5)
        return np.zeros((len(tiles), 4), dtype="float32")

reg_encoder(EncoderSpec(name="slowfake", dim=4, role="primary", modality="he",
                        unit="tile", trained_on="nothing - a test stand-in",
                        extras=()), lambda: Slow())

import asyncio
from temporalio.client import Client
from omicstra.dispatch.temporal import build_worker, register
from omicstra.protocols.encode import he_shard_body
from omicstra.settings import settings

# the body is named, not passed. this process never spoke to the submitter.
register("encode_he", he_shard_body)
addr, out_dir, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])

async def main():
    client = await Client.connect(addr, namespace=settings.temporal_namespace)
    async with build_worker(client, settings.temporal_task_queue):
        deadline = time.time() + 120
        while time.time() < deadline:
            if len(list(out_dir.glob("*.npy"))) >= n:
                return
            await asyncio.sleep(0.2)

asyncio.run(main())
print("WORKED", flush=True)
'''


def test_the_durable_run_is_finished_by_a_worker_that_never_met_the_submitter(tmp_path):
    """the temporal promise, through the surface.

    the submitting process exits at submission having encoded nothing. no
    re-dispatch, no resume, nobody re-submitting: the history is the handle, and
    a worker that was not running when the work was accepted finishes it.
    """
    pytest.importorskip("temporalio", reason="the durable extra is not installed")
    pytest.importorskip("PIL")
    import asyncio

    from temporalio.testing import WorkflowEnvironment

    root = _cohort(tmp_path)
    ck = tmp_path / "ck.sqlite"
    rid = "kill-temporal"
    out_dir = root / "data" / "embeddings" / "slowfake_niche"

    async def main():
        async with await WorkflowEnvironment.start_local() as env:
            addr = env.client.service_client.config.target_host

            # 1. submit, and EXIT. nothing is polling the queue yet.
            r = await asyncio.to_thread(_run, root, ck, "start", rid,
                                        temporal=addr, timeout=180)
            assert r.returncode == 0, r.stderr[-1500:]
            start = _said(r.stdout, "START")
            assert start["report"]["executor"] == "temporal", start["report"]
            assert start["report"]["durable"]["workflow_id"].startswith("omicstra-encode-")
            assert not list(out_dir.glob("*.npy")), "no worker existed; nothing may have run"

            # 2. a worker in its own process, built from the package's helper and
            #    holding the body by NAME.
            w = (root / "worker.py")
            w.write_text(WORKER)
            wr = await asyncio.to_thread(functools.partial(
                subprocess.run, [sys.executable, str(w), addr, str(out_dir), str(len(SIDS))],
                env=_env(root, ck, addr), capture_output=True,
                text=True, timeout=240, check=False))

            # 3. the handle still resolves, from a THIRD process
            st = await asyncio.to_thread(_run, root, ck, "status", rid, temporal=addr)
            return start, wr, _said(st.stdout, "STATUS")

    _, wr, st = asyncio.run(main())
    assert "WORKED" in wr.stdout, wr.stderr[-1500:]
    assert {f.stem for f in out_dir.glob("*.npy")} == set(SIDS)
    assert st["executor"] == "temporal"
    assert st["status"] == "complete" and st["missing"] == []
