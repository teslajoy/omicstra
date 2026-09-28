"""the shared tail of a compute node: choose an executor, hand over, record it.

not a gate. it is the part of two compute nodes that must not drift - the arms
differ in what they resolve and shard, not in how a run is handed over, so a fix
to one executor must reach both.

the record says which executor ran because they promise different things:
  in_process   files + checkpoint survive a restart; the thread does not, and
               runs_resume re-dispatches what is missing
  temporal     the history survives; a worker that never met the submitter
               finishes the run
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def dispatch(state: dict, shards: list, out: Path, body: Callable,
             *, fn_name: str, cohort_run: Callable[[list], Any],
             report_extra: dict | None = None) -> dict:
    """submit the pending shards and RETURN. never waits for the work.

    `fn_name` is the queue's contract between submitter and worker, so it is an
    argument rather than `body.__name__` - renaming a function would orphan every
    run in flight.

    `cohort_run` runs the shards here, loading the model once; `body` is the same
    work as a named activity for a worker elsewhere.

    returning at submission is the point of a run handle: the first version
    blocked for the whole encode and a ~150 s call failed client-side while the
    server finished it.
    """
    out.mkdir(parents=True, exist_ok=True)
    pending = [s for s in shards if not s.is_done()]
    t0 = time.time()

    from omicstra.dispatch import assert_writable
    from omicstra.settings import settings

    # before either backend: a refusal after the first write came too late.
    assert_writable(shards, state.get("project_id"))

    if settings.temporal_address:
        from omicstra.dispatch.temporal import register, submit_shards_durably

        # named here so a worker in THIS process (a test, a laptop worker) can
        # serve the queue. a remote worker registers the same name itself.
        register(fn_name, body)
        handle = submit_shards_durably(pending, address=settings.temporal_address,
                                       fn_name=fn_name)
        executor, note = "temporal", (
            "the scheduler holds the run. this process may exit - the history "
            "outlives it and any worker polling the queue finishes the shards.")
    else:
        # the ShardReport was dropped, so a failed shard read as a slow one:
        # runs_status counts files, and "absent" means "not yet" or "never". one
        # hest section failed on an unmappable gene axis and said `running` for
        # ten minutes. writing it beside the outputs is what lets a later process
        # tell the two apart.
        failures = out / "_failed.json"

        def _work():
            import json as _j

            try:
                rep = cohort_run(pending)
            except Exception as e:                        # noqa: BLE001
                # the dispatch died, not a shard - a different thing, recorded
                # differently.
                failures.write_text(_j.dumps(
                    {"dispatch_error": f"{type(e).__name__}: {e}",
                     "shards": [s.id for s in pending]}, indent=1))
                return
            bad = [{"sample": r.id, "error": r.error, "attempts": r.attempts}
                   for r in rep.results if r.status == "failed"]
            if bad:
                failures.write_text(_j.dumps({"failed": bad}, indent=1))
            elif failures.exists():
                failures.unlink()          # a resume that succeeded clears it

        threading.Thread(target=_work, daemon=True,
                         name=f"encode-{state.get('project_id') or 'cohort'}").start()
        handle, executor, note = {}, "in_process", (
            "a thread keeps the run. the files and the checkpoint survive a "
            "restart; the thread does not, and runs_resume re-dispatches the "
            "shards still missing. a shard that fails every attempt is written to "
            "_failed.json beside the outputs, because an absent file alone cannot "
            "tell a failure from a wait.")

    wall = round(time.time() - t0, 1)
    return {
        "shards": [{"sample": s.id, "output": str(s.output)} for s in shards],
        # the ledger's copy. the report answers THIS call; the record is what a
        # later process reads out of the checkpoint, and which executor ran a
        # shard is the first thing a resume needs - re-dispatching a thread and
        # re-attaching to a workflow are not the same recovery.
        "records": [*state.get("records", []),
                    {"step_id": "encode_dispatch", "kind": "dispatch", "status": "pass",
                     "actor": "system", "executor": executor,
                     "n_shards": len(shards), "n_pending": len(pending),
                     "out_dir": str(out), "encoder": state.get("encoder"),
                     **({"durable": handle} if handle else {})}],
        "report": {"status": "submitted", "executor": executor, "executor_note": note,
                   **({"durable": handle} if handle else {}),
                   "n_pending": len(pending), "n_already": len(shards) - len(pending),
                   "n_shards": len(shards),
                   "max_concurrent_tasks": (state.get("plan") or {}).get(
                       "max_concurrent_tasks"),
                   "encoder": state.get("encoder"),
                   # progress is NOT reported here. this returns at submission, so
                   # any count would be a guess - runs_status counts the output
                   # files, which are the truth and stay true across a restart.
                   "submit_seconds": wall, "out_dir": str(out),
                   "answers": state.get("answers") or {},
                   "plan": state.get("plan", {}),
                   **(report_extra or {})}}
