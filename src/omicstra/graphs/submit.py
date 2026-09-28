"""the shared tail of a compute node: choose an executor, hand over, record it.

NOT a gate, and not a level of its own. `graphs/` holds one module per gate and
this is neither - it is the part of two compute nodes that must not drift, because
both arms make the same two promises about recovery and a fix applied to one of
them would silently miss the other. the H&E node and the ST node differ in what
they resolve and what they shard; they do not differ in how a run is handed over.

what each executor promises, since the record has to say which one ran:

  in_process   the OUTPUT FILES and the checkpoint survive a restart. the thread
               does not, and runs_resume re-dispatches whatever is missing.
  temporal     the HISTORY survives. a worker that never met the submitting
               process finishes the run, and the submitter may exit.
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

    `fn_name` is the activity's name on the queue. it is an ARGUMENT and not
    `body.__name__`: the name is the contract between a submitter and a worker in
    another process, and deriving it from a python identifier means renaming a
    function silently orphans every run already in flight.

    `cohort_run` is the in-process path - the protocol's own `encode_*_cohort`,
    which loads the model once and runs the shards in this process. `body` is the
    same work as a named activity, for the durable path, where the shard list is
    all that crosses and a worker holds the body.

    submission returning is the whole point of a run handle. the first version of
    the H&E node blocked for the entire encode and a ~150 s call failed on the
    client side while the server finished it.
    """
    out.mkdir(parents=True, exist_ok=True)
    pending = [s for s in shards if not s.is_done()]
    t0 = time.time()

    from omicstra.dispatch import assert_writable
    from omicstra.settings import settings

    # before either backend is chosen. a refusal that arrives after the first
    # shard has written is a refusal that came too late.
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
        # the thread's ShardReport used to be DROPPED, and a shard that failed all
        # its attempts then looked exactly like one still working: runs_status
        # counts output files, so "absent" meant "not yet" whatever had happened.
        # one hest section failed on an unmappable gene axis and read as `running`
        # for ten minutes. the report is written next to the outputs so a later
        # process - the one that has to answer for the run - can tell a failure
        # from a wait.
        failures = out / "_failed.json"

        def _work():
            import json as _j

            try:
                rep = cohort_run(pending)
            except Exception as e:                        # noqa: BLE001
                # the whole dispatch died, not one shard. that is a different
                # thing from a failed shard and must not read as one.
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
