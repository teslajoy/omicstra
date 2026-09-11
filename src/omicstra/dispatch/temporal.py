"""the durable backend. same shards, same idempotency, a scheduler underneath.

no plugin and no graph. the langgraph integration wraps graph nodes as
activities, which is the right tool for a workflow that interrupts and asks a
person - and the encode path does neither. what it does is run N independent
shards for a long time on a machine that may be reclaimed, so the whole content
of "durable" here is: one activity per shard, three attempts, a heartbeat, and a
history that outlives the worker.

why this module imports temporalio AT MODULE SCOPE
--------------------------------------------------
the SDK refuses a workflow class defined inside a function - "local classes
unsupported" - because it registers definitions by module path and a closure has
none. so the decorators have to run at import time, and the extras discipline is
kept one level up instead: `dispatch/__init__.py` imports this module only when
an address is configured, and a test asserts it never does so at package scope.

determinism, and where it lives
-------------------------------
workflow code REPLAYS, so it must be deterministic: no clocks, no filesystem, no
randomness. all of that - including the idempotency check, which reads a file -
lives in the ACTIVITY. the workflow only decides which shard ids to run and in
what order, which is a pure function of its argument.

that is also why `Shard` crosses the boundary as `ShardSpec`: a workflow argument
is serialised into history, so it has to be JSON, not a Path.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

# the workflow SANDBOX re-imports this module and forbids non-deterministic calls
# during that import - `settings` resolves paths at import time, which trips it.
# passing these through tells the sandbox to reuse the already-imported host
# modules instead of re-executing them. it is safe precisely because nothing the
# WORKFLOW body touches comes from them: the workflow sees only ShardSpec and
# str, and every filesystem call lives in the activity.
with workflow.unsafe.imports_passed_through():
    from omicstra.dispatch import (
        Shard,
        ShardFn,
        ShardReport,
        ShardResult,
        _refuse_total_failure,
    )
    from omicstra.settings import settings

# the activity body is process-global rather than passed through history: a
# function is not serialisable, and the worker that executes the activity is the
# one that must already hold it. the workflow only ever names shards.
_REGISTERED: dict[str, ShardFn] = {}


def register(name: str, fn: ShardFn) -> None:
    """name an activity body so a worker in another process can run it."""
    _REGISTERED[name] = fn


@dataclass
class ShardSpec:
    """`Shard` as it crosses the workflow boundary - JSON, not a Path."""
    id: str
    output: str
    params: dict = field(default_factory=dict)

    @classmethod
    def of(cls, s: Shard) -> ShardSpec:
        return cls(id=s.id, output=str(s.output), params=dict(s.params))

    def to_shard(self) -> Shard:
        return Shard(id=self.id, output=Path(self.output), params=self.params)


@activity.defn(name="omicstra_run_shard")
def run_shard(spec: ShardSpec, fn_name: str) -> str:
    """SYNCHRONOUS on purpose, and the worker gives it a thread pool.

    an async activity that offloads the body to `run_in_executor` cannot
    heartbeat from inside it: the activity context is a contextvar, the worker
    thread does not inherit one, and copying the context across only moves the
    failure from "Not in activity context" to "no running event loop". a sync
    activity runs IN the worker's executor thread with the context already
    established, which is what makes `heartbeat()` work from a body that is
    busy for minutes.

    the cost is that the worker must be constructed with an `activity_executor`.
    `build_worker` does it, which is why callers should use that rather than
    assembling a Worker themselves.
    """
    sh = spec.to_shard()
    # the idempotency check is HERE and not in the workflow: it reads the
    # filesystem, so it is non-deterministic and would break replay. it belongs
    # here for a second reason too - after a worker dies mid-shard the retry
    # re-enters this function, and what it must see is what actually landed.
    if sh.is_done():
        return "skipped"

    _REGISTERED[fn_name](sh, lambda msg="": activity.heartbeat(msg or sh.id))
    return "done"


@workflow.defn(name="omicstra_encode")
class EncodeWorkflow:
    @workflow.run
    async def run(self, specs: list[ShardSpec], fn_name: str,
                  max_attempts: int, heartbeat_s: int) -> list[list[str]]:
        out: list[list[str]] = []
        for spec in specs:
            # sequential on purpose. each shard holds a 631M-parameter model and
            # a decoded slide in memory; fanning them out on one worker is how a
            # laptop run dies of memory rather than of time. parallelism is a
            # worker-count decision, not a workflow one.
            try:
                status = await workflow.execute_activity(
                    run_shard, args=[spec, fn_name],
                    start_to_close_timeout=timedelta(hours=2),
                    heartbeat_timeout=timedelta(seconds=max(heartbeat_s * 3, 30)),
                    retry_policy=RetryPolicy(maximum_attempts=max_attempts))
                out.append([spec.id, status, ""])
            except Exception as e:                        # noqa: BLE001
                # recorded and the run continues, the same rule as local: one
                # unreadable image must not cost the other 279.
                out.append([spec.id, "failed", f"{type(e).__name__}: {e}"])
        return out


def workflow_runner():
    """the sandbox configuration a worker MUST use, supplied by the package.

    the sandbox re-imports the workflow's module to check it for
    non-determinism, and that re-import pulls in `omicstra.settings`, which
    resolves paths while it is being constructed - so validation fails before any
    workflow runs. passing the whole `omicstra` package through is correct here
    rather than a workaround: the workflow body sees only `ShardSpec` and `str`,
    and every filesystem call lives in the activity, which is not sandboxed.

    it lives here because a caller cannot be expected to rediscover it. a worker
    built without this fails with "Failed validating workflow omicstra_encode",
    which names neither the cause nor the fix.
    """
    from temporalio.worker.workflow_sandbox import (
        SandboxedWorkflowRunner,
        SandboxRestrictions,
    )

    return SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules("omicstra"))


def activities_and_workflows():
    """what a worker must register."""
    return run_shard, EncodeWorkflow


def build_worker(client, task_queue: str | None = None):
    """the one correct way to build a worker for this workflow.

    registration, the activity, and the sandbox configuration in one place, so
    the three cannot drift apart across a test, a CLI and a deployment.
    """
    from concurrent.futures import ThreadPoolExecutor

    from temporalio.worker import Worker

    # the executor is REQUIRED, not a tuning knob: `run_shard` is a synchronous
    # activity, and temporal refuses to register one without somewhere to run it.
    # one thread by default because each shard holds a 631M-parameter model and a
    # decoded slide - concurrency here is a memory decision, and the safe default
    # is none.
    return Worker(client, task_queue=task_queue or settings.temporal_task_queue,
                  workflows=[EncodeWorkflow], activities=[run_shard],
                  activity_executor=ThreadPoolExecutor(max_workers=1),
                  workflow_runner=workflow_runner())


async def _submit(shards, fn_name, address, namespace, task_queue,
                  max_attempts, heartbeat_s):
    from temporalio.client import Client

    client = await Client.connect(address, namespace=namespace)
    specs = [ShardSpec.of(s) for s in shards]
    # the id is derived from the shard set, so re-submitting the same cohort
    # attaches to the run already in flight rather than starting a second one.
    wf_id = "omicstra-encode-" + str(abs(hash(tuple(s.id for s in shards))))
    return await client.execute_workflow(
        EncodeWorkflow.run, args=[specs, fn_name, max_attempts, heartbeat_s],
        id=wf_id, task_queue=task_queue)


def run_shards_durably(shards: list[Shard], fn: ShardFn, *, address: str,
                       max_attempts: int = 3, heartbeat_s: int = 30,
                       fn_name: str = "encode_he",
                       on_event: Callable[[str, ShardResult], None] | None = None
                       ) -> ShardReport:
    """submit the shards and wait. the caller sees the same ShardReport."""
    register(fn_name, fn)
    rows = asyncio.run(_submit(shards, fn_name, address, settings.temporal_namespace,
                               settings.temporal_task_queue, max_attempts, heartbeat_s))
    report = ShardReport(backend=f"temporal:{address}")
    for sid, status, err in rows:
        r = ShardResult(sid, status, error=err)
        report.results.append(r)
        if on_event:
            on_event(status, r)
    # the same refusal as local. a durable run that fails every shard has burned
    # a scheduler slot to produce nothing, which is the more expensive place to
    # find that out quietly.
    _refuse_total_failure(report)
    return report
