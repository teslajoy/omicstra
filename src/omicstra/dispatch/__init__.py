"""running N shards of the same work, durably or not.

this is not a level. levels 0/1/2 say what code is ALLOWED to do; dispatch says
where the work runs, which is orthogonal to all three - the same protocol step
is the same step whether it executes in this process or on a scheduler.

the shape, and why each piece is there
--------------------------------------
one activity per subarray. not one per cohort: a 280-sample encode that fails at
sample 200 must not redo 199, and the unit of retry has to be the unit of work.

idempotent on the OUTPUT FILE, not on a database. the file is the thing the next
stage reads, so its existence is the only honest evidence the shard is done - a
ledger that says "done" beside a missing file is worse than no ledger. this is
also what makes kill-and-resume work with no coordinator at all.

fallthrough when TEMPORAL_ADDRESS is unset, and it is the DEFAULT. a laptop run
needs no server, CI has none, and a dependency that is only exercised in
production is a dependency nobody has tested. the durable backend is the same
function with a different executor underneath.

three attempts, then the shard is recorded failed and the RUN CONTINUES. one
unreadable image must not cost the other 279.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from omicstra.settings import settings

_log = logging.getLogger(__name__)

# the activity signature. `heartbeat` is a callable the body invokes to say it is
# still alive; locally it is a no-op, on temporal it reports progress so a long
# shard is not mistaken for a hung worker.
ShardFn = Callable[["Shard", Callable[[str], None]], None]


class WouldOverwriteOracle(RuntimeError):
    """a shard's output points inside a declared read-only input.

    this is not a data-loss guard, or not only. the artifacts that produced the
    published grid are what every port is diffed against, so writing into one
    means the next diff compares a port against its own output - and PASSES.
    the failure is silent and it invalidates the verification rather than the
    data, which is why it refuses rather than warning.
    """


class AllShardsFailed(RuntimeError):
    """every shard failed, which is never a data problem.

    one unreadable image is data. 280 unreadable images is a missing model, a
    wrong path, a permissions wall, or a writer that never produced its output -
    and the np.save suffix bug was exactly that: three retries each, a quiet
    `failed` on every row, and a report that read like a bad cohort.

    a partial failure is recorded and the run continues. a TOTAL failure raises,
    because continuing past it only produces an empty output directory and a
    summary nobody reads until much later.
    """


@dataclass(frozen=True)
class Shard:
    """one unit of work, and the file that proves it finished.

    `output` is load-bearing rather than descriptive: it is what `is_done` reads,
    so a shard whose function writes somewhere else will be re-run forever and a
    shard that writes a truncated file will be considered done. the writer's
    contract is therefore: write to a temporary path, then rename - which is what
    `atomic_write` below is for.
    """
    id: str
    output: Path
    params: dict = field(default_factory=dict)

    def is_done(self) -> bool:
        try:
            return self.output.exists() and self.output.stat().st_size > 0
        except OSError:
            return False


@dataclass
class ShardResult:
    id: str
    status: str                 # done | skipped | failed
    attempts: int = 0
    seconds: float = 0.0
    error: str = ""


@dataclass
class ShardReport:
    backend: str
    results: list[ShardResult] = field(default_factory=list)

    @property
    def done(self) -> list[str]:
        return [r.id for r in self.results if r.status == "done"]

    @property
    def skipped(self) -> list[str]:
        return [r.id for r in self.results if r.status == "skipped"]

    @property
    def failed(self) -> list[str]:
        return [r.id for r in self.results if r.status == "failed"]

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        return (f"{self.backend}: {len(self.done)} done, {len(self.skipped)} already "
                f"present, {len(self.failed)} failed")


def atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    """write via a temporary file and rename, so a killed process leaves no
    half-written output that `is_done` would believe.

    this is the other half of file-based idempotency. without it, a SIGKILL
    during a write produces a non-empty file and the shard is skipped forever
    with corrupt contents - the failure mode that makes people distrust resume
    and re-run everything by hand.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # the temp name keeps the real SUFFIX - `x.partial.npy`, not `x.npy.partial`.
    # numpy's save() appends `.npy` to any path that lacks it, so the suffix-last
    # spelling had it write `x.npy.partial.npy`, leaving the rename target absent,
    # the shard failed three times, and a stray file behind. writers that rewrite
    # the name they are handed are common enough that the temp path has to look
    # like the destination.
    tmp = path.with_name(f"{path.stem}.partial{path.suffix}")
    try:
        write(tmp)
        if not tmp.exists():
            raise FileNotFoundError(
                f"the writer did not produce {tmp.name}. some writers rewrite the "
                "path they are given - numpy.save appends '.npy' - so pass a file "
                "handle rather than a path, or write to exactly the path handed in.")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def read_only_inputs(project_id: str | None = None) -> list[Path]:
    """paths a cohort declares as oracle, resolved absolute.

    absent is not an error - a cohort with no published results has no oracle to
    protect - but a cohort that HAS one must say so, because nothing else can
    tell a cache from an output directory by looking at it.
    """
    from omicstra.contracts.project import ProjectConfig

    try:
        cfg = ProjectConfig.load(project_id)
    except Exception:                       # noqa: BLE001 - undeclared is fine
        return []
    return read_only_paths(cfg)


def read_only_paths(cfg) -> list[Path]:
    """the same resolution, from a config already in hand.

    the align and eval stages hold a ProjectConfig rather than a project id, and
    loading it a second time to ask one question is how two callers end up
    reading two different declarations.
    """
    decl = cfg.read_only_inputs or {}
    if decl.get("policy") != "refuse_writes":
        return []
    out = []
    for rel in decl.get("paths") or []:
        try:
            out.append(cfg.path(rel).resolve())
        except (OSError, ValueError) as e:
            # an unresolvable declared path protects nothing, and silently
            # dropping it would leave a gap in the guard - say so.
            _log.warning("read_only_inputs: cannot resolve %r (%s)", rel, e)
    return out


def protected_by(target: Path, protected: Iterable[Path]) -> Path | None:
    """the declared read-only path `target` sits in, or None."""
    t = Path(target).resolve()
    return next((ro for ro in protected if t == ro or ro in t.parents), None)


def assert_writable(shards: Iterable[Shard], project_id: str | None = None) -> None:
    """refuse before a single byte moves if any output lands in an oracle."""
    protected = read_only_inputs(project_id)
    if not protected:
        return
    for sh in shards:
        ro = protected_by(sh.output, protected)
        if ro is not None:
            raise WouldOverwriteOracle(
                f"shard {sh.id!r} would write {Path(sh.output).resolve()} inside {ro}, "
                "which this cohort declares read-only. that directory is what the ports "
                "are diffed against - writing into it makes the next diff "
                "compare a port against its own output and pass. choose a "
                "run-scoped output directory instead.")


def run_shards(shards: Iterable[Shard], fn: ShardFn, *, address: str | None = None,
               max_attempts: int = 3, heartbeat_s: int = 30, project_id: str | None = None,
               on_event: Callable[[str, ShardResult], None] | None = None) -> ShardReport:
    """run every shard once, skipping those already finished.

    `address` defaults to `settings.temporal_address`. unset means in-process,
    and that is the normal path rather than a fallback - see the module docstring.
    """
    shards = list(shards)
    # before anything runs, and before either backend is chosen: a refusal that
    # arrives after the first shard has written is a refusal that came too late.
    assert_writable(shards, project_id)
    addr = address if address is not None else settings.temporal_address
    if addr:
        from omicstra.dispatch.temporal import run_shards_durably

        return run_shards_durably(shards, fn, address=addr, max_attempts=max_attempts,
                                  heartbeat_s=heartbeat_s, on_event=on_event)
    return run_shards_locally(shards, fn, max_attempts=max_attempts, on_event=on_event)


def run_shards_locally(shards: Iterable[Shard], fn: ShardFn, *, max_attempts: int = 3,
                       on_event: Callable[[str, ShardResult], None] | None = None
                       ) -> ShardReport:
    """the in-process executor. retries and idempotency, no server.

    kill this mid-run and start it again: every shard whose output landed is
    skipped, and the run picks up where it stopped. that property comes from the
    output file, not from the executor, which is why the durable backend gets it
    for free rather than implementing it twice.
    """
    report = ShardReport(backend="local")
    for sh in shards:
        if sh.is_done():
            r = ShardResult(sh.id, "skipped")
            report.results.append(r)
            if on_event:
                on_event("skipped", r)
            continue

        r = ShardResult(sh.id, "failed")
        t0 = time.time()
        for attempt in range(1, max_attempts + 1):
            r.attempts = attempt
            try:
                fn(sh, lambda _msg="": None)
                r.status = "done"
                r.error = ""
                break
            except Exception as e:                        # noqa: BLE001 - recorded, not swallowed
                # the error travels on the result. a shard that fails three times
                # is a recorded failure and the run continues: one unreadable
                # image must not cost the other 279.
                r.error = f"{type(e).__name__}: {e}"
        r.seconds = round(time.time() - t0, 2)
        report.results.append(r)
        if on_event:
            on_event(r.status, r)
    _refuse_total_failure(report)
    return report


def _refuse_total_failure(report: ShardReport) -> None:
    """raise when nothing at all succeeded. shared by both backends."""
    attempted = [r for r in report.results if r.status != "skipped"]
    if attempted and all(r.status == "failed" for r in attempted):
        errs = sorted({r.error for r in attempted if r.error})
        raise AllShardsFailed(
            f"all {len(attempted)} shard(s) failed after their retries - that is a "
            f"setup problem, not a data problem. distinct error(s): "
            + ("; ".join(errs[:3]) if errs else "none recorded"))
