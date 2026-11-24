"""process-level compute cache for evaluation arrays.

the expensive part of every `ask` call is loading embedding matrices from
disk: ~150-200 MB of npy/parquet per run, taking 30-60s before any
computation starts. this module holds those arrays in memory for the life
of the server process.

the cache key is (project_root, run). arrays() checks it first; only the
first call per run touches disk. a warm_cache MCP tool pre-populates it at
startup so the first real question is also fast.

nothing here is persisted - a server restart clears everything.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

_STORE: dict[tuple[str, str], dict[str, Any]] = {}


def key(project_dir: str | Path, run: str) -> tuple[str, str]:
    return (str(Path(project_dir).resolve()), run)


def get(project_dir: str | Path, run: str) -> dict[str, Any] | None:
    return _STORE.get(key(project_dir, run))


def put(project_dir: str | Path, run: str, ctx: dict[str, Any]) -> None:
    _STORE[key(project_dir, run)] = ctx


def clear(project_dir: str | Path | None = None, run: str | None = None) -> int:
    if project_dir is None and run is None:
        n = len(_STORE)
        _STORE.clear()
        return n
    to_drop = [k for k in _STORE
               if (project_dir is None or k[0] == str(Path(project_dir).resolve()))
               and (run is None or k[1] == run)]
    for k in to_drop:
        del _STORE[k]
    return len(to_drop)


def status() -> dict[str, Any]:
    import sys
    total = sum(sys.getsizeof(v) for v in _STORE.values())
    return {
        "entries": len(_STORE),
        "runs": [{"project": k[0], "run": k[1]} for k in _STORE],
        "approx_mb": round(total / 1_048_576, 1),
    }
