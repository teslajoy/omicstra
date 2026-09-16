"""the whole chain in one call, and the diff that says it still agrees.

step 8. every stage already exists and each is tested on its own; what was
missing is the statement that they compose - that a cohort's declarations,
caches and scored artifacts carry from inventory through to a rendered report
without a person joining them up by hand.

    resolve   DEFAULT. read what previous runs produced, compute nothing.
    compute   pass compute=True to train and score; refused into any declared
              read-only path, and staged elsewhere by run_eval.

the acceptance is the DIFF, not the run. finishing proves plumbing; reproducing
the curated pack's numbers from the artifacts proves the chain still describes
what produced the published results. `diff_against` returns the disagreements,
and an empty list is the only passing state.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omicstra.contracts.project import ProjectConfig
from omicstra.protocols.align import (
    ComputeUnavailable,
    run_align,
    run_embed_he,
    run_embed_st,
    run_eval,
    run_niche_join,
)
from omicstra.protocols.promote import propose, provenance
from omicstra.report import gather, render
from omicstra.settings import settings


def _rec(records: list[dict], rec: Any) -> None:
    records.append(rec.model_dump(mode="json") if hasattr(rec, "model_dump") else dict(rec))


def run_cohort(project_id: str | None = None, *, compute: bool = False,
               out_dir: Path | None = None, hypotheses: tuple[str, ...] = ("H1",),
               out_root: Path | None = None) -> dict:
    """inventory -> join -> grid -> eval -> pack -> report, recording each stage.

    a stage that cannot resolve does NOT abort the run: it is recorded with the
    reason and the chain continues as far as the cohort's artifacts allow. a
    cohort missing its grid should still produce a report saying exactly that.
    """
    cfg = ProjectConfig.load(project_id)
    root = cfg.project_dir or settings.project_root(project_id)
    records: list[dict] = []
    stages: dict[str, str] = {}

    def stage(name: str, fn):
        try:
            value, rec = fn()
            _rec(records, rec)
            stages[name] = "resolved" if (rec.params or {}).get("resolves_only") else "computed"
            return value
        except (ComputeUnavailable, FileNotFoundError, KeyError) as e:
            stages[name] = f"unavailable: {e}"
            return None

    he = stage("embed_he", lambda: run_embed_he(cfg, project_id))
    st = stage("embed_st", lambda: run_embed_st(cfg, project_id))
    nj = stage("niche_join", lambda: run_niche_join(cfg, project_id))
    grid = None
    if nj is not None:
        grid = stage("align", lambda: run_align(cfg, nj, project_id=project_id, compute=compute))
    if grid is not None:
        stage("eval", lambda: run_eval(cfg, grid, project_id, compute=compute,
                                       hypotheses=hypotheses, out_root=out_root))

    pack, prov = None, None
    src_path = root / "evidence_sources.json"
    if src_path.is_file():
        src = json.loads(src_path.read_text())
        runs_root = (root / src["runs_root"]).resolve()
        if runs_root.is_dir():
            pack = propose(src, runs_root, src["runs"])
            prov = provenance(pack)
            stages["promote"] = "proposed"
        else:
            stages["promote"] = f"unavailable: no grid at {runs_root}"
    else:
        stages["promote"] = "unavailable: the cohort declares no evidence sources"

    bundle = gather(root)
    if pack:
        bundle |= {"pack": pack, "provenance": prov}
    bundle["records"] = records
    text = render(bundle, f"{root.name} evaluation report")

    out = {"project_id": root.name, "stages": stages, "records": records,
           "pack": pack, "provenance": prov, "report": text,
           "artifacts": {"he": he.model_dump() if he else None,
                         "st": st.model_dump() if st else None,
                         "niche_join": nj.model_dump() if nj else None}}
    if out_dir:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "report.md").write_text(text)
        (d / "records.json").write_text(json.dumps(records, indent=2) + "\n")
        if pack:
            (d / "pack.json").write_text(json.dumps(pack, indent=2) + "\n")
        (d / "stages.json").write_text(json.dumps(stages, indent=2) + "\n")
        out["written"] = str(d)
    return out


def diff_against(pack: dict, curated: dict, tol: float = 5e-4) -> list[dict]:
    """what the rebuilt pack and the curated one disagree about, per candidate.

    three kinds, kept distinct because they mean different things: a value that
    moved (the chain no longer produces the published number), a method the
    curated pack does not carry (it was omitted), and one the rebuild cannot
    reach (its artifact is gone or its reader is not declared).
    """
    out = []
    ours = pack.get("tasks") or {}
    theirs = curated.get("tasks") or {}
    for task_id, t in ours.items():
        if t.get("outcome") == "not_available" or task_id not in theirs:
            continue
        hand = {c["id"]: c for c in theirs[task_id].get("candidates", [])}
        mine = {c["id"]: c for c in t.get("candidates", [])}
        for rid, c in mine.items():
            h = hand.get(rid)
            if h is None:
                out.append({"task": task_id, "method": rid, "kind": "absent_from_curated",
                            "computed": c["value"]})
                continue
            if abs(float(h["value"]) - float(c["value"])) > tol:
                out.append({"task": task_id, "method": rid, "kind": "value_moved",
                            "curated": h["value"], "computed": c["value"]})
            elif h.get("ci") and c.get("ci") and any(
                    abs(a - b) > tol for a, b in zip(h["ci"], c["ci"], strict=False)):
                out.append({"task": task_id, "method": rid, "kind": "interval_moved",
                            "curated": h["ci"], "computed": c["ci"]})
        for rid in set(hand) - set(mine):
            out.append({"task": task_id, "method": rid, "kind": "not_reachable_by_rebuild",
                        "curated": hand[rid]["value"]})
    return out
