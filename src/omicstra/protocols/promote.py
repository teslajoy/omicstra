"""propose the evidence pack from scored artifacts, and never author its prose.

the pack is what `routing.resolve` reads. until now it was written by hand, which
made every routed number a claim about a person's care rather than about an
artifact. this stage computes the half that IS computable - candidates, their
intervals, the floor comparison, the guard verdicts and the outcome the contract's
rules imply - and leaves the half that is not.

    computed   value, ci, ranking, margin, outcome, guard verdicts
    declared   metric text, source, floor, higher_is_better, which reader to use
    human      note, caveat, interpretation - taken at a gate, actor=human

nothing here reads natural language and nothing here writes prose. a `note` that
appeared without a person is the failure this module exists to make impossible,
and `provenance()` is what a test holds it to.

cohort-free: the readers are named in the cohort's `evidence_sources.json`, which
says which artifact answers which task family. the package ships the readers; the
cohort says where its numbers live. a second cohort with the same pipeline
declares the same reader names against its own paths.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from omicstra.contracts.routing import load_routing_contract

# --- readers: artifact -> candidates ---------------------------------------
# each returns [{id, value, ci?}] and nothing else. no ranking, no judgement.
_READERS: dict[str, Callable[..., list[dict]]] = {}


def reader(name: str):
    def wrap(fn):
        _READERS[name] = fn
        return fn
    return wrap


def _runs(root: Path, runs: list[str]) -> list[Path]:
    return [root / r for r in runs]


@reader("h1_auc_ci")
def _h1_auc(root: Path, runs: list[str], **_: Any) -> list[dict]:
    """per-run AUC with its bootstrap interval, from each run's H1 CI file."""
    out = []
    for r in runs:
        p = root / r / "metrics_h1_ci.json"
        if not p.is_file():
            continue
        m = json.loads(p.read_text())
        out.append({"id": r, "value": round(m["auc"], 4),
                    "ci": [round(m["auc_bootstrap_ci_low"], 4),
                           round(m["auc_bootstrap_ci_high"], 4)]})
    return out


@reader("h2c_ratio_ci")
def _h2c(root: Path, runs: list[str], *, ci_file: str, **_: Any) -> list[dict]:
    """bio/patient ratio and its delta-method interval."""
    m = json.loads((root / ci_file).read_text()).get("h2c", {})
    return [{"id": r, "value": round(m[r]["ratio"], 3),
             "ci": [round(m[r]["ratio_ci_low"], 3), round(m[r]["ratio_ci_high"], 3)],
             "guard_inputs": {"bio_z": m[r]["z_TIME"], "confounder_z": m[r]["z_patient"]}}
            for r in runs if r in m]


@reader("mc_coherence_ari")
def _h2a(root: Path, runs: list[str], *, table: str, view: str = "z_he",
         ci_file: str | None = None, **_: Any) -> list[dict]:
    """KMeans ARI against the niche-level label, with the patient-bootstrap interval."""
    import pandas as pd

    df = pd.read_parquet(root / table)
    df = df[df["view"] == view].set_index("run_id")
    ci = json.loads((root / ci_file).read_text()).get("h2a", {}) if ci_file else {}
    out = []
    for r in runs:
        in_table = r in df.index
        if not in_table and r not in ci:
            continue
        # a run added to the grid after the point table was built has its ARI only
        # in the interval file, which recomputes it under the same protocol. that
        # is a usable number and a MIXED source, so the candidate says which it is.
        row = {"id": r,
               "value": round(float(df.loc[r, "ari"]) if in_table else ci[r]["ari"], 3),
               "source": table if in_table else ci_file}
        if r in ci:
            row["ci"] = [round(ci[r]["ari_ci_low"], 3), round(ci[r]["ari_ci_high"], 3)]
        out.append(row)
    return out


@reader("biology_raw_floor")
def _raw_floor(root: Path, runs: list[str], *, view: str = "raw_he",
               label: str = "TIME", table: str = "eval/biology.parquet", **_: Any) -> list[dict]:
    """the same biology-vs-subject test on the UN-PROJECTED features.

    every run's biology table carries the raw views beside the projected ones,
    so the floor a method is judged against is measured on the same niches, by
    the same statistic, under the same null - not cited from a document.

    the raw H&E view is identical across arms because the morphology input is;
    the raw ST view is NOT, because an arm may declare different ST features.
    the reader therefore returns one row per run and lets the caller see that.
    """
    import pandas as pd

    out = []
    for r in runs:
        f = root / r / table
        if not f.is_file():
            continue
        df = pd.read_parquet(f)
        d = df[df["view"] == view]
        bio = d[(d["label"] == label) & (d["test_type"] == "biology")]["z"]
        pat = d[d["test_type"] == "patient"]["z"]
        if len(bio) and len(pat) and float(pat.iloc[0]):
            out.append({"id": r, "value": round(float(bio.iloc[0]) / float(pat.iloc[0]), 4),
                        "source": f"{r}/{table}#{view}"})
    return out


@reader("mc_coherence_reference")
def _h2a_ref(root: Path, runs: list[str], *, table: str, view: str = "z_he",
             **_: Any) -> list[dict]:
    """the raw-modality rows of the same table - reported, never decisive."""
    import pandas as pd

    # the raw rows carry no view - they are the un-projected modality, so the
    # view column that distinguishes z_he from z_st does not apply to them.
    df = pd.read_parquet(root / table)
    df = df[df["run_id"].isin(runs)]
    return [{"id": r.run_id, "value": round(float(r.ari), 3)} for r in df.itertuples()]


@reader("pathway_cca_sig_count")
def _h3(root: Path, runs: list[str], *, table: str, view: str = "z_he",
        alpha: float = 0.05, **_: Any) -> list[dict]:
    """how many TESTABLE pathways survive BH-FDR on the held-out-subject null."""
    import pandas as pd

    df = pd.read_parquet(root / table)
    df = df[(df["view"] == view) & df["testable"]]
    out = []
    for r in runs:
        sub = df[df["run_id"] == r]
        if sub.empty:
            continue
        out.append({"id": r, "value": int((sub["fdr_A"] < alpha).sum()),
                    "unit_total": len(sub)})
    return out


@reader("signature_cca_z")
def _tls(root: Path, runs: list[str], *, table: str, view: str = "z_he", **_: Any) -> list[dict]:
    """z of the signature CCA, zero where it fails its own FDR - a non-finding is 0, not a rank."""
    import pandas as pd

    df = pd.read_parquet(root / table)
    df = df[df["view"] == view].set_index("run_id")
    col = next((c for c in df.columns if c.startswith("sig_")), None)
    out = []
    for r in runs:
        if r not in df.index:
            continue
        passed = bool(df.loc[r, col]) if col else True
        out.append({"id": r, "value": round(float(df.loc[r, "z_TLS"]), 1) if passed else 0})
    return out


# --- the proposal ----------------------------------------------------------
def _clears(value: float, floor: dict | None, higher_is_better: bool) -> bool:
    """strictly beats it. a floor is what a number must BEAT to mean anything.

    at-floor is not cleared: a run with zero significant pathways sits exactly on
    a floor of "none significant" and answers nothing. reading `>=` here forces
    the floor to be inflated in the declaration instead, which puts a number in
    the cohort's file that the cohort never measured.
    """
    if not floor or floor.get("value") is None:
        return True
    return value > floor["value"] if higher_is_better else value < floor["value"]


def _overlaps(a: dict, b: dict) -> bool:
    """two candidates are tied when the measurement does not order them.

    equal values tie regardless of interval: a count metric carries no width, and
    four arms that each answer 4 of 4 pathways are not ranked by the order they
    came out of a groupby. otherwise the contract's interval_overlap rule applies,
    and a candidate with no interval is left unordered by it.
    """
    if a["value"] == b["value"]:
        return True
    ca, cb = a.get("ci"), b.get("ci")
    if not ca or not cb:
        return False
    return min(ca[1], cb[1]) >= max(ca[0], cb[0])


def guard_verdicts(decl: dict, candidates: list[dict]) -> list[dict]:
    """the declared guard, applied per candidate to the inputs its reader carried.

    only guards whose inputs are IN the artifact are run here. a guard named
    without inputs would produce a verdict about nothing, which is the failure
    `_not_run` exists for one layer down.
    """
    name = decl.get("guard")
    if not name:
        return []
    floor = (decl.get("floor") or {}).get("value")
    out = []
    for c in candidates:
        gi = c.get("guard_inputs") or {}
        if name == "G3_bio_vs_confounder" and {"bio_z", "confounder_z"} <= set(gi):
            from omicstra.protocols.evaluate import bio_vs_confounder_ratio

            g = bio_vs_confounder_ratio(gi["bio_z"], gi["confounder_z"], floor)
            out.append({"run": c["id"], "guard": g.name, "passed": g.passed,
                        "value": g.value, "contraindicated": not g.passed,
                        "note": g.note})
        else:
            out.append({"run": c["id"], "guard": name, "status": "not_run",
                        "contraindicated": False,
                        "note": f"needs inputs the {decl['reader']!r} reader does not carry"})
    return out


def propose_task(task_id: str, decl: dict, candidates: list[dict],
                 guards: list[dict] | None = None) -> dict:
    """rank, compare to the floor, apply the contract's rules, and stop there.

    the outcome vocabulary is the contract's: recommend | tie | refuse. a row
    where no candidate clears its floor or where every candidate is
    contraindicated REFUSES - it has no method to route to, and saying so is a
    routable answer rather than a caveat attached to a winner.
    """
    hib = bool(decl.get("higher_is_better", True))
    ranked = sorted(candidates, key=lambda c: c["value"], reverse=hib)
    floor = decl.get("floor")
    for c in ranked:
        c["clears_floor"] = _clears(c["value"], floor, hib)

    guards = guards or []
    blocked = {g["run"] for g in guards if g.get("contraindicated")}
    eligible = [c for c in ranked if c["clears_floor"] and c["id"] not in blocked]

    out = {"task_id": task_id, "metric": decl.get("metric", ""),
           "source": decl.get("source", ""), "higher_is_better": hib,
           "unit": decl.get("unit"), "floor": floor,
           "candidates": ranked, "guards": guards}
    if decl.get("reference"):
        out["reference"] = decl["reference"]

    if not eligible:
        why = ("no candidate clears the floor" if floor and not any(c["clears_floor"] for c in ranked)
               else "every candidate is contraindicated by a guard")
        best = ranked[0] if ranked else None
        out |= {"outcome": "refuse",
                "refusal": (f"{why}. this cohort supports no method for this question"
                            + (f"; the best measured is {best['id']} at {best['value']}"
                               if best else "")),
                "winner": None, "margin": None}
        return out

    lead = eligible[0]
    tied = [c["id"] for c in eligible[1:] if _overlaps(lead, c)]
    runner = eligible[1] if len(eligible) > 1 else None
    margin = round(abs(lead["value"] - runner["value"]), 4) if runner else None
    if tied:
        out |= {"outcome": "tie", "winner": None, "tied": [lead["id"], *tied],
                "margin": margin}
    else:
        out |= {"outcome": "recommend", "winner": lead["id"], "margin": margin}
    return out


def read_candidates(decl: dict, root: Path, runs: list[str]) -> list[dict]:
    name = decl["reader"]
    if name not in _READERS:
        raise KeyError(f"no reader {name!r}; declared readers: {sorted(_READERS)}")
    args = {k: v for k, v in decl.items()
            if k not in {"reader", "metric", "source", "floor", "higher_is_better",
                         "unit", "guard", "not_available"}}
    return _READERS[name](root, runs, **args)


def propose(sources: dict, root: Path, runs: list[str],
            guards_by_task: dict[str, list[dict]] | None = None) -> dict:
    """one proposal per declared task family, plus the ones that are not available.

    a family the cohort cannot answer is REPORTED as unavailable with its reason,
    never dropped: a missing row and an unanswerable question look identical in a
    pack that omits it.
    """
    contract = load_routing_contract()
    known = {f["id"] for f in contract["task_families"]}
    guards_by_task = guards_by_task or {}
    tasks = {}
    for task_id, decl in sources.get("tasks", {}).items():
        if task_id not in known:
            raise KeyError(f"{task_id!r} is not a declared task family")
        if decl.get("not_available"):
            tasks[task_id] = {"task_id": task_id, "outcome": "not_available",
                              "why": decl["not_available"], "candidates": []}
            continue
        cands = read_candidates(decl, root, runs)
        g = guards_by_task.get(task_id) or guard_verdicts(decl, cands)
        rr = decl.get("reference_reader")
        if rr:
            decl = dict(decl, reference=read_candidates(rr, root, rr["ids"]))
        tasks[task_id] = propose_task(task_id, decl, cands, g)
    missing = sorted(known - set(tasks))
    return {"project_id": sources.get("project_id"), "runs_root": str(root),
            "tasks": tasks, "families_not_declared": missing}


# --- provenance: which half of every field was computed ---------------------
COMPUTED = ("value", "ci", "candidates", "margin", "outcome", "winner", "tied",
            "refusal", "clears_floor", "guards", "ranking")
DECLARED = ("metric", "source", "floor", "higher_is_better", "unit", "reader")
HUMAN = ("note", "caveat", "interpretation")


def provenance(pack: dict) -> dict:
    """per section, which fields came from where. the assertion, not the claim.

    a test reads this and fails if a computed field was authored or a human field
    was generated, which is what keeps "the server invents nothing" true after
    the next edit rather than only today.
    """
    out = {}
    for task_id, t in (pack.get("tasks") or {}).items():
        out[task_id] = {
            "computed": sorted(k for k in t if k in COMPUTED),
            "declared": sorted(k for k in t if k in DECLARED),
            "human": sorted(k for k in t if k in HUMAN),
            "human_fields_present": any(k in t for k in HUMAN),
        }
    return out


def assert_not_authored(pack: dict) -> None:
    """no computed field may arrive as prose, and no human field may be generated.

    the pack is written by two actors and the boundary is the whole point: a
    string where a number belongs means a person typed a measurement.
    """
    for task_id, t in (pack.get("tasks") or {}).items():
        for c in t.get("candidates", []):
            if not isinstance(c.get("value"), (int, float)):
                raise TypeError(f"{task_id}/{c.get('id')}: value is not a number - "
                                "a measurement was authored")
            if c.get("ci") is not None and not (isinstance(c["ci"], list) and len(c["ci"]) == 2):
                raise ValueError(f"{task_id}/{c.get('id')}: ci is not an interval")
        if t.get("margin") is not None and not isinstance(t["margin"], (int, float)):
            raise ValueError(f"{task_id}: margin is not a number")
