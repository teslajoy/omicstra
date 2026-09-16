"""the report: a pack and a ledger in, markdown out, no cohort in the renderer.

the template, not anyone's report. every line comes from a declaration, a
record or the pack; nothing here knows a platform, a patient count or an
encoder name. the test is that a synthetic two-method cohort and the seed cohort
render through the same code.

three rules hold it to that:

  - a section appears because its INPUT exists, never because the code expects
    one. a cohort that never ran a hypothesis has no block for it and says so in
    one line, which is a true statement about that cohort rather than a gap.
  - computed and authored stay marked. the matrix carries measurements; the
    caveats carry prose, attributed. a reader can always tell which they are
    reading, because `promote` kept them apart and this file does not merge them.
  - nothing is recomputed here. a renderer that derives a number is a second
    source for it, and two sources for one number is how they disagree.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# --- the bundle ------------------------------------------------------------
# the renderer takes ONE dict. assembling it from a cohort root is a separate
# function, so a caller with a pack in memory never has to write files first.
_DECLARATIONS = ("cohort", "platform", "project", "inventory", "eda")


def gather(root: Path, pack_name: str = "routing_evidence.json") -> dict:
    """read what a cohort has, skip what it does not. absence is not an error."""
    root = Path(root)
    files = {"cohort": "cohort.json", "platform": "platform.json",
             "project": "project.json", "inventory": "inventory.json",
             "eda": "eda_summary.json", "pack": pack_name}
    out: dict[str, Any] = {"root": str(root)}
    for key, name in files.items():
        p = root / name
        if p.is_file():
            try:
                out[key] = json.loads(p.read_text())
            except json.JSONDecodeError as e:
                out.setdefault("unreadable", []).append({"file": name, "error": str(e)})
    return out


# --- small helpers ---------------------------------------------------------
def _h(level: int, text: str) -> str:
    return f"{'#' * level} {text}"


def _kv(rows: list[tuple[str, Any]]) -> list[str]:
    rows = [(k, v) for k, v in rows if v not in (None, "", [], {})]
    if not rows:
        return []
    out = ["| field | value |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return out


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return []
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join("" if c is None else str(c) for c in r) + " |" for r in rows]
    return out


def _absent(what: str, why: str) -> list[str]:
    return [f"*not applicable: {why}*" if why else f"*{what} not recorded for this cohort*"]


# --- sections --------------------------------------------------------------
def section_cohort(b: dict) -> list[str]:
    """what was declared, what bound, what was refused."""
    lines = [_h(2, "1 · cohort")]
    decl = b.get("project") or {}
    cohort = b.get("cohort") or {}
    plat = b.get("platform") or {}
    platforms = list((plat.get("platforms") or {}))
    lines += _kv([
        ("project", decl.get("project_id")),
        ("platform", decl.get("platform") or ", ".join(platforms)),
        ("subject column", cohort.get("subject_id_column")),
        ("encoders", ", ".join(f"{e.get('name')} ({e.get('dim')}d, {e.get('role')})"
                               for e in decl.get("encoders", []))),
        ("unit", f"niche = centre spot + {decl['k_neighbors']} neighbours"
                 if decl.get("k_neighbors") else None),
        ("split", decl.get("split")),
        ("compute backend", cohort.get("compute_backend")),
    ])
    inv = b.get("inventory") or {}
    bind = inv.get("bind") or {}
    if bind:
        lines += ["", *_kv([("samples bound", bind.get("n_bound")),
                            ("declared without data", bind.get("n_declared_without_data")),
                            ("undeclared", bind.get("n_undeclared"))])]
    undeclared = (plat.get("samples_declared_without_data") or {})
    if undeclared:
        lines += ["", "declared without data, and why:"]
        lines += _table(["sample", "reason"],
                        [[k, v.get("why", "")] for k, v in undeclared.items()])
    if len(lines) == 1:
        lines += _absent("cohort declarations", "")
    return lines


def section_admissibility(b: dict) -> list[str]:
    """which checks ran, which did not, and the verdict."""
    lines = [_h(2, "2 · admissibility")]
    eda = b.get("eda") or {}
    if not eda:
        return lines + _absent("EDA", "no eda_summary.json - this cohort has not been gated")
    verdict = eda.get("verdict") or eda.get("gate") or {}
    lines += _kv([("verdict", verdict if isinstance(verdict, str) else verdict.get("verdict")),
                  ("run", eda.get("eda_date")),
                  ("produced by", eda.get("produced_by"))])
    checks = eda.get("checks") or eda.get("steps") or {}
    rows = []
    if isinstance(checks, dict):
        for name, c in checks.items():
            if isinstance(c, dict):
                rows.append([name, c.get("status", ""), c.get("why_not") or c.get("result", "")])
    if rows:
        lines += ["", *_table(["check", "status", "result"], rows)]
    return lines


def section_computed(b: dict) -> list[str]:
    """what was computed, from the records the run wrote."""
    lines = [_h(2, "3 · what was computed")]
    recs = b.get("records") or []
    if not recs:
        return lines + _absent("run records", "no ledger for this run")
    # one row per STEP, not per record. a cohort answers the same question many
    # times and a ledger dump buries the pipeline under its own routing history.
    agg: dict[tuple, dict] = {}
    for r in recs:
        step = str(r.get("step_id", ""))
        kind = "decision" if step.startswith("route::") else "pipeline"
        key = (kind, step.split("::")[0], r.get("actor"), r.get("status"))
        a = agg.setdefault(key, {"n": 0, "seconds": 0.0})
        a["n"] += 1
        a["seconds"] += float(r.get("duration_s") or 0)
    rows = [[kind, step, actor, status, v["n"], round(v["seconds"], 1) or ""]
            for (kind, step, actor, status), v in sorted(agg.items())]
    lines += _table(["kind", "step", "actor", "status", "n", "seconds"], rows)
    lines += ["", f"*{len(recs)} record(s) in the ledger, grouped by step.*"]
    return lines


def section_hypotheses(b: dict) -> list[str]:
    """one block per task family that produced evidence; one line for each that did not."""
    lines = [_h(2, "4 · per question")]
    tasks = ((b.get("pack") or {}).get("tasks") or {})
    if not tasks:
        return lines + _absent("evidence", "no pack - this cohort is not routable")
    for task_id, t in tasks.items():
        lines += ["", _h(3, task_id)]
        if t.get("outcome") == "not_available":
            lines += _absent(task_id, t.get("why", ""))
            continue
        lines += _kv([("asks", t.get("metric")),
                      ("measured on", t.get("source")),
                      ("unit", t.get("unit")),
                      ("floor", f"{(t.get('floor') or {}).get('id')} = "
                                f"{(t.get('floor') or {}).get('value')}"
                                if t.get("floor") else None)])
        guards = [g for g in (t.get("guards") or []) if g.get("contraindicated")]
        if guards:
            lines += ["", f"guard: {guards[0].get('guard')} contraindicates "
                          f"{len(guards)} of {len(t.get('guards', []))} candidates"]
        cands = t.get("candidates") or []
        if cands:
            lines += ["", *_table(
                ["method", "value", "interval", "clears floor", "source"],
                [[c.get("id"), c.get("value"),
                  f"{c['ci'][0]} - {c['ci'][1]}" if c.get("ci") else "",
                  "" if c.get("clears_floor") is None else ("yes" if c["clears_floor"] else "no"),
                  Path(c["source"]).name if c.get("source") else ""]
                 for c in cands])]
        ref = t.get("reference")
        if ref:
            # a pack may carry one reference or several. both shapes are read,
            # because the renderer describes what a cohort recorded rather than
            # requiring it to have recorded it one way.
            rows = ref if isinstance(ref, list) else [ref]
            lines += ["", "reference (reported, not decisive):",
                      *_table(["id", "value"],
                              [[r.get("id"), r.get("value")] for r in rows
                               if isinstance(r, dict)])]
    return lines


def section_matrix(b: dict) -> list[str]:
    """question x method: the routable answer per family, from the pack."""
    lines = [_h(2, "5 · the matrix")]
    tasks = ((b.get("pack") or {}).get("tasks") or {})
    if not tasks:
        return lines + _absent("matrix", "no pack")
    rows = []
    for task_id, t in tasks.items():
        # a pack written before promote existed carries evidence but no outcome.
        # the renderer says which is missing rather than deriving it: computing
        # an outcome here would make this a second source for the decision.
        outcome = t.get("outcome") or "not_derived"
        if outcome == "not_available":
            rows.append([task_id, outcome, "-", "-", (t.get("why") or "")[:60]])
            continue
        answer = t.get("winner") or ", ".join(t.get("tied", [])) or "-"
        best = (t.get("candidates") or [{}])[0]
        note = t.get("refusal") or ("no outcome recorded - run promote to derive it"
                                    if outcome == "not_derived" else "")
        rows.append([task_id, outcome, answer, best.get("value"), note[:60]])
    lines += _table(["question", "outcome", "method", "value", "note"], rows)
    lines += ["", "*outcome vocabulary is the routing contract's: recommend, tie, "
                  "refuse, not_available. a refusal is an answer.*"]
    return lines


def section_notes(b: dict) -> list[str]:
    """the human fields, verbatim and marked as such."""
    lines = [_h(2, "6 · caveats and notes")]
    tasks = ((b.get("pack") or {}).get("tasks") or {})
    rows = []
    for task_id, t in tasks.items():
        for field in ("note", "caveat", "interpretation"):
            if t.get(field):
                rows.append([task_id, field, t[field]])
    if not rows:
        return lines + _absent("notes", "no human fields in the pack")
    lines += _table(["question", "field", "text (authored)"], rows)
    lines += ["", "*authored by a person at the promote gate. every other number in "
                  "this report was computed from an artifact.*"]
    return lines


def section_provenance(b: dict) -> list[str]:
    lines = [_h(2, "7 · provenance")]
    pack = b.get("pack") or {}
    prov = b.get("provenance") or {}
    lines += _kv([("pack version", pack.get("evidence_version")),
                  ("runs root", pack.get("runs_root")),
                  ("families not declared", ", ".join(pack.get("families_not_declared", []))),
                  ("records", len(b.get("records") or []) or None),
                  ("package version", b.get("version"))])
    if prov:
        rows = [[k, ", ".join(v.get("computed", [])), ", ".join(v.get("human", []))]
                for k, v in prov.items()]
        lines += ["", *_table(["question", "computed fields", "authored fields"], rows)]
    return lines


SECTIONS = (section_cohort, section_admissibility, section_computed,
            section_hypotheses, section_matrix, section_notes, section_provenance)


def render(bundle: dict, title: str = "evaluation report") -> str:
    """the whole report. one heading, then every section that has an input."""
    out = [_h(1, title), ""]
    for fn in SECTIONS:
        out += fn(bundle)
        out += [""]
    return "\n".join(out).rstrip() + "\n"
