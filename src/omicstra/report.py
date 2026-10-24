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


def gather(root: Path, pack_name: str = "routing_evidence.json",
           ledger_path: Path | None = None) -> dict:
    """read what a cohort has, skip what it does not. absence is not an error.

    the LEDGER is read too, which it was not: this file's own first line says
    "a pack and a ledger in", and for a while only the pack arrived. so a report
    stated what a cohort measured and nothing about who decided what on the way -
    which gates were answered, by which actor, on which executor.

    `ledger_path` is for a caller that holds one elsewhere; by default it is the
    cohort's own `runs/{id}/records/decisions.jsonl`, the same file the arms
    append to.
    """
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

    lp = Path(ledger_path) if ledger_path is not None else _ledger_path(root, out)
    if lp is not None and lp.is_file():
        from omicstra.records import RecordStore

        store = RecordStore(lp)
        out["records"] = [r.model_dump(mode="json") for r in store.all()]
        out["actor_ratio"] = store.actor_ratio()
        out["ledger"] = str(lp)
    return out


def _ledger_path(root: Path, bundle: dict) -> Path | None:
    """the cohort's own ledger, resolved the way the arms resolve it.

    it is NOT under the project root. `runs/` is settings-resolved, and a cohort
    whose declaration points its runs root outside its own directory keeps its
    ledger there - so this asks settings rather than joining a path onto `root`.
    """
    from omicstra.settings import settings

    pid = ((bundle.get("project") or {}).get("project_id")
           or (bundle.get("cohort") or {}).get("project_id") or root.name)
    try:
        return settings.work_root(pid) / "records" / "decisions.jsonl"
    except (ValueError, OSError):
        return None


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
    platforms = list(plat.get("platforms") or {})
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


def _field(r: dict, *names: str) -> str:
    """a record's value for the first of `names` that is set, params included.

    `records.coerce` puts any key a Record does not declare into `params`, which
    is the field that exists for exactly that - so a reader that looked only at
    the top level saw an empty cause on every record an arm wrote.
    """
    params = r.get("params") or {}
    for n in names:
        v = r.get(n) if r.get(n) not in (None, "") else params.get(n)
        if v not in (None, ""):
            return str(v)
    return ""


def _declared_step_ids() -> tuple[set[str], set[str]]:
    """the inventory steps and EDA checks the CONTRACTS declare.

    read from the contracts rather than matched by shape, so a record is
    classified as an inventory step or an EDA check because something declared
    it one - not because its id happened to look like one.
    """
    from omicstra.contracts.eda import load_contract
    from omicstra.protocols.inventory import INVENTORY_STEPS

    checks = load_contract()["checks"]
    ids = {c["id"] for c in (checks if isinstance(checks, list) else checks.values())}
    return {s.id for s in INVENTORY_STEPS}, ids


def _measured_steps(b: dict) -> list[str]:
    """what the chain measured THIS run, and where it disagrees with the summary.

    section 2 reports what `eda_summary.json` DECLARES, which on a curated cohort
    is hand-written and says so. this reports what the chain computed, from the
    ledger - so the two can be compared rather than one standing for both.

    a refusal is a result here, not an omission: a step that could not run names
    what it needed, which is the only thing that distinguishes "measured and
    found nothing" from "never measured".
    """
    recs = b.get("records") or []
    if not recs:
        return []
    try:
        inv_ids, eda_ids = _declared_step_ids()
    except Exception:                                    # noqa: BLE001
        return []

    out: list[str] = []
    for title, ids in (("inventory", inv_ids), ("EDA", eda_ids)):
        rows = [[r.get("step_id"), r.get("status"),
                 _field(r, "result", "why_not_applicable", "decision", "reason")[:68]]
                for r in recs if r.get("step_id") in ids]
        if rows:
            out += ["", f"**{title}, as this run measured it**", "",
                    *_table(["step", "status", "result"], rows)]

    refused = [[r.get("step_id"), r.get("status"),
                _field(r, "reason", "why", "decision", "result")[:68]]
               for r in recs
               if r.get("status") in ("not_run", "error", "fail")
               and not str(r.get("step_id", "")).startswith("verdict::")]
    if refused:
        out += ["", "**refused, with the cause**", "",
                *_table(["step", "status", "cause"], refused)]

    # the drift that matters: a declared verdict standing over a computed one.
    eda = b.get("eda") or {}
    declared = eda.get("verdict")
    declared = declared if isinstance(declared, str) else (declared or {}).get("verdict")
    computed = next((_field(r, "result") for r in recs
                     if r.get("step_id") == "eda_gate"), None)
    if declared and computed and declared != computed:
        out += ["", (f"**the summary declares `{declared}` and this run computed "
                     f"`{computed}`.** a declared verdict is a person's judgement "
                     "that the cautions were acceptable; it does not make the two "
                     "agree, and the difference travels with every number below.")]
    return out


def section_findings(b: dict) -> list[str]:
    """COMPUTED. what this cohort's own chain measured, pack or no pack.

    the report takes a pack and a ledger, and for a while only the pack reached
    its substance - so a cohort that had just measured everything rendered as
    "not routable" because nobody had curated it yet. these rows come from the
    verdict records the evaluate stage writes, which means they appear for a
    cohort with no pack at all. that is the whole point.
    """
    lines = [_h(2, "4 · measured on this cohort")]
    recs = b.get("records") or []
    if not recs:
        return lines + _absent("findings", "nothing in this cohort's ledger")
    v = [r for r in recs if str(r.get("step_id", "")).startswith("verdict::")]

    # the hypothesis verdict is ONE block of this section, not its gate. a
    # cohort that has run its inventory and its EDA and nothing else has
    # findings, and reporting none because no hypothesis was evaluated is the
    # same mistake this whole section was added to fix.
    rows = []
    for r in sorted(v, key=lambda r: r["step_id"]):
        g = r.get("guards") or []
        failed = [x["id"] for x in g if not x.get("passed")]
        o = r.get("observed") or {}
        miss = ((o.get("rollup") or {}).get("coverage") or {}).get("missing") or []
        rows.append([r["step_id"].split("::")[-1], r.get("result"),
                     f"{len(g) - len(failed)}/{len(g)}" if g else "none ran",
                     ", ".join(sorted({f.split("/")[-1] for f in failed}))[:34],
                     ", ".join(miss) or "complete"])
    if rows:
        lines += _table(["hypothesis", "claim", "guards passed", "failed",
                         "arms not scored"], rows)
    lines += ["", ("*computed from the artifacts this cohort's declaration names. "
                   "`diagnostic` means a required guard ran and failed, so the "
                   "number stands and may not be cited as biology alone; "
                   "`unguarded` means nothing checked it.*")] if rows else []

    lines += _measured_steps(b)

    # R3: the pack routes numbers. a routed number whose hypothesis is not clean
    # is the disagreement worth surfacing, and it is the one a reader of the
    # pack alone cannot see.
    claims = {r["step_id"].split("::")[-1]: r.get("result") for r in v}
    unclean = {h for h, c in claims.items() if c in ("diagnostic", "unguarded")}
    routed = [t for t, d in (((b.get("pack") or {}).get("tasks")) or {}).items()
              if d.get("candidates")]
    if unclean and routed:
        lines += ["", (f"**the pack routes {len(routed)} question(s) while "
                       f"{', '.join(sorted(unclean))} "
                       f"{'is' if len(unclean) == 1 else 'are'} not a clean "
                       "result.** a routed answer inherits its hypothesis's claim, "
                       "and the pack does not restate it.")]
    return lines


def section_hypotheses(b: dict) -> list[str]:
    """one block per task family that produced evidence; one line for each that did not."""
    lines = [_h(2, "5 · per question")]
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
            lines += ["", (f"guard: {guards[0].get('guard')} contraindicates "
                           f"{len(guards)} of {len(t.get('guards', []))} candidates")]
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
    """question x METHOD, every cell carrying whether it may be compared.

    this was question x WINNER - one row per question, the top value. so a reader
    saw a ranked list per question with nothing marking the entries that are not
    comparable: a method contraindicated for that question, or one that does not
    beat the baseline the hypothesis is stated against. "cell" here is a cell of
    the TABLE, never a biological one. the admissibility is computed in
    `protocols.evaluate`; nothing is decided here.
    """
    lines = [_h(2, "6 · the matrix")]
    pack = b.get("pack") or {}
    if not (pack.get("tasks") or {}):
        return lines + _absent("matrix", "no pack")

    from omicstra.protocols.evaluate import matrix

    claims = {r["step_id"].split("::")[-1]: r.get("result")
              for r in (b.get("records") or [])
              if str(r.get("step_id", "")).startswith("verdict::")}
    m = matrix(pack, claims)

    mark = {"admissible": "", "contraindicated": "✗ contraindicated",
            "below_floor": "✗ below floor"}
    for task_id, row in m.items():
        if row["status"] != "scored":
            lines += ["", _h(3, task_id),
                      *_absent(task_id, row.get("why") or "no candidates recorded")]
            continue
        head = f"{task_id} · {row['hypothesis']}"
        if row.get("claim") and row["claim"] != "biological":
            head += f" · {row['claim']}"
        lines += ["", _h(3, head)]
        # the pack's OUTCOME is reported, never derived here: a renderer that
        # computed one would be a second source for the decision, and two
        # sources for one decision is how two reports disagree.
        t = (pack.get("tasks") or {})[task_id]
        outcome = t.get("outcome") or "not_derived"
        f = row.get("floor") or {}
        lines += _kv([("outcome", outcome),
                      ("leads", t.get("winner") or ", ".join(t.get("tied", [])) or None),
                      ("floor", f"{f.get('id')} = {f.get('value')}" if f else None),
                      ("comparable", f"{row['n_rankable']} of {row['n_cells']}")])
        if outcome == "not_derived":
            lines += ["", "*no outcome recorded - run promote to derive it.*"]
        lines += ["", *_table(
            ["method", "value", "admissible", "why not"],
            [[c["method"], c["value"], mark.get(c["status"], c["status"]) or "yes",
              (c["why"] or "")[:72]] for c in row["cells"]])]
        if row.get("claim") in ("diagnostic", "unguarded", "not_run"):
            lines += ["", (f"*{row['hypothesis']} is `{row['claim']}` on this cohort, "
                           "so every cell above inherits that caveat equally.*")]

    lines += ["", ("*each entry above - one question crossed with one method, not "
                   "a biological cell - is a value AND whether it may be compared. "
                   "a contraindicated method is not ranked; the reason stands where "
                   "its number would. a method below the floor has not answered the "
                   "question, whatever its rank, because every hypothesis here is "
                   "stated against a baseline.*")]
    return lines


def section_notes(b: dict) -> list[str]:
    """the human fields, verbatim and marked as such."""
    lines = [_h(2, "7 · caveats and notes")]
    tasks = ((b.get("pack") or {}).get("tasks") or {})
    rows = []
    for task_id, t in tasks.items():
        for field in ("note", "caveat", "interpretation"):
            if t.get(field):
                rows.append([task_id, field, t[field]])
    if not rows:
        return lines + _absent("notes", "no human fields in the pack")
    lines += _table(["question", "field", "text (authored)"], rows)
    lines += ["", ("*authored by a person at the promote gate. every other number in "
                   "this report was computed from an artifact.*")]
    return lines


def section_decisions(b: dict) -> list[str]:
    """the ledger. who decided what, and how much of it was a rule.

    the reproducibility claim is a number rather than a claim: `actor_ratio` is
    the count per actor, and `client` is deliberately not `human` - an answer
    arrived through the surface and the server observed a tool call, not a
    person.
    """
    lines = [_h(2, "8 · decisions")]
    recs = b.get("records") or []
    if not recs:
        return lines + _absent("decisions", "this cohort's ledger is empty or absent")
    ratio = b.get("actor_ratio") or {}
    lines += _kv([("decisions", len(recs)),
                  *[(f"by {a}", n) for a, n in ratio.items() if n],
                  ("ledger", b.get("ledger"))])
    rows = [[r.get("step_id"), r.get("kind"), r.get("status"), r.get("actor"),
             (r.get("params") or {}).get("executor") or r.get("executor") or ""]
            for r in recs[-20:]]
    lines += ["", *_table(["step", "kind", "status", "actor", "executor"], rows)]
    if len(recs) > 20:
        lines += ["", f"*the last 20 of {len(recs)}. the ledger is the whole record.*"]
    return lines


def section_provenance(b: dict) -> list[str]:
    lines = [_h(2, "9 · provenance")]
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
            section_findings, section_hypotheses, section_matrix, section_notes,
            section_decisions, section_provenance)


def render(bundle: dict, title: str = "evaluation report") -> str:
    """the whole report. one heading, then every section that has an input."""
    out = [_h(1, title), ""]
    for fn in SECTIONS:
        out += fn(bundle)
        out += [""]
    return "\n".join(out).rstrip() + "\n"
