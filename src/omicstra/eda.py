"""the EDA gate - validator.

evaluates a cohort's eda_summary.json against configs/eda_contract.json and emits
a verdict with per-check results. read-only: no data access, no R, no GPU.

the contract is generalizable (thresholds + verdict logic, no cohort numbers);
the summary is cohort-specific. a new cohort is evaluated by the same contract.

contract 1.0 reports `learned_checks` as advisories - findings the tnbc-92
notebooks established that EDA.md never encoded. advisories never change a verdict.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from .settings import settings

Status = Literal["pass", "fail", "not_run", "not_applicable"]
Verdict = Literal["proceed", "proceed_with_caution", "stop"]


class CheckResult(BaseModel):
    id: str
    status: Status
    required: bool
    observed: Any = None
    note: str = ""


class GateReport(BaseModel):
    project_id: str
    verdict: Verdict
    declared_verdict: str | None = None
    agrees_with_declared: bool = True
    checks: list[CheckResult] = []
    violations: list[str] = []
    cautions: list[str] = []
    advisories: list[str] = []

    @property
    def passed(self) -> bool:
        return self.verdict != "stop"


def load_contract(configs_dir: str | Path | None = None) -> dict:
    """the contract ships with the PACKAGE - it is cohort-free by construction."""
    base = Path(configs_dir) if configs_dir else settings.configs_dir
    return json.loads((settings.resolve(base) / "eda_contract.json").read_text())


def load_summary(project_id: str | None = None) -> dict:
    """the summary belongs to the PROJECT - resolved through the boundary."""
    p = settings.project_root(project_id) / "eda_summary.json"
    if not p.exists():
        raise FileNotFoundError(
            f"no eda_summary.json in {p.parent} - the EDA gate has not been run for this cohort"
        )
    return json.loads(p.read_text())


# --- rule evaluators -------------------------------------------------------
def _apply(rule: str, value: Any, spec: dict) -> tuple[Status, str]:
    if rule == "is_true":
        if value is None:
            return "not_run", "field is null"
        return ("pass", "") if value is True else ("fail", f"expected true, got {value!r}")

    if rule == "is_false":
        if value is None:
            return "not_run", "field is null"
        return ("pass", "") if value is False else ("fail", f"expected false, got {value!r}")

    if rule == "present_and_positive":
        if value is None:
            return "not_run", "field is null"
        try:
            return ("pass", "") if float(value) > 0 else ("fail", f"got {value!r}")
        except (TypeError, ValueError):
            return "fail", f"not numeric: {value!r}"

    if rule == "resolved_string":
        if value in spec.get("unresolved_values", ["", None]):
            return "fail", "unresolved - blocking per EDA.md"
        return "pass", ""

    if rule == "all_encoders_assessed":
        if not value:
            return "not_run", "no encoders assessed"
        ok = set(spec.get("acceptable", []))
        esc = set(spec.get("escalate", []))
        bad, uncertain = [], []
        for name, info in value.items():
            fit = (info or {}).get("fit")
            if fit in esc:
                uncertain.append(f"{name}={fit}")
            elif fit not in ok:
                bad.append(f"{name}={fit}")
        if bad:
            return "fail", "unacceptable fit: " + ", ".join(bad)
        if uncertain:
            return "pass", "escalate (uncertain): " + ", ".join(uncertain)
        return "pass", ""

    return "not_run", f"unknown rule {rule!r}"


def _advisories(summary: dict, contract: dict) -> list[str]:
    """learned checks - report what the summary cannot answer. never verdict-changing."""
    out: list[str] = []
    for item in contract.get("learned_checks", {}).get("items", []):
        field = item.get("expected_field")
        if field and field not in summary:
            out.append(f"{item['id']}: absent - {item['description']}")

    # single-sample statistic: the one instance we can detect structurally
    mi = summary.get("morans_i_summary") or {}
    if mi.get("subarray") and "n_subarrays" not in mi:
        n = summary.get("samples_with_counts") or summary.get("samples")
        out.append(
            f"single_sample_statistic: Moran's I computed on '{mi['subarray']}' alone"
            + (f" of {n} samples" if n else "")
            + " - this statistic gates the molecular encoder branch"
        )
    return out


def evaluate(summary: dict, contract: dict, project_id: str = "") -> GateReport:
    checks: list[CheckResult] = []
    violations: list[str] = []
    cautions: list[str] = []

    for spec in contract["checks"]:
        field = spec["field"]
        present = field in summary
        value = summary.get(field)

        # a conditional check whose field is explicitly null = not applicable,
        # but EDA.md does not distinguish that from "not measured". say so.
        if not spec["required"] and value is None:
            note = ("null - cannot distinguish 'not applicable' from 'not measured'; "
                    "EDA.md schema has no marker for this")
            checks.append(CheckResult(id=spec["id"], status="not_applicable",
                                      required=False, observed=None, note=note))
            cautions.append(f"{spec['id']}: {note}")
            continue

        if not present:
            checks.append(CheckResult(id=spec["id"], status="not_run",
                                      required=spec["required"],
                                      note=f"field '{field}' missing from summary"))
            (violations if spec["required"] else cautions).append(
                f"{spec['id']}: field '{field}' missing")
            continue

        status, note = _apply(spec["rule"], value, spec)
        checks.append(CheckResult(id=spec["id"], status=status,
                                  required=spec["required"],
                                  observed=value if not isinstance(value, dict) else None,
                                  note=note))
        if status == "fail" and spec["required"]:
            violations.append(f"{spec['id']}: {note}")
        elif status == "not_run" and spec["required"]:
            violations.append(f"{spec['id']}: not measured ({note})")
        elif note:
            cautions.append(f"{spec['id']}: {note}")

    # unaccepted risks -> caution, per EDA.md
    risks = summary.get("risks") or []
    if risks:
        cautions.append(f"risks[]: {len(risks)} entry(s) recorded and not marked accepted")

    verdict: Verdict = ("stop" if violations
                        else "proceed_with_caution" if cautions
                        else "proceed")

    declared = summary.get("verdict")
    declared_norm = (declared or "").replace(" ", "_")

    return GateReport(
        project_id=project_id,
        verdict=verdict,
        declared_verdict=declared,
        agrees_with_declared=(declared_norm == verdict),
        checks=checks,
        violations=violations,
        cautions=cautions,
        advisories=_advisories(summary, contract),
    )


def run_gate(project_id: str | None = None,
             configs_dir: str | Path | None = None) -> GateReport:
    summary = load_summary(project_id)
    resolved = project_id or summary.get("project_id") or settings.project_root(project_id).name
    return evaluate(summary, load_contract(configs_dir), resolved)