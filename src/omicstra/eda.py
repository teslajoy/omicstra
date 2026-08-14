"""the EDA gate - validator.

evaluates a cohort's eda_summary.json against configs/eda_contract.json and emits
a verdict with per-check results. read-only: no data access, no R, no GPU.

the contract is generalizable (thresholds + verdict logic, no cohort numbers);
the summary is cohort-specific. a new cohort is evaluated by the same contract.

contract 1.1 reports `learned_checks` as advisories - findings the notebooks
established that EDA.md never encoded. advisories never change a verdict.

every check declares an `authority`, which says where its criterion draws its
warrant from and therefore what happens on a cohort that did not produce it:

    universal          apply silently
    cohort_calibrated  escalate on a cohort that has not derived it. do not
                       inherit the default, do not substitute one
    advisory           report, never gate

the calibrated values themselves stay in the package as defaults; the warrant
for them lives in the cohort's own eda_calibration.json, because the warrant is
the thing a second cohort must not inherit.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from omicstra.settings import settings

Status = Literal["pass", "fail", "not_run", "not_applicable", "escalate"]
Verdict = Literal["proceed", "proceed_with_caution", "stop"]
Authority = Literal["universal", "cohort_calibrated", "advisory"]

AUTHORITIES = ("universal", "cohort_calibrated", "advisory")
CALIBRATED_STATUSES = ("derived", "adopted")


class CheckResult(BaseModel):
    id: str
    status: Status
    required: bool
    authority: str = ""
    observed: Any = None
    note: str = ""
    provenance: str = ""


class GateReport(BaseModel):
    project_id: str
    verdict: Verdict
    declared_verdict: str | None = None
    agrees_with_declared: bool = True
    checks: list[CheckResult] = []
    violations: list[str] = []
    cautions: list[str] = []
    advisories: list[str] = []
    escalations: list[dict] = []

    @property
    def passed(self) -> bool:
        return self.verdict != "stop"


def _declared_authority(spec: dict, where: str) -> str:
    """a check with no authority is a load error, not a universal check.

    missing must not mean permissive - that is the whole point of the field. an
    unclassified threshold is exactly the one that gets silently inherited.
    """
    a = spec.get("authority")
    if a not in AUTHORITIES:
        raise ValueError(
            f"{where} {spec.get('id', '?')!r} declares authority {a!r}; expected one of "
            f"{', '.join(AUTHORITIES)}. a check with no declared authority is not "
            "assumed universal - classify it in configs/eda_contract.json"
        )
    return a


def load_contract(configs_dir: str | Path | None = None) -> dict:
    """the contract ships with the PACKAGE - it is cohort-free by construction."""
    base = Path(configs_dir) if configs_dir else settings.configs_dir
    contract = json.loads((settings.resolve(base) / "eda_contract.json").read_text())
    for spec in contract.get("checks", []):
        _declared_authority(spec, "check")
    for item in contract.get("learned_checks", {}).get("items", []):
        _declared_authority(item, "learned check")
    return contract


def load_calibration(project_id: str | None = None) -> dict:
    """which cohort_calibrated values THIS cohort derived, and on what evidence.

    belongs to the PROJECT, never the package. an absent file is the honest
    default for a new cohort: nothing is calibrated here, so every
    cohort_calibrated check escalates rather than inheriting a number.
    """
    try:
        p = settings.project_root(project_id) / "eda_calibration.json"
    except ValueError:
        return {}  # no cohort selected -> nothing is calibrated -> escalate
    return json.loads(p.read_text()) if p.exists() else {}


def _calibration_entry(calibration: dict, check_id: str) -> dict | None:
    entry = (calibration.get("calibrated") or {}).get(check_id)
    if not isinstance(entry, dict):
        return None
    return entry if entry.get("status") in CALIBRATED_STATUSES else None


def escalation(spec: dict, project_id: str = "") -> dict:
    """the payload for a calibrated criterion this cohort has not derived.

    same shape as the route_question tie: candidates, evidence, and what the
    choice forecloses. `no_default` is not decoration - the system declines to
    pick, and a payload that suggested one would defeat the escalation.
    """
    default = {k: v for k, v in spec.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return {
        "check": spec["id"],
        "authority": "cohort_calibrated",
        "question": f"{spec['id']}: this criterion was calibrated on another cohort. "
                    f"derive it here, adopt it deliberately, or stop?",
        "cohort": project_id,
        "package_default": default,
        "why_not_inherited": spec.get("authority_basis", "provenance unknown"),
        "consequence": spec.get("forecloses", ""),
        "no_default": "the system does not substitute a value",
        "resolve_by": "record the decision in {project_root}/eda_calibration.json",
    }


def cohort_escalations(contract: dict, calibration: dict, project_id: str = "",
                       only: set[str] | None = None) -> list[dict]:
    """every cohort_calibrated check this cohort has no calibration record for."""
    ids = only
    return [escalation(spec, project_id) for spec in contract["checks"]
            if spec["authority"] == "cohort_calibrated"
            and (ids is None or spec["id"] in ids)
            and _calibration_entry(calibration, spec["id"]) is None]


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


def evaluate(summary: dict, contract: dict, project_id: str = "",
             calibration: dict | None = None) -> GateReport:
    checks: list[CheckResult] = []
    violations: list[str] = []
    cautions: list[str] = []
    escalations: list[dict] = []
    advisory_notes: list[str] = []
    calibration = calibration or {}

    for spec in contract["checks"]:
        field = spec["field"]
        present = field in summary
        value = summary.get(field)
        authority = spec["authority"]
        entry = _calibration_entry(calibration, spec["id"])

        # a calibrated criterion on a cohort that has not derived it is not
        # evaluated at all. substituting the package default here is exactly the
        # silent inheritance this field exists to stop.
        if authority == "cohort_calibrated" and entry is None:
            esc = escalation(spec, project_id)
            escalations.append(esc)
            checks.append(CheckResult(
                id=spec["id"], status="escalate", required=spec["required"],
                authority=authority,
                note="calibrated elsewhere and not derived here - escalated, not evaluated"))
            cautions.append(f"{spec['id']}: escalated - {esc['why_not_inherited']}")
            continue

        provenance = ""
        if entry is not None:
            provenance = f"{entry.get('status', '')}: {entry.get('provenance', '')}".strip(": ")
            if entry.get("scope"):
                provenance += f" [scope: {entry['scope']}]"

        # advisory informs, it never gates. its outcome reaches the reader
        # through advisories[] and goes no further - a verdict never moves on it.
        gates = authority != "advisory"

        # a conditional check whose field is explicitly null = not applicable,
        # but EDA.md does not distinguish that from "not measured". say so.
        if not spec["required"] and value is None:
            note = ("null - cannot distinguish 'not applicable' from 'not measured'; "
                    "EDA.md schema has no marker for this")
            checks.append(CheckResult(id=spec["id"], status="not_applicable",
                                      required=False, authority=authority,
                                      observed=None, note=note))
            (cautions if gates else advisory_notes).append(f"{spec['id']}: {note}")
            continue

        if not present:
            checks.append(CheckResult(id=spec["id"], status="not_run",
                                      required=spec["required"], authority=authority,
                                      note=f"field '{field}' missing from summary"))
            note = f"{spec['id']}: field '{field}' missing"
            if not gates:
                advisory_notes.append(note)
            else:
                (violations if spec["required"] else cautions).append(note)
            continue

        status, note = _apply(spec["rule"], value, spec)
        checks.append(CheckResult(id=spec["id"], status=status,
                                  required=spec["required"], authority=authority,
                                  observed=value if not isinstance(value, dict) else None,
                                  note=note, provenance=provenance))

        if not gates:
            if note or status != "pass":
                advisory_notes.append(f"{spec['id']}: {status}" + (f" - {note}" if note else ""))
            continue
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
        advisories=advisory_notes + _advisories(summary, contract),
        escalations=escalations,
    )


def run_gate(project_id: str | None = None,
             configs_dir: str | Path | None = None) -> GateReport:
    summary = load_summary(project_id)
    resolved = project_id or summary.get("project_id") or settings.project_root(project_id).name
    return evaluate(summary, load_contract(configs_dir), resolved,
                    calibration=load_calibration(project_id))