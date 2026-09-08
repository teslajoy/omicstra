"""reads of the EDA contract, a cohort's calibration, and its gate summary.

the same split as routing: `configs/eda_contract.json` ships with the PACKAGE
and declares which checks exist and what each is judged against.
`{project_root}/eda_calibration.json` and `eda_summary.json` belong to the
PROJECT.

deciding what those add up to is `evaluate()` in `omicstra.eda`, and asking a
person about it is `graphs/eda.py`. this file only reads.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omicstra.settings import settings


AUTHORITIES = ("universal", "cohort_calibrated", "advisory")


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


def load_summary(project_id: str | None = None) -> dict:
    """the summary belongs to the PROJECT - resolved through the boundary."""
    p = settings.project_root(project_id) / "eda_summary.json"
    if not p.exists():
        raise FileNotFoundError(
            f"no eda_summary.json in {p.parent} - the EDA gate has not been run for this cohort"
        )
    return json.loads(p.read_text())
