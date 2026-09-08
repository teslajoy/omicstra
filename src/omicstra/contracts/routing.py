"""reads of the routing contract and a cohort's evidence. no decision.

the split this file exists to hold: `configs/routing_contract.json` ships with
the PACKAGE - the task taxonomy, the outcome vocabulary, the rules. it names no
cohort. `{project_root}/routing_evidence.json` belongs to the PROJECT and holds
what that cohort measured.

an absent evidence file must NOT fall back to a default table. inheriting
another cohort's winner is the exact failure the split exists to stop, so the
loader returns nothing and the caller reports "not routable" rather than
answering from someone else's numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

from omicstra.records import RecordStore
from omicstra.settings import settings


def load_routing_contract(configs_dir: str | Path | None = None) -> dict:
    """ships with the PACKAGE - cohort-free by construction."""
    base = Path(configs_dir) if configs_dir else settings.configs_dir
    return json.loads((settings.resolve(base) / "routing_contract.json").read_text())


def load_routing_evidence(project_id: str | None = None) -> dict:
    """belongs to the PROJECT. absent = this cohort has not been evaluated.

    an absent file must not fall back to a default routing table. inheriting
    another cohort's winner is the exact failure this split exists to stop.
    """
    try:
        p = settings.project_root(project_id) / "routing_evidence.json"
    except ValueError:
        return {}
    return json.loads(p.read_text()) if p.exists() else {}


def ledger(project_id: str | None = None) -> RecordStore:
    """the file-backed decision ledger for a cohort."""
    try:
        pid = project_id or settings.project_root(project_id).name
    except ValueError:
        pid = "unrouted"
    return RecordStore(settings.resolve(settings.runs_dir) / pid / "records" / "decisions.jsonl")


