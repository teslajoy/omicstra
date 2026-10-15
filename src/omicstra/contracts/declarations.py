"""a cohort's declarations, merged across the study it belongs to.

a STUDY groups projects the way GDC/Gen3 nests project under program, and most
of what a project would otherwise repeat belongs to the study: the platform, the
encoder set, the classification, the subject-column convention, the split
policy. a project overrides only what actually differs.

that turns this package's first rule - declared, not assumed; absent is a
refusal - into a TWO-LEVEL resolution:

    project level present    use it
    study level present      use it
    neither                  refuse, with a reason

and it makes a second thing necessary rather than optional. the ledger already
records WHERE a declared answer came from (`gate_record` writes
`source: "cohort.json#field"`), because a declaration is a person's act and
version control attests the file. with inheritance the file matters more, not
less: a study-level declaration binds every project under it, which is a bigger
act than a project-level one, and an actor ledger that could not tell them apart
would under-report what was decided.

so `resolve` returns the value AND the file it came from, and callers that
record provenance use the pair.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# project files win over study files; later wins within a level. cohort.json
# holds gate answers and data declarations, project.json the run layout - which
# is a layout decision rather than a semantic one, so a caller asking "what did
# this cohort declare" should not have to know which.
_STUDY_FILES = ("study.json",)
_PROJECT_FILES = ("cohort.json", "project.json")


def _read(p: Path) -> dict:
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def sources(project_id: str | None = None) -> list[tuple[str, Path]]:
    """every declaration file that applies, weakest first.

    the order IS the precedence, and it is returned rather than applied so a
    caller can report what was consulted even when nothing answered.
    """
    from omicstra.settings import settings

    out: list[tuple[str, Path]] = []
    try:
        root = settings.project_root(project_id)
        study = settings.study_root(project_id)
    except (ValueError, OSError):
        return out
    if study is not None:
        out += [(n, study / n) for n in _STUDY_FILES]
    out += [(n, root / n) for n in _PROJECT_FILES]
    return out


def merged(project_id: str | None = None) -> dict[str, Any]:
    """every declaration that applies, with the strongest winning."""
    out: dict[str, Any] = {}
    for _, p in sources(project_id):
        out.update(_read(p))
    return out


def resolve(field: str, project_id: str | None = None) -> tuple[Any, str | None]:
    """one declared field, and the FILE it came from.

    the file is returned rather than inferred because the ledger records it: a
    study-level answer binds every project under the study, and
    `source: "study.json#field"` says so where `source: "declared"` would not.
    """
    value, origin = None, None
    for name, p in sources(project_id):
        d = _read(p)
        if field in d and d[field] not in (None, ""):
            value, origin = d[field], f"{name}#{field}"
    return value, origin


def provenance(project_id: str | None = None) -> dict[str, str]:
    """field -> the file that last set it. the ledger's view of a cohort."""
    out: dict[str, str] = {}
    for name, p in sources(project_id):
        for k, v in _read(p).items():
            if v not in (None, ""):
                out[k] = name
    return out
