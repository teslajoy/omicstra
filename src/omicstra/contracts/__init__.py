"""reads of the declared files. no logic, no decisions.

a contract module answers "what did someone declare", never "what follows from
it". the package contracts (`configs/*.json`) say which checks exist and what
each is judged against; a cohort's files (`project.json`, `cohort.json`,
`platform.json`, `routing_evidence.json`) say what this cohort declared and
measured.

deciding is one level up: `graphs/` asks people, `protocols/` runs methods.
keeping the read here is what lets a caller load a declaration without pulling
in the machinery that acts on it - which is also why `settings.py` stays below
this and imports nothing from omicstra at all.
"""
from omicstra.contracts.project import EncoderSpec, ProjectConfig

__all__ = ["EncoderSpec", "ProjectConfig"]
