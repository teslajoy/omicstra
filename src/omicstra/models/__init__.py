"""frozen encoders, resolved by the name a cohort declares.

nothing here fine-tunes anything. an encoder is a function from a batch of
inputs to a batch of vectors, plus the metadata needed to decide whether it may
be used on a given platform at all.

why a registry rather than an import
------------------------------------
`project.json` names an encoder as a string. if the package imported Virchow2
directly, "pluggable by declaration" would be false - adding a modality would
mean editing package code, which MODALITY_TEMPLATE.md says is a bug. so the name
is looked up here, and a cohort that declares an unknown one gets a refusal that
names what IS available rather than an ImportError.

why the heavy imports are inside the functions
----------------------------------------------
`pip install omicstra` must keep serving, routing and gating without torch. this
module is importable with no tensor library present; only calling `load()`
requires one, and it says so when it is missing.
"""
from __future__ import annotations

from .encoders import (
    EncoderSpec,
    EncoderUnavailable,
    available,
    load,
    register,
    spec,
)

__all__ = ["EncoderSpec", "EncoderUnavailable", "available", "load", "register", "spec"]