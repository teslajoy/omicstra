"""seed resolution and application.

the seed is DECLARED in project.json, CARRIED in run config and provenance,
and PASSED to each stochastic operation. global RNG state is not inherited
across process boundaries, resumed checkpoints, or worker dispatch - so
every callsite that needs determinism receives the seed explicitly.

operation-specific seeds (e.g. PCA random_state=42 in historical runs) are
preserved during reproduction; the project seed governs new work.
"""
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """set global RNG state for the current process.

    call at process entry - graph launch, worker start, or resume. does
    not propagate to subprocesses or other workers; those receive the
    seed through their run config and call this themselves.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass