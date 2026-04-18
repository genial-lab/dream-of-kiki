"""S4 attention prior guard — each component in [0, 1], sum bounded.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §5.2
Invariant S4 — WARN. Enforced on canal-4 (ATTENTION_PRIOR) emission.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


DEFAULT_BUDGET_ATTENTION = 1.5


class AttentionGuardError(Exception):
    """Raised when S4 invariant is violated."""


def check_attention_bounded(
    prior: NDArray,
    budget: float = DEFAULT_BUDGET_ATTENTION,
) -> None:
    """Verify attention prior is bounded per S4.

    - each component in [0, 1]
    - sum(prior) <= budget

    Raises AttentionGuardError on first violation.
    """
    arr = np.asarray(prior)
    if (arr < 0.0).any() or (arr > 1.0).any():
        raise AttentionGuardError(
            f"attention components must be in [0, 1], got "
            f"min={float(arr.min())}, max={float(arr.max())}"
        )
    total = float(arr.sum())
    if total > budget:
        raise AttentionGuardError(
            f"attention sum {total} exceeds budget {budget}"
        )
