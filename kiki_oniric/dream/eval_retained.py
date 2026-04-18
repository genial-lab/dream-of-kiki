"""Retained benchmark evaluation bridge — S1 guard input.

Connects `harness.benchmarks.retained.RetainedBenchmark` to the
`swap_atomic` S1 (retained non-regression) guard. Substrate-agnostic
by design: `model` is any callable `(item: dict) -> str` that
predicts the expected outcome for a benchmark item.

For MLX-backed models, wrap the model invocation in a lambda that
projects predictions to strings — see S9.4 (P_min E2E) for the
MLX-specific wrapper.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §5.2
Invariant S1 — BLOCKING.
"""
from __future__ import annotations

from typing import Callable

from harness.benchmarks.retained.retained import RetainedBenchmark


ItemPredictor = Callable[[dict], str]


def evaluate_retained(
    model: ItemPredictor,
    benchmark: RetainedBenchmark,
) -> float:
    """Compute accuracy of `model` over `benchmark.items` in [0, 1].

    Returns 1.0 for an empty benchmark (vacuous pass — common when
    the retained set is being constructed). Otherwise returns
    fraction of items where `model(item) == item["expected"]`.

    Designed to be passed as the `retained_eval` callable in
    `swap_atomic`. The closure over the model is the caller's
    responsibility (see S9.4 for the canonical pattern).
    """
    items = benchmark.items
    if not items:
        return 1.0

    correct = sum(
        1 for item in items if model(item) == item["expected"]
    )
    return correct / len(items)
