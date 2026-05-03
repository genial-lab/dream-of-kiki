"""TDD for kiki_oniric/substrates/wake_sleep_cl_baseline.py.

The baseline is an *adapter*, not a DR-3 substrate. It must :
1. expose a stable name + version,
2. round-trip seeds deterministically (R1 contract),
3. emit a CellResult-shaped dict scoring on M1.a + M1.b,
4. NOT register as a 4-op substrate (`substrate_components()`
   helper must omit the op-factory keys).

Reference :
  docs/superpowers/plans/2026-05-02-wake-sleep-cl-ablation-baseline.md
  Task 3.
"""
from __future__ import annotations

import pytest

from kiki_oniric.substrates.wake_sleep_cl_baseline import (
    WAKE_SLEEP_BASELINE_NAME,
    WAKE_SLEEP_BASELINE_VERSION,
    WakeSleepCLBaseline,
    wake_sleep_substrate_components,
)


def test_name_and_version_are_pinned() -> None:
    assert WAKE_SLEEP_BASELINE_NAME == "wake_sleep_cl_baseline"
    # Version must lift to the post-bump tag (Plan E).
    assert WAKE_SLEEP_BASELINE_VERSION == "C-v0.12.0+PARTIAL"


def test_evaluate_continual_returns_M1ab_keys() -> None:
    bl = WakeSleepCLBaseline()
    out = bl.evaluate_continual(seed=42, task_split="cifar10_5tasks_buffer500")
    assert {"forgetting_rate", "avg_accuracy", "n_tasks", "seed"} <= out.keys()
    assert out["seed"] == 42
    forgetting = float(out["forgetting_rate"])
    avg_acc = float(out["avg_accuracy"])
    assert 0.0 <= forgetting <= 1.0
    assert 0.0 <= avg_acc <= 1.0


def test_evaluate_continual_is_deterministic_per_seed() -> None:
    """R1 contract : same (seed, task_split) -> same numbers."""
    bl = WakeSleepCLBaseline()
    a = bl.evaluate_continual(seed=42, task_split="cifar10_5tasks_buffer500")
    b = bl.evaluate_continual(seed=42, task_split="cifar10_5tasks_buffer500")
    assert a == b


def test_evaluate_continual_seed_round_trips() -> None:
    bl = WakeSleepCLBaseline()
    a = bl.evaluate_continual(seed=42, task_split="cifar10_5tasks_buffer500")
    b = bl.evaluate_continual(seed=7, task_split="cifar10_5tasks_buffer500")
    # Numbers may match by coincidence (variant c) but seeds must
    # round-trip in output.
    assert a["seed"] == 42
    assert b["seed"] == 7


def test_unknown_task_split_raises() -> None:
    bl = WakeSleepCLBaseline()
    with pytest.raises(ValueError, match="task_split"):
        bl.evaluate_continual(seed=42, task_split="unsupported_xyz")


def test_components_helper_omits_op_factories() -> None:
    """The baseline is NOT DR-3 conformant ; no op-factory keys."""
    comps = wake_sleep_substrate_components()
    for op in ("replay", "downscale", "restructure", "recombine"):
        assert op not in comps
    assert "evaluate_continual" in comps
    assert "predictor" in comps


def test_source_flag_marks_published_reference() -> None:
    """Variant c must surface the published-reference flag in the output."""
    bl = WakeSleepCLBaseline()
    out = bl.evaluate_continual(seed=42, task_split="cifar10_5tasks_buffer500")
    assert "source" in out
    assert "published_reference" in str(out["source"])
    assert "alfarano" in str(out["source"]).lower()


def test_n_tasks_default_is_five() -> None:
    bl = WakeSleepCLBaseline()
    out = bl.evaluate_continual(seed=42, task_split="cifar10_5tasks_buffer500")
    assert out["n_tasks"] == 5
