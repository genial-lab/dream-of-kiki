"""End-to-end tests for P_min profile (replay + downscale + swap)."""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from harness.benchmarks.retained.retained import RetainedBenchmark
from kiki_oniric.dream.swap import SwapAborted, SwapResult
from kiki_oniric.profiles.p_min import PMinProfile


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


def _always_correct_predictor(item: dict) -> str:
    return item["expected"]


def _always_wrong_predictor(item: dict) -> str:
    return "WRONG"


def test_p_min_swap_now_commits_on_clean_state() -> None:
    """swap_now should succeed when retained eval matches pre-acc."""
    model = TinyMLP()
    profile = PMinProfile(model=model)

    benchmark = RetainedBenchmark(
        items=[{"id": "r-0", "context": "c", "expected": "y",
                "domain": "d"}],
        hash_verified=True,
        source_hash="0" * 64,
    )

    result = profile.swap_now(
        retained_pre_acc=1.0,
        benchmark=benchmark,
        model_predictor=_always_correct_predictor,
        delta_regression=0.02,
    )

    assert isinstance(result, SwapResult)
    assert result.committed is True
    assert result.retained_post_acc == 1.0


def test_p_min_swap_now_aborts_on_degraded_retained() -> None:
    """swap_now should abort with SwapAborted when accuracy crashes."""
    model = TinyMLP()
    profile = PMinProfile(model=model)

    benchmark = RetainedBenchmark(
        items=[
            {"id": "r-0", "context": "c", "expected": "y", "domain": "d"},
            {"id": "r-1", "context": "c2", "expected": "y2", "domain": "d"},
        ],
        hash_verified=True,
        source_hash="0" * 64,
    )

    with pytest.raises(SwapAborted, match="S1"):
        profile.swap_now(
            retained_pre_acc=1.0,
            benchmark=benchmark,
            model_predictor=_always_wrong_predictor,
            delta_regression=0.02,
        )
