"""Tests for the Path B inference-only adaptation shim."""
from __future__ import annotations

import numpy as np

from experiments.g6_mmlu_stream.micro_kiki_inference import (
    InferenceOnlyAdapter,
    adapt_subdomain,
)
from harness.real_benchmarks.mmlu import MMLURecord


def _record(subject: str, idx: int, gold: int = 0) -> MMLURecord:
    return MMLURecord(
        question=f"{subject}-Q{idx}?",
        choices=("A", "B", "C", "D"),
        answer=gold,
        subject=subject,
    )


def test_inference_adapter_starts_with_zero_delta() -> None:
    adapter = InferenceOnlyAdapter(
        out_dim=8, rank=4, seed=0,
    )
    delta = adapter.current_delta("layer_0_lora_B")
    assert delta.shape == (8, 4)
    assert np.allclose(delta, 0.0)


def test_adapt_subdomain_mutates_delta() -> None:
    adapter = InferenceOnlyAdapter(out_dim=8, rank=4, seed=0)
    train_records = [_record("anatomy", i) for i in range(4)]
    before = adapter.current_delta("layer_0_lora_B").copy()
    adapt_subdomain(
        adapter=adapter,
        subdomain="anatomy",
        train=train_records,
        seed=0,
    )
    after = adapter.current_delta("layer_0_lora_B")
    # Delta must change by a non-trivial amount to model "training".
    assert not np.allclose(before, after)
    # Magnitude must be bounded so dream-episode downscale can still
    # detect a change (sanity bound; not load-bearing).
    assert np.max(np.abs(after)) < 10.0


def test_adapt_subdomain_is_deterministic() -> None:
    a1 = InferenceOnlyAdapter(out_dim=8, rank=4, seed=0)
    a2 = InferenceOnlyAdapter(out_dim=8, rank=4, seed=0)
    train = [_record("anatomy", i) for i in range(4)]
    adapt_subdomain(adapter=a1, subdomain="anatomy", train=train, seed=42)
    adapt_subdomain(adapter=a2, subdomain="anatomy", train=train, seed=42)
    np.testing.assert_array_equal(
        a1.current_delta("layer_0_lora_B"),
        a2.current_delta("layer_0_lora_B"),
    )


def test_adapt_subdomain_different_subjects_yield_different_deltas() -> None:
    a1 = InferenceOnlyAdapter(out_dim=8, rank=4, seed=0)
    a2 = InferenceOnlyAdapter(out_dim=8, rank=4, seed=0)
    train_a = [_record("anatomy", i) for i in range(4)]
    train_b = [_record("astronomy", i) for i in range(4)]
    adapt_subdomain(adapter=a1, subdomain="anatomy", train=train_a, seed=0)
    adapt_subdomain(adapter=a2, subdomain="astronomy", train=train_b, seed=0)
    assert not np.allclose(
        a1.current_delta("layer_0_lora_B"),
        a2.current_delta("layer_0_lora_B"),
    )
