"""Unit tests for ``harness.real_benchmarks.dataset_registry``.

Cycle-3 pre-cycle-3 lock #3 : SHA pins for MMLU + HellaSwag.
Mirrors the contract of ``tests/unit/test_base_model_registry.py``
(pre-cycle-3 lock #2) so both pin surfaces share a single QA
pattern : local-only, deterministic, no network dependency.
"""
from __future__ import annotations

import re

import pytest

from harness.real_benchmarks.dataset_registry import (
    DATASET_REGISTRY,
    DatasetPin,
    get_dataset_pin,
    verify_all_datasets,
)

_SHA40_RE = re.compile(r"^[a-f0-9]{40}$")

_EXPECTED_DATASETS: tuple[str, ...] = ("mmlu", "hellaswag")


def test_registry_has_both_required_datasets() -> None:
    """Both MMLU + HellaSwag entries are registered.

    Cycle-3 spec §5 pre-cycle-3 lock #3 names these two
    benchmarks explicitly ; a missing one would block the real-
    benchmark replacement of the synthetic retained suite.
    """
    for key in _EXPECTED_DATASETS:
        assert key in DATASET_REGISTRY, (
            f"missing dataset pin : {key}"
        )


@pytest.mark.parametrize("key", _EXPECTED_DATASETS)
def test_every_dataset_entry_is_well_formed(key: str) -> None:
    """Every pin has a 40-char revision SHA + non-empty fields."""
    pin = DATASET_REGISTRY[key]
    assert isinstance(pin, DatasetPin)
    assert pin.hf_repo_id
    assert "/" in pin.hf_repo_id
    assert _SHA40_RE.match(pin.revision_sha), (
        f"revision_sha for {key} is not 40-char lowercase hex"
    )
    assert pin.n_examples > 0
    assert pin.protocol
    assert pin.license_id


def test_get_dataset_pin_returns_registered_entry() -> None:
    """``get_dataset_pin`` returns the same object as direct lookup."""
    pin = get_dataset_pin("mmlu")
    assert pin is DATASET_REGISTRY["mmlu"]


def test_get_dataset_pin_raises_keyerror_with_hint() -> None:
    """Unknown key raises ``KeyError`` listing available keys."""
    with pytest.raises(KeyError) as excinfo:
        get_dataset_pin("no-such-dataset")
    msg = str(excinfo.value)
    assert "no-such-dataset" in msg
    assert "mmlu" in msg


def test_verify_all_datasets_returns_bool_mapping() -> None:
    """``verify_all_datasets`` (local-only) passes for all pins."""
    results = verify_all_datasets(live=False)
    assert set(results.keys()) == set(DATASET_REGISTRY.keys())
    assert all(isinstance(v, bool) for v in results.values())
    assert all(results.values()), (
        f"self-consistency regression : {results}"
    )


def test_mmlu_protocol_is_5shot_hellaswag_is_zeroshot() -> None:
    """Protocol strings match cycle-3 spec §5 / Open LLM defaults.

    MMLU is 5-shot per Hendrycks et al. 2020 + Open LLM Leaderboard ;
    HellaSwag is zero-shot per lm-evaluation-harness defaults.
    """
    assert get_dataset_pin("mmlu").protocol == "5-shot"
    assert get_dataset_pin("hellaswag").protocol == "zero-shot"


def test_dataset_pin_is_frozen() -> None:
    """``DatasetPin`` dataclass is frozen so entries stay immutable."""
    pin = get_dataset_pin("mmlu")
    with pytest.raises(Exception):
        pin.hf_repo_id = "hacked"  # type: ignore[misc]
