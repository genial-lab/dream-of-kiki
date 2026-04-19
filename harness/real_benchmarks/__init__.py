"""Real benchmark dataset registry + loaders for cycle-3.

Exposes :

- :mod:`harness.real_benchmarks.dataset_registry` — SHA-pinned
  dataset entries (MMLU full + HellaSwag) per cycle-3 spec §5
  pre-cycle-3 lock #3.

Future cycle-3 deliverables under this package : loaders with
``.sha256`` artifact hashes, eval-protocol runners (5-shot for
MMLU, zero-shot for HellaSwag), and the real-benchmark replay
cache wired to the ``harness/storage/run_registry.py`` R1
contract.
"""
from __future__ import annotations

from harness.real_benchmarks.dataset_registry import (
    DATASET_REGISTRY,
    DatasetPin,
    get_dataset_pin,
    verify_all_datasets,
)

__all__ = [
    "DATASET_REGISTRY",
    "DatasetPin",
    "get_dataset_pin",
    "verify_all_datasets",
]
