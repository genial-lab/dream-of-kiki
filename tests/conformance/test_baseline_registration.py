"""DR-3 exemption contract for baseline-only adapters.

The Wake-Sleep CL baseline (Alfarano 2024) is registered in
`kiki_oniric.substrates.__init__` for ablation-matrix
discoverability, but it is NOT a DR-3-conformant substrate.
This test pins that distinction so future contributors do not
accidentally enrol the baseline as a 4-op substrate.

Reference :
  docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
  docs/superpowers/plans/2026-05-02-wake-sleep-cl-ablation-baseline.md
  Task 4.
"""
from __future__ import annotations

import kiki_oniric.substrates as substrates
from kiki_oniric.substrates.wake_sleep_cl_baseline import (
    wake_sleep_substrate_components,
)


def test_baseline_is_in_public_api() -> None:
    assert "WakeSleepCLBaseline" in substrates.__all__
    assert "wake_sleep_substrate_components" in substrates.__all__
    assert "WAKE_SLEEP_BASELINE_NAME" in substrates.__all__
    assert "WAKE_SLEEP_BASELINE_VERSION" in substrates.__all__


def test_baseline_lacks_dr3_op_factories() -> None:
    """DR-3 exemption pin : no op factory keys in the components map."""
    comps = wake_sleep_substrate_components()
    for op in ("replay", "downscale", "restructure", "recombine"):
        assert op not in comps, (
            f"Baseline must NOT register a {op} op factory — "
            f"that would imply DR-3 conformance which the baseline "
            f"does not claim."
        )


def test_dr3_substrates_unchanged() -> None:
    """The 3 real substrates still expose their components helpers."""
    assert hasattr(substrates, "mlx_substrate_components")
    assert hasattr(substrates, "esnn_substrate_components")
    assert hasattr(substrates, "micro_kiki_substrate_components")
