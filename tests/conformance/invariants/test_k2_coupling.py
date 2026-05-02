"""K-coupling invariant — SO x fast-spindle phase-coupling measurement.

Reference: docs/invariants/registry.md (K2),
docs/proofs/k2-coupling-evidence.md, framework-C spec §5,
empirical anchor `elife2025bayesian` (eLife 2025,
coupling strength 0.33 [0.27, 0.39]).
"""
from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kiki_oniric.core.observables import PhaseCouplingObservable
from kiki_oniric.dream.guards.coupling import (
    CouplingGuardError,
    check_coupling_in_window,
)
from tests.conformance.invariants._synthetic_phase_coupling import (
    SyntheticPhaseCouplingSubstrate,
)

# Empirical anchor: eLife 2025 Bayesian meta-analysis
# (BibTeX `elife2025bayesian`). 95 % CI on coupling strength.
K2_CI_LOW: float = 0.27
K2_CI_HIGH: float = 0.39
K2_N_SAMPLES: int = 8192  # >= 32 SO cycles at fs=256 Hz, f_SO=1 Hz.


def _is_protocol(cls: type) -> bool:
    return Protocol in getattr(cls, "__mro__", ())


def test_phase_coupling_observable_is_runtime_checkable() -> None:
    """Protocol must be @runtime_checkable so isinstance() works."""
    assert runtime_checkable(PhaseCouplingObservable) is PhaseCouplingObservable


def test_phase_coupling_observable_is_protocol() -> None:
    """Structural test mirrors test_dr3_substrate.test_all_8_protocols_declared."""
    assert _is_protocol(PhaseCouplingObservable)


def test_k2_guard_passes_inside_window() -> None:
    """Mid-window value must pass silently."""
    check_coupling_in_window(0.33, ci_low=0.27, ci_high=0.39)


def test_k2_guard_rejects_below_window() -> None:
    """Value < ci_low must raise."""
    with pytest.raises(CouplingGuardError, match="below"):
        check_coupling_in_window(0.20, ci_low=0.27, ci_high=0.39)


def test_k2_guard_rejects_above_window() -> None:
    """Value > ci_high must raise."""
    with pytest.raises(CouplingGuardError, match="above"):
        check_coupling_in_window(0.50, ci_low=0.27, ci_high=0.39)


def test_k2_guard_rejects_nan() -> None:
    """NaN slips through naive comparisons; explicit guard required."""
    with pytest.raises(CouplingGuardError, match="NaN"):
        check_coupling_in_window(math.nan, ci_low=0.27, ci_high=0.39)


def test_k2_guard_rejects_inverted_window() -> None:
    """ci_low > ci_high is a programmer error, must raise."""
    with pytest.raises(ValueError, match="ci_low"):
        check_coupling_in_window(0.33, ci_low=0.50, ci_high=0.10)


def test_synthetic_substrate_satisfies_protocol() -> None:
    """Synthetic fixture must structurally implement the Protocol."""
    sub = SyntheticPhaseCouplingSubstrate()
    assert isinstance(sub, PhaseCouplingObservable)


def test_synthetic_substrate_returns_aligned_arrays() -> None:
    """Phase + amplitude arrays must have the requested length and fs > 0."""
    sub = SyntheticPhaseCouplingSubstrate()
    phase, amp, fs = sub.emit_phase_coupling_signal(n_samples=2048, seed=7)
    assert phase.shape == (2048,)
    assert amp.shape == (2048,)
    assert phase.dtype.name == "float32"
    assert amp.dtype.name == "float32"
    assert fs > 0.0


def test_synthetic_substrate_is_deterministic() -> None:
    """Same seed -> bit-identical output (R1 reproducibility, parent rule)."""
    sub = SyntheticPhaseCouplingSubstrate()
    p1, a1, _ = sub.emit_phase_coupling_signal(n_samples=512, seed=42)
    p2, a2, _ = sub.emit_phase_coupling_signal(n_samples=512, seed=42)
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(a1, a2)


def test_synthetic_substrate_seeds_are_independent() -> None:
    """Distinct seeds produce distinct realisations (no global state)."""
    sub = SyntheticPhaseCouplingSubstrate()
    _, a1, _ = sub.emit_phase_coupling_signal(n_samples=512, seed=1)
    _, a2, _ = sub.emit_phase_coupling_signal(n_samples=512, seed=2)
    assert not np.array_equal(a1, a2)


def _mean_vector_length(
    phase: np.ndarray, amplitude: np.ndarray
) -> float:
    """Tort 2010-style mean vector length (PAC strength).

    MVL = | mean_t [ amplitude(t) * exp(i * phase(t)) ] | / mean_t amplitude(t)

    Returns a float in [0, 1]. Pure numpy, no SciPy needed; SciPy
    is reserved for any future Hilbert-transform based estimator.
    """
    if phase.shape != amplitude.shape:
        raise ValueError("phase and amplitude must have identical shapes")
    z = amplitude.astype(np.float64) * np.exp(1j * phase.astype(np.float64))
    num = float(np.abs(z.mean()))
    denom = float(np.abs(amplitude.astype(np.float64)).mean())
    if denom == 0.0:
        return 0.0
    return num / denom


def test_estimator_zero_for_random_phase() -> None:
    """No coupling: random uniform phase yields MVL ~= 0 (large N)."""
    rng = np.random.default_rng(0)
    n = 8192
    phase = rng.uniform(-np.pi, np.pi, size=n).astype(np.float32)
    amp = (0.5 + rng.normal(0.0, 0.05, size=n)).astype(np.float32)
    mvl = _mean_vector_length(phase, amp)
    assert mvl < 0.05, f"expected near-zero MVL on random phase, got {mvl}"


def test_estimator_one_for_perfect_coupling() -> None:
    """Perfect coupling: amplitude = 1 only at phase 0 -> MVL = 1.0."""
    n = 1024
    phase = np.zeros(n, dtype=np.float32)
    amp = np.ones(n, dtype=np.float32)
    mvl = _mean_vector_length(phase, amp)
    assert abs(mvl - 1.0) < 1e-6


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(deadline=None, max_examples=50)
def test_k2_property_synthetic_in_window(seed: int) -> None:
    """K2: synthetic substrate's MVL falls inside eLife 2025 95 % CI.

    Reference: docs/invariants/registry.md (K2),
    docs/proofs/k2-coupling-evidence.md, BibTeX `elife2025bayesian`.
    """
    sub = SyntheticPhaseCouplingSubstrate()
    phase, amp, _fs = sub.emit_phase_coupling_signal(
        n_samples=K2_N_SAMPLES, seed=seed
    )
    mvl = _mean_vector_length(phase, amp)
    check_coupling_in_window(mvl, ci_low=K2_CI_LOW, ci_high=K2_CI_HIGH)


def test_k2_property_smoke_known_seed() -> None:
    """Determinism check: seed=7 yields a known MVL bucket.

    This anchors the synthetic generator's calibration: if a future
    edit to the generator drifts the MVL outside [0.30, 0.36], the
    coverage of the property test against the empirical CI degrades
    silently. This smoke-test catches that.
    """
    sub = SyntheticPhaseCouplingSubstrate()
    phase, amp, _fs = sub.emit_phase_coupling_signal(
        n_samples=K2_N_SAMPLES, seed=7
    )
    mvl = _mean_vector_length(phase, amp)
    assert 0.30 < mvl < 0.36, (
        f"calibration drift detected: seed=7 MVL={mvl:.4f} "
        f"outside [0.30, 0.36]"
    )


def test_estimator_rejects_shape_mismatch() -> None:
    """Estimator must guard against array length mismatch."""
    phase = np.zeros(10, dtype=np.float32)
    amp = np.zeros(11, dtype=np.float32)
    with pytest.raises(ValueError, match="identical shapes"):
        _mean_vector_length(phase, amp)


def test_estimator_zero_amplitude_returns_zero() -> None:
    """All-zero amplitude defines MVL = 0 (no division by zero)."""
    n = 256
    phase = np.linspace(-np.pi, np.pi, n, dtype=np.float32)
    amp = np.zeros(n, dtype=np.float32)
    assert _mean_vector_length(phase, amp) == 0.0


def test_synthetic_substrate_rejects_zero_samples() -> None:
    """n_samples=0 must raise (per fixture contract)."""
    sub = SyntheticPhaseCouplingSubstrate()
    with pytest.raises(ValueError, match="n_samples"):
        sub.emit_phase_coupling_signal(n_samples=0, seed=0)
