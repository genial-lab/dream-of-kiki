"""Unit tests for the CCA Studyforrest alignment module
(cycle-3 C3.17 — Phase 2 track c).

Runs WITHOUT ``scikit-learn`` installed — the aligner ships a
numpy + scipy SVD fallback so the test path is independent of
the optional dependency. The sklearn branch is exercised only
when the env has it (``pragma: no cover`` on the import-
dependent code).

References :
- docs/interfaces/fmri-schema.yaml (v0.7.0+PARTIAL, schema)
- kiki_oniric/eval/state_alignment.py (C3.16 HMM alignment)
- framework-C spec §6.2 (DR-3 condition 2), §3 track (c)
"""
from __future__ import annotations

import numpy as np

from kiki_oniric.eval.cca_alignment import CcaAligner, CcaResult


def test_cca_recovers_planted_linear_correlation() -> None:
    """TDD-1 — when X and Y share the same latent up to a linear
    transform + small noise, the top canonical correlation is
    recovered at ≥ 0.9 and the result is declared significant.
    """
    rng = np.random.default_rng(seed=0)
    n_samples = 400
    latent = rng.standard_normal((n_samples, 3))

    # X = latent · W_x + noise ; Y = latent · W_y + noise, both
    # well-conditioned so the top canonical correlation is high.
    W_x = rng.standard_normal((3, 5))
    W_y = rng.standard_normal((3, 4))
    X = latent @ W_x + 0.05 * rng.standard_normal((n_samples, 5))
    Y = latent @ W_y + 0.05 * rng.standard_normal((n_samples, 4))

    aligner = CcaAligner(
        n_components=3, n_permutations=50, seed=0, alpha=0.05,
    )
    result = aligner.fit(X, Y)

    assert isinstance(result, CcaResult)
    # Top canonical correlation must be >= 0.9
    assert float(result.canonical_correlations[0]) >= 0.9, (
        f"top canonical r {result.canonical_correlations[0]:.3f} "
        "below the 0.9 recovery bar"
    )
    # Projections have the expected shape
    assert result.x_projection.shape == (n_samples, 3)
    assert result.y_projection.shape == (n_samples, 3)
    # Planted signal clears the permutation null.
    assert result.is_significant


def test_permutation_null_below_planted_signal() -> None:
    """TDD-2 — the permutation null built by shuffling Y's rows
    has a 95th-percentile canonical-correlation lower than the
    planted mean r̄. Validates the null-construction procedure
    is actually breaking the X↔Y correspondence.
    """
    rng = np.random.default_rng(seed=1)
    n_samples = 300
    latent = rng.standard_normal((n_samples, 2))

    X = latent + 0.05 * rng.standard_normal((n_samples, 2))
    Y = latent + 0.05 * rng.standard_normal((n_samples, 2))

    aligner = CcaAligner(
        n_components=2, n_permutations=100, seed=0, alpha=0.05,
    )
    result = aligner.fit(X, Y)

    assert result.null_distribution.shape == (100,)
    # Observed mean r̄
    observed = float(np.mean(result.canonical_correlations))
    # 95th-percentile of the null distribution
    null_95 = float(np.quantile(result.null_distribution, 0.95))
    assert null_95 < observed, (
        f"null 95th {null_95:.3f} >= observed {observed:.3f} — "
        "permutation did not break the X↔Y link"
    )
    # p-value strictly below alpha on this well-separated signal
    assert result.p_value < 0.05
    assert result.is_significant


def test_cca_chance_null_is_non_significant() -> None:
    """TDD-3 — effect-size check : when X and Y are independent
    the canonical correlations do NOT clear the permutation null
    reliably, so is_significant stays False and p_value is large.
    """
    rng = np.random.default_rng(seed=2)
    n_samples = 300
    X = rng.standard_normal((n_samples, 3))
    Y = rng.standard_normal((n_samples, 3))

    aligner = CcaAligner(
        n_components=2, n_permutations=200, seed=0, alpha=0.05,
    )
    result = aligner.fit(X, Y)

    # Under pure independence the observed r̄ should fall well
    # inside the null support ; p_value ≥ alpha ; not significant.
    assert not result.is_significant, (
        f"independence-null returned significant result "
        f"(p={result.p_value:.3f}, r̄={float(np.mean(result.canonical_correlations)):.3f})"
    )
    assert result.p_value >= 0.05
    # Null distribution length matches n_permutations
    assert result.null_distribution.shape == (200,)
    # Canonical correlations still well-defined (non-NaN) and
    # bounded in [0, 1] up to numerical precision.
    assert np.all(np.isfinite(result.canonical_correlations))
    assert float(result.canonical_correlations.max()) <= 1.0 + 1e-6
    assert float(result.canonical_correlations.min()) >= -1e-6
