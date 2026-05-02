"""Unit tests for empirical effect-size targets (Hu 2020 + Javadi 2024).

Targets are typed, frozen constants encoding published meta-analytic
Hedges' g and 95% CIs. Every constant must be immutable and resolve
to a real BibTeX key in docs/papers/paper1/references.bib.
"""
from dataclasses import FrozenInstanceError

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from harness.benchmarks.effect_size_targets import (
    HU_2020_OVERALL,
    EffectSizeTarget,
)


def test_target_constructs_with_all_fields() -> None:
    target = EffectSizeTarget(
        name="dummy_overall",
        hedges_g=0.29,
        ci_low=0.21,
        ci_high=0.38,
        sample_size_n=2004,
        k_studies=91,
        source_bibtex_key="hu2020tmr",
        profile_target="P_equ",
        stratum=None,
    )
    assert target.name == "dummy_overall"
    assert target.hedges_g == 0.29


def test_target_is_frozen() -> None:
    target = EffectSizeTarget(
        name="dummy",
        hedges_g=0.29,
        ci_low=0.21,
        ci_high=0.38,
        sample_size_n=2004,
        k_studies=91,
        source_bibtex_key="hu2020tmr",
        profile_target="P_equ",
        stratum=None,
    )
    with pytest.raises(FrozenInstanceError):
        target.hedges_g = 0.99  # type: ignore[misc]


def test_target_rejects_inverted_ci() -> None:
    """ci_low must be <= hedges_g <= ci_high (sanity, not stat rule)."""
    with pytest.raises(ValueError, match="ci_low.*ci_high"):
        EffectSizeTarget(
            name="bad",
            hedges_g=0.29,
            ci_low=0.50,   # > ci_high — invalid
            ci_high=0.10,
            sample_size_n=10,
            k_studies=1,
            source_bibtex_key="hu2020tmr",
            profile_target="P_equ",
            stratum=None,
        )


def test_target_rejects_g_outside_ci() -> None:
    with pytest.raises(ValueError, match="hedges_g.*ci"):
        EffectSizeTarget(
            name="bad",
            hedges_g=0.99,    # outside [0.21, 0.38]
            ci_low=0.21,
            ci_high=0.38,
            sample_size_n=10,
            k_studies=1,
            source_bibtex_key="hu2020tmr",
            profile_target="P_equ",
            stratum=None,
        )


# ----------------------------------------------------------------------
# Hu 2020 TMR meta-analysis constants
# ----------------------------------------------------------------------


def test_hu2020_overall_matches_published() -> None:
    """[@hu2020tmr] overall Hedges' g = 0.29 [0.21, 0.38], k=91, N=2004."""
    from harness.benchmarks.effect_size_targets import HU_2020_OVERALL

    assert HU_2020_OVERALL.hedges_g == 0.29
    assert HU_2020_OVERALL.ci_low == 0.21
    assert HU_2020_OVERALL.ci_high == 0.38
    assert HU_2020_OVERALL.k_studies == 91
    assert HU_2020_OVERALL.sample_size_n == 2004
    assert HU_2020_OVERALL.source_bibtex_key == "hu2020tmr"
    assert HU_2020_OVERALL.profile_target == "P_equ"
    assert HU_2020_OVERALL.stratum is None


def test_hu2020_nrem2_matches_published() -> None:
    """[@hu2020tmr] NREM2 stratum g = 0.32 [0.04, 0.60]."""
    from harness.benchmarks.effect_size_targets import HU_2020_NREM2

    assert HU_2020_NREM2.hedges_g == 0.32
    assert HU_2020_NREM2.ci_low == 0.04
    assert HU_2020_NREM2.ci_high == 0.60
    assert HU_2020_NREM2.stratum == "NREM2"
    assert HU_2020_NREM2.profile_target == "P_equ"
    assert HU_2020_NREM2.source_bibtex_key == "hu2020tmr"


def test_hu2020_sws_matches_published() -> None:
    """[@hu2020tmr] SWS stratum g = 0.27 [0.20, 0.35]."""
    from harness.benchmarks.effect_size_targets import HU_2020_SWS

    assert HU_2020_SWS.hedges_g == 0.27
    assert HU_2020_SWS.ci_low == 0.20
    assert HU_2020_SWS.ci_high == 0.35
    assert HU_2020_SWS.stratum == "SWS"
    assert HU_2020_SWS.profile_target == "P_equ"


# ----------------------------------------------------------------------
# Javadi 2024 sleep-restriction meta-analysis constants
# ----------------------------------------------------------------------


def test_javadi2024_overall_matches_published() -> None:
    """[@javadi2024sleeprestriction] g = 0.29 [0.13, 0.44], k=39, N=1234."""
    from harness.benchmarks.effect_size_targets import JAVADI_2024_OVERALL

    assert JAVADI_2024_OVERALL.hedges_g == 0.29
    assert JAVADI_2024_OVERALL.ci_low == 0.13
    assert JAVADI_2024_OVERALL.ci_high == 0.44
    assert JAVADI_2024_OVERALL.k_studies == 39
    assert JAVADI_2024_OVERALL.sample_size_n == 1234
    assert JAVADI_2024_OVERALL.source_bibtex_key == "javadi2024sleeprestriction"
    # P_min : sleep restriction = degraded substrate decrement floor
    assert JAVADI_2024_OVERALL.profile_target == "P_min"
    assert JAVADI_2024_OVERALL.stratum is None


def test_is_within_ci_inclusive_at_bounds() -> None:
    assert HU_2020_OVERALL.is_within_ci(HU_2020_OVERALL.ci_low) is True
    assert HU_2020_OVERALL.is_within_ci(HU_2020_OVERALL.ci_high) is True
    assert HU_2020_OVERALL.is_within_ci(HU_2020_OVERALL.hedges_g) is True


def test_is_within_ci_outside_returns_false() -> None:
    assert HU_2020_OVERALL.is_within_ci(0.0) is False
    assert HU_2020_OVERALL.is_within_ci(0.5) is False
    assert HU_2020_OVERALL.is_within_ci(-1.0) is False


@given(observed=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False))
@settings(max_examples=200, derandomize=True)
def test_is_within_ci_property(observed: float) -> None:
    """For any observed in [-5, 5], is_within_ci agrees with bounds check."""
    result = HU_2020_OVERALL.is_within_ci(observed)
    expected = HU_2020_OVERALL.ci_low <= observed <= HU_2020_OVERALL.ci_high
    assert result is expected
