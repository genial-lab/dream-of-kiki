"""Sharon 2025 SO-trough biomarker calibration tests for P_min / P_equ / P_max.

Reference: Sharon et al., Alzheimer's & Dementia 2025 (sharon2025alzdementia
in docs/papers/paper1/references.bib). hd-EEG, N=55 (21 healthy older /
28 aMCI / 6 AD). Cognitive performance decreases monotonically with
slow-wave trough amplitude and frontocentral synchronization.

These tests verify the qualitative calibration of the
``so_trough_amplitude_factor`` field on each profile: an informed
placeholder whose final empirical value lands at G2 / G4 pilots
(cf. ``scripts/pilot_g2.py``).
"""
from __future__ import annotations

import math

from kiki_oniric.profiles.p_equ import PEquProfile
from kiki_oniric.profiles.p_max import PMaxProfile
from kiki_oniric.profiles.p_min import PMinProfile


def test_p_min_so_trough_amplitude_factor_default_value() -> None:
    """P_min default factor = 0.45 (aMCI midpoint, Sharon 2025)."""
    profile = PMinProfile()
    assert math.isclose(profile.so_trough_amplitude_factor, 0.45)


def test_p_equ_so_trough_amplitude_factor_default_value() -> None:
    """P_equ default factor = 1.0 (healthy-older anchor, Sharon 2025)."""
    profile = PEquProfile()
    assert math.isclose(profile.so_trough_amplitude_factor, 1.0)


def test_p_max_so_trough_amplitude_factor_default_value() -> None:
    """P_max default factor = 1.0 (intact substrate, healthy-young anchor)."""
    profile = PMaxProfile()
    assert math.isclose(profile.so_trough_amplitude_factor, 1.0)
