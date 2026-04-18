"""Conformance test for S4 invariant — attention prior bounded."""
from __future__ import annotations

import numpy as np
import pytest

from kiki_oniric.dream.guards.attention import (
    AttentionGuardError,
    check_attention_bounded,
)


def test_s4_passes_valid_prior() -> None:
    """Valid prior (each in [0,1], sum <= budget) passes silently."""
    prior = np.array([0.3, 0.4, 0.2])
    check_attention_bounded(prior, budget=1.0)


def test_s4_rejects_out_of_unit_interval() -> None:
    """S4 must reject any component outside [0, 1]."""
    bad = np.array([0.5, -0.2, 0.4])
    with pytest.raises(AttentionGuardError):
        check_attention_bounded(bad, budget=1.5)
