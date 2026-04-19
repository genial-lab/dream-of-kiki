"""SNN-substrate downscale op — spike-rate shrinkage (cycle-3 C3.12).

Norse-substrate counterpart to
:mod:`kiki_oniric.dream.operations.downscale_real`. Shrinks spike
rates (not raw weights) by ``shrink_factor``, then projects back to
the weight domain via the inverse-sigmoid map.

Why shrink rates and not weights ? Multiplicative scaling of
weights does not map cleanly to the spike-rate domain under a
sigmoid nonlinearity. Shrinking rates directly preserves the SHY
semantics (Tononi : global downscale of firing activity during
NREM) while remaining round-trip-stable via the rate proxy.

Contract :

- ``shrink_factor`` read from ``episode.input_slice`` ; must lie
  in ``(0, 1]`` (value-error on violation, same message as
  :mod:`downscale_real`).
- ``rates_after = rates_before * shrink_factor``, clipped to
  ``[eps, max_rate - eps]``.
- S2 finite guard invoked on the rate array after shrink.
- ``state.compound_factor *= shrink_factor`` accumulates the
  multiplicative drift (same contract as the MLX variant).
- ``state.last_compute_flops`` tagged at ~1 multiply per scalar.

Reference :
  docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2, §6.2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from kiki_oniric.dream.episode import DreamEpisode
from kiki_oniric.dream.guards.finite import FiniteGuardError, check_finite
from kiki_oniric.dream.operations.replay_snn import (
    spike_rates_to_weights,
    weights_to_spike_rates,
)


@dataclass
class DownscaleSNNState:
    """K1-tagged SNN-proxy downscale state across multiple episodes."""

    compound_factor: float = 1.0
    last_compute_flops: int = 0
    total_compute_flops: int = 0


def downscale_snn_handler(
    state: DownscaleSNNState,
    *,
    weights: np.ndarray,
    max_rate: float = 100.0,
) -> Callable[[DreamEpisode], None]:
    """Build an SNN-proxy downscale handler bound to ``state``.

    ``weights`` is mutated in-place ; callers keep the reference.
    """

    def handler(episode: DreamEpisode) -> None:
        factor = episode.input_slice.get("shrink_factor", 1.0)
        if not (0.0 < factor <= 1.0):
            raise ValueError(
                f"shrink_factor must be in (0, 1], got {factor}"
            )

        rates = weights_to_spike_rates(weights, max_rate=max_rate)
        new_rates = rates * factor
        new_rates = np.clip(new_rates, 1e-6, max_rate - 1e-6)
        # S2 guard : reject any NaN / Inf that could sneak in via
        # a poisoned input. check_finite raises FiniteGuardError.
        check_finite(new_rates)

        new_weights = spike_rates_to_weights(
            new_rates, max_rate=max_rate
        )
        weights[...] = new_weights

        state.compound_factor *= float(factor)
        flops = max(int(weights.size), 1)
        state.last_compute_flops = flops
        state.total_compute_flops += flops

    return handler


__all__ = [
    "DownscaleSNNState",
    "downscale_snn_handler",
    # Re-export FiniteGuardError so test imports read naturally.
    "FiniteGuardError",
]
