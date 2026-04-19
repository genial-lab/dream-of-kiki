"""SNN-substrate restructure op — rate-channel swap (cycle-3 C3.12).

Norse-substrate counterpart to
:mod:`kiki_oniric.dream.operations.restructure_real`. The narrow
production use-case here is ``"reroute"`` : swap two
spike-rate channels in the weight proxy. This is the SNN analogue
of an MLX layer swap — but at the per-neuron channel granularity,
not at the layer granularity (SNNs have no concept of "layer"
reorderings ; the restructuring lives at the spike-rate axis).

Contract :

- ``topo_op`` read from ``episode.input_slice`` ; only ``"reroute"``
  supported in the cycle-3 proxy. Unknown ops raise a
  ``ValueError`` whose message contains the literal ``"S3"`` tag
  (matches :mod:`restructure_real` for cross-substrate test
  parametrisation).
- ``swap_indices`` defaults to ``[0, 1]`` ; must be a length-2
  sequence of valid weight-channel indices.
- ``weights`` mutated in place (mirrors MLX layer swap).
- ``state.diff_history`` grows by one entry per call.
- ``state.last_compute_flops`` is ~O(n_channels) for a pointer
  swap in numpy.

Reference :
  docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2, §6.2
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from kiki_oniric.dream.episode import DreamEpisode
from kiki_oniric.dream.operations.replay_snn import (
    spike_rates_to_weights,
    weights_to_spike_rates,
)


# Subset of topo_ops supported by the SNN proxy in cycle-3.
# The skeleton restructure.py accepts {"add", "remove", "reroute"},
# but per the C3.12 plan only "reroute" lands here.
_SUPPORTED_TOPO_OPS: frozenset[str] = frozenset({"reroute"})


@dataclass
class RestructureSNNState:
    """K1-tagged SNN-proxy restructure state across episodes."""

    diff_history: list[str] = field(default_factory=list)
    last_compute_flops: int = 0
    total_compute_flops: int = 0


def restructure_snn_handler(
    state: RestructureSNNState,
    *,
    weights: np.ndarray,
    max_rate: float = 100.0,
) -> Callable[[DreamEpisode], None]:
    """Build an SNN-proxy restructure handler bound to ``state``.

    Only ``"reroute"`` is supported — unknown ops raise a
    :class:`ValueError` with an ``"S3"``-tagged message.
    """

    def handler(episode: DreamEpisode) -> None:
        topo_op = episode.input_slice.get("topo_op", "")
        if topo_op not in _SUPPORTED_TOPO_OPS:
            raise ValueError(
                f"S3: DE {episode.episode_id!r}: unknown topo_op "
                f"{topo_op!r} ; SNN-proxy op supports "
                f"{sorted(_SUPPORTED_TOPO_OPS)}"
            )

        swap_indices = episode.input_slice.get("swap_indices", [0, 1])
        if len(swap_indices) != 2:
            raise ValueError(
                "S3: reroute requires swap_indices of length 2"
            )
        i, j = swap_indices
        # weights may be multi-dimensional ; we swap along axis 0
        # (the canonical "channel" axis in the rate proxy).
        n_chan = weights.shape[0] if weights.ndim >= 1 else 0
        if not (
            isinstance(i, int)
            and isinstance(j, int)
            and 0 <= i < n_chan
            and 0 <= j < n_chan
        ):
            raise ValueError(
                f"S3: reroute swap_indices {swap_indices!r} out of "
                f"bounds for weights axis-0 of length {n_chan}"
            )

        rates = weights_to_spike_rates(weights, max_rate=max_rate)
        # np fancy-indexing swap : rates[[i, j]] = rates[[j, i]]
        rates[[i, j]] = rates[[j, i]]
        new_weights = spike_rates_to_weights(
            rates, max_rate=max_rate
        )
        weights[...] = new_weights

        state.diff_history.append(topo_op)
        flops = max(int(n_chan), 1)
        state.last_compute_flops = flops
        state.total_compute_flops += flops

    return handler


__all__ = [
    "RestructureSNNState",
    "restructure_snn_handler",
]
