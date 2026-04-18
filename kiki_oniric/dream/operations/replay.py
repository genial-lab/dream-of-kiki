"""Replay operation — A-Walker/Stickgold consolidation source.

Skeleton version (S5.4): counts consumed records, logs episode.
Real gradient-based replay (sample β → forward through W → update
via retention-objective gradient) lands alongside MLX integration
S7+ with swap protocol.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from kiki_oniric.dream.episode import DreamEpisode


@dataclass
class ReplayOpState:
    """Mutable counter state for replay op across multiple episodes."""

    total_records_consumed: int = 0
    total_episodes_handled: int = 0


def replay_handler(state: ReplayOpState) -> Callable[[DreamEpisode], None]:
    """Build a replay handler bound to a state instance.

    Handler consumes all `beta_records` in the DE's input_slice,
    updates the state counters. No-op on weights for now
    (skeleton) — gradient integration S7+ with MLX.
    """

    def handler(episode: DreamEpisode) -> None:
        records = episode.input_slice.get("beta_records", [])
        state.total_records_consumed += len(records)
        state.total_episodes_handled += 1

    return handler
