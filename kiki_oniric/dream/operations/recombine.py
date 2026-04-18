"""Recombine operation — C-Hobson VAE light source (creative branch).

Skeleton "light" version (S11.1): linear interpolation between two
randomly-sampled latents from `delta_latents` input. Real VAE
sampling (encoder/decoder pair) lands S13+ alongside concurrent
dream worker.

Mathematical role (per docs/proofs/op-pair-analysis.md): canonical
parallel branch (§4.3) — recombine runs in parallel with the
serial A-B-D branch to preserve generative diversity. Sampling is
non-deterministic by design; rng injection enables reproducible
tests.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from kiki_oniric.dream.episode import DreamEpisode


@dataclass
class RecombineOpState:
    """Mutable state for recombine op across episodes."""

    total_episodes_handled: int = 0
    total_samples_emitted: int = 0
    last_sample: list[float] | None = None
    sample_history: list[list[float]] = field(default_factory=list)


def _interpolate(
    a: list[float], b: list[float], alpha: float
) -> list[float]:
    """Linear interpolation: alpha*a + (1-alpha)*b component-wise."""
    if len(a) != len(b):
        raise ValueError(
            f"latent dimensions mismatch: {len(a)} vs {len(b)}"
        )
    return [alpha * x + (1.0 - alpha) * y for x, y in zip(a, b)]


def recombine_handler(
    state: RecombineOpState,
    rng: random.Random | None = None,
) -> Callable[[DreamEpisode], None]:
    """Build a recombine handler bound to a state instance.

    Handler reads `delta_latents` from input_slice (must contain
    >= 2 latents), samples 2 distinct indices via rng, interpolates
    with alpha ~ U(0, 1), updates state. Real VAE sampling lands
    S13+ with MLX integration.
    """
    if rng is None:
        rng = random.Random()

    def handler(episode: DreamEpisode) -> None:
        latents = episode.input_slice.get("delta_latents", [])
        if len(latents) < 2:
            raise ValueError(
                f"delta_latents must contain at least 2 latents, "
                f"got {len(latents)}"
            )
        idx_a, idx_b = rng.sample(range(len(latents)), 2)
        alpha = rng.random()
        sample = _interpolate(latents[idx_a], latents[idx_b], alpha)
        state.total_episodes_handled += 1
        state.total_samples_emitted += 1
        state.last_sample = sample
        state.sample_history.append(sample)

    return handler
