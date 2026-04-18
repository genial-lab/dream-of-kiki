"""Unit tests for recombine operation (C-Hobson VAE light)."""
from __future__ import annotations

import random

import pytest

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.operations.recombine import (
    RecombineOpState,
    recombine_handler,
)
from kiki_oniric.dream.runtime import DreamRuntime


def make_recombine_episode(
    ep_id: str, latents: list[list[float]]
) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"delta_latents": latents},
        operation_set=(Operation.RECOMBINE,),
        output_channels=(OutputChannel.LATENT_SAMPLE,),
        budget=BudgetCap(flops=15_000, wall_time_s=1.5, energy_j=0.15),
        episode_id=ep_id,
    )


def test_recombine_emits_one_sample_per_episode() -> None:
    state = RecombineOpState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.RECOMBINE,
        recombine_handler(state, rng=random.Random(42)),
    )
    runtime.execute(make_recombine_episode(
        "de-rc0", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    ))
    assert state.total_episodes_handled == 1
    assert state.total_samples_emitted == 1
    assert state.last_sample is not None
    assert len(state.last_sample) == 3
    # Linear interpolation: each component in [0, 1]
    for value in state.last_sample:
        assert 0.0 <= value <= 1.0


def test_recombine_rejects_too_few_latents() -> None:
    state = RecombineOpState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.RECOMBINE,
        recombine_handler(state, rng=random.Random(42)),
    )
    with pytest.raises(ValueError, match="at least 2"):
        runtime.execute(make_recombine_episode(
            "de-rc1", [[1.0, 2.0, 3.0]]
        ))


def test_recombine_accumulates_across_episodes() -> None:
    state = RecombineOpState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.RECOMBINE,
        recombine_handler(state, rng=random.Random(0)),
    )
    latents = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    runtime.execute(make_recombine_episode("de-rc2", latents))
    runtime.execute(make_recombine_episode("de-rc3", latents))
    runtime.execute(make_recombine_episode("de-rc4", latents))
    assert state.total_episodes_handled == 3
    assert state.total_samples_emitted == 3
    assert len(state.sample_history) == 3
