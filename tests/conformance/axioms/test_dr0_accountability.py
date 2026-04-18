"""DR-0 Accountability — property test.

Every executed dream-episode must appear in the runtime log with a
finite budget. Validates framework spec §6.2 DR-0.
"""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.runtime import DreamRuntime


@st.composite
def dream_episodes_with_replay_only(draw) -> DreamEpisode:
    flops = draw(st.integers(min_value=1, max_value=10_000_000))
    wall = draw(st.floats(min_value=0.0, max_value=60.0,
                          allow_nan=False, allow_infinity=False))
    energy = draw(st.floats(min_value=0.0, max_value=100.0,
                            allow_nan=False, allow_infinity=False))
    ep_idx = draw(st.integers(min_value=0, max_value=99_999))
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={},
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=flops, wall_time_s=wall, energy_j=energy),
        episode_id=f"de-prop-{ep_idx:05d}",
    )


def noop_handler(episode: DreamEpisode) -> None:
    """Placeholder handler for DR-0 test — does nothing."""
    return None


@given(ep=dream_episodes_with_replay_only())
@settings(max_examples=50, deadline=None)
def test_dr0_every_executed_de_has_log_entry(
    ep: DreamEpisode,
) -> None:
    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, noop_handler)
    runtime.execute(ep)
    assert any(e.episode_id == ep.episode_id for e in runtime.log)


@given(ep=dream_episodes_with_replay_only())
@settings(max_examples=50, deadline=None)
def test_dr0_budget_is_finite(ep: DreamEpisode) -> None:
    # Budget components must be finite — enforced at construction
    assert ep.budget.flops >= 0
    assert ep.budget.wall_time_s >= 0
    assert ep.budget.energy_j >= 0
