"""Dream-episode 5-tuple dataclass — DR-0 accountability unit.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.1
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EpisodeTrigger(str, Enum):
    SCHEDULED = "scheduled"
    SATURATION = "saturation"
    EXTERNAL = "external"


class Operation(str, Enum):
    REPLAY = "replay"
    DOWNSCALE = "downscale"
    RESTRUCTURE = "restructure"
    RECOMBINE = "recombine"


class OutputChannel(str, Enum):
    WEIGHT_DELTA = "weight_delta"        # canal 1
    LATENT_SAMPLE = "latent_sample"      # canal 2
    HIERARCHY_CHG = "hierarchy_chg"      # canal 3
    ATTENTION_PRIOR = "attention_prior"  # canal 4


@dataclass(frozen=True)
class BudgetCap:
    """Resource cap per DE — K1 invariant enforcement unit."""

    flops: int
    wall_time_s: float
    energy_j: float

    def __post_init__(self) -> None:
        if self.flops < 0:
            raise ValueError(f"flops must be non-negative, got {self.flops}")
        if self.wall_time_s < 0:
            raise ValueError(
                f"wall_time_s must be non-negative, got {self.wall_time_s}"
            )
        if self.energy_j < 0:
            raise ValueError(
                f"energy_j must be non-negative, got {self.energy_j}"
            )


@dataclass(frozen=True)
class DreamEpisode:
    """5-tuple (trigger, input_slice, operation_set, output_channels,
    budget) + episode_id for DR-0 traceability.
    """

    trigger: EpisodeTrigger
    input_slice: dict[str, Any]
    operation_set: tuple[Operation, ...]
    output_channels: tuple[OutputChannel, ...]
    budget: BudgetCap
    episode_id: str

    def __post_init__(self) -> None:
        if not self.operation_set:
            raise ValueError(
                "operation_set must be non-empty — DE with zero "
                "operations has no effect"
            )
