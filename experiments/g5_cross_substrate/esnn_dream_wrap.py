"""E-SNN dream-episode wrapper for the G5 cross-substrate pilot.

Mirrors `experiments.g4_split_fmnist.dream_wrap.dream_episode` but
constructs profiles whose op handlers are rebound to the E-SNN
substrate's factory handlers, wrapped in adapters that translate
between the runtime's ``OperationHandler =
Callable[[DreamEpisode], None]`` contract and the E-SNN factories'
typed signatures.

The function `dream_episode(classifier, profile, seed)` is a free
function (not a method) because the classifier owns no mutable
runtime state — the runtime lives on the profile.

DR-0 accountability is automatic : every call to `dream_episode`
appends one `EpisodeLogEntry` to `profile.runtime.log` regardless
of handler outcome. Classifier weights are **not** mutated by this
call (G5 isolates the E-SNN substrate dispatch path ; the G4-bis
weight-mutating coupling is left as future work for a G5-bis
follow-up).

Reference :
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §3.1
    kiki_oniric/substrates/esnn_thalamocortical.py
    experiments/g4_split_fmnist/dream_wrap.py (sister module)
"""
from __future__ import annotations

import random
from typing import Any, Callable

import numpy as np

from experiments.g5_cross_substrate.esnn_classifier import EsnnG5Classifier
from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.profiles.p_equ import PEquProfile
from kiki_oniric.profiles.p_max import PMaxProfile
from kiki_oniric.profiles.p_min import PMinProfile
from kiki_oniric.substrates.esnn_thalamocortical import EsnnSubstrate


ProfileT = PMinProfile | PEquProfile | PMaxProfile


PROFILE_FACTORIES: dict[str, Callable[..., ProfileT]] = {
    "P_min": PMinProfile,
    "P_equ": PEquProfile,
    "P_max": PMaxProfile,
}


def build_esnn_profile(name: str, seed: int) -> ProfileT:
    """Construct a profile and rebind its op handlers to E-SNN adapters.

    Profiles in `kiki_oniric.profiles.*` ship default MLX/numpy
    handlers ; G5 needs the same profile *shape* but with the
    E-SNN substrate's handlers bound. We rebind via the runtime's
    public `register_handler(op, handler)` method, wrapping each
    E-SNN factory in a `Callable[[DreamEpisode], None]` adapter.
    The profile API is unchanged from the caller's point of view —
    `runtime.execute(episode)` dispatches through the rebound
    handlers, which is precisely what makes the pilot a
    *cross-substrate* test.
    """
    if name not in PROFILE_FACTORIES:
        raise ValueError(
            f"unknown profile {name!r} — expected one of "
            f"{sorted(PROFILE_FACTORIES)}"
        )
    factory = PROFILE_FACTORIES[name]
    profile: ProfileT
    if name == "P_min":
        profile = factory()
    else:
        profile = factory(rng=random.Random(seed))
    _rebind_to_esnn(profile)
    return profile


def _make_replay_adapter(
    substrate: EsnnSubstrate,
) -> Callable[[DreamEpisode], None]:
    """Wrap the E-SNN replay factory in a DreamEpisode-shaped handler."""
    inner = substrate.replay_handler_factory()

    def handler(episode: DreamEpisode) -> None:
        records = episode.input_slice.get("beta_records", [])
        n_steps = int(episode.input_slice.get("n_steps", 20))
        # Discard return value : DR-0 logging is enough for the
        # cross-substrate dispatch comparison.
        inner(list(records), n_steps)

    handler.__esnn__ = True  # type: ignore[attr-defined]
    return handler


def _make_downscale_adapter(
    substrate: EsnnSubstrate,
) -> Callable[[DreamEpisode], None]:
    inner = substrate.downscale_handler_factory()

    def handler(episode: DreamEpisode) -> None:
        # Use a small dummy weight tensor — the cross-substrate
        # comparison cares about dispatch, not about the magnitude
        # of the returned scaling.
        weights = np.ones(4, dtype=float)
        factor = float(episode.input_slice.get("shrink_factor", 0.99))
        inner(weights, factor)

    handler.__esnn__ = True  # type: ignore[attr-defined]
    return handler


def _make_restructure_adapter(
    substrate: EsnnSubstrate,
) -> Callable[[DreamEpisode], None]:
    inner = substrate.restructure_handler_factory()

    def handler(episode: DreamEpisode) -> None:
        conn = np.eye(4, dtype=float)
        op = str(episode.input_slice.get("topo_op", "reroute"))
        swap = list(episode.input_slice.get("swap_indices", [0, 1]))
        src = int(swap[0]) if len(swap) >= 1 else 0
        dst = int(swap[1]) if len(swap) >= 2 else 1
        inner(conn, op, src, dst)

    handler.__esnn__ = True  # type: ignore[attr-defined]
    return handler


def _make_recombine_adapter(
    substrate: EsnnSubstrate,
) -> Callable[[DreamEpisode], None]:
    inner = substrate.recombine_handler_factory()

    def handler(episode: DreamEpisode) -> None:
        delta = episode.input_slice.get("delta_latents", [])
        if not delta:
            latents = np.zeros((2, 4), dtype=float)
        else:
            arr = np.asarray(
                [list(d) for d in delta], dtype=float
            )
            if arr.shape[0] < 2:
                arr = np.vstack(
                    [arr, np.zeros((2 - arr.shape[0], arr.shape[1]))]
                )
            latents = arr
        seed = int(episode.input_slice.get("seed", 0))
        inner(latents, seed=seed, n_steps=10)

    handler.__esnn__ = True  # type: ignore[attr-defined]
    return handler


def _rebind_to_esnn(profile: ProfileT) -> None:
    """Overwrite `profile.runtime`'s op handlers with E-SNN adapters.

    Uses ``DreamRuntime.register_handler`` (the public API) to
    overwrite each of the four ops with an adapter that wraps the
    matching ``EsnnSubstrate.<op>_handler_factory()`` and
    translates `DreamEpisode.input_slice` kwargs to the factory's
    typed signature.
    """
    substrate = EsnnSubstrate()
    adapters: dict[Operation, Callable[[DreamEpisode], None]] = {
        Operation.REPLAY: _make_replay_adapter(substrate),
        Operation.DOWNSCALE: _make_downscale_adapter(substrate),
        Operation.RESTRUCTURE: _make_restructure_adapter(substrate),
        Operation.RECOMBINE: _make_recombine_adapter(substrate),
    }
    # Overwrite only the keys the profile already exposes — we
    # don't extend the profile's op set (DR-4 inclusion is
    # preserved). Inspect the runtime's private handler map only
    # to read the registered op set ; rebinding goes through the
    # public ``register_handler`` API.
    registered = list(
        profile.runtime._handlers.keys()  # type: ignore[attr-defined]
    )
    for op in registered:
        if op in adapters:
            profile.runtime.register_handler(op, adapters[op])


def _sample_beta_records(
    seed: int, n_records: int, feat_dim: int
) -> list[dict[str, Any]]:
    """Re-derive `n_records` deterministic beta records from `seed`.

    Mirrors `experiments.g4_split_fmnist.dream_wrap.sample_beta_records`
    plus an `input` key carrying a third feature vector — required
    by the E-SNN replay handler factory which reads
    ``record["input"]`` (not ``record["x"]``).
    """
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for _ in range(n_records):
        out.append(
            {
                "x": rng.standard_normal(feat_dim)
                .astype(np.float32)
                .tolist(),
                "y": rng.standard_normal(feat_dim)
                .astype(np.float32)
                .tolist(),
                "input": rng.standard_normal(feat_dim)
                .astype(np.float32)
                .tolist(),
            }
        )
    return out


def dream_episode(
    classifier: EsnnG5Classifier, profile: ProfileT, seed: int
) -> None:
    """Drive one `DreamEpisode` through the E-SNN-rebound profile.

    Builds an episode whose `operation_set` matches the profile's
    wired handlers (P_min : replay+downscale ; P_equ/P_max :
    +restructure+recombine), and dispatches via
    `profile.runtime.execute`. The classifier weights are **not**
    mutated by this call — see module docstring.
    """
    profile_name = type(profile).__name__
    if isinstance(profile, PMinProfile):
        ops: tuple[Operation, ...] = (
            Operation.REPLAY,
            Operation.DOWNSCALE,
        )
        channels: tuple[OutputChannel, ...] = (
            OutputChannel.WEIGHT_DELTA,
        )
    else:
        ops = (
            Operation.REPLAY,
            Operation.DOWNSCALE,
            Operation.RESTRUCTURE,
            Operation.RECOMBINE,
        )
        channels = (
            OutputChannel.WEIGHT_DELTA,
            OutputChannel.HIERARCHY_CHG,
            OutputChannel.LATENT_SAMPLE,
        )
    beta_records = _sample_beta_records(
        seed=seed, n_records=4, feat_dim=4
    )
    rng = np.random.default_rng(seed + 10_000)
    delta_latents = [
        rng.standard_normal(4).astype(np.float32).tolist()
        for _ in range(2)
    ]
    # Touch classifier reference once to satisfy lint without
    # mutating it — keeps the spectator-only contract explicit.
    _ = classifier.W_out.shape
    episode = DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={
            "beta_records": beta_records,
            "shrink_factor": 0.99,
            "topo_op": "reroute",
            "swap_indices": [0, 1],
            "delta_latents": delta_latents,
            "seed": seed,
            "n_steps": 20,
        },
        operation_set=ops,
        output_channels=channels,
        budget=BudgetCap(
            flops=10_000_000, wall_time_s=10.0, energy_j=1.0
        ),
        episode_id=f"g5-{profile_name}-seed{seed}",
    )
    profile.runtime.execute(episode)
