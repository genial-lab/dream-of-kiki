"""Private test-DSL for axiom conformance tests.

Purpose: centralise the boilerplate that recurs in two or more
``tests/conformance/axioms/`` modules — MLX substrate fixtures
(tiny MLP / encoder / decoder), runtime assembly wired with the
four canonical real-weight handlers, byte-equality snapshots and
diff-aware assertions, and a thin ``DreamEpisode`` factory with
sensible defaults.

Leading underscore marks this file as **private**; pytest does not
collect it as a test module (``_dsl.py`` does not match
``test_*.py``), and downstream tests import its helpers explicitly.

Design rules (cf. ``tests/CLAUDE.md``) :

- No hidden global state — every helper is pure or returns a fresh
  object tree.
- Seeds are *inputs*, never defaults mutated in-place (R1 contract).
- Snapshot equality is byte-exact via ``np.array_equal`` + dtype +
  shape check. A diff enumerates the first key that differs so
  failures are actionable.
- Only helpers that appear in 2+ existing tests (or that an
  in-progress test is about to reuse) live here. Single-use helpers
  stay inline in their test module.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §5
"""
from __future__ import annotations

from typing import Any

import numpy as np

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.operations.downscale_real import (
    DownscaleRealState,
    downscale_real_handler,
)
from kiki_oniric.dream.operations.recombine_real import (
    RecombineRealState,
    recombine_real_handler,
)
from kiki_oniric.dream.operations.replay_real import (
    ReplayRealState,
    replay_real_handler,
)
from kiki_oniric.dream.operations.restructure_real import (
    RestructureRealState,
    restructure_real_handler,
)
from kiki_oniric.dream.runtime import DreamRuntime


# ---------------------------------------------------------------------------
# MLX fixture factories — tiny deterministic shapes mirroring the
# ones used by tests/unit/test_real_ops.py and the DR-2' test.
# ---------------------------------------------------------------------------


def make_tiny_mlp() -> Any:
    """Return a fresh :class:`mlx.nn.Module` MLP with layers [4→8, 8→2].

    The attribute surface (``layers`` list, ``input_dim``) matches the
    contract assumed by every real-weight op in ``kiki_oniric.dream.operations``.
    """
    # MLX does not ship type stubs, so under ``mypy --strict`` every
    # reference to ``nn.Module`` / ``nn.Linear`` / ``nn.relu`` surfaces
    # as ``name-defined`` / ``attr-defined`` — identical to the
    # noise present in ``tests/unit/test_real_ops.py``. Silenced
    # locally so this DSL module type-checks clean.
    import mlx.nn as nn

    class _TinyMLP(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.layers = [nn.Linear(4, 8), nn.Linear(8, 2)]  # type: ignore[attr-defined]
            self.input_dim = 4

        def __call__(self, x: Any) -> Any:
            h = nn.relu(self.layers[0](x))  # type: ignore[attr-defined]
            return self.layers[1](h)

    return _TinyMLP()


def make_tiny_encoder() -> Any:
    """Fresh encoder returning ``(mu, log_var=0)`` — sigma collapses to 1."""
    import mlx.nn as nn

    class _TinyEncoder(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)  # type: ignore[attr-defined]

        def __call__(self, x: Any) -> tuple[Any, Any]:
            h = self.fc(x)
            mu = h
            log_var = h * 0.0
            return mu, log_var

    return _TinyEncoder()


def make_tiny_decoder() -> Any:
    """Fresh decoder — single Linear(4, 4)."""
    import mlx.nn as nn

    class _TinyDecoder(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)  # type: ignore[attr-defined]

        def __call__(self, z: Any) -> Any:
            return self.fc(z)

    return _TinyDecoder()


# ---------------------------------------------------------------------------
# Canonical input-slice + episode factory
# ---------------------------------------------------------------------------


# Profile→channel mapping matching ``tests/conformance/axioms/test_dr4_profile_inclusion.py``.
_PROFILE_CHANNELS: dict[str, tuple[OutputChannel, ...]] = {
    "P_min": (OutputChannel.WEIGHT_DELTA,),
    "P_equ": (
        OutputChannel.WEIGHT_DELTA,
        OutputChannel.HIERARCHY_CHG,
        OutputChannel.LATENT_SAMPLE,
    ),
}


def _default_input_slice() -> dict[str, Any]:
    """Minimal input-slice that every real-weight handler will accept.

    Every key is present so the same slice drives REPLAY, DOWNSCALE,
    RESTRUCTURE and RECOMBINE without edits — callers can override
    individual keys through ``make_episode(..., input_slice=...)``.
    """
    return {
        "beta_records": [
            {"x": [0.1, 0.2, 0.3, 0.4], "y": [1.0, 0.0]},
            {"x": [0.5, 0.6, 0.7, 0.8], "y": [0.0, 1.0]},
        ],
        "shrink_factor": 0.97,
        "topo_op": "reroute",
        "swap_indices": [0, 1],
        "delta_latents": [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ],
    }


def make_episode(
    *,
    ops: tuple[Operation, ...],
    seed: int,
    profile: str = "P_min",
    episode_id: str | None = None,
    input_slice: dict[str, Any] | None = None,
) -> DreamEpisode:
    """Build a :class:`DreamEpisode` with sane defaults.

    Parameters
    ----------
    ops :
        Tuple of operations composing the DE. Must be non-empty.
    seed :
        Integer used both as deterministic tag in ``episode_id`` and
        relayed to callers that want to seed MLX with it — this
        factory does **not** touch ``mx.random`` (see
        :func:`seeded_runtime`).
    profile :
        ``"P_min"`` or ``"P_equ"`` — selects the output-channel
        tuple. Other profiles raise ``KeyError`` to force the caller
        to extend this DSL deliberately.
    episode_id :
        Override for the generated id. Default is
        ``f"de-dsl-{seed:05d}"``.
    input_slice :
        Override for the default input slice. Accepts a plain dict;
        :class:`DreamEpisode` wraps it into a :class:`MappingProxyType`.
    """
    channels = _PROFILE_CHANNELS[profile]
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice=input_slice if input_slice is not None else _default_input_slice(),
        operation_set=ops,
        output_channels=channels,
        budget=BudgetCap(flops=1_000_000, wall_time_s=1.0, energy_j=0.1),
        episode_id=episode_id or f"de-dsl-{seed:05d}",
    )


# ---------------------------------------------------------------------------
# Runtime assembly
# ---------------------------------------------------------------------------


class WiredRuntime:
    """Container returned by :func:`seeded_runtime`.

    Bundles the ``DreamRuntime`` together with the fresh substrate
    objects (model, encoder, decoder) and every handler state. Tests
    read whichever fields they need; unused fields cost nothing.
    """

    __slots__ = (
        "runtime",
        "model",
        "encoder",
        "decoder",
        "replay_state",
        "downscale_state",
        "restructure_state",
        "recombine_state",
    )

    def __init__(
        self,
        *,
        runtime: DreamRuntime,
        model: Any,
        encoder: Any,
        decoder: Any,
        replay_state: ReplayRealState,
        downscale_state: DownscaleRealState,
        restructure_state: RestructureRealState,
        recombine_state: RecombineRealState,
    ) -> None:
        self.runtime = runtime
        self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self.replay_state = replay_state
        self.downscale_state = downscale_state
        self.restructure_state = restructure_state
        self.recombine_state = recombine_state


def seeded_runtime(seed: int) -> WiredRuntime:
    """Seed MLX, build fresh substrate, wire all 4 real-weight handlers.

    The process-wide MLX RNG is reset via ``mx.random.seed(seed)`` so
    two calls with the same ``seed`` produce substrate objects whose
    weights are byte-identical at construction time (the R1 anchor
    used by DR-2' and by the DR-2 empirical test).

    The recombine handler is built with ``seed=seed`` as well, so its
    per-episode RNG key derivation is deterministic too.
    """
    import mlx.core as mx

    mx.random.seed(seed)

    model = make_tiny_mlp()
    encoder = make_tiny_encoder()
    decoder = make_tiny_decoder()

    replay_state = ReplayRealState()
    downscale_state = DownscaleRealState()
    restructure_state = RestructureRealState()
    recombine_state = RecombineRealState()

    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.REPLAY,
        replay_real_handler(replay_state, model=model, lr=0.01),
    )
    runtime.register_handler(
        Operation.DOWNSCALE,
        downscale_real_handler(downscale_state, model=model),
    )
    runtime.register_handler(
        Operation.RESTRUCTURE,
        restructure_real_handler(restructure_state, model=model),
    )
    runtime.register_handler(
        Operation.RECOMBINE,
        recombine_real_handler(
            recombine_state, encoder=encoder, decoder=decoder, seed=seed
        ),
    )

    return WiredRuntime(
        runtime=runtime,
        model=model,
        encoder=encoder,
        decoder=decoder,
        replay_state=replay_state,
        downscale_state=downscale_state,
        restructure_state=restructure_state,
        recombine_state=recombine_state,
    )


# ---------------------------------------------------------------------------
# Snapshots + byte-equality assertions
# ---------------------------------------------------------------------------


def snapshot_state(wired: WiredRuntime) -> dict[str, Any]:
    """Serialisable snapshot of a wired runtime — tensors as numpy arrays.

    Keys are stable strings so two snapshots of two runs can be
    compared key-by-key. The snapshot covers :

    - MLP weights + biases (``mlp_layer_{i}_{weight,bias}``)
    - Encoder / decoder weights + biases
    - Restructure ``diff_history`` (topology mutation trace)
    - Recombine ``last_sample`` (latent draw)
    - Compound downscale factor
    - Total replay records consumed

    Values are plain python / numpy types so downstream code can
    json-dump, hash, or diff them without extra conversion.
    """
    snapshot: dict[str, Any] = {}

    for idx, layer in enumerate(wired.model.layers):
        snapshot[f"mlp_layer_{idx}_weight"] = np.asarray(layer.weight).copy()
        snapshot[f"mlp_layer_{idx}_bias"] = np.asarray(layer.bias).copy()

    snapshot["encoder_fc_weight"] = np.asarray(wired.encoder.fc.weight).copy()
    snapshot["encoder_fc_bias"] = np.asarray(wired.encoder.fc.bias).copy()
    snapshot["decoder_fc_weight"] = np.asarray(wired.decoder.fc.weight).copy()
    snapshot["decoder_fc_bias"] = np.asarray(wired.decoder.fc.bias).copy()

    snapshot["restructure_diff_history"] = list(
        wired.restructure_state.diff_history
    )
    snapshot["recombine_last_sample"] = (
        list(wired.recombine_state.last_sample)
        if wired.recombine_state.last_sample is not None
        else None
    )
    snapshot["downscale_compound_factor"] = (
        wired.downscale_state.compound_factor
    )
    snapshot["replay_total_records"] = (
        wired.replay_state.total_records_consumed
    )

    return snapshot


def _values_equal(a: Any, b: Any) -> bool:
    """Byte-equality over mixed scalar / ndarray / list values."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and a.dtype == b.dtype and bool(
            np.array_equal(a, b)
        )
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return False
    return bool(a == b)


def assert_states_equal(
    a: dict[str, Any], b: dict[str, Any], *, msg: str = ""
) -> None:
    """Byte-for-byte comparison of two snapshots.

    Fails with a diff pinpointing the first key that differs so the
    caller sees **which** piece of state diverged, not merely that
    something did. Missing keys on either side are reported with the
    offending key name.
    """
    keys_a = set(a.keys())
    keys_b = set(b.keys())
    if keys_a != keys_b:
        missing_in_b = sorted(keys_a - keys_b)
        missing_in_a = sorted(keys_b - keys_a)
        prefix = f"{msg}: " if msg else ""
        raise AssertionError(
            f"{prefix}snapshot key mismatch "
            f"(missing in b: {missing_in_b}, missing in a: {missing_in_a})"
        )

    for key in sorted(keys_a):
        if not _values_equal(a[key], b[key]):
            prefix = f"{msg}: " if msg else ""
            raise AssertionError(
                f"{prefix}snapshot differs at key {key!r}: "
                f"left={a[key]!r} right={b[key]!r}"
            )


__all__ = [
    "WiredRuntime",
    "assert_states_equal",
    "make_episode",
    "make_tiny_decoder",
    "make_tiny_encoder",
    "make_tiny_mlp",
    "seeded_runtime",
    "snapshot_state",
]
