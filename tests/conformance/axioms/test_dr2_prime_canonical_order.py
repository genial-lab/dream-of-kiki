"""DR-2' Canonical-order compositionality — determinism conformance test.

DR-2' (fallback contract — DR-2 is now proved under precondition,
see docs/proofs/dr2-compositionality.md v0.2 (2026-04-21) ; DR-2'
retained as stricter canonical-order contract for
P_min/P_equ/P_max profiles) states
that operations applied in the canonical order are compositional
within the operation set :

    canonical order = replay < downscale < restructure < recombine

Operationally, this test verifies that chaining the four canonical
operations through a single :class:`DreamRuntime` run under an
identical seed produces a **byte-identical** final state across
two independent runs. This is the empirical contract retained by
the G2/G4 pilots until a strict DR-2 proof is available.

Reference :
  docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
  (DR-2 unproven working axiom) and the DR-2' fallback definition
  in the same section.
"""
from __future__ import annotations

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx.nn")

from kiki_oniric.dream.episode import Operation  # noqa: E402

from tests.conformance.axioms._dsl import (  # noqa: E402
    assert_states_equal,
    make_episode,
    seeded_runtime,
    snapshot_state,
)


_CANONICAL_OPS: tuple[Operation, ...] = (
    Operation.REPLAY,
    Operation.DOWNSCALE,
    Operation.RESTRUCTURE,
    Operation.RECOMBINE,
)


def test_dr2_prime_canonical_order_is_deterministic() -> None:
    """DR-2' (fallback, DR-2 unproven — see spec §6.2).

    Applying REPLAY → DOWNSCALE → RESTRUCTURE → RECOMBINE in a single
    DE under identical initial conditions (same seed, same input
    slice) must yield byte-identical final model weights and an
    identical recombine latent sample across two independent runs.
    """
    wired1 = seeded_runtime(seed=7)
    wired1.runtime.execute(
        make_episode(
            ops=_CANONICAL_OPS, seed=7, profile="P_equ", episode_id="de-dr2p-1"
        )
    )

    wired2 = seeded_runtime(seed=7)
    wired2.runtime.execute(
        make_episode(
            ops=_CANONICAL_OPS, seed=7, profile="P_equ", episode_id="de-dr2p-2"
        )
    )

    # Byte-identical final state — weights, topology trace, latent sample,
    # compound factor, records consumed.
    assert_states_equal(
        snapshot_state(wired1),
        snapshot_state(wired2),
        msg=(
            "DR-2' violated — canonical-order composition is not "
            "deterministic under identical seed"
        ),
    )

    # Runtime log records the canonical-order execution identically.
    log1 = wired1.runtime.log
    log2 = wired2.runtime.log
    assert len(log1) == len(log2) == 1
    assert log1[0].completed is True
    assert log2[0].completed is True
    assert log1[0].operations_executed == log2[0].operations_executed == _CANONICAL_OPS

    # Restructure diff trace is identical (already covered by the
    # snapshot, asserted here for intent clarity).
    assert wired1.restructure_state.diff_history == ["reroute"]
    assert wired2.restructure_state.diff_history == ["reroute"]
