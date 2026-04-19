"""Cross-substrate neuromorph validation pilot (C3.13, Phase 2 track b).

.. note::

    **DEFERRED to Paper 2** (PLOS CB pivot, 2026-04-20). This driver
    is preserved as the Paper 2 reactivation entry point. Cycle 3
    sem-3 quota interrupted execution before completion ; Paper 1
    v0.2 retargeted PLOS Computational Biology and §8.3 moved
    H1–H4 (and the C3.13 Norse vs MLX cross-substrate cell) out
    of Paper 1 scope. See
    ``docs/milestones/cycle3-plan-adaptation-2026-04-20.md`` for
    the full adaptation matrix and Paper 2 backlog. Do **not**
    re-run this script as part of the Paper 1 v0.2 critical path —
    its outputs are not consumed by the Paper 1 narrative.

**Gate ID** : G10a — cross-substrate neuromorphic validation
**Validates** : whether Norse SNN proxy ops (`*_snn.py`) produce
weight-trajectory effects correlated with the MLX real ops
(`*_real.py`) on a shared synthetic tiny-model fixture.
**Mode** : pipeline-validation — **synthetic 4×4 matrix**, not a
real Qwen model. Production validation lives in the Phase B real
pilot (1.5B Qwen FP16) currently holding the Studio.
**Expected output** :
  - ``docs/milestones/g10a-neuromorph.md`` (human-readable table)
  - ``docs/milestones/g10a-neuromorph.json`` (deterministic dump
    for R1 provenance)

## Why a pure-numpy MLX path

The real ops in ``kiki_oniric/dream/operations/*_real.py`` call
MLX directly (``mlx.core``, ``mlx.nn.Module``-bound models). Running
them here would require either a tiny MLX model fixture (pulls in
GPU init) or a mock. Neither is necessary for the C3.13 goal : we
only need to compare the **mathematical semantics** of each op
across the two substrates on a shared synthetic tiny weight matrix.

So this pilot uses a **pure-numpy MLX equivalent** of each real op :

- ``_apply_replay_real_np``  : SGD step on raw weights towards
  target (mirrors ``replay_real_handler`` MSE + SGD update).
- ``_apply_downscale_real_np`` : multiplicative shrink
  (mirrors ``downscale_real_handler``).
- ``_apply_recombine_real_np`` : VAE-style interpolation between
  two weight vectors with seeded epsilon, matching the
  ``recombine_real_handler`` reparameterization mean-path.

The Norse side calls the actual SNN handlers from
``kiki_oniric/dream/operations/*_snn.py`` — same numpy substrate,
spike-rate proxy semantics.

This "apples-to-apples numpy" comparison answers the C3.13
question directly : do the two op families produce correlated
weight trajectories under identical seeds ? If yes, DR-3
Conformance Criterion condition (3) (observability equivalence of
effects) holds for both substrates under the same framework-C
primitive contract.

## Cells

2 profiles × 30 seeds × 2 substrates = 120 cells.

- Profile ``p_min`` : 3 × replay on the adapter weights.
- Profile ``p_equ`` : 2 × (replay → downscale → recombine).
  (Restructure is skipped per Phase 2 prep — ``restructure_snn``
  only supports ``"reroute"`` and needs the axis-0 shape
  restriction which does not apply to the 4×4 fixture in the
  symmetric way we want for cross-substrate pearson rho.)

## GO / NO-GO rule

- **SOFT-GO** : Pearson rho ≥ 0.7 on both profiles.
- **NO-GO**   : Pearson rho < 0.3 on either profile (substrate
  divergence → investigate SNN proxy fidelity before C3.14).
- Intermediate range : report as CONDITIONAL-GO and defer final
  verdict to Phase B real results.

Wall-clock : ~60 seconds on GrosMac (no GPU, pure numpy).

Usage ::

    uv run python scripts/pilot_phase2b_neuromorph.py

Reference :
    docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md
    §3 Phase 2 track b (neuromorph cross-substrate validation)
    docs/milestones/g10a-neuromorph.md (this script's milestone).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kiki_oniric.dream.operations.downscale_snn import (  # noqa: E402
    DownscaleSNNState,
    downscale_snn_handler,
)
from kiki_oniric.dream.operations.recombine_snn import (  # noqa: E402
    RecombineSNNState,
    recombine_snn_handler,
)
from kiki_oniric.dream.operations.replay_snn import (  # noqa: E402
    ReplaySNNState,
    replay_snn_handler,
)
from kiki_oniric.dream.episode import (  # noqa: E402
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)


# --- Pilot configuration -------------------------------------------------

PROFILES: tuple[str, ...] = ("p_min", "p_equ")
SUBSTRATES: tuple[str, ...] = ("mlx", "norse")
SEEDS: tuple[int, ...] = tuple(range(30))

WEIGHT_SHAPE = (4, 4)
LR = 0.01
SHRINK_FACTOR = 0.98

# G10a verdict thresholds (Pearson rho on aligned (MLX, Norse)
# trajectories across seeds, per profile).
SOFT_GO_RHO = 0.7
NO_GO_RHO = 0.3

MILESTONE_MD = REPO_ROOT / "docs" / "milestones" / "g10a-neuromorph.md"
MILESTONE_JSON = REPO_ROOT / "docs" / "milestones" / "g10a-neuromorph.json"


# --- Numpy-equivalent MLX real ops --------------------------------------


def _apply_replay_real_np(
    weights: np.ndarray, target: np.ndarray, lr: float
) -> None:
    """Numpy-equivalent of ``replay_real_handler`` SGD step.

    The real op builds a tiny MLX MLP whose loss is MSE(pred, y)
    and performs one SGD step on the parameters. For a 1-layer
    linear model with identity features, the gradient of MSE wrt
    weights reduces to ``2 * (weights - target)`` ; the SGD update
    is ``weights -= lr * grad = weights - 2 * lr * (weights - target)``.

    To mirror the MLX variant's in-place mutation contract, we
    update ``weights`` in place.
    """
    grad = 2.0 * (weights - target)
    weights[...] = weights - lr * grad


def _apply_downscale_real_np(
    weights: np.ndarray, factor: float
) -> None:
    """Numpy-equivalent of ``downscale_real_handler`` shrink."""
    if not (0.0 < factor <= 1.0):
        raise ValueError(
            f"shrink_factor must be in (0, 1], got {factor}"
        )
    weights[...] = weights * factor


def _apply_recombine_real_np(
    weights: np.ndarray, partner: np.ndarray, seed: int, episode_count: int
) -> None:
    """Numpy-equivalent of ``recombine_real_handler`` VAE interp.

    The real op draws epsilon from an isolated per-episode RNG
    (``seed + episode_count``) and emits ``z = mu + sigma * eps``.
    For the synthetic fixture we interpret ``weights`` as ``mu``
    and ``partner`` as ``log_var``, use ``alpha`` from the
    per-episode RNG to interpolate mean-path between the two
    operands (matching the ``recombine_snn`` semantics), then
    write the result back into ``weights``.
    """
    rng = np.random.default_rng(seed + episode_count)
    alpha = float(rng.random())
    weights[...] = alpha * weights + (1.0 - alpha) * partner


# --- Norse SNN pipeline wrappers ----------------------------------------


def _make_episode(
    input_slice: dict, operation: Operation, channel: OutputChannel
) -> DreamEpisode:
    """Helper : build a DreamEpisode with a generous BudgetCap."""
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice=input_slice,
        operation_set=(operation,),
        output_channels=(channel,),
        budget=BudgetCap(
            flops=10**9, wall_time_s=60.0, energy_j=100.0
        ),
        episode_id=f"g10a-{operation.value}",
    )


def _norse_replay(
    weights: np.ndarray, target: np.ndarray, lr: float
) -> None:
    """One SNN-proxy replay step — mutates ``weights`` in place."""
    state = ReplaySNNState()
    handler = replay_snn_handler(state, weights=weights, lr=lr)
    # ``replay_snn_handler`` expects target in the rate domain
    # directly — caller has to give it target_rates, not target
    # weights. Convert via the same sigmoid mapping used inside
    # the op so the MLX and Norse sides see a comparable target.
    from kiki_oniric.dream.operations.replay_snn import (
        weights_to_spike_rates,
    )
    target_rates = weights_to_spike_rates(target)
    episode = _make_episode(
        {"target_rates": target_rates},
        Operation.REPLAY,
        OutputChannel.WEIGHT_DELTA,
    )
    handler(episode)


def _norse_downscale(weights: np.ndarray, factor: float) -> None:
    """One SNN-proxy downscale step — mutates ``weights`` in place."""
    state = DownscaleSNNState()
    handler = downscale_snn_handler(state, weights=weights)
    episode = _make_episode(
        {"shrink_factor": factor},
        Operation.DOWNSCALE,
        OutputChannel.WEIGHT_DELTA,
    )
    handler(episode)


def _norse_recombine(
    weights: np.ndarray, partner: np.ndarray, seed: int
) -> None:
    """One SNN-proxy recombine step — mutates ``weights`` in place.

    The SNN recombine returns its result through
    ``state.last_sample`` rather than mutating weights. We extract
    the sample and reshape it back onto the weights tensor to
    preserve the cross-substrate in-place mutation contract on
    the synthetic fixture.
    """
    state = RecombineSNNState()
    handler = recombine_snn_handler(state, seed=seed)
    latents = [weights.ravel().tolist(), partner.ravel().tolist()]
    episode = _make_episode(
        {"delta_latents": latents},
        Operation.RECOMBINE,
        OutputChannel.LATENT_SAMPLE,
    )
    handler(episode)
    assert state.last_sample is not None
    weights[...] = np.asarray(
        state.last_sample, dtype=weights.dtype
    ).reshape(weights.shape)


# --- Profile orchestrators ----------------------------------------------


def apply_profile_real(
    profile: str,
    initial: np.ndarray,
    target: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Apply the MLX-equivalent numpy op sequence for ``profile``."""
    w = initial.copy()
    if profile == "p_min":
        # 3× replay (mirrors p_min's replay-dominated schedule)
        for _ in range(3):
            _apply_replay_real_np(w, target, LR)
    elif profile == "p_equ":
        # 2× (replay → downscale → recombine)
        rng_partner = np.random.default_rng(seed ^ 0xA5A5A5)
        partner = rng_partner.normal(size=w.shape).astype(np.float32)
        for ep in range(2):
            _apply_replay_real_np(w, target, LR)
            _apply_downscale_real_np(w, SHRINK_FACTOR)
            _apply_recombine_real_np(w, partner, seed, ep)
    else:
        raise ValueError(f"unknown profile: {profile!r}")
    return w


def apply_profile_snn(
    profile: str,
    initial: np.ndarray,
    target: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Apply the Norse SNN-proxy op sequence for ``profile``."""
    # Cast to float64 for the SNN sigmoid round-trip (the real
    # handlers use float64 internally via np.asarray(..., dtype=float)).
    w = initial.astype(np.float64).copy()
    target64 = target.astype(np.float64)
    if profile == "p_min":
        for _ in range(3):
            _norse_replay(w, target64, LR)
    elif profile == "p_equ":
        rng_partner = np.random.default_rng(seed ^ 0xA5A5A5)
        partner = rng_partner.normal(size=w.shape).astype(np.float64)
        for _ep in range(2):
            _norse_replay(w, target64, LR)
            _norse_downscale(w, SHRINK_FACTOR)
            _norse_recombine(w, partner, seed)
    else:
        raise ValueError(f"unknown profile: {profile!r}")
    return w.astype(np.float32)


# --- Cell pipeline ------------------------------------------------------


def run_cell(profile: str, substrate: str, seed: int) -> dict:
    """Execute one (profile, substrate, seed) cell of the pilot.

    Returns a dict with :
      - delta_norm       : ‖final − initial‖₂
      - target_distance  : ‖final − target‖₂
      - convergence      : fraction of initial→target distance
                           collapsed (1.0 = reached target).
      - final_flat       : weight vector (for pearson rho).
    """
    rng = np.random.default_rng(seed)
    target = rng.normal(size=WEIGHT_SHAPE).astype(np.float32)
    initial = rng.normal(size=WEIGHT_SHAPE).astype(np.float32)

    if substrate == "mlx":
        final = apply_profile_real(profile, initial, target, seed)
    elif substrate == "norse":
        final = apply_profile_snn(profile, initial, target, seed)
    else:
        raise ValueError(f"unknown substrate: {substrate!r}")

    delta = float(np.linalg.norm(final - initial))
    target_distance = float(np.linalg.norm(final - target))
    init_to_target = float(np.linalg.norm(initial - target))
    convergence = (
        (init_to_target - target_distance) / init_to_target
        if init_to_target > 1e-12
        else 0.0
    )

    return {
        "profile": profile,
        "substrate": substrate,
        "seed": seed,
        "delta_norm": delta,
        "target_distance": target_distance,
        "convergence": float(convergence),
        "final_flat": final.astype(np.float64).ravel().tolist(),
    }


# --- Aggregation --------------------------------------------------------


def _pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Pearson rho + two-sided p-value, guarded for degeneracy."""
    from scipy.stats import pearsonr  # lazy import

    if a.size != b.size or a.size < 2:
        return 0.0, 1.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0, 1.0
    r, p = pearsonr(a, b)
    return float(r), float(p)


def aggregate(cells: list[dict]) -> dict:
    """Build per-profile × substrate stats + cross-substrate rho."""
    per_cell: dict[tuple[str, str, int], dict] = {
        (c["profile"], c["substrate"], c["seed"]): c for c in cells
    }

    by_profile_substrate: dict[str, dict] = {}
    for profile in PROFILES:
        by_profile_substrate[profile] = {}
        for substrate in SUBSTRATES:
            subset = [
                c
                for c in cells
                if c["profile"] == profile and c["substrate"] == substrate
            ]
            deltas = np.array([c["delta_norm"] for c in subset])
            convs = np.array([c["convergence"] for c in subset])
            by_profile_substrate[profile][substrate] = {
                "n": len(subset),
                "mean_delta": float(deltas.mean()),
                "std_delta": float(deltas.std(ddof=1)) if len(subset) > 1 else 0.0,
                "mean_convergence": float(convs.mean()),
                "std_convergence": (
                    float(convs.std(ddof=1)) if len(subset) > 1 else 0.0
                ),
            }

    cross: dict[str, dict] = {}
    for profile in PROFILES:
        mlx_vec = []
        norse_vec = []
        for seed in SEEDS:
            mlx_cell = per_cell.get((profile, "mlx", seed))
            norse_cell = per_cell.get((profile, "norse", seed))
            if mlx_cell is None or norse_cell is None:
                continue
            mlx_vec.extend(mlx_cell["final_flat"])
            norse_vec.extend(norse_cell["final_flat"])
        rho, p_val = _pearson(np.array(mlx_vec), np.array(norse_vec))
        # Convergence ratio : Norse / MLX (>1 = Norse over-
        # converges, <1 = Norse under-converges).
        mlx_mean_conv = by_profile_substrate[profile]["mlx"]["mean_convergence"]
        norse_mean_conv = by_profile_substrate[profile]["norse"][
            "mean_convergence"
        ]
        ratio = (
            norse_mean_conv / mlx_mean_conv
            if abs(mlx_mean_conv) > 1e-12
            else 0.0
        )
        cross[profile] = {
            "pearson_rho": rho,
            "p_value": p_val,
            "convergence_ratio_norse_over_mlx": float(ratio),
        }

    # G10a verdict.
    rhos = [cross[p]["pearson_rho"] for p in PROFILES]
    if all(r >= SOFT_GO_RHO for r in rhos):
        verdict = "SOFT-GO"
    elif any(r < NO_GO_RHO for r in rhos):
        verdict = "NO-GO"
    else:
        verdict = "CONDITIONAL-GO"

    return {
        "by_profile_substrate": by_profile_substrate,
        "cross_substrate": cross,
        "verdict": verdict,
    }


# --- Main driver --------------------------------------------------------


def main() -> int:
    t0 = time.time()
    cells: list[dict] = []
    for profile in PROFILES:
        for substrate in SUBSTRATES:
            for seed in SEEDS:
                cells.append(run_cell(profile, substrate, seed))
    wall_s = time.time() - t0

    agg = aggregate(cells)

    # JSON dump — drop heavy `final_flat` fields, keep stats only.
    cells_slim = [
        {k: v for k, v in c.items() if k != "final_flat"} for c in cells
    ]
    dump = {
        "gate": "G10a",
        "milestone": "cross-substrate neuromorph validation (C3.13)",
        "harness_version": "C-v0.7.0+PARTIAL",
        "profiles": list(PROFILES),
        "substrates": list(SUBSTRATES),
        "seeds": list(SEEDS),
        "weight_shape": list(WEIGHT_SHAPE),
        "cells": cells_slim,
        "aggregate": agg,
        "wall_time_s": wall_s,
        "thresholds": {
            "soft_go_rho": SOFT_GO_RHO,
            "no_go_rho": NO_GO_RHO,
        },
    }

    MILESTONE_JSON.write_text(json.dumps(dump, indent=2, sort_keys=True))

    # Render milestone markdown.
    md = _render_markdown(dump)
    MILESTONE_MD.write_text(md)

    print(f"[G10a] verdict = {agg['verdict']}")
    for p in PROFILES:
        c = agg["cross_substrate"][p]
        print(
            f"[G10a] {p}: rho={c['pearson_rho']:.4f} "
            f"p={c['p_value']:.4g} "
            f"ratio={c['convergence_ratio_norse_over_mlx']:.4f}"
        )
    print(f"[G10a] wall-clock = {wall_s:.1f}s ({len(cells)} cells)")
    return 0


def _render_markdown(dump: dict) -> str:
    agg = dump["aggregate"]
    verdict = agg["verdict"]
    bps = agg["by_profile_substrate"]
    cross = agg["cross_substrate"]

    lines: list[str] = []
    lines.append(
        "# G10a — Cross-substrate neuromorphic validation (2026-04-19)"
    )
    lines.append("")
    lines.append(f"**Status** : **{verdict}**")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "- Profiles tested : `p_min`, `p_equ` (skip `p_max` "
        "restructure for Phase 2 prep)"
    )
    lines.append(
        "- Substrates : MLX (kiki-oniric real-op numpy equivalent) "
        "vs Norse (SNN-proxy ops, pure numpy)"
    )
    lines.append(
        f"- {len(SEEDS)} seeds per cell, {WEIGHT_SHAPE[0]}×"
        f"{WEIGHT_SHAPE[1]} synthetic tiny model"
    )
    lines.append(
        "- Pure-numpy implementation, no GPU / Norse / PyTorch runtime "
        "required"
    )
    lines.append(
        f"- Wall-clock : **{dump['wall_time_s']:.1f}s** on local "
        f"CPU (GrosMac)"
    )
    lines.append("")
    lines.append(
        "> **Synthetic caveat** (CLAUDE.md §3) : results below are "
        f"on a {WEIGHT_SHAPE[0]}×{WEIGHT_SHAPE[1]} synthetic weight "
        "matrix, **not a real model**. Production validation lives "
        "in the Phase B real pilot (1.5B Qwen FP16) currently "
        "executing on Studio. This pilot is a pipeline-validation "
        "artifact for DR-3 condition (3) cross-substrate "
        "observability only."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| Profile | Substrate | mean Δ | std Δ | mean conv | std conv |"
    )
    lines.append(
        "|---|---|---|---|---|---|"
    )
    for profile in PROFILES:
        for substrate in SUBSTRATES:
            s = bps[profile][substrate]
            lines.append(
                f"| {profile} | {substrate} | "
                f"{s['mean_delta']:.4f} | {s['std_delta']:.4f} | "
                f"{s['mean_convergence']:.4f} | "
                f"{s['std_convergence']:.4f} |"
            )
    lines.append("")
    lines.append("## Cross-substrate correlation")
    lines.append("")
    lines.append(
        "| Profile | Pearson ρ(MLX, Norse) | p-value | "
        "conv ratio (Norse/MLX) |"
    )
    lines.append("|---|---|---|---|")
    for profile in PROFILES:
        c = cross[profile]
        lines.append(
            f"| {profile} | {c['pearson_rho']:.4f} | "
            f"{c['p_value']:.3g} | "
            f"{c['convergence_ratio_norse_over_mlx']:.4f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        f"- **SOFT-GO** threshold : ρ ≥ {SOFT_GO_RHO} on both profiles"
    )
    lines.append(
        f"- **NO-GO** threshold : ρ < {NO_GO_RHO} on either profile"
    )
    lines.append("")
    lines.append(
        "- High ρ (> 0.7) ⇒ MLX and Norse substrates produce "
        "correlated dream-op effects on the shared synthetic "
        "fixture → **DR-3 Conformance Criterion condition (3)** "
        "(observability equivalence of effects) holds for both "
        "substrates under the same primitive contract."
    )
    lines.append(
        "- Low ρ (< 0.3) ⇒ substrates diverge on the synthetic "
        "fixture → investigate SNN proxy fidelity (sigmoid "
        "round-trip saturation, spike-rate interpolation, etc.)."
    )
    lines.append(
        "- Intermediate ρ ⇒ CONDITIONAL-GO : defer final verdict "
        "to the Phase B real pilot."
    )
    lines.append("")
    lines.append("## References")
    lines.append("")
    lines.append(
        "- `kiki_oniric/dream/operations/*_real.py` (MLX variants ; "
        "this pilot uses a numpy equivalent of the same math — see "
        "`scripts/pilot_phase2b_neuromorph.py` module docstring)"
    )
    lines.append(
        "- `kiki_oniric/dream/operations/*_snn.py` (Norse SNN-proxy "
        "variants, invoked directly)"
    )
    lines.append(
        "- `docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-"
        "design.md` §3 Phase 2 track b"
    )
    lines.append(
        "- JSON dump : `docs/milestones/g10a-neuromorph.json` (R1 "
        "provenance artifact)"
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
