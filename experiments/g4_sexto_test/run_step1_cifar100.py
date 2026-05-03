"""G4-sexto Step 1 driver — H6-A CIFAR-100 + RECOMBINE-strategy placebo.

Reuses the ``G4SmallCNN`` substrate (with multi-class
``n_classes = 10`` per-task head) and threads a
``RecombineStrategy`` switch ({mog, none}) through a local
``_dream_episode_strategy`` mirror of G4-quinto Step 3. AE strategy
is dropped — H6-A is the placebo mog vs none only.

2 strategies x 4 arms x N seeds = 240 cells (N = 30 default,
Option B locked).

H6-A : Welch two-sided between (P_max with mog) and (P_max with
none), alpha = 0.05 / 3 = 0.0167. **Failing to reject** confirms
the G4-quinto H5-C finding generalises to mid-large class count :
RECOMBINE remains empirically empty on the CNN at CIFAR-100
100-class scale (positive empirical claim mog ≈ none).

Outputs :
    docs/milestones/g4-sexto-step1-2026-05-03.{json,md}

Pre-reg : docs/osf-prereg-g4-sexto-pilot.md sec 2 (H6-A).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
import numpy as np  # noqa: E402

from experiments.g4_quater_test.recombine_strategies import (  # noqa: E402
    RecombineStrategy,
    sample_synthetic_latents,
)
from experiments.g4_quinto_test.small_cnn import G4SmallCNN  # noqa: E402
from experiments.g4_sexto_test.cifar100_dataset import (  # noqa: E402
    SplitCIFAR100Task,
    load_split_cifar100_10tasks_auto,
)
from experiments.g4_split_fmnist.dream_wrap import (  # noqa: E402
    build_profile,
)
from experiments.g4_ter_hp_sweep.hp_grid import (  # noqa: E402
    HPCombo,
    representative_combo,
)
from harness.storage.run_registry import RunRegistry  # noqa: E402
from kiki_oniric.eval.statistics import compute_hedges_g  # noqa: E402

C_VERSION = "C-v0.12.0+PARTIAL"
ARMS: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
STRATEGIES: tuple[RecombineStrategy, ...] = ("mog", "none")  # H6-A : no AE
DEFAULT_N_SEEDS = 30
N_CLASSES_PER_TASK = 10
LATENT_DIM = 64
RANDOM_CHANCE = 1.0 / N_CLASSES_PER_TASK  # 0.10 for CIFAR-100 head
EXCLUSION_THRESHOLD = 2.0 * RANDOM_CHANCE  # 0.20 — § 4 multi-class adapt
DEFAULT_DATA_DIR = (
    REPO_ROOT / "experiments" / "g4_sexto_test" / "data"
)
DEFAULT_OUT_JSON = (
    REPO_ROOT / "docs" / "milestones" / "g4-sexto-step1-2026-05-03.json"
)
DEFAULT_OUT_MD = (
    REPO_ROOT / "docs" / "milestones" / "g4-sexto-step1-2026-05-03.md"
)
DEFAULT_REGISTRY_DB = REPO_ROOT / ".run_registry.sqlite"
SMOKE_OUT_JSON = Path("/tmp") / "g4-sexto-step1-smoke.json"
SMOKE_OUT_MD = Path("/tmp") / "g4-sexto-step1-smoke.md"
RETENTION_EPS = 1e-6
BETA_BUFFER_CAPACITY = 256
BETA_BUFFER_FILL_PER_TASK = 32
RECOMBINE_N_SYNTHETIC = 16
RECOMBINE_LR = 0.01
RESTRUCTURE_FACTOR = 0.05


class _BetaRecordCNN(TypedDict):
    x: list[list[list[float]]]
    y: int
    latent: list[float] | None


class _BetaBufferCNN:
    """Bounded FIFO buffer for CNN records (NHWC + latent)."""

    def __init__(self, capacity: int) -> None:
        self._records: deque[_BetaRecordCNN] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._records)

    def push(
        self, *, x: np.ndarray, y: int, latent: np.ndarray | None
    ) -> None:
        record: _BetaRecordCNN = {
            "x": x.astype(np.float32).tolist(),
            "y": int(y),
            "latent": (
                latent.astype(np.float32).tolist()
                if latent is not None
                else None
            ),
        }
        self._records.append(record)

    def sample(self, n: int, seed: int) -> list[_BetaRecordCNN]:
        n_avail = len(self._records)
        if n_avail == 0:
            return []
        rng = np.random.default_rng(seed)
        n_take = min(n, n_avail)
        idx = rng.choice(n_avail, size=n_take, replace=False)
        snapshot = list(self._records)
        return [snapshot[i] for i in sorted(idx.tolist())]

    def latents(self) -> list[tuple[list[float], int]]:
        return [
            (list(r["latent"]), int(r["y"]))
            for r in self._records
            if r["latent"] is not None
        ]


def _resolve_commit_sha() -> str:
    env_sha = os.environ.get("DREAMOFKIKI_COMMIT_SHA")
    if env_sha:
        return env_sha
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _strategy_aware_recombine(
    cnn: G4SmallCNN,
    *,
    strategy: RecombineStrategy,
    beta_buffer: _BetaBufferCNN,
    n_synthetic: int,
    lr: float,
    seed: int,
) -> None:
    """RECOMBINE step dispatched by ``strategy`` on the CNN head.

    For ``mog`` strategy, synthetic latents are produced by
    ``sample_synthetic_latents`` and projected through ``cnn._fc2``
    only. For ``none``, this function is a no-op (placebo control
    isolating REPLAY+DOWNSCALE).
    """
    populated = beta_buffer.latents()
    if not populated:
        return
    latents_arr = np.asarray(
        [lat for lat, _ in populated], dtype=np.float32
    )
    labels_arr = np.asarray(
        [lbl for _, lbl in populated], dtype=np.int64
    )
    batch = sample_synthetic_latents(
        strategy=strategy,
        latents=latents_arr,
        labels=labels_arr,
        n_synthetic=n_synthetic,
        seed=seed,
    )
    if batch is None:
        return

    x = mx.array(batch["x"])
    y = mx.array(batch["y"].astype(np.int32))
    opt = optim.SGD(learning_rate=lr)

    def loss_fn(layer: nn.Linear, xb: mx.array, yb: mx.array) -> mx.array:
        return nn.losses.cross_entropy(layer(xb), yb, reduction="mean")

    loss_and_grad = nn.value_and_grad(cnn._fc2, loss_fn)
    _loss, grads = loss_and_grad(cnn._fc2, x, y)
    opt.update(cnn._fc2, grads)
    mx.eval(cnn._fc2.parameters(), opt.state)


def _dream_episode_strategy(
    cnn: G4SmallCNN,
    profile: object,
    seed: int,
    *,
    beta_buffer: _BetaBufferCNN,
    combo: HPCombo,
    restructure_factor: float,
    recombine_strategy: RecombineStrategy,
    recombine_n_synthetic: int,
    recombine_lr: float,
) -> None:
    """Mirror of G4-quinto Step 3 dream-episode wrapper.

    DR-0 spectator runtime path is preserved.
    """
    from kiki_oniric.dream.episode import (
        BudgetCap,
        DreamEpisode,
        EpisodeTrigger,
        Operation,
        OutputChannel,
    )
    from kiki_oniric.profiles.p_min import PMinProfile

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

    rng = np.random.default_rng(seed + 10_000)
    synthetic_records = [
        {
            "x": rng.standard_normal(4).astype(np.float32).tolist(),
            "y": rng.standard_normal(4).astype(np.float32).tolist(),
        }
        for _ in range(4)
    ]
    delta_latents = [
        rng.standard_normal(4).astype(np.float32).tolist() for _ in range(2)
    ]
    episode = DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={
            "beta_records": synthetic_records,
            "shrink_factor": 0.99,
            "topo_op": "reroute",
            "swap_indices": [0, 1],
            "delta_latents": delta_latents,
        },
        operation_set=ops,
        output_channels=channels,
        budget=BudgetCap(
            flops=10_000_000, wall_time_s=10.0, energy_j=1.0
        ),
        episode_id=(
            f"g4-sexto-step1-{type(profile).__name__}-"
            f"{recombine_strategy}-seed{seed}"
        ),
    )
    profile.runtime.execute(episode)  # type: ignore[attr-defined]

    if Operation.REPLAY in ops:
        sampled = beta_buffer.sample(n=combo.replay_batch, seed=seed)
        records = [{"x": r["x"], "y": r["y"]} for r in sampled]
        cnn.replay_optimizer_step(
            records, lr=combo.replay_lr, n_steps=combo.replay_n_steps
        )
    if Operation.DOWNSCALE in ops:
        cnn.downscale_step(factor=combo.downscale_factor)
    if Operation.RESTRUCTURE in ops:
        cnn.restructure_step(
            factor=restructure_factor, seed=seed + 20_000
        )
    if Operation.RECOMBINE in ops:
        _strategy_aware_recombine(
            cnn,
            strategy=recombine_strategy,
            beta_buffer=beta_buffer,
            n_synthetic=recombine_n_synthetic,
            lr=recombine_lr,
            seed=seed + 30_000,
        )


def _run_cell(
    arm: str,
    seed: int,
    combo: HPCombo,
    strategy: RecombineStrategy,
    tasks: list[SplitCIFAR100Task],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict[str, Any]:
    start = time.time()
    cnn = G4SmallCNN(
        latent_dim=LATENT_DIM,
        n_classes=N_CLASSES_PER_TASK,
        seed=seed,
    )
    buffer = _BetaBufferCNN(capacity=BETA_BUFFER_CAPACITY)
    fill_rng = np.random.default_rng(seed + 5_000)

    def _push_task(task: SplitCIFAR100Task) -> None:
        n = task["x_train_nhwc"].shape[0]
        n_take = min(BETA_BUFFER_FILL_PER_TASK, n)
        if n_take == 0:
            return
        idx = fill_rng.choice(n, size=n_take, replace=False)
        for i in idx.tolist():
            x = task["x_train_nhwc"][i]
            latent = cnn.latent(x[None, ...])[0]
            buffer.push(x=x, y=int(task["y_train"][i]), latent=latent)

    cnn.train_task(
        tasks[0]["x_train_nhwc"],
        tasks[0]["y_train"],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    acc_initial = cnn.eval_accuracy(
        tasks[0]["x_test_nhwc"], tasks[0]["y_test"]
    )
    _push_task(tasks[0])

    profile = None
    if arm != "baseline":
        profile = build_profile(arm, seed=seed)

    for k in range(1, len(tasks)):
        if profile is not None:
            _dream_episode_strategy(
                cnn,
                profile,
                seed=seed + k,
                beta_buffer=buffer,
                combo=combo,
                restructure_factor=RESTRUCTURE_FACTOR,
                recombine_strategy=strategy,
                recombine_n_synthetic=RECOMBINE_N_SYNTHETIC,
                recombine_lr=RECOMBINE_LR,
            )
        cnn.train_task(
            tasks[k]["x_train_nhwc"],
            tasks[k]["y_train"],
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )
        _push_task(tasks[k])

    acc_final = cnn.eval_accuracy(
        tasks[0]["x_test_nhwc"], tasks[0]["y_test"]
    )
    retention = acc_final / max(acc_initial, RETENTION_EPS)
    excluded = bool(acc_initial < EXCLUSION_THRESHOLD)
    return {
        "arm": arm,
        "seed": seed,
        "hp_combo_id": combo.combo_id,
        "recombine_strategy": strategy,
        "acc_task1_initial": float(acc_initial),
        "acc_task1_final": float(acc_final),
        "retention": float(retention),
        "excluded_underperforming_baseline": excluded,
        "wall_time_s": time.time() - start,
    }


def _retention_by_arm_strategy(
    cells: list[dict[str, Any]], strategy: RecombineStrategy
) -> dict[str, list[float]]:
    arms = sorted({c["arm"] for c in cells})
    out: dict[str, list[float]] = {a: [] for a in arms}
    for c in cells:
        if c["excluded_underperforming_baseline"]:
            continue
        if c["recombine_strategy"] != strategy:
            continue
        out[c["arm"]].append(c["retention"])
    return out


def _h6a_verdict(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """H6-A — Welch two-sided P_max(mog) vs P_max(none) on CIFAR-100.

    Failing to reject H0 at alpha = 0.05 / 3 = 0.0167 confirms H6-A
    (RECOMBINE empirically empty universalises to CIFAR-100 at
    100-class scale).
    """
    from scipy import stats as scistats

    ret_mog = _retention_by_arm_strategy(cells, "mog")
    ret_none = _retention_by_arm_strategy(cells, "none")
    p_max_mog = ret_mog.get("P_max", [])
    p_max_none = ret_none.get("P_max", [])
    if len(p_max_mog) < 2 or len(p_max_none) < 2:
        return {
            "insufficient_samples": True,
            "n_p_max_mog": len(p_max_mog),
            "n_p_max_none": len(p_max_none),
        }
    alpha = 0.05 / 3
    welch = scistats.ttest_ind(p_max_mog, p_max_none, equal_var=False)
    fail_to_reject = bool(welch.pvalue > alpha)
    g = compute_hedges_g(p_max_mog, p_max_none)
    return {
        "n_p_max_mog": len(p_max_mog),
        "n_p_max_none": len(p_max_none),
        "mean_p_max_mog": float(np.mean(p_max_mog)),
        "mean_p_max_none": float(np.mean(p_max_none)),
        "welch_t": float(welch.statistic),
        "welch_p_two_sided": float(welch.pvalue),
        "alpha_per_test": alpha,
        "fail_to_reject_h0": fail_to_reject,
        "h6a_recombine_empty_confirmed": fail_to_reject,
        "hedges_g_mog_vs_none": g,
    }


def _render_md_report(payload: dict[str, Any]) -> str:
    h = payload["verdict"]["h6a_recombine_strategy"]
    lines: list[str] = [
        "# G4-sexto Step 1 — H6-A CIFAR-100+RECOMBINE strategy + placebo",
        "",
        f"**Date** : {payload['date']}",
        f"**c_version** : `{payload['c_version']}`",
        f"**commit_sha** : `{payload['commit_sha']}`",
        (
            f"**Cells** : {len(payload['cells'])} "
            f"({len(STRATEGIES)} strategies x {len(ARMS)} arms x "
            f"{payload['n_seeds']} seeds)"
        ),
        f"**Wall time** : {payload['wall_time_s']:.1f}s",
        f"**Smoke** : {payload['smoke']}",
        "",
        (
            "**Multi-class exclusion threshold** : "
            f"acc_initial < 2 × random_chance = "
            f"{EXCLUSION_THRESHOLD:.2f} (random_chance = "
            f"{RANDOM_CHANCE:.2f} for n_classes = {N_CLASSES_PER_TASK})."
        ),
        "",
        "## Pre-registered hypothesis",
        "",
        "Pre-registration : `docs/osf-prereg-g4-sexto-pilot.md`",
        "",
        "### H6-A — universality of RECOMBINE-empty (CIFAR-100, n_classes=10)",
        "",
        (
            "Welch two-sided test of `retention(P_max with mog)` vs "
            "`retention(P_max with none)` on the small CNN with a "
            "10-class per-task head. **Failing** to reject H0 at "
            "alpha = 0.05 / 3 = 0.0167 confirms H6-A : the G4-quinto "
            "H5-C RECOMBINE-empty finding generalises to mid-large "
            "class count (CIFAR-100, 100 fine classes split into 10 "
            "tasks of 10 classes each)."
        ),
        "",
    ]
    if h.get("insufficient_samples"):
        lines.append(
            f"INSUFFICIENT SAMPLES "
            f"(mog={h.get('n_p_max_mog')}, "
            f"none={h.get('n_p_max_none')})"
        )
    else:
        lines += [
            (
                f"- mean retention P_max (mog) : "
                f"{h['mean_p_max_mog']:.4f} (N={h['n_p_max_mog']})"
            ),
            (
                f"- mean retention P_max (none) : "
                f"{h['mean_p_max_none']:.4f} (N={h['n_p_max_none']})"
            ),
            (
                f"- Hedges' g (mog vs none) : "
                f"{h['hedges_g_mog_vs_none']:.4f}"
            ),
            f"- Welch t : {h['welch_t']:.4f}",
            (
                f"- Welch p (two-sided, alpha = "
                f"{h['alpha_per_test']:.4f}) : "
                f"{h['welch_p_two_sided']:.4f} -> "
                f"fail_to_reject_h0 = {h['fail_to_reject_h0']}"
            ),
            "",
            (
                f"**H6-A verdict** : RECOMBINE empty confirmed "
                f"= {h['h6a_recombine_empty_confirmed']} "
                "(positive empirical claim mog ≈ none if True)."
            ),
            "",
            (
                "*Honest reading* : Welch fail-to-reject = absence "
                "of evidence at this N for a difference between "
                "mog and none — under H6-A specifically, this **is** "
                "the predicted positive empirical claim that "
                "RECOMBINE adds nothing measurable beyond "
                "REPLAY+DOWNSCALE on the small CNN substrate at "
                "CIFAR-100 100-class scale."
            ),
        ]
    lines += [
        "",
        "## Provenance",
        "",
        (
            "- Pre-registration : "
            "[docs/osf-prereg-g4-sexto-pilot.md]"
            "(../osf-prereg-g4-sexto-pilot.md)"
        ),
        (
            "- Driver : `experiments/g4_sexto_test/"
            "run_step1_cifar100.py`"
        ),
        (
            "- Substrate : `experiments.g4_quinto_test."
            "small_cnn.G4SmallCNN` (n_classes=10)"
        ),
        (
            "- Loader : `experiments.g4_sexto_test."
            "cifar100_dataset.load_split_cifar100_10tasks_auto`"
        ),
        (
            "- Strategies : `experiments.g4_quater_test."
            "recombine_strategies.sample_synthetic_latents`"
        ),
        (
            "- Run registry : `harness/storage/run_registry.RunRegistry`"
            " (db `.run_registry.sqlite`)"
        ),
        "",
    ]
    return "\n".join(lines)


def run_pilot(
    *,
    data_dir: Path,
    seeds: tuple[int, ...],
    strategies: tuple[RecombineStrategy, ...],
    out_json: Path,
    out_md: Path,
    registry_db: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    smoke: bool = False,
) -> dict[str, Any]:
    tasks = load_split_cifar100_10tasks_auto(data_dir)
    if len(tasks) != 10:
        raise RuntimeError(
            f"Split-CIFAR-100 loader returned {len(tasks)} tasks "
            f"(expected 10)"
        )

    registry = RunRegistry(registry_db)
    commit_sha = _resolve_commit_sha()
    c5 = representative_combo()
    cells: list[dict[str, Any]] = []
    sweep_start = time.time()

    for strategy in strategies:
        for arm in ARMS:
            for seed in seeds:
                cell = _run_cell(
                    arm,
                    seed,
                    c5,
                    strategy,
                    tasks,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                )
                profile_key = (
                    f"g4-sexto/step1/{arm}/{c5.combo_id}/{strategy}"
                )
                run_id = registry.register(
                    c_version=C_VERSION,
                    profile=profile_key,
                    seed=seed,
                    commit_sha=commit_sha,
                )
                cell["run_id"] = run_id
                cells.append(cell)
                # Incremental JSON checkpoint for overnight resilience.
                _checkpoint(
                    out_json, cells, c_version=C_VERSION,
                    commit_sha=commit_sha, seeds=seeds,
                    strategies=strategies, data_dir=data_dir,
                    smoke=smoke, sweep_start=sweep_start,
                )

    wall = time.time() - sweep_start
    verdict = {"h6a_recombine_strategy": _h6a_verdict(cells)}
    payload = {
        "date": "2026-05-03",
        "c_version": C_VERSION,
        "commit_sha": commit_sha,
        "n_seeds": len(seeds),
        "strategies": list(strategies),
        "arms": list(ARMS),
        "data_dir": str(data_dir),
        "wall_time_s": wall,
        "smoke": smoke,
        "cells": cells,
        "verdict": verdict,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    out_md.write_text(_render_md_report(payload))
    return payload


def _checkpoint(
    out_json: Path,
    cells: list[dict[str, Any]],
    *,
    c_version: str,
    commit_sha: str,
    seeds: tuple[int, ...],
    strategies: tuple[RecombineStrategy, ...],
    data_dir: Path,
    smoke: bool,
    sweep_start: float,
) -> None:
    """Write a partial JSON snapshot so a watchdog kill mid-run is recoverable.

    The verdict block is only computed once the full sweep
    completes ; the partial JSON carries cells but no verdict,
    flagged ``"partial": True``.
    """
    out_json.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "date": "2026-05-03",
        "c_version": c_version,
        "commit_sha": commit_sha,
        "n_seeds": len(seeds),
        "strategies": list(strategies),
        "arms": list(ARMS),
        "data_dir": str(data_dir),
        "wall_time_s": time.time() - sweep_start,
        "smoke": smoke,
        "partial": True,
        "cells": cells,
        "verdict": None,
    }
    out_json.write_text(json.dumps(snapshot, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G4-sexto Step 1 driver")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument(
        "--registry-db", type=Path, default=DEFAULT_REGISTRY_DB
    )
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--strategies", type=str, nargs="+",
        default=list(STRATEGIES),
    )
    args = parser.parse_args(argv)

    if args.smoke:
        seeds: tuple[int, ...] = (0,)
        strategies: tuple[RecombineStrategy, ...] = ("mog",)
        out_json = args.out_json or SMOKE_OUT_JSON
        out_md = args.out_md or SMOKE_OUT_MD
    else:
        if args.seeds is not None:
            seeds = tuple(args.seeds)
        else:
            seeds = tuple(range(args.n_seeds))
        strategies = tuple(args.strategies)  # type: ignore[assignment]
        out_json = args.out_json or DEFAULT_OUT_JSON
        out_md = args.out_md or DEFAULT_OUT_MD

    payload = run_pilot(
        data_dir=args.data_dir,
        seeds=seeds,
        strategies=strategies,
        out_json=out_json,
        out_md=out_md,
        registry_db=args.registry_db,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        smoke=args.smoke,
    )
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Cells : {len(payload['cells'])}")
    h = payload["verdict"]["h6a_recombine_strategy"]
    if not h.get("insufficient_samples"):
        print(
            f"H6-A : RECOMBINE empty confirmed = "
            f"{h['h6a_recombine_empty_confirmed']} "
            f"(p={h['welch_p_two_sided']:.4f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
