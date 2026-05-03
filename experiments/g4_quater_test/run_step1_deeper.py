"""G4-quater Step 1 driver — H4-A test with 5-layer deeper substrate.

Layout : 4 arms x N seeds x 1 HP (C5 anchor) on
``G4HierarchicalDeeperClassifier`` (hidden = 64-32-16-8). Per-cell
pipeline mirrors ``experiments/g4_ter_hp_sweep/run_g4_ter._run_cell_richer``
modulo the substrate swap.

Outputs :
    docs/milestones/g4-quater-step1-2026-05-03.{json,md}

Pre-reg : docs/osf-prereg-g4-quater-pilot.md sec 2 (H4-A).
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

import numpy as np  # noqa: E402

from experiments.g4_quater_test.deeper_classifier import (  # noqa: E402
    G4HierarchicalDeeperClassifier,
)
from experiments.g4_split_fmnist.dataset import (  # noqa: E402
    SplitFMNISTTask,
    load_split_fmnist_5tasks,
)
from experiments.g4_split_fmnist.dream_wrap import (  # noqa: E402
    build_profile,
)
from experiments.g4_ter_hp_sweep.hp_grid import (  # noqa: E402
    HPCombo,
    representative_combo,
)
from harness.storage.run_registry import RunRegistry  # noqa: E402
from kiki_oniric.eval.statistics import jonckheere_trend  # noqa: E402

C_VERSION = "C-v0.12.0+PARTIAL"
ARMS: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
DEFAULT_N_SEEDS = 95
DEFAULT_DATA_DIR = REPO_ROOT / "experiments" / "g4_split_fmnist" / "data"
DEFAULT_OUT_JSON = (
    REPO_ROOT / "docs" / "milestones" / "g4-quater-step1-2026-05-03.json"
)
DEFAULT_OUT_MD = (
    REPO_ROOT / "docs" / "milestones" / "g4-quater-step1-2026-05-03.md"
)
DEFAULT_REGISTRY_DB = REPO_ROOT / ".run_registry.sqlite"
RETENTION_EPS = 1e-6
BETA_BUFFER_CAPACITY = 256
BETA_BUFFER_FILL_PER_TASK = 32
HIDDEN = (64, 32, 16, 8)
RECOMBINE_N_SYNTHETIC = 16
RECOMBINE_LR = 0.01
RESTRUCTURE_FACTOR = 0.05


class Cell(TypedDict):
    arm: str
    seed: int
    hp_combo_id: str
    acc_task1_initial: float
    acc_task1_final: float
    retention: float
    excluded_underperforming_baseline: bool
    wall_time_s: float
    run_id: str


class _BetaRecord(TypedDict):
    x: list[float]
    y: int
    latent: list[float] | None


class _BetaBufferDeeper:
    """Bounded FIFO buffer with optional latents (h3-dim).

    Mirrors ``BetaBufferHierFIFO`` but stores hidden_3 activations
    of the deeper head as the RECOMBINE Gaussian-MoG support set.
    """

    def __init__(self, capacity: int) -> None:
        self._records: deque[_BetaRecord] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._records)

    def push(
        self, *, x: np.ndarray, y: int, latent: np.ndarray | None
    ) -> None:
        record: _BetaRecord = {
            "x": x.astype(np.float32).tolist(),
            "y": int(y),
            "latent": (
                latent.astype(np.float32).tolist()
                if latent is not None
                else None
            ),
        }
        self._records.append(record)

    def sample(self, n: int, seed: int) -> list[_BetaRecord]:
        n_avail = len(self._records)
        if n_avail == 0:
            return []
        rng = np.random.default_rng(seed)
        n_take = min(n, n_avail)
        idx = rng.choice(n_avail, size=n_take, replace=False)
        snapshot = list(self._records)
        return [
            {
                "x": list(snapshot[i]["x"]),
                "y": int(snapshot[i]["y"]),
                "latent": (
                    list(snapshot[i]["latent"])
                    if snapshot[i]["latent"] is not None
                    else None
                ),
            }
            for i in sorted(idx.tolist())
        ]

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


def _dream_episode_deeper(
    clf: G4HierarchicalDeeperClassifier,
    profile: object,
    seed: int,
    *,
    beta_buffer: _BetaBufferDeeper,
    combo: HPCombo,
    restructure_factor: float,
    recombine_n_synthetic: int,
    recombine_lr: float,
) -> None:
    """Drive one DreamEpisode and mutate deeper-classifier weights.

    Coupling map (P_min : REPLAY+DOWNSCALE only ; P_equ/P_max :
    full {REPLAY, DOWNSCALE, RESTRUCTURE, RECOMBINE}). DR-0 spectator
    runtime path preserved (synthetic input_slice values).
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
        episode_id=f"g4-quater-step1-{type(profile).__name__}-seed{seed}",
    )
    profile.runtime.execute(episode)  # type: ignore[attr-defined]

    if Operation.REPLAY in ops:
        sampled = beta_buffer.sample(n=combo.replay_batch, seed=seed)
        records = [{"x": r["x"], "y": r["y"]} for r in sampled]
        clf.replay_optimizer_step(
            records, lr=combo.replay_lr, n_steps=combo.replay_n_steps
        )
    if Operation.DOWNSCALE in ops:
        clf.downscale_step(factor=combo.downscale_factor)
    if Operation.RESTRUCTURE in ops:
        clf.restructure_step(
            factor=restructure_factor, seed=seed + 20_000
        )
    if Operation.RECOMBINE in ops:
        latents_records = beta_buffer.latents()
        clf.recombine_step(
            latents=latents_records,
            n_synthetic=recombine_n_synthetic,
            lr=recombine_lr,
            seed=seed + 30_000,
        )


def _run_cell(
    arm: str,
    seed: int,
    combo: HPCombo,
    tasks: list[SplitFMNISTTask],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict[str, Any]:
    start = time.time()
    feat_dim = tasks[0]["x_train"].shape[1]
    clf = G4HierarchicalDeeperClassifier(
        in_dim=feat_dim,
        hidden=HIDDEN,
        n_classes=2,
        seed=seed,
    )
    buffer = _BetaBufferDeeper(capacity=BETA_BUFFER_CAPACITY)
    fill_rng = np.random.default_rng(seed + 5_000)

    def _push_task(task: SplitFMNISTTask) -> None:
        n = task["x_train"].shape[0]
        n_take = min(BETA_BUFFER_FILL_PER_TASK, n)
        idx = fill_rng.choice(n, size=n_take, replace=False)
        for i in idx.tolist():
            x = task["x_train"][i]
            latent = clf.latent(x[None, :])[0]
            buffer.push(x=x, y=int(task["y_train"][i]), latent=latent)

    clf.train_task(tasks[0], epochs=epochs, batch_size=batch_size, lr=lr)
    acc_initial = clf.eval_accuracy(tasks[0]["x_test"], tasks[0]["y_test"])
    _push_task(tasks[0])

    profile = None
    if arm != "baseline":
        profile = build_profile(arm, seed=seed)

    for k in range(1, len(tasks)):
        if profile is not None:
            _dream_episode_deeper(
                clf,
                profile,
                seed=seed + k,
                beta_buffer=buffer,
                combo=combo,
                restructure_factor=RESTRUCTURE_FACTOR,
                recombine_n_synthetic=RECOMBINE_N_SYNTHETIC,
                recombine_lr=RECOMBINE_LR,
            )
        clf.train_task(
            tasks[k], epochs=epochs, batch_size=batch_size, lr=lr
        )
        _push_task(tasks[k])

    acc_final = clf.eval_accuracy(tasks[0]["x_test"], tasks[0]["y_test"])
    retention = acc_final / max(acc_initial, RETENTION_EPS)
    excluded = bool(acc_initial < 0.5)
    return {
        "arm": arm,
        "seed": seed,
        "hp_combo_id": combo.combo_id,
        "acc_task1_initial": float(acc_initial),
        "acc_task1_final": float(acc_final),
        "retention": float(retention),
        "excluded_underperforming_baseline": excluded,
        "wall_time_s": time.time() - start,
    }


def _retention_by_arm(cells: list[dict[str, Any]]) -> dict[str, list[float]]:
    arms = sorted({c["arm"] for c in cells})
    out: dict[str, list[float]] = {a: [] for a in arms}
    for c in cells:
        if c["excluded_underperforming_baseline"]:
            continue
        out[c["arm"]].append(c["retention"])
    return out


def _h4a_verdict(retention: dict[str, list[float]]) -> dict[str, Any]:
    """H4-A — Jonckheere on retention across (P_min, P_equ, P_max)."""
    groups = [
        retention.get("P_min", []),
        retention.get("P_equ", []),
        retention.get("P_max", []),
    ]
    if any(len(g) < 2 for g in groups):
        return {
            "insufficient_samples": True,
            "n_per_arm": [len(g) for g in groups],
        }
    alpha = 0.05 / 3
    res = jonckheere_trend(groups, alpha=alpha)
    mean_p_min = float(sum(groups[0]) / len(groups[0]))
    mean_p_equ = float(sum(groups[1]) / len(groups[1]))
    mean_p_max = float(sum(groups[2]) / len(groups[2]))
    return {
        "j_statistic": res.statistic,
        "p_value": res.p_value,
        "reject_h0": res.reject_h0,
        "mean_p_min": mean_p_min,
        "mean_p_equ": mean_p_equ,
        "mean_p_max": mean_p_max,
        "monotonic_observed": (
            mean_p_min <= mean_p_equ <= mean_p_max
        ),
        "alpha_per_test": alpha,
    }


def _render_md_report(payload: dict[str, Any]) -> str:
    h = payload["verdict"]["h4a_deeper_substrate"]
    lines: list[str] = [
        "# G4-quater Step 1 — H4-A deeper substrate",
        "",
        f"**Date** : {payload['date']}",
        f"**c_version** : `{payload['c_version']}`",
        f"**commit_sha** : `{payload['commit_sha']}`",
        (
            f"**Cells** : {len(payload['cells'])} "
            f"({len(ARMS)} arms x {payload['n_seeds']} seeds x 1 HP)"
        ),
        f"**Hidden** : {HIDDEN}",
        f"**Wall time** : {payload['wall_time_s']:.1f}s",
        "",
        "## Pre-registered hypothesis",
        "",
        "Pre-registration : `docs/osf-prereg-g4-quater-pilot.md`",
        "",
        "### H4-A — substrate-depth (5-layer deeper hierarchical head)",
    ]
    if h.get("insufficient_samples"):
        lines.append(
            f"INSUFFICIENT SAMPLES (n_per_arm={h['n_per_arm']})"
        )
    else:
        lines += [
            f"- mean retention P_min : {h['mean_p_min']:.4f}",
            f"- mean retention P_equ : {h['mean_p_equ']:.4f}",
            f"- mean retention P_max : {h['mean_p_max']:.4f}",
            (
                "- monotonic observed P_max >= P_equ >= P_min : "
                f"{h['monotonic_observed']}"
            ),
            f"- Jonckheere J statistic : {h['j_statistic']:.4f}",
            (
                f"- one-sided p (alpha = {h['alpha_per_test']:.4f}) : "
                f"{h['p_value']:.4f} -> reject_h0 = {h['reject_h0']}"
            ),
            "",
            (
                "*Honest reading* : reject_h0 means there is evidence "
                "for the predicted ordering at this N ; failure to "
                "reject means no evidence at this N (absence of "
                "evidence vs evidence of absence)."
            ),
        ]
    lines += [
        "",
        "## Provenance",
        "",
        (
            "- Pre-registration : "
            "[docs/osf-prereg-g4-quater-pilot.md]"
            "(../osf-prereg-g4-quater-pilot.md)"
        ),
        "- Driver : `experiments/g4_quater_test/run_step1_deeper.py`",
        (
            "- Substrate : `experiments.g4_quater_test."
            "deeper_classifier.G4HierarchicalDeeperClassifier`"
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
    out_json: Path,
    out_md: Path,
    registry_db: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    smoke: bool = False,
) -> dict[str, Any]:
    tasks = load_split_fmnist_5tasks(data_dir)
    if len(tasks) != 5:
        raise RuntimeError(
            f"Split-FMNIST loader returned {len(tasks)} tasks (expected 5)"
        )

    registry = RunRegistry(registry_db)
    commit_sha = _resolve_commit_sha()
    c5 = representative_combo()
    cells: list[dict[str, Any]] = []
    sweep_start = time.time()

    for arm in ARMS:
        for seed in seeds:
            cell = _run_cell(
                arm,
                seed,
                c5,
                tasks,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
            )
            run_id = registry.register(
                c_version=C_VERSION,
                profile=f"g4-quater/step1/{arm}/{c5.combo_id}",
                seed=seed,
                commit_sha=commit_sha,
            )
            cell["run_id"] = run_id
            cells.append(cell)

    wall = time.time() - sweep_start
    retention = _retention_by_arm(cells)
    verdict = {
        "h4a_deeper_substrate": _h4a_verdict(retention),
        "retention_by_arm": retention,
    }
    payload = {
        "date": "2026-05-03",
        "c_version": C_VERSION,
        "commit_sha": commit_sha,
        "n_seeds": len(seeds),
        "arms": list(ARMS),
        "hidden": list(HIDDEN),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G4-quater Step 1 driver")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument(
        "--registry-db", type=Path, default=DEFAULT_REGISTRY_DB
    )
    parser.add_argument(
        "--n-seeds", type=int, default=DEFAULT_N_SEEDS
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args(argv)

    if args.smoke:
        seeds: tuple[int, ...] = (0,)
    elif args.seeds is not None:
        seeds = tuple(args.seeds)
    else:
        seeds = tuple(range(args.n_seeds))

    payload = run_pilot(
        data_dir=args.data_dir,
        seeds=seeds,
        out_json=args.out_json,
        out_md=args.out_md,
        registry_db=args.registry_db,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        smoke=args.smoke,
    )
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Cells : {len(payload['cells'])}")
    h = payload["verdict"]["h4a_deeper_substrate"]
    if not h.get("insufficient_samples"):
        print(
            f"H4-A : monotonic={h['monotonic_observed']} "
            f"J={h['j_statistic']:.3f} p={h['p_value']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
