"""G4-quater Step 2 driver — H4-B test (RESTRUCTURE factor sweep).

Reuses the existing G4-ter 3-layer ``G4HierarchicalClassifier``.
Sweeps RESTRUCTURE factor in {0.85, 0.95, 0.99}.
3 factors x 4 arms x N seeds = 360 cells (N=30 default).

Outputs :
    docs/milestones/g4-quater-step2-2026-05-03.{json,md}

Pre-reg : docs/osf-prereg-g4-quater-pilot.md sec 2 (H4-B).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from experiments.g4_split_fmnist.dataset import (  # noqa: E402
    SplitFMNISTTask,
    load_split_fmnist_5tasks,
)
from experiments.g4_split_fmnist.dream_wrap import (  # noqa: E402
    build_profile,
)
from experiments.g4_ter_hp_sweep.dream_wrap_hier import (  # noqa: E402
    BetaBufferHierFIFO,
    G4HierarchicalClassifier,
)
from experiments.g4_ter_hp_sweep.hp_grid import (  # noqa: E402
    HPCombo,
    representative_combo,
)
from harness.storage.run_registry import RunRegistry  # noqa: E402
from kiki_oniric.eval.statistics import jonckheere_trend  # noqa: E402

C_VERSION = "C-v0.12.0+PARTIAL"
ARMS: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
RESTRUCTURE_FACTORS: tuple[float, ...] = (0.85, 0.95, 0.99)
DEFAULT_N_SEEDS = 30
DEFAULT_DATA_DIR = REPO_ROOT / "experiments" / "g4_split_fmnist" / "data"
DEFAULT_OUT_JSON = (
    REPO_ROOT / "docs" / "milestones" / "g4-quater-step2-2026-05-03.json"
)
DEFAULT_OUT_MD = (
    REPO_ROOT / "docs" / "milestones" / "g4-quater-step2-2026-05-03.md"
)
DEFAULT_REGISTRY_DB = REPO_ROOT / ".run_registry.sqlite"
RETENTION_EPS = 1e-6
BETA_BUFFER_CAPACITY = 256
BETA_BUFFER_FILL_PER_TASK = 32
HIDDEN_1 = 32
HIDDEN_2 = 16
RECOMBINE_N_SYNTHETIC = 16
RECOMBINE_LR = 0.01


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


def _run_cell(
    arm: str,
    seed: int,
    combo: HPCombo,
    factor: float,
    tasks: list[SplitFMNISTTask],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict[str, Any]:
    start = time.time()
    feat_dim = tasks[0]["x_train"].shape[1]
    clf = G4HierarchicalClassifier(
        in_dim=feat_dim,
        hidden_1=HIDDEN_1,
        hidden_2=HIDDEN_2,
        n_classes=2,
        seed=seed,
    )
    buffer = BetaBufferHierFIFO(capacity=BETA_BUFFER_CAPACITY)
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
            clf.dream_episode_hier(
                profile,
                seed=seed + k,
                beta_buffer=buffer,
                replay_n_records=combo.replay_batch,
                replay_n_steps=combo.replay_n_steps,
                replay_lr=combo.replay_lr,
                downscale_factor=combo.downscale_factor,
                restructure_factor=factor,
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
        "restructure_factor": float(factor),
        "acc_task1_initial": float(acc_initial),
        "acc_task1_final": float(acc_final),
        "retention": float(retention),
        "excluded_underperforming_baseline": excluded,
        "wall_time_s": time.time() - start,
    }


def _retention_by_arm_factor(
    cells: list[dict[str, Any]], factor: float
) -> dict[str, list[float]]:
    arms = sorted({c["arm"] for c in cells})
    out: dict[str, list[float]] = {a: [] for a in arms}
    for c in cells:
        if c["excluded_underperforming_baseline"]:
            continue
        if abs(c["restructure_factor"] - factor) > 1e-9:
            continue
        out[c["arm"]].append(c["retention"])
    return out


def _h4b_per_factor_verdict(
    cells: list[dict[str, Any]], factor: float
) -> dict[str, Any]:
    retention = _retention_by_arm_factor(cells, factor)
    groups = [
        retention.get("P_min", []),
        retention.get("P_equ", []),
        retention.get("P_max", []),
    ]
    if any(len(g) < 2 for g in groups):
        return {
            "factor": factor,
            "insufficient_samples": True,
            "n_per_arm": [len(g) for g in groups],
        }
    alpha = 0.05 / 9  # 3 factors x 3 hypotheses
    res = jonckheere_trend(groups, alpha=alpha)
    mean_p_min = float(sum(groups[0]) / len(groups[0]))
    mean_p_equ = float(sum(groups[1]) / len(groups[1]))
    mean_p_max = float(sum(groups[2]) / len(groups[2]))
    return {
        "factor": factor,
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
    per_factor = payload["verdict"]["h4b_per_factor"]
    lines: list[str] = [
        "# G4-quater Step 2 — H4-B RESTRUCTURE factor sweep",
        "",
        f"**Date** : {payload['date']}",
        f"**c_version** : `{payload['c_version']}`",
        f"**commit_sha** : `{payload['commit_sha']}`",
        (
            f"**Cells** : {len(payload['cells'])} "
            f"({len(RESTRUCTURE_FACTORS)} factors x {len(ARMS)} arms "
            f"x {payload['n_seeds']} seeds)"
        ),
        f"**Wall time** : {payload['wall_time_s']:.1f}s",
        "",
        "## Pre-registered hypothesis",
        "",
        "Pre-registration : `docs/osf-prereg-g4-quater-pilot.md`",
        "",
        "### H4-B — HP-calibration (RESTRUCTURE factor sweep)",
        "",
        (
            "Per-factor Jonckheere on retention across "
            "(P_min, P_equ, P_max). Multiplicity-adjusted "
            "alpha = 0.05 / 9 = 0.0056."
        ),
        "",
    ]
    for entry in per_factor:
        lines.append(f"#### factor = {entry['factor']}")
        if entry.get("insufficient_samples"):
            lines.append(
                f"INSUFFICIENT SAMPLES (n_per_arm={entry['n_per_arm']})"
            )
        else:
            lines += [
                f"- mean retention P_min : {entry['mean_p_min']:.4f}",
                f"- mean retention P_equ : {entry['mean_p_equ']:.4f}",
                f"- mean retention P_max : {entry['mean_p_max']:.4f}",
                (
                    "- monotonic P_max >= P_equ >= P_min : "
                    f"{entry['monotonic_observed']}"
                ),
                f"- Jonckheere J : {entry['j_statistic']:.4f}",
                (
                    f"- one-sided p (alpha = "
                    f"{entry['alpha_per_test']:.4f}) : "
                    f"{entry['p_value']:.4f} -> reject_h0 "
                    f"= {entry['reject_h0']}"
                ),
            ]
        lines.append("")
    lines += [
        "*Honest reading* : even one factor cell with reject_h0=True "
        "and monotonic_observed=True is sufficient to confirm H4-B "
        "(at the multiplicity-adjusted alpha). All factors failing "
        "to reject is consistent with H4-B refutation at this N.",
        "",
        "## Provenance",
        "",
        (
            "- Pre-registration : "
            "[docs/osf-prereg-g4-quater-pilot.md]"
            "(../osf-prereg-g4-quater-pilot.md)"
        ),
        (
            "- Driver : `experiments/g4_quater_test/"
            "run_step2_restructure_sweep.py`"
        ),
        (
            "- Substrate : `experiments.g4_ter_hp_sweep."
            "dream_wrap_hier.G4HierarchicalClassifier`"
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
    factors: tuple[float, ...],
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

    for factor in factors:
        for arm in ARMS:
            for seed in seeds:
                cell = _run_cell(
                    arm,
                    seed,
                    c5,
                    factor,
                    tasks,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                )
                profile = (
                    f"g4-quater/step2/{arm}/{c5.combo_id}/f{factor:.2f}"
                )
                run_id = registry.register(
                    c_version=C_VERSION,
                    profile=profile,
                    seed=seed,
                    commit_sha=commit_sha,
                )
                cell["run_id"] = run_id
                cells.append(cell)

    wall = time.time() - sweep_start
    per_factor_verdicts = [
        _h4b_per_factor_verdict(cells, f) for f in factors
    ]
    verdict = {
        "h4b_per_factor": per_factor_verdicts,
        "any_factor_recovers_ordering": any(
            (not v.get("insufficient_samples"))
            and v.get("reject_h0")
            and v.get("monotonic_observed")
            for v in per_factor_verdicts
        ),
    }
    payload = {
        "date": "2026-05-03",
        "c_version": C_VERSION,
        "commit_sha": commit_sha,
        "n_seeds": len(seeds),
        "factors": list(factors),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G4-quater Step 2 driver")
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
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--factors", type=float, nargs="+",
        default=list(RESTRUCTURE_FACTORS),
    )
    args = parser.parse_args(argv)

    if args.smoke:
        seeds: tuple[int, ...] = (0,)
        factors: tuple[float, ...] = (0.95,)
    else:
        seeds = tuple(args.seeds) if args.seeds else tuple(range(args.n_seeds))
        factors = tuple(args.factors)

    payload = run_pilot(
        data_dir=args.data_dir,
        seeds=seeds,
        factors=factors,
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
    print(
        "H4-B any factor recovers ordering : "
        f"{payload['verdict']['any_factor_recovers_ordering']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
