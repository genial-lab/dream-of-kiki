"""G6 pilot driver — micro-kiki Qwen-35B × MMLU subdomain CL stream.

**Gate ID** : G6 — first empirical evidence on a real production LLM.
**Validates** : whether observed Hedges' g for retention on a 5-subdomain
MMLU continual-learning stream matches Hu 2020 / Javadi 2024 floors when
the substrate is `kiki_oniric.substrates.micro_kiki.MicroKikiSubstrate`
(Qwen3.6-35B-A3B + r=16 LoRA).

**Path branches** (locked at Task 0.5) :
- Path A — full pilot (Studio + KIKI-Mac_tunner + mlx_lm.lora). Future
  work on this M1 Max host (KIKI-Mac_tunner absent ; raises
  ``NotImplementedError`` in the cell runner Path A leg).
- Path B — inference-only exploratory (any host, no fine-tune). Default.

**Mode** : empirical claim at first-pilot scale (3 seeds × 4 arms = 12
sequences). Pre-registered as exploratory for absolute g magnitudes.
**Expected output** :
    - docs/milestones/g6-pilot-pathB-2026-05-03.json
    - docs/milestones/g6-pilot-pathB-2026-05-03.md

Reference :
    docs/superpowers/plans/2026-05-03-g6-micro-kiki-mmlu-cl.md
    docs/osf-prereg-g6-pilot.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ARMS: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
DEFAULT_SUBDOMAINS: tuple[str, ...] = (
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
)
C_VERSION = "C-v0.12.0+PARTIAL"  # synced with STATUS.md at plan-write time


def run_pilot(**kwargs: Any) -> dict[str, Any]:
    """Run the G6 pilot. Stub — full implementation lands in later tasks."""
    raise NotImplementedError(
        "G6 run_pilot is implemented incrementally — see plan tasks 3..7"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="G6 pilot driver — micro-kiki Qwen-35B × MMLU CL stream",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 1 cell (P_min, seed 0, S1 only) to validate the pipeline.",
    )
    parser.add_argument(
        "--path",
        choices=("A", "B"),
        default="B",
        help="Path A (full LoRA pilot) or B (inference-only). Default B.",
    )
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--inner-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.parse_args(argv)
    print("G6 pilot stub — implementation lands in plan tasks 3..7")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
