"""Import-and-CLI smoke for the G6 pilot driver."""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_g6_module_importable() -> None:
    mod = importlib.import_module("experiments.g6_mmlu_stream.run_g6")
    assert hasattr(mod, "run_pilot"), "run_g6 must export run_pilot"
    assert hasattr(mod, "main"), "run_g6 must export main"
    assert hasattr(mod, "ARMS"), "run_g6 must export ARMS"
    assert tuple(mod.ARMS) == ("baseline", "P_min", "P_equ", "P_max")


def test_g6_help_smokes() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "experiments.g6_mmlu_stream.run_g6", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, (
        f"--help failed: stderr={result.stderr!r}"
    )
    assert "G6" in result.stdout or "g6" in result.stdout
    assert "--smoke" in result.stdout
    assert "--path" in result.stdout
