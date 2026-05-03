"""Smoke + structural tests for the G4-quater Step 1 driver."""
from __future__ import annotations

from pathlib import Path

from experiments.g4_quater_test.run_step1_deeper import (
    ARMS,
    HIDDEN,
    run_pilot,
)


def test_constants_match_prereg() -> None:
    assert ARMS == ("baseline", "P_min", "P_equ", "P_max")
    assert HIDDEN == (64, 32, 16, 8)


def test_run_pilot_smoke(tmp_path: Path) -> None:
    data_dir = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "g4_split_fmnist"
        / "data"
    )
    out_json = tmp_path / "step1.json"
    out_md = tmp_path / "step1.md"
    registry_db = tmp_path / "registry.sqlite"
    payload = run_pilot(
        data_dir=data_dir,
        seeds=(0,),
        out_json=out_json,
        out_md=out_md,
        registry_db=registry_db,
        epochs=1,
        batch_size=64,
        lr=0.01,
        smoke=True,
    )
    assert len(payload["cells"]) == len(ARMS)
    for cell in payload["cells"]:
        assert "run_id" in cell
        assert "retention" in cell
    assert out_json.exists()
    assert out_md.exists()
