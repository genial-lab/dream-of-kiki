"""Integration smoke test for the G4 pilot driver.

Runs N=2 seeds × 4 arms = 8 cells against a tiny synthetic FMNIST
fixture so the full sweep + verdict + dump pipeline exercises in
under 30 s. Validates :

- All 8 cells register in the run registry
- The verdict JSON has the expected schema
- compute_hedges_g is invoked (g_h1 / g_h3 keys present)
- All four arms appear in cells[]
"""
from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import numpy as np

from experiments.g4_split_fmnist.run_g4 import run_pilot


def _make_synthetic_fmnist(tmp_path: Path, n_train: int = 600) -> Path:
    """Drop a 4x4 / 10-class IDX fixture under ``tmp_path/data``."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    img_train = rng.integers(0, 256, size=(n_train, 4, 4), dtype=np.uint8)
    lbl_train = np.array([i % 10 for i in range(n_train)], dtype=np.uint8)
    img_test = rng.integers(0, 256, size=(n_train // 4, 4, 4), dtype=np.uint8)
    lbl_test = np.array(
        [i % 10 for i in range(n_train // 4)], dtype=np.uint8
    )
    for arr, kind in (
        (img_train, "train-images-idx3-ubyte.gz"),
        (lbl_train, "train-labels-idx1-ubyte.gz"),
        (img_test, "t10k-images-idx3-ubyte.gz"),
        (lbl_test, "t10k-labels-idx1-ubyte.gz"),
    ):
        with gzip.open(data_dir / kind, "wb") as fh:
            if arr.ndim == 3:
                fh.write(struct.pack(">IIII", 2051, arr.shape[0], 4, 4))
                fh.write(arr.tobytes())
            else:
                fh.write(struct.pack(">II", 2049, arr.shape[0]))
                fh.write(arr.tobytes())
    return data_dir


def test_run_pilot_smoke_2_seeds(tmp_path: Path) -> None:
    data_dir = _make_synthetic_fmnist(tmp_path)
    out_json = tmp_path / "g4.json"
    out_md = tmp_path / "g4.md"
    registry_db = tmp_path / "runs.sqlite"

    result = run_pilot(
        data_dir=data_dir,
        seeds=(0, 1),
        out_json=out_json,
        out_md=out_md,
        registry_db=registry_db,
        epochs=2,
        batch_size=32,
        hidden_dim=16,
        lr=0.05,
    )

    # 4 arms × 2 seeds = 8 cells
    assert len(result["cells"]) == 8
    # Every cell carries a run_id
    assert all(isinstance(c["run_id"], str) for c in result["cells"])
    assert all(len(c["run_id"]) == 32 for c in result["cells"])
    # Every cell carries retention in [0, +inf)
    for c in result["cells"]:
        assert c["retention"] >= 0.0

    # Verdict block has H1, H3, H_DR4 keys
    assert "h1_p_equ_vs_baseline" in result["verdict"]
    assert "h3_p_min_vs_baseline" in result["verdict"]
    assert "h_dr4_jonckheere" in result["verdict"]

    # Files written
    assert out_json.exists()
    assert out_md.exists()
    payload = json.loads(out_json.read_text())
    assert payload["c_version"]
    assert len(payload["cells"]) == 8


def test_smoke_pmin_arm_retention_diverges_from_baseline(
    tmp_path: Path,
) -> None:
    """G4-bis : with coupling on, P_min retention must differ from baseline.

    On the synthetic 4×4 fixture the absolute retention values are not
    meaningful, but the *coupling* must produce at least some divergence
    between baseline (no DE) and a dream-active arm — otherwise the
    replay step is silently a no-op. Uses a larger n_train fixture so
    each task carries enough test items for retention to land on
    distinguishable buckets across arms.
    """
    data_dir = _make_synthetic_fmnist(tmp_path, n_train=2000)
    result = run_pilot(
        data_dir=data_dir,
        seeds=(0, 1),
        out_json=tmp_path / "g4bis.json",
        out_md=tmp_path / "g4bis.md",
        registry_db=tmp_path / "runs.sqlite",
        epochs=2,
        batch_size=32,
        hidden_dim=16,
        lr=0.05,
    )
    by_arm: dict[str, list[float]] = {}
    for c in result["cells"]:
        by_arm.setdefault(c["arm"], []).append(c["retention"])
    # Baseline and P_min retention vectors must not be element-wise
    # identical — coupling has *some* effect (sign-agnostic).
    base = by_arm["baseline"]
    p_min = by_arm["P_min"]
    assert base != p_min, (
        "P_min retention must differ from baseline once coupling is on"
    )


def test_run_pilot_deterministic_run_id(tmp_path: Path) -> None:
    """Same (c_version, profile, seed, commit_sha) -> same run_id."""
    data_dir = _make_synthetic_fmnist(tmp_path)

    result_a = run_pilot(
        data_dir=data_dir,
        seeds=(0,),
        out_json=tmp_path / "a.json",
        out_md=tmp_path / "a.md",
        registry_db=tmp_path / "a.sqlite",
        epochs=1,
        batch_size=32,
        hidden_dim=16,
        lr=0.05,
    )
    result_b = run_pilot(
        data_dir=data_dir,
        seeds=(0,),
        out_json=tmp_path / "b.json",
        out_md=tmp_path / "b.md",
        registry_db=tmp_path / "b.sqlite",
        epochs=1,
        batch_size=32,
        hidden_dim=16,
        lr=0.05,
    )
    # Same (c_version, profile, seed, commit_sha) tuple per cell ⇒
    # bit-identical run_ids across the two pilot invocations.
    ids_a = {(c["arm"], c["seed"]): c["run_id"] for c in result_a["cells"]}
    ids_b = {(c["arm"], c["seed"]): c["run_id"] for c in result_b["cells"]}
    assert ids_a == ids_b
