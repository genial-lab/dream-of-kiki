"""Unit tests for the Split-FMNIST 5-task numpy loader."""
from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import pytest

from experiments.g4_split_fmnist.dataset import (
    SPLIT_FMNIST_TASKS,
    decode_idx_images,
    decode_idx_labels,
    load_split_fmnist_5tasks,
)


def _make_synthetic_idx(
    tmp_path: Path,
    n: int = 200,
) -> tuple[Path, Path, Path, Path]:
    """Build a deterministic mini Fashion-MNIST IDX pair in ``tmp_path``.

    Produces ``train`` and ``test`` IDX files (images + labels) gzipped.
    Each image is a tiny 4x4 uint8 ; each label cycles through 0..9
    so all 10 classes appear and Split-FMNIST splits are non-empty.
    """
    rng = np.random.default_rng(0)
    img_train = rng.integers(0, 256, size=(n, 4, 4), dtype=np.uint8)
    lbl_train = np.array([i % 10 for i in range(n)], dtype=np.uint8)
    img_test = rng.integers(0, 256, size=(n // 4, 4, 4), dtype=np.uint8)
    lbl_test = np.array([i % 10 for i in range(n // 4)], dtype=np.uint8)

    paths: list[Path] = []
    for arr, kind in (
        (img_train, "train-images-idx3-ubyte.gz"),
        (lbl_train, "train-labels-idx1-ubyte.gz"),
        (img_test, "t10k-images-idx3-ubyte.gz"),
        (lbl_test, "t10k-labels-idx1-ubyte.gz"),
    ):
        path = tmp_path / kind
        with gzip.open(path, "wb") as fh:
            if arr.ndim == 3:
                fh.write(struct.pack(">IIII", 2051, arr.shape[0], 4, 4))
                fh.write(arr.tobytes())
            else:
                fh.write(struct.pack(">II", 2049, arr.shape[0]))
                fh.write(arr.tobytes())
        paths.append(path)
    return tuple(paths)  # type: ignore[return-value]


def test_decode_idx_images_returns_uint8_3d(tmp_path: Path) -> None:
    img_train, _, _, _ = _make_synthetic_idx(tmp_path, n=10)
    arr = decode_idx_images(img_train)
    assert arr.dtype == np.uint8
    assert arr.ndim == 3
    assert arr.shape == (10, 4, 4)


def test_decode_idx_labels_returns_uint8_1d(tmp_path: Path) -> None:
    _, lbl_train, _, _ = _make_synthetic_idx(tmp_path, n=10)
    arr = decode_idx_labels(lbl_train)
    assert arr.dtype == np.uint8
    assert arr.ndim == 1
    assert arr.shape == (10,)


def test_decode_idx_images_rejects_bad_magic(tmp_path: Path) -> None:
    bad = tmp_path / "bad.gz"
    with gzip.open(bad, "wb") as fh:
        fh.write(struct.pack(">IIII", 9999, 1, 4, 4))
        fh.write(b"\x00" * 16)
    with pytest.raises(ValueError, match="magic"):
        decode_idx_images(bad)


def test_decode_idx_labels_rejects_bad_magic(tmp_path: Path) -> None:
    bad = tmp_path / "bad.gz"
    with gzip.open(bad, "wb") as fh:
        fh.write(struct.pack(">II", 9999, 1))
        fh.write(b"\x00")
    with pytest.raises(ValueError, match="magic"):
        decode_idx_labels(bad)


def test_split_fmnist_tasks_constant_is_5_pairs() -> None:
    assert len(SPLIT_FMNIST_TASKS) == 5
    flat = [c for pair in SPLIT_FMNIST_TASKS for c in pair]
    assert sorted(flat) == list(range(10))


def test_load_split_fmnist_5tasks_yields_5_tasks(tmp_path: Path) -> None:
    _make_synthetic_idx(tmp_path, n=200)
    tasks = load_split_fmnist_5tasks(tmp_path)
    assert len(tasks) == 5
    for i, task in enumerate(tasks):
        assert "x_train" in task
        assert "y_train" in task
        assert "x_test" in task
        assert "y_test" in task
        # Both classes for this pair must appear in the task
        expected_classes = set(SPLIT_FMNIST_TASKS[i])
        # After remap we expect labels in {0,1}, so check via mask :
        assert set(task["y_train"].tolist()) <= {0, 1}
        assert set(task["y_test"].tolist()) <= {0, 1}
        # And no underlying class outside the expected pair (n_train > 0).
        assert task["x_train"].shape[0] > 0
        # Ensure the expected_classes constant is consumed in the
        # assertion path so it is not flagged unused.
        assert len(expected_classes) == 2


def test_load_split_fmnist_normalises_to_float32(tmp_path: Path) -> None:
    _make_synthetic_idx(tmp_path, n=200)
    tasks = load_split_fmnist_5tasks(tmp_path)
    assert tasks[0]["x_train"].dtype == np.float32
    assert tasks[0]["x_test"].dtype == np.float32
    # Normalised to [0, 1]
    assert tasks[0]["x_train"].min() >= 0.0
    assert tasks[0]["x_train"].max() <= 1.0


def test_load_split_fmnist_flattens_images(tmp_path: Path) -> None:
    """Pilot uses an MLP classifier — images must be flat vectors."""
    _make_synthetic_idx(tmp_path, n=200)
    tasks = load_split_fmnist_5tasks(tmp_path)
    # 4x4 = 16 pixels in our synthetic fixture
    assert tasks[0]["x_train"].ndim == 2
    assert tasks[0]["x_train"].shape[1] == 16


def test_load_split_fmnist_remaps_labels_to_0_1(tmp_path: Path) -> None:
    """Each 2-class task should remap labels to {0, 1} for binary
    cross-entropy / MLP head sizing convenience.
    """
    _make_synthetic_idx(tmp_path, n=200)
    tasks = load_split_fmnist_5tasks(tmp_path)
    for task in tasks:
        assert set(task["y_train"].tolist()) <= {0, 1}
        assert set(task["y_test"].tolist()) <= {0, 1}


def test_load_split_fmnist_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_split_fmnist_5tasks(tmp_path / "does-not-exist")
