"""Split-FMNIST 5-task loader — pure numpy, no torchvision.

Decodes the canonical Fashion-MNIST IDX files (gzipped magic
2051 / 2049) into a list of 5 task dicts, each carrying flat
float32 images normalised to [0, 1] plus binary labels remapped
to {0, 1}.

Task split (canonical class-incremental Split-FMNIST) :

    task 0 : classes {0, 1}  -> remapped to {0, 1}
    task 1 : classes {2, 3}  -> remapped to {0, 1}
    task 2 : classes {4, 5}  -> remapped to {0, 1}
    task 3 : classes {6, 7}  -> remapped to {0, 1}
    task 4 : classes {8, 9}  -> remapped to {0, 1}

The remap to {0, 1} keeps the classifier head fixed at 2 outputs
across tasks (binary head shared, weights drift between tasks
exactly the way that drives catastrophic forgetting).

Reference :
    Hsu et al. 2018 — "Re-evaluating continual learning"
    Fashion-MNIST mirror : https://github.com/zalandoresearch/fashion-mnist
"""
from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Final, TypedDict

import numpy as np


SPLIT_FMNIST_TASKS: Final[tuple[tuple[int, int], ...]] = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
)


class SplitFMNISTTask(TypedDict):
    """One Split-FMNIST 2-class task : (train, test) flat float32."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def decode_idx_images(path: Path) -> np.ndarray:
    """Decode an IDX-3 (image) gzipped file into a uint8 3-D array.

    Magic = 2051 (per Fashion-MNIST spec). Returns shape
    ``(N, H, W)`` with H = W = 28 for canonical FMNIST (smaller in
    test fixtures).

    Raises :
        ValueError : magic mismatch or truncated payload.
    """
    with gzip.open(path, "rb") as fh:
        header = fh.read(16)
        magic, n, h, w = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(
                f"bad image-IDX magic {magic} (expected 2051) in {path}"
            )
        payload = fh.read(n * h * w)
        if len(payload) != n * h * w:
            raise ValueError(
                f"truncated image payload in {path}: got "
                f"{len(payload)} expected {n * h * w}"
            )
    return np.frombuffer(payload, dtype=np.uint8).reshape(n, h, w)


def decode_idx_labels(path: Path) -> np.ndarray:
    """Decode an IDX-1 (label) gzipped file into a uint8 1-D array.

    Magic = 2049. Returns shape ``(N,)``.

    Raises :
        ValueError : magic mismatch or truncated payload.
    """
    with gzip.open(path, "rb") as fh:
        header = fh.read(8)
        magic, n = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(
                f"bad label-IDX magic {magic} (expected 2049) in {path}"
            )
        payload = fh.read(n)
        if len(payload) != n:
            raise ValueError(
                f"truncated label payload in {path}: got "
                f"{len(payload)} expected {n}"
            )
    return np.frombuffer(payload, dtype=np.uint8)


def load_split_fmnist_5tasks(data_dir: Path) -> list[SplitFMNISTTask]:
    """Load Split-FMNIST as 5 sequential 2-class binary tasks.

    Expects the four canonical FMNIST gzipped files in
    ``data_dir`` :

        train-images-idx3-ubyte.gz
        train-labels-idx1-ubyte.gz
        t10k-images-idx3-ubyte.gz
        t10k-labels-idx1-ubyte.gz

    Returns a list of 5 :class:`SplitFMNISTTask` dicts, each with
    flattened float32 images normalised to ``[0, 1]`` and labels
    remapped to ``{0, 1}`` (binary head shared across tasks).

    Raises :
        FileNotFoundError : ``data_dir`` does not exist or any of
                            the four IDX files is missing.
    """
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(
            f"FMNIST data dir does not exist : {data_dir}"
        )
    files = {
        "x_train": data_dir / "train-images-idx3-ubyte.gz",
        "y_train": data_dir / "train-labels-idx1-ubyte.gz",
        "x_test": data_dir / "t10k-images-idx3-ubyte.gz",
        "y_test": data_dir / "t10k-labels-idx1-ubyte.gz",
    }
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"FMNIST {name} file missing : {path}"
            )

    x_train_raw = decode_idx_images(files["x_train"])
    y_train_raw = decode_idx_labels(files["y_train"])
    x_test_raw = decode_idx_images(files["x_test"])
    y_test_raw = decode_idx_labels(files["y_test"])

    n_train, h, w = x_train_raw.shape
    feat_dim = h * w
    x_train = (x_train_raw.astype(np.float32) / 255.0).reshape(
        n_train, feat_dim
    )
    x_test = (x_test_raw.astype(np.float32) / 255.0).reshape(
        x_test_raw.shape[0], feat_dim
    )

    tasks: list[SplitFMNISTTask] = []
    for class_a, class_b in SPLIT_FMNIST_TASKS:
        train_mask = (y_train_raw == class_a) | (y_train_raw == class_b)
        test_mask = (y_test_raw == class_a) | (y_test_raw == class_b)
        y_train_task = np.where(
            y_train_raw[train_mask] == class_a, 0, 1
        ).astype(np.int64)
        y_test_task = np.where(
            y_test_raw[test_mask] == class_a, 0, 1
        ).astype(np.int64)
        tasks.append(
            SplitFMNISTTask(
                x_train=x_train[train_mask],
                y_train=y_train_task,
                x_test=x_test[test_mask],
                y_test=y_test_task,
            )
        )
    return tasks
