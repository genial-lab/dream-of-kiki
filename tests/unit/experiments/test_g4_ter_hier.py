"""Unit tests for the G4HierarchicalClassifier (Plan G4-ter Task 2)."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.g4_ter_hp_sweep.dream_wrap_hier import (
    BetaBufferHierFIFO,
    G4HierarchicalClassifier,
)


def test_classifier_has_three_linear_layers() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=784, hidden_1=32, hidden_2=16, n_classes=2, seed=7
    )
    # Public attributes for RESTRUCTURE site identification.
    assert clf.hidden_1 == 32
    assert clf.hidden_2 == 16
    assert clf.n_classes == 2


def test_predict_logits_shape() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=784, hidden_1=32, hidden_2=16, n_classes=2, seed=7
    )
    x = np.zeros((4, 784), dtype=np.float32)
    logits = clf.predict_logits(x)
    assert logits.shape == (4, 2)


def test_seed_determinism() -> None:
    a = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=42
    )
    b = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=42
    )
    x = np.ones((1, 10), dtype=np.float32)
    np.testing.assert_array_equal(a.predict_logits(x), b.predict_logits(x))


def _toy_task(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 64
    x = rng.standard_normal((n, 10)).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.int32)
    return {
        "x_train": x[:48],
        "y_train": y[:48],
        "x_test": x[48:],
        "y_test": y[48:],
    }


def test_train_task_then_eval_accuracy() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=10, hidden_1=32, hidden_2=16, n_classes=2, seed=11
    )
    task = _toy_task(seed=11)
    clf.train_task(task, epochs=2, batch_size=16, lr=0.05)
    acc = clf.eval_accuracy(task["x_test"], task["y_test"])
    assert 0.0 <= acc <= 1.0


def test_buffer_push_pop_with_latent() -> None:
    buf = BetaBufferHierFIFO(capacity=4)
    x = np.zeros(10, dtype=np.float32)
    latent = np.ones(16, dtype=np.float32)
    buf.push(x=x, y=1, latent=latent)
    snap = buf.snapshot()
    assert len(snap) == 1
    assert snap[0]["y"] == 1
    assert snap[0]["latent"] == [1.0] * 16


def test_buffer_sample_deterministic() -> None:
    buf = BetaBufferHierFIFO(capacity=8)
    for i in range(8):
        buf.push(
            x=np.full(4, float(i), dtype=np.float32),
            y=i % 2,
            latent=np.full(2, float(i), dtype=np.float32),
        )
    a = buf.sample(n=3, seed=42)
    b = buf.sample(n=3, seed=42)
    assert [r["x"] for r in a] == [r["x"] for r in b]


def test_buffer_sample_no_latent() -> None:
    """latent=None records are still sampleable (legacy compat)."""
    buf = BetaBufferHierFIFO(capacity=2)
    buf.push(x=np.zeros(2, dtype=np.float32), y=0, latent=None)
    snap = buf.snapshot()
    assert snap[0]["latent"] is None
