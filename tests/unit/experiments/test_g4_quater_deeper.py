"""Unit tests for the 5-layer deeper hierarchical classifier
(G4-quater Step 1 H4-A substrate)."""
from __future__ import annotations

import numpy as np

from experiments.g4_quater_test.deeper_classifier import (
    G4HierarchicalDeeperClassifier,
)


def test_deeper_classifier_shape() -> None:
    m = G4HierarchicalDeeperClassifier(
        in_dim=784,
        hidden=(64, 32, 16, 8),
        n_classes=2,
        seed=0,
    )
    logits = m.predict_logits(np.zeros((3, 784), dtype=np.float32))
    assert logits.shape == (3, 2)


def test_deeper_classifier_seeded_determinism() -> None:
    x = np.random.RandomState(0).randn(5, 784).astype(np.float32)
    a = G4HierarchicalDeeperClassifier(
        in_dim=784, hidden=(64, 32, 16, 8), n_classes=2, seed=42
    )
    b = G4HierarchicalDeeperClassifier(
        in_dim=784, hidden=(64, 32, 16, 8), n_classes=2, seed=42
    )
    np.testing.assert_array_equal(
        a.predict_logits(x), b.predict_logits(x)
    )


def test_deeper_classifier_latent_shape() -> None:
    m = G4HierarchicalDeeperClassifier(
        in_dim=784,
        hidden=(64, 32, 16, 8),
        n_classes=2,
        seed=0,
    )
    lat = m.latent(np.zeros((3, 784), dtype=np.float32))
    # latent() returns hidden_3 activations (16-dim).
    assert lat.shape == (3, 16)


def test_deeper_classifier_train_then_eval() -> None:
    rng = np.random.RandomState(0)
    x = rng.randn(40, 32).astype(np.float32)
    y = (rng.rand(40) > 0.5).astype(np.int64)
    m = G4HierarchicalDeeperClassifier(
        in_dim=32, hidden=(16, 8, 8, 4), n_classes=2, seed=0
    )
    task = {"x_train": x, "y_train": y}
    m.train_task(task, epochs=2, batch_size=8, lr=0.05)
    acc = m.eval_accuracy(x, y)
    assert 0.0 <= acc <= 1.0
