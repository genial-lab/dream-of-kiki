"""Unit tests for `EsnnG5BisHierarchicalClassifier` (Plan G5-bis Task 2)."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.g5_bis_richer_esnn.esnn_hier_classifier import (
    EsnnG5BisHierarchicalClassifier,
)


def _toy_task(seed: int, in_dim: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 24
    x = rng.standard_normal((n, in_dim)).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    return {
        "x_train": x[:18],
        "y_train": y[:18],
        "x_test": x[18:],
        "y_test": y[18:],
    }


def test_classifier_init_shapes_match_layer_sizes() -> None:
    """W_in / W_h / W_out shapes are derived from layer sizes."""
    clf = EsnnG5BisHierarchicalClassifier(
        in_dim=4, hidden_1=8, hidden_2=6, n_classes=2, seed=0
    )
    assert clf.W_in.shape == (4, 8)
    assert clf.W_h.shape == (8, 6)
    assert clf.W_out.shape == (6, 2)


def test_classifier_rejects_invalid_dims() -> None:
    """Zero-sized layers and < 2 classes are rejected at construction."""
    with pytest.raises(ValueError):
        EsnnG5BisHierarchicalClassifier(
            in_dim=0, hidden_1=4, hidden_2=4, n_classes=2, seed=0
        )
    with pytest.raises(ValueError):
        EsnnG5BisHierarchicalClassifier(
            in_dim=4, hidden_1=0, hidden_2=4, n_classes=2, seed=0
        )
    with pytest.raises(ValueError):
        EsnnG5BisHierarchicalClassifier(
            in_dim=4, hidden_1=4, hidden_2=0, n_classes=2, seed=0
        )
    with pytest.raises(ValueError):
        EsnnG5BisHierarchicalClassifier(
            in_dim=4, hidden_1=4, hidden_2=4, n_classes=1, seed=0
        )


def test_predict_logits_shape_matches_n_classes() -> None:
    """Forward pass returns shape `(N, n_classes)`."""
    clf = EsnnG5BisHierarchicalClassifier(
        in_dim=4, hidden_1=8, hidden_2=6, n_classes=2, seed=0, n_steps=5
    )
    x = np.zeros((3, 4), dtype=np.float32)
    logits = clf.predict_logits(x)
    assert logits.shape == (3, 2)


def test_predict_logits_empty_returns_empty() -> None:
    """Empty input returns shape `(0, n_classes)` (no LIF call)."""
    clf = EsnnG5BisHierarchicalClassifier(
        in_dim=4, hidden_1=8, hidden_2=6, n_classes=2, seed=0, n_steps=5
    )
    x = np.zeros((0, 4), dtype=np.float32)
    logits = clf.predict_logits(x)
    assert logits.shape == (0, 2)


def test_eval_accuracy_in_unit_interval() -> None:
    """Accuracy is a float in [0, 1]."""
    clf = EsnnG5BisHierarchicalClassifier(
        in_dim=4, hidden_1=8, hidden_2=6, n_classes=2, seed=0, n_steps=5
    )
    x = np.zeros((4, 4), dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    acc = clf.eval_accuracy(x, y)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_latent_returns_hidden_2_spike_rates() -> None:
    """`latent(x)` returns `(N, hidden_2)` rates in [0, 1]."""
    clf = EsnnG5BisHierarchicalClassifier(
        in_dim=4, hidden_1=8, hidden_2=6, n_classes=2, seed=0, n_steps=5
    )
    x = np.zeros((3, 4), dtype=np.float32)
    lat = clf.latent(x)
    assert lat.shape == (3, 6)
    assert (lat >= 0.0).all() and (lat <= 1.0).all()


def test_train_task_changes_weights_deterministically() -> None:
    """Same seed + same task -> bit-identical W_in/W_h/W_out, and W_in moves."""
    task = _toy_task(seed=11)
    a = EsnnG5BisHierarchicalClassifier(
        in_dim=4, hidden_1=8, hidden_2=6, n_classes=2, seed=99, n_steps=5
    )
    b = EsnnG5BisHierarchicalClassifier(
        in_dim=4, hidden_1=8, hidden_2=6, n_classes=2, seed=99, n_steps=5
    )
    w_in_before = a.W_in.copy()
    a.train_task(task, epochs=2, batch_size=4, lr=0.1)
    b.train_task(task, epochs=2, batch_size=4, lr=0.1)
    np.testing.assert_array_equal(a.W_in, b.W_in)
    np.testing.assert_array_equal(a.W_h, b.W_h)
    np.testing.assert_array_equal(a.W_out, b.W_out)
    assert not np.allclose(a.W_in, w_in_before)
