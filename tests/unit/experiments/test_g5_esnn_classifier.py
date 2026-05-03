"""Unit tests for `EsnnG5Classifier` — DR-3 cross-substrate validation."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.g5_cross_substrate.esnn_classifier import EsnnG5Classifier


def test_classifier_constructs_with_seeded_weights() -> None:
    """Same seed -> same `W_in` / `W_out` (R1 determinism)."""
    a = EsnnG5Classifier(in_dim=8, hidden_dim=4, n_classes=2, seed=42)
    b = EsnnG5Classifier(in_dim=8, hidden_dim=4, n_classes=2, seed=42)
    np.testing.assert_array_equal(a.W_in, b.W_in)
    np.testing.assert_array_equal(a.W_out, b.W_out)
    assert a.W_in.shape == (8, 4)
    assert a.W_out.shape == (4, 2)


def test_classifier_predict_logits_shape() -> None:
    """`predict_logits(x)` returns shape (N, n_classes) with finite values."""
    clf = EsnnG5Classifier(in_dim=4, hidden_dim=3, n_classes=2, seed=0)
    x = np.zeros((5, 4), dtype=np.float32)
    logits = clf.predict_logits(x)
    assert logits.shape == (5, 2)
    assert np.isfinite(logits).all()


def test_classifier_eval_accuracy_in_unit_range() -> None:
    """`eval_accuracy(x, y)` returns a float in [0, 1] for non-empty input."""
    clf = EsnnG5Classifier(in_dim=4, hidden_dim=3, n_classes=2, seed=0)
    x = np.zeros((4, 4), dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    acc = clf.eval_accuracy(x, y)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_classifier_eval_accuracy_empty_returns_zero() -> None:
    """Empty input returns 0.0 (matches G4Classifier convention)."""
    clf = EsnnG5Classifier(in_dim=4, hidden_dim=3, n_classes=2, seed=0)
    x = np.zeros((0, 4), dtype=np.float32)
    y = np.zeros((0,), dtype=np.int64)
    assert clf.eval_accuracy(x, y) == 0.0


def test_classifier_train_task_changes_weights() -> None:
    """`train_task(...)` mutates `W_out` (the trainable readout)."""
    clf = EsnnG5Classifier(
        in_dim=4, hidden_dim=3, n_classes=2, seed=0, n_steps=8
    )
    w_out_before = clf.W_out.copy()
    rng = np.random.default_rng(7)
    task = {
        "x_train": rng.standard_normal((20, 4)).astype(np.float32),
        "y_train": rng.integers(0, 2, size=20).astype(np.int64),
        "x_test": rng.standard_normal((4, 4)).astype(np.float32),
        "y_test": rng.integers(0, 2, size=4).astype(np.int64),
    }
    clf.train_task(task, epochs=2, batch_size=4, lr=0.1)
    assert not np.allclose(clf.W_out, w_out_before)


def test_classifier_train_task_deterministic() -> None:
    """Two classifiers with same seed + same task -> same final weights."""
    rng = np.random.default_rng(11)
    task = {
        "x_train": rng.standard_normal((16, 4)).astype(np.float32),
        "y_train": rng.integers(0, 2, size=16).astype(np.int64),
        "x_test": rng.standard_normal((4, 4)).astype(np.float32),
        "y_test": rng.integers(0, 2, size=4).astype(np.int64),
    }
    a = EsnnG5Classifier(
        in_dim=4, hidden_dim=3, n_classes=2, seed=99, n_steps=8
    )
    b = EsnnG5Classifier(
        in_dim=4, hidden_dim=3, n_classes=2, seed=99, n_steps=8
    )
    a.train_task(task, epochs=2, batch_size=4, lr=0.1)
    b.train_task(task, epochs=2, batch_size=4, lr=0.1)
    np.testing.assert_allclose(a.W_in, b.W_in)
    np.testing.assert_allclose(a.W_out, b.W_out)


def test_classifier_predict_uses_lif_simulation() -> None:
    """The forward path drives the LIF population (n_steps > 0 changes output).

    Sanity check : with n_steps=0 the population produces zero firing
    rate, so logits collapse to bias only ; with n_steps=20 the rates
    are non-trivial and logits differ.
    """
    clf_lo = EsnnG5Classifier(
        in_dim=4, hidden_dim=3, n_classes=2, seed=0, n_steps=0
    )
    clf_hi = EsnnG5Classifier(
        in_dim=4, hidden_dim=3, n_classes=2, seed=0, n_steps=20
    )
    # Identical weights via same seed
    np.testing.assert_array_equal(clf_lo.W_in, clf_hi.W_in)
    x = np.ones((2, 4), dtype=np.float32)
    logits_lo = clf_lo.predict_logits(x)
    logits_hi = clf_hi.predict_logits(x)
    assert not np.allclose(logits_lo, logits_hi)


def test_classifier_validates_dim_constraints() -> None:
    """Reject zero / negative dims at construction."""
    with pytest.raises(ValueError, match="in_dim"):
        EsnnG5Classifier(in_dim=0, hidden_dim=4, n_classes=2, seed=0)
    with pytest.raises(ValueError, match="hidden_dim"):
        EsnnG5Classifier(in_dim=4, hidden_dim=0, n_classes=2, seed=0)
    with pytest.raises(ValueError, match="n_classes"):
        EsnnG5Classifier(in_dim=4, hidden_dim=4, n_classes=1, seed=0)
