"""Unit tests for the MLX classifier + dream-episode wrapper."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.g4_split_fmnist.dataset import SplitFMNISTTask
from experiments.g4_split_fmnist.dream_wrap import (
    G4Classifier,
    PROFILE_FACTORIES,
    build_profile,
    sample_beta_records,
)


@pytest.fixture()
def tiny_task() -> SplitFMNISTTask:
    """16-feature 2-class binary task ; 100 train + 40 test."""
    rng = np.random.default_rng(0)
    return SplitFMNISTTask(
        x_train=rng.standard_normal((100, 16)).astype(np.float32),
        y_train=rng.integers(0, 2, size=(100,), dtype=np.int64),
        x_test=rng.standard_normal((40, 16)).astype(np.float32),
        y_test=rng.integers(0, 2, size=(40,), dtype=np.int64),
    )


def test_classifier_constructs_with_seed() -> None:
    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    assert clf.seed == 42
    assert clf.in_dim == 16


def test_classifier_seed_determines_initial_weights() -> None:
    clf_a = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    clf_b = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    clf_c = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=99)
    # Same seed → identical initial logits at zero input
    np.testing.assert_array_equal(
        clf_a.predict_logits(np.zeros((1, 16), dtype=np.float32)),
        clf_b.predict_logits(np.zeros((1, 16), dtype=np.float32)),
    )
    # Different seed → different initial logits
    assert not np.allclose(
        clf_a.predict_logits(np.zeros((1, 16), dtype=np.float32)),
        clf_c.predict_logits(np.zeros((1, 16), dtype=np.float32)),
    )


def test_classifier_train_task_increases_train_accuracy(
    tiny_task: SplitFMNISTTask,
) -> None:
    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    pre = clf.eval_accuracy(tiny_task["x_train"], tiny_task["y_train"])
    clf.train_task(tiny_task, epochs=5, batch_size=16, lr=0.05)
    post = clf.eval_accuracy(tiny_task["x_train"], tiny_task["y_train"])
    assert post >= pre  # weak monotonicity — 5 epochs cannot reduce train acc


def test_classifier_eval_accuracy_in_unit_interval(tiny_task: SplitFMNISTTask) -> None:
    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    acc = clf.eval_accuracy(tiny_task["x_test"], tiny_task["y_test"])
    assert 0.0 <= acc <= 1.0


def test_profile_factories_keys_are_canonical() -> None:
    assert set(PROFILE_FACTORIES) == {"P_min", "P_equ", "P_max"}


@pytest.mark.parametrize("name", ["P_min", "P_equ", "P_max"])
def test_build_profile_returns_runtime_with_handlers(name: str) -> None:
    from kiki_oniric.dream.episode import Operation

    profile = build_profile(name, seed=7)
    expected_ops = (
        {Operation.REPLAY, Operation.DOWNSCALE}
        if name == "P_min"
        else {
            Operation.REPLAY,
            Operation.DOWNSCALE,
            Operation.RESTRUCTURE,
            Operation.RECOMBINE,
        }
    )
    # Each expected op must have a handler registered on the runtime
    for op in expected_ops:
        assert op in profile.runtime._handlers


def test_sample_beta_records_seeded_reproducible() -> None:
    a = sample_beta_records(seed=42, n_records=4, feat_dim=16)
    b = sample_beta_records(seed=42, n_records=4, feat_dim=16)
    assert len(a) == 4
    for ra, rb in zip(a, b, strict=True):
        np.testing.assert_array_equal(ra["x"], rb["x"])
        np.testing.assert_array_equal(ra["y"], rb["y"])


def test_sample_beta_records_different_seeds_differ() -> None:
    a = sample_beta_records(seed=42, n_records=4, feat_dim=16)
    b = sample_beta_records(seed=99, n_records=4, feat_dim=16)
    # At least one record differs
    assert any(
        not np.array_equal(ra["x"], rb["x"])
        for ra, rb in zip(a, b, strict=True)
    )


def test_dream_episode_executes_pmin_handlers(tiny_task: SplitFMNISTTask) -> None:
    """P_min episode must add at least 1 entry to runtime.log (DR-0)."""
    from experiments.g4_split_fmnist.dream_wrap import BetaBufferFIFO

    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    clf.train_task(tiny_task, epochs=2, batch_size=16, lr=0.05)
    profile = build_profile("P_min", seed=7)
    buf = BetaBufferFIFO(capacity=8)
    log_before = len(profile.runtime.log)
    clf.dream_episode(profile, seed=7, beta_buffer=buf)
    assert len(profile.runtime.log) == log_before + 1


def test_dream_episode_executes_pequ_with_4_ops(tiny_task: SplitFMNISTTask) -> None:
    """P_equ episode must execute 4 ops (replay/downscale/restructure/recombine)."""
    from kiki_oniric.dream.episode import Operation
    from experiments.g4_split_fmnist.dream_wrap import BetaBufferFIFO

    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    clf.train_task(tiny_task, epochs=2, batch_size=16, lr=0.05)
    profile = build_profile("P_equ", seed=7)
    buf = BetaBufferFIFO(capacity=8)
    clf.dream_episode(profile, seed=7, beta_buffer=buf)
    last = profile.runtime.log[-1]
    assert last.completed
    assert set(last.operations_executed) == {
        Operation.REPLAY,
        Operation.DOWNSCALE,
        Operation.RESTRUCTURE,
        Operation.RECOMBINE,
    }


def test_dream_episode_baseline_returns_no_op() -> None:
    """build_profile('baseline') must raise — baseline runs no DE."""
    with pytest.raises(ValueError, match="baseline"):
        build_profile("baseline", seed=7)


def test_replay_optimizer_step_changes_weights() -> None:
    """A replay step on non-empty records must mutate weights."""
    from experiments.g4_split_fmnist.dream_wrap import BetaBufferFIFO

    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    # Snapshot a layer's weight before any update.
    w_before = np.asarray(clf._model.layers[0].weight).copy()

    # Build a 4-record sample with deterministic RNG.
    rng = np.random.default_rng(0)
    buf = BetaBufferFIFO(capacity=8)
    for i in range(4):
        buf.push(rng.standard_normal(16).astype(np.float32), i % 2)

    records = buf.sample(n=4, seed=0)
    clf._replay_optimizer_step(records, lr=0.1, n_steps=2)

    w_after = np.asarray(clf._model.layers[0].weight)
    assert not np.allclose(w_before, w_after), (
        "weights must change after replay step"
    )


def test_replay_optimizer_step_empty_records_noop() -> None:
    """Empty record list must leave weights untouched."""
    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    w_before = np.asarray(clf._model.layers[0].weight).copy()
    clf._replay_optimizer_step([], lr=0.1, n_steps=2)
    w_after = np.asarray(clf._model.layers[0].weight)
    np.testing.assert_array_equal(w_before, w_after)


def test_replay_optimizer_step_seeded_reproducible() -> None:
    """Two classifiers with same seed + same records → same final weights."""
    from experiments.g4_split_fmnist.dream_wrap import BetaBufferFIFO

    rng = np.random.default_rng(0)
    buf = BetaBufferFIFO(capacity=8)
    for i in range(4):
        buf.push(rng.standard_normal(16).astype(np.float32), i % 2)
    records = buf.sample(n=4, seed=42)

    clf_a = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    clf_b = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    clf_a._replay_optimizer_step(records, lr=0.1, n_steps=2)
    clf_b._replay_optimizer_step(records, lr=0.1, n_steps=2)

    np.testing.assert_array_equal(
        np.asarray(clf_a._model.layers[0].weight),
        np.asarray(clf_b._model.layers[0].weight),
    )


def test_downscale_step_multiplies_weights_by_factor() -> None:
    """Each layer's weight must scale by exactly ``factor``."""
    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    w_before = [
        np.asarray(layer.weight).copy()
        for layer in clf._model.layers
        if hasattr(layer, "weight") and layer.weight is not None
    ]
    clf._downscale_step(factor=0.95)
    w_after = [
        np.asarray(layer.weight)
        for layer in clf._model.layers
        if hasattr(layer, "weight") and layer.weight is not None
    ]
    assert len(w_before) == len(w_after) > 0
    for wb, wa in zip(w_before, w_after, strict=True):
        np.testing.assert_allclose(wa, wb * 0.95, rtol=1e-6)


def test_downscale_step_factor_bounds() -> None:
    """``factor`` outside (0, 1] must raise ValueError."""
    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    with pytest.raises(ValueError, match="shrink_factor|factor"):
        clf._downscale_step(factor=0.0)
    with pytest.raises(ValueError, match="shrink_factor|factor"):
        clf._downscale_step(factor=1.5)
    with pytest.raises(ValueError, match="shrink_factor|factor"):
        clf._downscale_step(factor=-0.1)


def test_dream_episode_pmin_mutates_weights_when_buffer_nonempty(
    tiny_task: SplitFMNISTTask,
) -> None:
    """P_min dream_episode with a populated β buffer must mutate weights."""
    from experiments.g4_split_fmnist.dream_wrap import BetaBufferFIFO

    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    clf.train_task(tiny_task, epochs=2, batch_size=16, lr=0.05)
    profile = build_profile("P_min", seed=7)

    rng = np.random.default_rng(0)
    buf = BetaBufferFIFO(capacity=8)
    for i in range(4):
        buf.push(rng.standard_normal(16).astype(np.float32), i % 2)

    w_before = np.asarray(clf._model.layers[0].weight).copy()
    clf.dream_episode(profile, seed=7, beta_buffer=buf)
    w_after = np.asarray(clf._model.layers[0].weight)
    assert not np.allclose(w_before, w_after), (
        "P_min dream_episode with non-empty buffer must mutate weights"
    )


def test_dream_episode_pmin_empty_buffer_only_downscales(
    tiny_task: SplitFMNISTTask,
) -> None:
    """Empty buffer → no replay step, but DOWNSCALE still fires (P_min ops)."""
    from experiments.g4_split_fmnist.dream_wrap import BetaBufferFIFO

    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    clf.train_task(tiny_task, epochs=2, batch_size=16, lr=0.05)
    profile = build_profile("P_min", seed=7)
    buf = BetaBufferFIFO(capacity=8)  # left empty

    w_before = np.asarray(clf._model.layers[0].weight).copy()
    clf.dream_episode(profile, seed=7, beta_buffer=buf)
    w_after = np.asarray(clf._model.layers[0].weight)
    # Downscale by 0.95 must scale visibly.
    np.testing.assert_allclose(w_after, w_before * 0.95, rtol=1e-6)


def test_dream_episode_baseline_arm_does_not_call_dream_episode() -> None:
    """build_profile('baseline') must still raise — driver-level guard."""
    with pytest.raises(ValueError, match="baseline"):
        build_profile("baseline", seed=7)
