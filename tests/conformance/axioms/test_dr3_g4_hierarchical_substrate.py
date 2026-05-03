"""DR-3 Conformance - G4-ter hierarchical substrate.

Verifies the richer substrate (G4HierarchicalClassifier) satisfies
DR-3 conditions (1) typed Protocols, (2) executable primitives,
(3) primitives chain without raising.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md sec 6.2
"""
from __future__ import annotations

import numpy as np

from experiments.g4_split_fmnist.dream_wrap import build_profile
from experiments.g4_ter_hp_sweep.dream_wrap_hier import (
    BetaBufferHierFIFO,
    G4HierarchicalClassifier,
)


def _fill_buffer(buf: BetaBufferHierFIFO, clf: G4HierarchicalClassifier,
                 n_per_class: int = 4) -> None:
    rng = np.random.default_rng(0)
    for cls in (0, 1):
        for _ in range(n_per_class):
            x = rng.standard_normal(10).astype(np.float32)
            latent = clf.latent(x[None, :])[0]
            buf.push(x=x, y=cls, latent=latent)


def test_dr3_g4_hier_protocols_present() -> None:
    """DR-3 (1) - substrate exposes the public coupling methods."""
    clf = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=0
    )
    assert hasattr(clf, "predict_logits")
    assert hasattr(clf, "latent")
    assert hasattr(clf, "train_task")
    assert hasattr(clf, "eval_accuracy")
    assert hasattr(clf, "dream_episode_hier")


def test_dr3_g4_hier_primitives_executable() -> None:
    """DR-3 (2) - REPLAY/DOWNSCALE/RESTRUCTURE/RECOMBINE all dispatchable."""
    clf = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=0
    )
    buf = BetaBufferHierFIFO(capacity=16)
    _fill_buffer(buf, clf)
    profile = build_profile("P_max", seed=0)
    clf.dream_episode_hier(
        profile,
        seed=0,
        beta_buffer=buf,
        replay_n_records=4,
        replay_n_steps=1,
        replay_lr=0.01,
        downscale_factor=0.95,
        restructure_factor=0.05,
        recombine_n_synthetic=4,
        recombine_lr=0.01,
    )
    assert len(profile.runtime.log) == 1


def test_dr3_g4_hier_primitives_chain() -> None:
    """DR-3 (3) - three consecutive episodes do not raise."""
    clf = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=0
    )
    buf = BetaBufferHierFIFO(capacity=16)
    _fill_buffer(buf, clf)
    profile = build_profile("P_max", seed=0)
    for k in range(3):
        clf.dream_episode_hier(
            profile,
            seed=k,
            beta_buffer=buf,
            replay_n_records=4,
            replay_n_steps=1,
            replay_lr=0.01,
            downscale_factor=0.95,
            restructure_factor=0.05,
            recombine_n_synthetic=4,
            recombine_lr=0.01,
        )
    assert len(profile.runtime.log) == 3
