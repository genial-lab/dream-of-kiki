"""Hierarchical MLX MLP classifier + dream-episode wrapper for G4-ter.

Architecture: input -> Linear(in_dim, hidden_1) -> ReLU ->
Linear(hidden_1, hidden_2) -> ReLU -> Linear(hidden_2, n_classes).

Compared to ``experiments.g4_split_fmnist.dream_wrap.G4Classifier``
the hierarchy exposes a *middle* hidden layer (hidden_2) that is
addressable as a RESTRUCTURE site (perturb its weight tensor without
touching input projection nor output classifier) and a latent
representation (hidden_2 activations) that is addressable as a
RECOMBINE site (Gaussian-MoG synthetic-latent injection).

DR-0 accountability is automatic: every call to ``dream_episode_hier``
appends one EpisodeLogEntry to ``profile.runtime.log`` regardless of
handler outcome.

Reference :
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md sec 3.1
    docs/osf-prereg-g4-ter-pilot.md sec 2-3
    docs/superpowers/plans/2026-05-03-g4-ter-hp-sweep-richer-substrate.md
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TypedDict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


class BetaRecordHier(TypedDict, total=False):
    """One curated episodic exemplar for the hierarchical head.

    Compared to ``BetaRecord`` (G4-bis), adds an optional
    ``latent`` field that holds the hidden_2 activation captured at
    push time, used as the support set for the RECOMBINE Gaussian-
    MoG sampler.
    """

    x: list[float]
    y: int
    latent: list[float] | None


@dataclass
class G4HierarchicalClassifier:
    """Hierarchical MLP classifier for Split-FMNIST 2-class tasks.

    Layers : Linear(in_dim, hidden_1) -> ReLU -> Linear(hidden_1,
    hidden_2) -> ReLU -> Linear(hidden_2, n_classes). Deterministic
    under a fixed ``seed`` via ``mx.random.seed`` at construction.
    """

    in_dim: int
    hidden_1: int
    hidden_2: int
    n_classes: int
    seed: int
    _l1: nn.Linear = field(init=False, repr=False)
    _l2: nn.Linear = field(init=False, repr=False)
    _l3: nn.Linear = field(init=False, repr=False)
    _model: nn.Module = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mx.random.seed(self.seed)
        np.random.seed(self.seed)
        self._l1 = nn.Linear(self.in_dim, self.hidden_1)
        self._l2 = nn.Linear(self.hidden_1, self.hidden_2)
        self._l3 = nn.Linear(self.hidden_2, self.n_classes)
        self._model = nn.Sequential(
            self._l1, nn.ReLU(), self._l2, nn.ReLU(), self._l3
        )
        mx.eval(self._model.parameters())

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        """Return raw logits as a numpy array shape ``(N, n_classes)``."""
        out = self._model(mx.array(x))
        mx.eval(out)
        return np.asarray(out)

    def latent(self, x: np.ndarray) -> np.ndarray:
        """Return hidden_2 activations shape ``(N, hidden_2)``.

        Used by the beta buffer to capture per-record latents at push
        time for the RECOMBINE Gaussian-MoG sampler.
        """
        h1 = nn.relu(self._l1(mx.array(x)))
        h2 = nn.relu(self._l2(h1))
        mx.eval(h2)
        return np.asarray(h2)

    def eval_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(x) == 0:
            return 0.0
        logits = self.predict_logits(x)
        pred = logits.argmax(axis=1)
        return float((pred == y).mean())

    def train_task(
        self,
        task: dict[str, np.ndarray],
        *,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> None:
        x = mx.array(task["x_train"])
        y = mx.array(task["y_train"])
        n = x.shape[0]
        opt = optim.SGD(learning_rate=lr)
        rng = np.random.default_rng(self.seed)

        def loss_fn(model: nn.Module, xb: mx.array, yb: mx.array) -> mx.array:
            return nn.losses.cross_entropy(model(xb), yb, reduction="mean")

        loss_and_grad = nn.value_and_grad(self._model, loss_fn)
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                if len(idx) == 0:
                    continue
                xb = x[mx.array(idx)]
                yb = y[mx.array(idx)]
                _loss, grads = loss_and_grad(self._model, xb, yb)
                opt.update(self._model, grads)
                mx.eval(self._model.parameters(), opt.state)


class BetaBufferHierFIFO:
    """Bounded curated episodic buffer with optional latents (beta channel).

    FIFO eviction at capacity. Compared to ``BetaBufferFIFO`` (G4-bis),
    each record carries an optional ``latent`` field - used by the
    RECOMBINE Gaussian-MoG sampler. ``latent=None`` is allowed for
    legacy / pre-classifier-warmup pushes.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(
                f"capacity must be positive, got {capacity}"
            )
        self._capacity = capacity
        self._records: deque[BetaRecordHier] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._records)

    def push(
        self, *, x: np.ndarray, y: int, latent: np.ndarray | None
    ) -> None:
        record: BetaRecordHier = {
            "x": x.astype(np.float32).tolist(),
            "y": int(y),
            "latent": (
                latent.astype(np.float32).tolist()
                if latent is not None
                else None
            ),
        }
        self._records.append(record)

    def snapshot(self) -> list[BetaRecordHier]:
        return [
            {
                "x": list(r["x"]),
                "y": int(r["y"]),
                "latent": (
                    list(r["latent"]) if r["latent"] is not None else None
                ),
            }
            for r in self._records
        ]

    def sample(self, n: int, seed: int) -> list[BetaRecordHier]:
        n_avail = len(self._records)
        if n_avail == 0:
            return []
        rng = np.random.default_rng(seed)
        n_take = min(n, n_avail)
        indices = rng.choice(n_avail, size=n_take, replace=False)
        snapshot = list(self._records)
        return [
            {
                "x": list(snapshot[i]["x"]),
                "y": int(snapshot[i]["y"]),
                "latent": (
                    list(snapshot[i]["latent"])
                    if snapshot[i]["latent"] is not None
                    else None
                ),
            }
            for i in sorted(indices.tolist())
        ]

    def latents(self) -> list[list[float]]:
        """Return the list of populated latents (skips None)."""
        return [
            list(r["latent"])
            for r in self._records
            if r["latent"] is not None
        ]
