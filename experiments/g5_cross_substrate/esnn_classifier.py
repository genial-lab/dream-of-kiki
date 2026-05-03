"""Spike-rate 2-class classifier on the E-SNN thalamocortical substrate.

Architecture ::

    x  ->  W_in (in_dim x hidden_dim)  ->  LIF population  ->
    mean spike rate per neuron  ->  W_out (hidden_dim x n_classes)  -> logits

Training uses a straight-through estimator on the mean-rate (rate
is a continuous proxy for spike counts) : gradients flow through
the linear projections only, the LIF non-linearity is treated as
identity in the backward pass. This is the standard rate-coded SNN
training trick (Wu et al. 2018, "Spatio-temporal backpropagation
for training high-performance spiking neural networks") and is
sufficient for a 2-class continual-learning pilot.

The classifier deliberately mirrors the G4-bis ``G4Classifier``
public surface (``train_task`` / ``eval_accuracy`` /
``predict_logits``) so the pilot driver in ``run_g5.py`` can be a
1-to-1 transposition of ``run_g4.py``. The dream-episode wrapper
lives in ``esnn_dream_wrap.py`` and is composed in Task 3.

Reference :
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
    kiki_oniric/substrates/esnn_thalamocortical.py
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from kiki_oniric.substrates.esnn_thalamocortical import (
    LIFState,
    simulate_lif_step,
)


@dataclass
class EsnnG5Classifier:
    """Tiny rate-coded SNN classifier for Split-FMNIST 2-class tasks.

    Parameters
    ----------
    in_dim, hidden_dim, n_classes
        Layer sizes. All must be > 0 ; ``n_classes`` must be >= 2.
    seed
        Numpy RNG seed — controls weight init + minibatch order.
    n_steps
        LIF simulation horizon per forward pass. Defaults to 20
        (matches `_simulate_population` default in
        `esnn_thalamocortical.py`). Set to 0 for ablation tests.
    tau, threshold
        LIF dynamics parameters (passed through to
        `simulate_lif_step`). Defaults match the substrate's
        canonical values.
    """

    in_dim: int
    hidden_dim: int
    n_classes: int
    seed: int
    n_steps: int = 20
    tau: float = 10.0
    threshold: float = 1.0
    W_in: NDArray[np.float32] = field(init=False, repr=False)
    W_out: NDArray[np.float32] = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.in_dim <= 0:
            raise ValueError(f"in_dim must be > 0, got {self.in_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0, got {self.hidden_dim}"
            )
        if self.n_classes < 2:
            raise ValueError(
                f"n_classes must be >= 2, got {self.n_classes}"
            )
        self._rng = np.random.default_rng(self.seed)
        # Xavier-style init — small random weights
        scale_in = float(np.sqrt(2.0 / self.in_dim))
        scale_out = float(np.sqrt(2.0 / self.hidden_dim))
        self.W_in = (
            self._rng.standard_normal((self.in_dim, self.hidden_dim))
            * scale_in
        ).astype(np.float32)
        self.W_out = (
            self._rng.standard_normal((self.hidden_dim, self.n_classes))
            * scale_out
        ).astype(np.float32)

    # -------------------- forward --------------------

    def _hidden_rates(
        self, x: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Drive the LIF population from `x @ W_in`, return mean rates.

        Per-sample loop (n_steps small, batch sizes modest — fine for
        a research pilot ; vectorising over batch is a pure perf
        optimisation deferred to a follow-up).
        """
        currents = (x @ self.W_in).astype(np.float32)
        n = currents.shape[0]
        rates = np.zeros((n, self.hidden_dim), dtype=np.float32)
        for i in range(n):
            state = LIFState(n_neurons=self.hidden_dim)
            spike_sum = np.zeros(self.hidden_dim, dtype=float)
            for _ in range(self.n_steps):
                state = simulate_lif_step(
                    state,
                    currents[i],
                    dt=1.0,
                    tau=self.tau,
                    threshold=self.threshold,
                )
                spike_sum += state.spikes
            denom = max(self.n_steps, 1)
            rates[i] = (spike_sum / denom).astype(np.float32)
        return rates

    def predict_logits(
        self, x: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Return logits of shape (N, n_classes)."""
        if x.shape[0] == 0:
            return np.zeros((0, self.n_classes), dtype=np.float32)
        rates = self._hidden_rates(x.astype(np.float32))
        return (rates @ self.W_out).astype(np.float32)

    def eval_accuracy(
        self, x: NDArray[np.float32], y: NDArray[np.int64]
    ) -> float:
        """Classification accuracy in [0, 1]."""
        if len(x) == 0:
            return 0.0
        logits = self.predict_logits(x)
        pred = logits.argmax(axis=1)
        return float((pred == y).mean())

    # -------------------- training --------------------

    def train_task(
        self,
        task: dict,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> None:
        """SGD training with straight-through gradient through LIF.

        Loss : softmax cross-entropy on the linear logits. Backward :
        d_logits = softmax - one_hot(y) ; d_W_out = rates.T @ d_logits ;
        d_rates = d_logits @ W_out.T ; **straight-through** :
        d_currents = d_rates ; d_W_in = x.T @ d_currents.

        Determinism : minibatch order is drawn from a numpy RNG
        seeded at ``self.seed`` so two classifiers built with the same
        seed and same task converge to the same weights bit-exactly.
        """
        x_train = task["x_train"].astype(np.float32)
        y_train = task["y_train"].astype(np.int64)
        n = x_train.shape[0]
        if n == 0:
            return
        rng = np.random.default_rng(self.seed)
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                if len(idx) == 0:
                    continue
                xb = x_train[idx]
                yb = y_train[idx]
                rates = self._hidden_rates(xb)  # (B, hidden)
                logits = rates @ self.W_out  # (B, n_classes)
                # Stable softmax
                logits_shift = logits - logits.max(axis=1, keepdims=True)
                exp = np.exp(logits_shift)
                probs = exp / exp.sum(axis=1, keepdims=True)
                one_hot = np.zeros_like(probs)
                one_hot[np.arange(len(yb)), yb] = 1.0
                d_logits = (probs - one_hot) / max(len(yb), 1)
                d_W_out = rates.T @ d_logits  # (hidden, n_classes)
                d_rates = d_logits @ self.W_out.T  # (B, hidden)
                # Straight-through : d_currents = d_rates
                d_W_in = xb.T @ d_rates  # (in, hidden)
                self.W_out = (
                    self.W_out - lr * d_W_out.astype(np.float32)
                ).astype(np.float32)
                self.W_in = (
                    self.W_in - lr * d_W_in.astype(np.float32)
                ).astype(np.float32)
