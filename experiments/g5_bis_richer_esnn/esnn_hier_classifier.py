"""3-layer rate-coded LIF SNN classifier for the G5-bis pilot.

Architecture ::

    x  ->  W_in  (in_dim x hidden_1)
        ->  LIF population_1  ->  mean spike rate r_h1  (hidden_1)
        ->  W_h   (hidden_1 x hidden_2)
        ->  LIF population_2  ->  mean spike rate r_h2  (hidden_2)
        ->  W_out (hidden_2 x n_classes)
        ->  logits

Backward pass uses a straight-through estimator (Wu et al. 2018) :
the LIF non-linearity is identity in the backward pass. Given
softmax-CE ``d_logits = (probs - one_hot) / N`` ::

    d_W_out = r_h2.T @ d_logits          ; d_r_h2 = d_logits @ W_out.T
    d_W_h   = r_h1.T @ d_r_h2            ; d_r_h1 = d_r_h2 @ W_h.T
    d_W_in  = x.T   @ d_r_h1

Public surface mirrors ``G4HierarchicalClassifier`` so the dream
wrapper and pilot driver can transpose ``run_g4_ter`` cell logic
mechanically.

Reference :
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md sec 6.2
    docs/osf-prereg-g5-bis-richer-esnn.md sec 2
    experiments/g5_cross_substrate/esnn_classifier.py (sister)
    experiments/g4_ter_hp_sweep/dream_wrap_hier.py (MLX sister)
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
class EsnnG5BisHierarchicalClassifier:
    """Two-LIF-population rate-coded classifier with STE backward.

    Parameters
    ----------
    in_dim, hidden_1, hidden_2, n_classes
        Layer sizes. All must be > 0 ; ``n_classes`` >= 2.
    seed
        Numpy RNG seed — controls weight init + minibatch order.
    n_steps
        LIF simulation horizon per forward pass (default 20).
    tau, threshold
        LIF dynamics parameters passed to ``simulate_lif_step``.
    """

    in_dim: int
    hidden_1: int
    hidden_2: int
    n_classes: int
    seed: int
    n_steps: int = 20
    tau: float = 10.0
    threshold: float = 1.0
    W_in: NDArray[np.float32] = field(init=False, repr=False)
    W_h: NDArray[np.float32] = field(init=False, repr=False)
    W_out: NDArray[np.float32] = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.in_dim <= 0:
            raise ValueError(f"in_dim must be > 0, got {self.in_dim}")
        if self.hidden_1 <= 0:
            raise ValueError(
                f"hidden_1 must be > 0, got {self.hidden_1}"
            )
        if self.hidden_2 <= 0:
            raise ValueError(
                f"hidden_2 must be > 0, got {self.hidden_2}"
            )
        if self.n_classes < 2:
            raise ValueError(
                f"n_classes must be >= 2, got {self.n_classes}"
            )
        self._rng = np.random.default_rng(self.seed)
        scale_in = float(np.sqrt(2.0 / self.in_dim))
        scale_h = float(np.sqrt(2.0 / self.hidden_1))
        scale_out = float(np.sqrt(2.0 / self.hidden_2))
        self.W_in = (
            self._rng.standard_normal((self.in_dim, self.hidden_1))
            * scale_in
        ).astype(np.float32)
        self.W_h = (
            self._rng.standard_normal((self.hidden_1, self.hidden_2))
            * scale_h
        ).astype(np.float32)
        self.W_out = (
            self._rng.standard_normal((self.hidden_2, self.n_classes))
            * scale_out
        ).astype(np.float32)

    # -------------------- forward --------------------

    def _lif_population_rates(
        self,
        currents: NDArray[np.float32],
        n_neurons: int,
    ) -> NDArray[np.float32]:
        """Drive ``currents`` through a LIF pop, return mean spike rates.

        Per-sample loop matches ``EsnnG5Classifier._hidden_rates`` —
        ``n_steps`` is small in research pilots ; vectorising over the
        batch axis is a deferred perf optimisation.
        """
        n = currents.shape[0]
        rates = np.zeros((n, n_neurons), dtype=np.float32)
        for i in range(n):
            state = LIFState(n_neurons=n_neurons)
            spike_sum = np.zeros(n_neurons, dtype=float)
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

    def _forward_with_caches(
        self, x: NDArray[np.float32]
    ) -> tuple[
        NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]
    ]:
        """Return ``(r_h1, r_h2, logits)`` for STE backward."""
        i_h1 = (x @ self.W_in).astype(np.float32)
        r_h1 = self._lif_population_rates(i_h1, self.hidden_1)
        i_h2 = (r_h1 @ self.W_h).astype(np.float32)
        r_h2 = self._lif_population_rates(i_h2, self.hidden_2)
        logits = (r_h2 @ self.W_out).astype(np.float32)
        return r_h1, r_h2, logits

    def latent(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return hidden_2 spike rates ``(N, hidden_2)``.

        Used by the beta buffer at push time as the support set for
        the RECOMBINE Gaussian-MoG sampler.
        """
        if x.shape[0] == 0:
            return np.zeros((0, self.hidden_2), dtype=np.float32)
        i_h1 = (x.astype(np.float32) @ self.W_in).astype(np.float32)
        r_h1 = self._lif_population_rates(i_h1, self.hidden_1)
        i_h2 = (r_h1 @ self.W_h).astype(np.float32)
        return self._lif_population_rates(i_h2, self.hidden_2)

    def predict_logits(
        self, x: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Return logits of shape ``(N, n_classes)``."""
        if x.shape[0] == 0:
            return np.zeros((0, self.n_classes), dtype=np.float32)
        _r_h1, _r_h2, logits = self._forward_with_caches(
            x.astype(np.float32)
        )
        return logits

    def eval_accuracy(
        self, x: NDArray[np.float32], y: NDArray[np.int64]
    ) -> float:
        """Classification accuracy in [0, 1]."""
        if len(x) == 0:
            return 0.0
        logits = self.predict_logits(x)
        pred = logits.argmax(axis=1)
        return float((pred == y).mean())

    # -------------------- backward / training --------------------

    def _ste_backward(
        self,
        x: NDArray[np.float32],
        y: NDArray[np.int64],
        lr: float,
    ) -> None:
        """One SGD step with the straight-through gradient.

        Loss : softmax cross-entropy on logits. STE : ``d_currents
        = d_rates`` for both LIF populations, so gradient flows
        through the linear projections only.
        """
        if x.shape[0] == 0:
            return
        x_f = x.astype(np.float32)
        r_h1, r_h2, logits = self._forward_with_caches(x_f)
        # Stable softmax
        logits_shift = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits_shift)
        probs = exp / exp.sum(axis=1, keepdims=True)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(y)), y] = 1.0
        d_logits = ((probs - one_hot) / max(len(y), 1)).astype(np.float32)
        # Backward through linears with STE through LIF non-linearities
        d_W_out = r_h2.T @ d_logits
        d_r_h2 = d_logits @ self.W_out.T
        d_W_h = r_h1.T @ d_r_h2  # d_i_h2 = d_r_h2 (STE)
        d_r_h1 = d_r_h2 @ self.W_h.T
        d_W_in = x_f.T @ d_r_h1  # d_i_h1 = d_r_h1 (STE)
        self.W_out = (
            self.W_out - lr * d_W_out.astype(np.float32)
        ).astype(np.float32)
        self.W_h = (
            self.W_h - lr * d_W_h.astype(np.float32)
        ).astype(np.float32)
        self.W_in = (
            self.W_in - lr * d_W_in.astype(np.float32)
        ).astype(np.float32)

    def train_task(
        self,
        task: dict,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> None:
        """Per-epoch seeded permutation + minibatch SGD with STE.

        Determinism : minibatch order is drawn from a numpy RNG
        seeded at ``self.seed`` so two classifiers built with the
        same seed and the same task converge to the same weights
        bit-exactly.
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
                self._ste_backward(xb, yb, lr)
