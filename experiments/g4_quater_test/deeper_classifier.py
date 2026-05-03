"""5-layer deeper hierarchical MLX MLP classifier for G4-quater Step 1.

Architecture :
``Linear(in_dim, h1) -> ReLU -> Linear(h1, h2) -> ReLU ->
Linear(h2, h3) -> ReLU -> Linear(h3, h4) -> ReLU ->
Linear(h4, n_classes)``.

Default ``hidden=(64, 32, 16, 8)`` per pre-reg §2 H4-A. RESTRUCTURE
perturbs the *middle* layer's weight (``_l3 : Linear(h2, h3)``),
preserving the input projection (``_l1``) and output classifier
(``_l5``). RECOMBINE samples synthetic latents from the activations
*after the third ReLU* (dimension ``h3``) as exposed by ``latent()``.

DR-0 accountability is provided by the dream-episode wrapper — this
module exposes only the model + train/eval primitives.

Reference :
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md sec 3.1
    docs/osf-prereg-g4-quater-pilot.md sec 2 (H4-A)
    docs/superpowers/plans/2026-05-03-g4-quater-restructure-recombine-test.md
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


@dataclass
class G4HierarchicalDeeperClassifier:
    """Deeper (5-layer) MLP classifier for Split-FMNIST 2-class tasks.

    Layers : five ``nn.Linear`` separated by four ReLU activations.
    Deterministic under a fixed ``seed`` via ``mx.random.seed`` at
    construction.
    """

    in_dim: int
    hidden: tuple[int, int, int, int]
    n_classes: int
    seed: int
    _l1: nn.Linear = field(init=False, repr=False)
    _l2: nn.Linear = field(init=False, repr=False)
    _l3: nn.Linear = field(init=False, repr=False)
    _l4: nn.Linear = field(init=False, repr=False)
    _l5: nn.Linear = field(init=False, repr=False)
    _model: nn.Module = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.hidden) != 4:
            raise ValueError(
                "G4HierarchicalDeeperClassifier expects 4 hidden "
                f"sizes (got {len(self.hidden)})"
            )
        h1, h2, h3, h4 = self.hidden
        mx.random.seed(self.seed)
        np.random.seed(self.seed)
        self._l1 = nn.Linear(self.in_dim, h1)
        self._l2 = nn.Linear(h1, h2)
        self._l3 = nn.Linear(h2, h3)
        self._l4 = nn.Linear(h3, h4)
        self._l5 = nn.Linear(h4, self.n_classes)
        self._model = nn.Sequential(
            self._l1, nn.ReLU(),
            self._l2, nn.ReLU(),
            self._l3, nn.ReLU(),
            self._l4, nn.ReLU(),
            self._l5,
        )
        mx.eval(self._model.parameters())

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        """Return raw logits as a numpy array shape ``(N, n_classes)``."""
        out = self._model(mx.array(x))
        mx.eval(out)
        return np.asarray(out)

    def latent(self, x: np.ndarray) -> np.ndarray:
        """Return hidden_3 activations shape ``(N, h3)``.

        These are the activations *after the third ReLU* — the
        RECOMBINE Gaussian-MoG sampling site for the deeper head.
        """
        h1 = nn.relu(self._l1(mx.array(x)))
        h2 = nn.relu(self._l2(h1))
        h3 = nn.relu(self._l3(h2))
        mx.eval(h3)
        return np.asarray(h3)

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

    def restructure_step(self, *, factor: float, seed: int) -> None:
        """Add ``factor * sigma * N(0, 1)`` to the middle (``_l3``)
        layer's weight only.

        ``sigma`` is the per-tensor std of ``self._l3.weight`` at
        call time. ``factor=0`` is a no-op ; ``factor < 0`` raises.
        """
        if factor < 0.0:
            raise ValueError(
                f"factor must be non-negative, got {factor}"
            )
        if factor == 0.0:
            return
        w = np.asarray(self._l3.weight)
        sigma = float(w.std()) if w.size > 0 else 0.0
        if sigma == 0.0:
            return
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(size=w.shape).astype(np.float32)
        new_w = w + factor * sigma * noise
        self._l3.weight = mx.array(new_w)
        mx.eval(self._l3.weight)

    def downscale_step(self, *, factor: float) -> None:
        """Multiply every weight + bias in ``self._model`` by ``factor``.

        Bounds : ``factor`` must lie in ``(0, 1]``.
        """
        if not (0.0 < factor <= 1.0):
            raise ValueError(
                f"shrink_factor must be in (0, 1], got {factor}"
            )
        for layer in (self._l1, self._l2, self._l3, self._l4, self._l5):
            layer.weight = layer.weight * factor
            if getattr(layer, "bias", None) is not None:
                layer.bias = layer.bias * factor
        mx.eval(self._model.parameters())

    def replay_optimizer_step(
        self,
        records: list[dict[str, list[float] | int]],
        *,
        lr: float,
        n_steps: int,
    ) -> None:
        if not records:
            return
        x = mx.array([r["x"] for r in records])
        y = mx.array([r["y"] for r in records])
        opt = optim.SGD(learning_rate=lr)

        def loss_fn(model: nn.Module, xb: mx.array, yb: mx.array) -> mx.array:
            return nn.losses.cross_entropy(model(xb), yb, reduction="mean")

        loss_and_grad = nn.value_and_grad(self._model, loss_fn)
        for _ in range(n_steps):
            _loss, grads = loss_and_grad(self._model, x, y)
            opt.update(self._model, grads)
            mx.eval(self._model.parameters(), opt.state)

    def recombine_step(
        self,
        *,
        latents: list[tuple[list[float], int]],
        n_synthetic: int,
        lr: float,
        seed: int,
    ) -> None:
        """Sample ``n_synthetic`` synthetic latents from a per-class
        Gaussian-MoG and run one CE-loss SGD pass through ``_l5`` only.

        Empty / single-class ``latents`` -> no-op (S1-trivial).
        """
        if not latents:
            return
        classes = sorted({lbl for _, lbl in latents})
        if len(classes) < 2:
            return

        rng = np.random.default_rng(seed)
        components: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for c in classes:
            arr = np.asarray(
                [lat for lat, lbl in latents if lbl == c],
                dtype=np.float32,
            )
            mean = arr.mean(axis=0)
            std = arr.std(axis=0) + 1e-6
            components[c] = (mean, std)

        per_class = max(1, n_synthetic // len(classes))
        synth_x: list[np.ndarray] = []
        synth_y: list[int] = []
        for c in classes:
            mean, std = components[c]
            for _ in range(per_class):
                synth_x.append(
                    mean + std * rng.standard_normal(size=mean.shape).astype(
                        np.float32
                    )
                )
                synth_y.append(c)

        # Synthetic latents have h3 dimension; map through _l4 -> ReLU
        # -> _l5 with grad on _l5 only.
        x_lat = mx.array(np.stack(synth_x).astype(np.float32))
        y = mx.array(np.asarray(synth_y, dtype=np.int32))
        h4 = nn.relu(self._l4(x_lat))
        mx.eval(h4)

        opt = optim.SGD(learning_rate=lr)

        def loss_fn(layer: nn.Linear, xb: mx.array, yb: mx.array) -> mx.array:
            return nn.losses.cross_entropy(layer(xb), yb, reduction="mean")

        loss_and_grad = nn.value_and_grad(self._l5, loss_fn)
        _loss, grads = loss_and_grad(self._l5, h4, y)
        opt.update(self._l5, grads)
        mx.eval(self._l5.parameters(), opt.state)
