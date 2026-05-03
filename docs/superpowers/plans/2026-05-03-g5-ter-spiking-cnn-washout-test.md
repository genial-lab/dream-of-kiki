# G5-ter pilot — port G4-quinto small CNN onto E-SNN substrate (LIF vs architecture washout test) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Disambiguate the G5-bis H7-B verdict (`g_h7a = +0.1043`, fail-to-reject) by porting the G4-quinto Step 2 small-CNN architecture onto the E-SNN substrate (Conv2d-LIF + Conv2d-LIF + FC-LIF + Linear, STE backward) and testing whether convolutional structure recovers a positive dream-arm effect that the richer-MLP-LIF stack failed to express.

**Architecture:** A new `experiments/g5_ter_spiking_cnn/` module reuses the G4-quinto Step 2 CIFAR-10 loader and the G4-ter HP grid C5 anchor but introduces a 4-layer **spiking-CNN** classifier (`EsnnG5TerSpikingCNN` : Conv2d(3→16) → LIF rates → Conv2d(16→32) → LIF rates → avg-pool 4×4 → flatten + Linear(2048→64) → LIF rates → Linear(64→2), STE backward over all three LIF stages). The dream-episode wrapper transposes the G5-bis four-op coupling onto this substrate (REPLAY=SGD-with-STE, DOWNSCALE=multiply all weight tensors, RESTRUCTURE=`W_c2` noise, RECOMBINE=Gaussian-MoG → final `W_out` linear). A separate aggregator runs four Welch two-sided tests at α/4 = 0.0125 between **G4-quinto Step 2 MLX CNN** retention and **G5-ter spiking-CNN** retention and emits an H8-A/B/C verdict to `docs/milestones/g5-ter-aggregate-2026-05-03.{json,md}`. EC stays PARTIAL, no FC bump.

**Tech Stack:** Python 3.12, numpy (LIF dynamics + pure-numpy Conv2d forward/backward + STE), existing `kiki_oniric.substrates.esnn_thalamocortical` (`simulate_lif_step`, `LIFState`), existing `experiments.g4_quinto_test.cifar10_dataset.{SplitCIFAR10Task, load_split_cifar10_5tasks_auto}`, existing `experiments.g4_ter_hp_sweep.hp_grid.representative_combo`, existing `experiments.g5_cross_substrate.esnn_dream_wrap.{PROFILE_FACTORIES, _rebind_to_esnn, ProfileT}`, existing `harness.storage.run_registry.RunRegistry`, existing `kiki_oniric.eval.statistics.{compute_hedges_g, welch_one_sided}`, pytest + Hypothesis (`derandomize=True`), conventional commits, no `Co-Authored-By` trailer.

---

## Hard prerequisites (block until satisfied)

1. `docs/milestones/g4-quinto-step2-2026-05-03.json` exists ; `verdict.retention_by_arm` is a `{baseline,P_min,P_equ,P_max}` dict (each list of N≥10 floats).
2. `docs/milestones/g5-bis-aggregate-2026-05-03.json` carries `h7_classification == "H7-B"` and `g_h7a_esnn` ≈ +0.1043.
3. `experiments/g4_quinto_test/small_cnn.py` exposes `G4SmallCNN` with `replay_optimizer_step`, `downscale_step`, `restructure_step`, `recombine_step`, `latent`.
4. `experiments/g4_quinto_test/cifar10_dataset.py` exposes `load_split_cifar10_5tasks_auto(data_dir)` (HF parquet fallback).
5. `experiments/g5_bis_richer_esnn/esnn_hier_classifier.py` exposes `EsnnG5BisHierarchicalClassifier` (LIF + STE template).
6. `experiments/g5_cross_substrate/esnn_dream_wrap.py` exposes `PROFILE_FACTORIES`, `_rebind_to_esnn`, `ProfileT`.

If any is missing : **stop and surface the blocker** before Task 1.

## Variant decision (locked) — Variant A

**Full spiking-CNN with native pure-numpy Conv2d forward/backward and STE on every LIF stage**, not "MLX small-CNN wrapped in E-SNN dispatch". Same rationale as G5-bis : H8-A/B/C is only meaningful if the classifier carries the substrate's native rate-coding state representation at every non-linear stage.

## Compute / power note

Per-cell cost ≈ 3-5× the G5-bis spiking-MLP cost because each Conv2d adds an `im2col`-style numpy matmul of shape `(N, k*k*C_in, H*W)` per layer per LIF time step. G5-bis measured ~21 min for 40 cells (Option B) on M1 Max → G5-ter ~30-60 min for 40 cells.

| Option | N seeds | Cells | Wall (M1 Max, est.) | Min g (80 % power, two-sided α=0.0125) |
|--------|---------|-------|---------------------|----------------------------------------|
| A | 30 | 120 | ~2-3 h | g ≈ 0.74 |
| B | 10 | 40 | ~30-60 min | g ≈ 1.27 |
| C | 1 (smoke) | 4 | ~3-5 min | exploratory only |

**Recommended : Option B (N=10) — locked in Task 0.5.** Sufficient to detect a G4-ter-scale effect (g ≈ +2.77). Option C is the smoke variant (`--smoke`, no milestone dump).

## Decision rule (pre-registered, locked)

Cross-substrate aggregator runs Welch two-sided per arm at α/4 = 0.0125 between G4-quinto Step 2 MLX CNN retention and G5-ter spiking-CNN retention.

Let `g_h8 = g(P_equ_esnn, baseline_esnn)` (own-substrate) and `g_p_equ_cross = g_mlx_minus_esnn(P_equ)`.

| Verdict | Rule |
|---------|------|
| **H8-A** (LIF non-linearity is the load-bearing washout) | `g_h8 < 0.5` AND own-Welch fails to reject AND `g_p_equ_cross >= 2.0`. |
| **H8-B** (architecture-dependent — CNN closes the gap) | `g_h8 >= 0.5` AND own-Welch rejects AND `g_p_equ_cross < 1.0`. |
| **H8-C** (partial — both contribute) | anything else. |

Thresholds 0.5 / 1.0 / 2.0 are locked at pre-reg time and may not be moved post hoc.

## File structure

| File | Role |
|------|------|
| `experiments/g5_ter_spiking_cnn/__init__.py` (create) | Package marker |
| `experiments/g5_ter_spiking_cnn/spiking_cnn.py` (create) | `EsnnG5TerSpikingCNN` 4-layer + STE + pure-numpy Conv2d |
| `experiments/g5_ter_spiking_cnn/dream_wrap_cnn.py` (create) | `dream_episode_cnn_esnn` + `EsnnCNNBetaBuffer` |
| `experiments/g5_ter_spiking_cnn/run_g5_ter.py` (create) | Pilot driver |
| `experiments/g5_ter_spiking_cnn/aggregator.py` (create) | H8-A/B/C cross-substrate verdict |
| `tests/unit/experiments/test_g5_ter_spiking_cnn.py` (create) | Classifier + Conv2d unit tests |
| `tests/unit/experiments/test_g5_ter_dream_wrap_cnn.py` (create) | Coupling unit tests |
| `tests/unit/experiments/test_g5_ter_run_smoke.py` (create) | 1-seed smoke (synthetic CIFAR shape) |
| `tests/unit/experiments/test_g5_ter_aggregator.py` (create) | Aggregator math tests |
| `docs/osf-prereg-g5-ter-spiking-cnn.md` (create) | OSF pre-reg, append-only |
| `docs/milestones/g5-ter-spiking-cnn-2026-05-03.{json,md}` (driver) | Per-cell + H8-A own-substrate verdict |
| `docs/milestones/g5-ter-aggregate-2026-05-03.{json,md}` (aggregator) | H8-A/B/C cross-substrate verdict |
| `docs/papers/paper2/results.md` (modify, +§7.1.10) | EN |
| `docs/papers/paper2-fr/results.md` (modify, +§7.1.10) | FR mirror |
| `docs/proofs/dr3-substrate-evidence.md` (modify, append) | DR-3 evidence row per H8 outcome |
| `CHANGELOG.md`, `STATUS.md` (modify) | `[Unreleased]` row + gates table |

`experiments/` is excluded from coverage scope per `pyproject.toml`. Unit tests live under `tests/unit/experiments/` and contribute to the project-wide ≥90 % coverage gate.

---

## Task 0: Investigate (read-only)

- [ ] **Step 1: Confirm G4-quinto Step 2 milestone schema (BLOCKER)**

`python -c "import json; p=json.load(open('docs/milestones/g4-quinto-step2-2026-05-03.json')); print(sorted(p['verdict'].keys())); print(sorted(p['verdict']['retention_by_arm'].keys()))"`
Expected: `verdict` carries `h5b_cnn_cifar`, `retention_by_arm` ; arms `[P_equ, P_max, P_min, baseline]`.

- [ ] **Step 2: Confirm G5-bis H7-B prior**

`python -c "import json; p=json.load(open('docs/milestones/g5-bis-aggregate-2026-05-03.json')); print(p['h7_classification'], p['g_h7a_esnn'])"`
Expected: `H7-B 0.1043...`.

- [ ] **Step 3: Confirm G4-quinto CNN op signatures + G5-bis STE template**

```bash
grep -nE "def (replay_optimizer_step|downscale_step|restructure_step|recombine_step|latent)" experiments/g4_quinto_test/small_cnn.py
grep -nE "_ste_backward|_lif_population_rates|_forward_with_caches" experiments/g5_bis_richer_esnn/esnn_hier_classifier.py
```
Expected: 5 method definitions on `G4SmallCNN` ; 3 method names on `EsnnG5BisHierarchicalClassifier`.

- [ ] **Step 4: Confirm `representative_combo()` returns C5**

`python -c "from experiments.g4_ter_hp_sweep.hp_grid import representative_combo; c=representative_combo(); print(c.combo_id, c.downscale_factor, c.replay_batch, c.replay_n_steps, c.replay_lr)"`
Expected: `C5 0.95 32 5 0.01`.

- [ ] **Step 5: No commit** — investigation only.

---

## Task 0.5: Decision — compute budget

- [ ] **Step 1: Lock Option B (N=10, 40 cells, ~30-60 min M1 Max).** Recorded in pre-reg in Task 1. Option C is smoke only, does NOT write the dated milestone.

---

## Task 1: OSF pre-reg — `docs/osf-prereg-g5-ter-spiking-cnn.md`

**Files:** Create `docs/osf-prereg-g5-ter-spiking-cnn.md`.

- [ ] **Step 1: Write the pre-reg**

Mirror `docs/osf-prereg-g5-bis-richer-esnn.md` sections 0–10. Required content :

- §0 **Background** : two priors — (i) G5-bis H7-B verdict (`g_h7a = +0.1043`, MLX-only artefact for the 3-layer LIF MLP) ; (ii) G4-quinto Step 2 H5-B (CNN MLX retention level on Split-CIFAR-10 as cross-substrate reference). Cite both milestone JSONs by full path.
- §1 **Hypotheses** : H8-A / H8-B / H8-C with the locked decision rule from this plan, verbatim. Threshold table (`g_h8` 0.5 ; `g_mlx_minus_esnn` 1.0 / 2.0) — LOCKED.
- §2 **Coupling map** : REPLAY=SGD-with-STE ; DOWNSCALE=`*factor` on every weight + bias ; RESTRUCTURE=`W_c2` Gaussian noise only ; RECOMBINE=Gaussian-MoG → `W_out` (skips LIF). Bound checks identical to G5-bis : `factor ∈ (0, 1]`.
- §3 **Substrate** : `EsnnG5TerSpikingCNN` 4-layer (Conv2d-LIF, Conv2d-LIF, Linear-LIF, Linear). LIF defaults : `tau=10.0`, `threshold=1.0`, `n_steps=20`. Pure-numpy Conv2d forward (im2col + matmul) and STE-backward (no `mlx`, no `torch`).
- §4 **Dataset** : Split-CIFAR-10 5-task (class pairs (0,1)/(2,3)/(4,5)/(6,7)/(8,9)), reuse `load_split_cifar10_5tasks_auto`. NHWC, float32 in [0, 1].
- §5 **Sweep** : 4 arms × N=10 seeds = 40 cells. HP combo C5. `epochs=2, batch_size=64, lr=0.05` (matches G5-bis).
- §6 **Statistics** : Welch one-sided per arm + cross-substrate two-sided, Bonferroni α/4 = 0.0125. Hedges' g via `compute_hedges_g`.
- §7 **R1 reproducibility** : `c_version = "C-v0.12.0+PARTIAL"`, registered via `RunRegistry`.
- §8 **Outputs** : the two milestone path pairs.
- §9 **Deviation envelope** : (i) HF parquet fallback for CIFAR-10 (cite G4-quinto §9.1) ; (ii) abort if `n_p_equ < 2` after exclusions.
- §10 **Cross-references** : G5-bis pre-reg, G4-quinto pre-reg, this plan.

- [ ] **Step 2: Verify presence**

`wc -l docs/osf-prereg-g5-ter-spiking-cnn.md` — expected ≥ 80 lines.

- [ ] **Step 3: Commit**

```bash
git add docs/osf-prereg-g5-ter-spiking-cnn.md
git commit -m "docs(g5ter): OSF pre-reg spiking-CNN H8" -m "Pre-register H8-A/B/C decision rule for the
G5-ter spiking-CNN cross-substrate test (locked
thresholds 0.5/1.0/2.0). Cites G5-bis H7-B and
G4-quinto Step 2 CNN MLX retention as priors."
```

---

## Task 2: `EsnnG5TerSpikingCNN` — 4-layer spiking CNN with STE backward

**Files:** Create `experiments/g5_ter_spiking_cnn/__init__.py` (one-line module docstring) and `experiments/g5_ter_spiking_cnn/spiking_cnn.py`. Test: `tests/unit/experiments/test_g5_ter_spiking_cnn.py`.

**Architecture (NHWC, pure numpy)**:

    (N, 32, 32, 3)
      -> Conv2d(3->16, 3x3, pad=1) -> LIF rates  (N, 32, 32, 16)
      -> Conv2d(16->32, 3x3, pad=1) -> LIF rates (N, 32, 32, 32)
      -> avg_pool 4x4 (deterministic)            (N, 8, 8, 32)
      -> flatten + Linear(2048, 64) -> LIF       (N, 64)
      -> Linear(64, 2) (no LIF)                  (N, 2) logits

LIF defaults : `tau=10.0`, `threshold=1.0`, `n_steps=20` (same as G5-bis). Pooling is **average-pool 4×4** (parameter-free, fully-differentiable through STE — replaces the MLX `MaxPool2d` which has no clean numpy backward).

Three LIF stages = three STE applications. Loss = softmax CE on logits.

- [ ] **Step 1: Write the failing tests**

```python
"""Unit tests for EsnnG5TerSpikingCNN."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.g5_ter_spiking_cnn.spiking_cnn import (
    EsnnG5TerSpikingCNN, avg_pool_4x4, avg_pool_4x4_backward,
    conv2d_backward, conv2d_forward,
)


def _make_clf(seed: int = 0) -> EsnnG5TerSpikingCNN:
    return EsnnG5TerSpikingCNN(n_classes=2, seed=seed, n_steps=4)


def test_init_shapes() -> None:
    clf = _make_clf()
    assert clf.W_c1.shape == (3, 3, 3, 16)
    assert clf.W_c2.shape == (3, 3, 16, 32)
    assert clf.W_fc1.shape == (2048, 64)
    assert clf.W_out.shape == (64, 2)
    for b, n in (
        (clf.b_c1, 16), (clf.b_c2, 32),
        (clf.b_fc1, 64), (clf.b_out, 2),
    ):
        assert b.shape == (n,) and b.dtype == np.float32


def test_predict_logits_shape_and_empty() -> None:
    clf = _make_clf()
    x = np.random.default_rng(0).standard_normal(
        (5, 32, 32, 3)
    ).astype(np.float32)
    assert clf.predict_logits(x).shape == (5, 2)
    assert clf.predict_logits(
        np.zeros((0, 32, 32, 3), np.float32)
    ).shape == (0, 2)


def test_latent_shape() -> None:
    clf = _make_clf()
    x = np.random.default_rng(1).standard_normal(
        (3, 32, 32, 3)
    ).astype(np.float32)
    assert clf.latent(x).shape == (3, 64)


def test_eval_accuracy_determinism() -> None:
    clf = _make_clf()
    x = np.zeros((4, 32, 32, 3), dtype=np.float32)
    y = np.zeros((4,), dtype=np.int64)
    assert clf.eval_accuracy(x, y) == clf.eval_accuracy(x, y)


def test_train_task_changes_weights() -> None:
    clf = _make_clf()
    rng = np.random.default_rng(7)
    x = rng.standard_normal((16, 32, 32, 3)).astype(np.float32)
    y = rng.integers(0, 2, size=16, dtype=np.int64)
    w0 = clf.W_out.copy()
    clf.train_task(
        {"x_train": x, "y_train": y},
        epochs=1, batch_size=8, lr=0.01,
    )
    assert not np.allclose(w0, clf.W_out)


def test_seed_determinism() -> None:
    a, b = _make_clf(seed=42), _make_clf(seed=42)
    np.testing.assert_array_equal(a.W_c1, b.W_c1)
    np.testing.assert_array_equal(a.W_out, b.W_out)


def test_avg_pool_4x4_shape_and_backward() -> None:
    x = np.ones((2, 32, 32, 4), dtype=np.float32)
    p = avg_pool_4x4(x)
    assert p.shape == (2, 8, 8, 4) and np.allclose(p, 1.0)
    grad = avg_pool_4x4_backward(np.ones_like(p), x.shape)
    assert grad.shape == x.shape and np.allclose(grad, 1.0 / 16.0)


def test_conv2d_forward_backward_shapes() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 8, 8, 3)).astype(np.float32)
    W = rng.standard_normal((3, 3, 3, 4)).astype(np.float32)
    b = np.zeros(4, dtype=np.float32)
    y = conv2d_forward(x, W, b, padding=1)
    assert y.shape == (1, 8, 8, 4)
    dx, dW, db = conv2d_backward(np.ones_like(y), x, W, padding=1)
    assert dx.shape == x.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape


def test_invalid_n_classes() -> None:
    with pytest.raises(ValueError):
        EsnnG5TerSpikingCNN(n_classes=1, seed=0)
```

- [ ] **Step 2: Run — expect ImportError**

`uv run pytest tests/unit/experiments/test_g5_ter_spiking_cnn.py -v --no-cov`

- [ ] **Step 3: Implement `experiments/g5_ter_spiking_cnn/spiking_cnn.py`**

Module exports the three free helpers (`conv2d_forward`, `conv2d_backward`, `avg_pool_4x4`, `avg_pool_4x4_backward`) plus the dataclass. Conv2d uses an im2col-style accumulation loop (NHWC, square kernels only). The class mirrors `EsnnG5BisHierarchicalClassifier` (G5-bis sister) at three levels :

1. **LIF rate-coding** : reuse the per-sample inner loop pattern from `EsnnG5BisHierarchicalClassifier._lif_population_rates` (drives `simulate_lif_step` × `n_steps`, accumulates `state.spikes`, divides by `n_steps`). Generalise to a flat `(N, n_neurons_per_sample)` interface — for conv layers the caller flattens spatial dims (`r_c1 = lif_rates(i_c1.reshape(N, -1)).reshape(N, 32, 32, 16)`).
2. **Forward with caches** : returns a `dict` carrying `{x, r_c1, r_c2, p2, flat, r_h1, logits}` so STE backward can reuse them.
3. **STE backward** : `d_currents = d_rates` at every LIF (Wu 2018). Wire the gradients :

```
d_logits = (probs - one_hot) / N           # softmax CE
d_W_out  = r_h1.T @ d_logits ;  d_b_out = d_logits.sum(axis=0)
d_r_h1   = d_logits @ W_out.T              # (N, 64)
d_i_h1   = d_r_h1                          # STE
d_W_fc1  = flat.T @ d_i_h1 ;   d_b_fc1 = d_i_h1.sum(axis=0)
d_flat   = d_i_h1 @ W_fc1.T                # (N, 2048)
d_p2     = d_flat.reshape(p2.shape)
d_r_c2   = avg_pool_4x4_backward(d_p2, r_c2.shape)
d_i_c2   = d_r_c2                          # STE
d_r_c1, d_W_c2, d_b_c2 = conv2d_backward(d_i_c2, r_c1, W_c2, padding=1)
d_i_c1   = d_r_c1                          # STE
_dx, d_W_c1, d_b_c1 = conv2d_backward(d_i_c1, x, W_c1, padding=1)
```

Then `param -= lr * d_param` for all 8 tensors (`W_c1, b_c1, W_c2, b_c2, W_fc1, b_fc1, W_out, b_out`), each cast to `np.float32`.

`train_task(task, *, epochs, batch_size, lr)` mirrors `EsnnG5BisHierarchicalClassifier.train_task` exactly (seeded permutation per epoch, minibatch SGD via `_ste_backward`).

Init : Kaiming `sqrt(2 / fan_in)` Gaussian for each weight tensor (fan-in = `k*k*c_in` for conv, `in_dim` for linear) ; biases zero. Validate `n_classes >= 2`.

`predict_logits`, `latent`, `eval_accuracy` follow the G5-bis sister API one-for-one.

Conv2d backward correctness is exercised by the shape-roundtrip test ; if a numeric finite-difference test is required later (gradient check), add a Hypothesis property test under `tests/unit/experiments/test_g5_ter_spiking_cnn.py` (`derandomize=True`).

- [ ] **Step 4: Run — expect 9/9 PASS**

`uv run pytest tests/unit/experiments/test_g5_ter_spiking_cnn.py -v --no-cov`

- [ ] **Step 5: Commit**

```bash
git add experiments/g5_ter_spiking_cnn/__init__.py \
        experiments/g5_ter_spiking_cnn/spiking_cnn.py \
        tests/unit/experiments/test_g5_ter_spiking_cnn.py
git commit -m "feat(g5ter): spiking-CNN classifier" -m "EsnnG5TerSpikingCNN — 4-layer pure-numpy CNN
with three LIF stages and STE backward. Conv2d
forward/backward via im2col-style numpy matmul,
avg-pool 4x4 deterministic. Mirrors the public
surface of EsnnG5BisHierarchicalClassifier so the
dream-episode wrapper can transpose mechanically."
```

---

## Task 3: Dream-episode wrapper for spiking-CNN

**Files:** Create `experiments/g5_ter_spiking_cnn/dream_wrap_cnn.py`. Test: `tests/unit/experiments/test_g5_ter_dream_wrap_cnn.py`.

Mirrors `experiments/g5_bis_richer_esnn/esnn_dream_wrap_hier.py` exactly with these substitutions :

- Buffer stores `x` as nested NHWC list `(32, 32, 3)` instead of flat 784-d ; reconstruction via `np.asarray(...).reshape(-1, 32, 32, 3)`.
- DOWNSCALE multiplies all 8 tensors `{W_c1, b_c1, W_c2, b_c2, W_fc1, b_fc1, W_out, b_out}` by `np.float32(factor)`.
- RESTRUCTURE perturbs `W_c2` only (analogue of `W_h` in G5-bis MLP / `_conv2.weight` in G4-quinto MLX CNN).
- RECOMBINE samples synthetic latents of dim 64 (post-fc1 LIF rates), runs **one** CE-loss SGD step on `(W_out, b_out)` only.
- `build_esnn_cnn_profile(name, seed)` reuses `experiments.g5_cross_substrate.esnn_dream_wrap.{PROFILE_FACTORIES, _rebind_to_esnn}` — same body as `build_esnn_richer_profile`.
- `dream_episode_cnn_esnn(...)` orchestration is a literal copy of `dream_episode_hier_esnn` (same op-set selection : P_min → REPLAY+DOWNSCALE ; P_equ/P_max → all four ; same DR-0 spectator `input_slice` synthesis ; `profile.runtime.execute(episode)` BEFORE substrate-side mutations).

- [ ] **Step 1: Write the failing tests**

```python
"""Unit tests for the G5-ter dream-episode wrapper."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.g5_ter_spiking_cnn.dream_wrap_cnn import (
    EsnnCNNBetaBuffer, _downscale_step, _recombine_step,
    _replay_step, _restructure_step, build_esnn_cnn_profile,
    dream_episode_cnn_esnn,
)
from experiments.g5_ter_spiking_cnn.spiking_cnn import (
    EsnnG5TerSpikingCNN,
)


def _make_clf() -> EsnnG5TerSpikingCNN:
    return EsnnG5TerSpikingCNN(n_classes=2, seed=0, n_steps=2)


def test_buffer_capacity_fifo() -> None:
    buf = EsnnCNNBetaBuffer(capacity=2)
    for k in range(3):
        buf.push(
            x=np.full((32, 32, 3), float(k), dtype=np.float32),
            y=k % 2, latent=None,
        )
    snap = buf.snapshot()
    assert len(snap) == 2 and snap[0]["y"] == 1


def test_buffer_capacity_zero_raises() -> None:
    with pytest.raises(ValueError):
        EsnnCNNBetaBuffer(capacity=0)


def test_replay_no_op_empty() -> None:
    clf = _make_clf()
    w0 = clf.W_c1.copy()
    _replay_step(clf, [], lr=0.1, n_steps=1)
    np.testing.assert_array_equal(clf.W_c1, w0)


def test_downscale_one_noop_then_half() -> None:
    clf = _make_clf()
    w0 = clf.W_c1.copy()
    _downscale_step(clf, factor=1.0)
    np.testing.assert_array_equal(clf.W_c1, w0)
    _downscale_step(clf, factor=0.5)
    np.testing.assert_allclose(clf.W_c1, 0.5 * w0)


def test_downscale_invalid_bounds() -> None:
    clf = _make_clf()
    with pytest.raises(ValueError):
        _downscale_step(clf, factor=0.0)
    with pytest.raises(ValueError):
        _downscale_step(clf, factor=1.5)


def test_restructure_zero_noop_else_W_c2_only() -> None:
    clf = _make_clf()
    w_c1_0, w_c2_0, w_out_0 = (
        clf.W_c1.copy(), clf.W_c2.copy(), clf.W_out.copy()
    )
    _restructure_step(clf, factor=0.0, seed=0)
    np.testing.assert_array_equal(clf.W_c2, w_c2_0)
    _restructure_step(clf, factor=0.1, seed=42)
    assert not np.allclose(clf.W_c2, w_c2_0)
    np.testing.assert_array_equal(clf.W_c1, w_c1_0)
    np.testing.assert_array_equal(clf.W_out, w_out_0)


def test_recombine_empty_or_single_class_no_op() -> None:
    clf = _make_clf()
    w0 = clf.W_out.copy()
    _recombine_step(clf, latents=[], n_synthetic=4, lr=0.1, seed=0)
    np.testing.assert_array_equal(clf.W_out, w0)
    _recombine_step(
        clf, latents=[([0.0] * 64, 0), ([0.1] * 64, 0)],
        n_synthetic=4, lr=0.1, seed=0,
    )
    np.testing.assert_array_equal(clf.W_out, w0)


def test_dream_episode_logs_one_entry_per_call() -> None:
    clf = _make_clf()
    profile = build_esnn_cnn_profile("P_equ", seed=0)
    buf = EsnnCNNBetaBuffer(capacity=4)
    rng = np.random.default_rng(0)
    for _ in range(4):
        buf.push(
            x=rng.standard_normal((32, 32, 3)).astype(np.float32),
            y=int(rng.integers(0, 2)),
            latent=rng.standard_normal(64).astype(np.float32),
        )
    n0 = len(profile.runtime.log)
    dream_episode_cnn_esnn(
        clf, profile, seed=0, beta_buffer=buf,
        replay_n_records=2, replay_n_steps=1, replay_lr=0.01,
        downscale_factor=0.99, restructure_factor=0.01,
        recombine_n_synthetic=4, recombine_lr=0.01,
    )
    assert len(profile.runtime.log) == n0 + 1
```

- [ ] **Step 2: Run — expect ImportError**

`uv run pytest tests/unit/experiments/test_g5_ter_dream_wrap_cnn.py -v --no-cov`

- [ ] **Step 3: Implement `experiments/g5_ter_spiking_cnn/dream_wrap_cnn.py`**

Reference structure : `experiments/g5_bis_richer_esnn/esnn_dream_wrap_hier.py` (read it ; it is the verbatim template). Substitute :

- Imports : `EsnnG5TerSpikingCNN` (Task 2) instead of `EsnnG5BisHierarchicalClassifier`.
- `EsnnCNNBetaBuffer` : same FIFO + `push(*, x, y, latent)` + `snapshot()` + `sample(n, seed)` + `__len__()` API as `EsnnHierBetaBuffer`. The only change is `x` is stored as the NHWC nested list `x.astype(np.float32).tolist()` (shape `(32, 32, 3)` round-trips through nested lists exactly as for the flat 784-d case).
- `_replay_step(clf, records, *, lr, n_steps)` : early-exit on empty records / `lr <= 0` / `n_steps <= 0`. Stack `np.asarray([r["x"] for r in records], dtype=np.float32)` (already shape `(N, 32, 32, 3)` after stacking), then `for _ in range(n_steps): clf._ste_backward(x, y, lr)`.
- `_downscale_step(clf, *, factor)` : validate `factor ∈ (0, 1]`, then iterate over the 8 tensor attribute names and multiply each by `np.float32(factor)`.
- `_restructure_step(clf, *, factor, seed)` : validate `factor >= 0`, early-exit on `factor == 0` or `clf.W_c2.size == 0` or `sigma == 0`. Add `factor * sigma * N(0, 1)` to `clf.W_c2`.
- `_recombine_step(clf, *, latents, n_synthetic, lr, seed)` : copy the G5-bis body verbatim, then add the bias gradient line :

```python
d_b_out = d_logits.sum(axis=0)
clf.b_out = (
    clf.b_out - np.float32(lr) * d_b_out.astype(np.float32)
).astype(np.float32)
```

- `build_esnn_cnn_profile(name, seed)` : copy `build_esnn_richer_profile` body verbatim (uses `PROFILE_FACTORIES`, `PMinProfile`, `_rebind_to_esnn`).
- `dream_episode_cnn_esnn(clf, profile, seed, *, beta_buffer, replay_n_records, replay_n_steps, replay_lr, downscale_factor, restructure_factor, recombine_n_synthetic, recombine_lr)` : copy `dream_episode_hier_esnn` body verbatim (same op-set selection, same DR-0 spectator `input_slice`, same call sequence). The only change is the type of `clf` and `beta_buffer`.

- [ ] **Step 4: Run — expect 7/7 PASS**

`uv run pytest tests/unit/experiments/test_g5_ter_dream_wrap_cnn.py -v --no-cov`

- [ ] **Step 5: Commit**

```bash
git add experiments/g5_ter_spiking_cnn/dream_wrap_cnn.py \
        tests/unit/experiments/test_g5_ter_dream_wrap_cnn.py
git commit -m "feat(g5ter): dream-episode coupling" -m "Coupling map : REPLAY=SGD+STE on full network,
DOWNSCALE on all 8 tensors, RESTRUCTURE on W_c2
only, RECOMBINE on (W_out, b_out) over per-class
Gaussian-MoG synthetic latents. DR-0 spectator
runtime preserved via build_esnn_cnn_profile."
```

---

## Task 4: Driver `run_g5_ter.py`

**Files:** Create `experiments/g5_ter_spiking_cnn/run_g5_ter.py`. Test: `tests/unit/experiments/test_g5_ter_run_smoke.py`.

Reference : `experiments/g5_bis_richer_esnn/run_g5_bis.py` (literal template). Substitution table :

| Aspect | G5-bis | G5-ter |
|--------|--------|--------|
| classifier | `EsnnG5BisHierarchicalClassifier(in_dim, hidden_1, hidden_2, n_classes=2, seed, n_steps)` | `EsnnG5TerSpikingCNN(n_classes=2, seed=seed, n_steps=n_steps)` |
| dataset loader import | `load_split_fmnist_5tasks` | `load_split_cifar10_5tasks_auto` |
| task input keys | `task["x_train"]`, `task["x_test"]` (flat) | `task["x_train_nhwc"]`, `task["x_test_nhwc"]` (NHWC) |
| `train_task` payload | `task` dict directly | `{"x_train": tasks[k]["x_train_nhwc"], "y_train": tasks[k]["y_train"]}` |
| `_push_task` per-record `x` | `task["x_train"][i]` 1-D | `task["x_train_nhwc"][i]` shape `(32, 32, 3)` |
| `clf.latent` input shape | `x[None, :]` (`(1, 784)`) | `x[None, ...]` (`(1, 32, 32, 3)`) |
| buffer / profile / episode | `EsnnHierBetaBuffer`, `build_esnn_richer_profile`, `dream_episode_hier_esnn` | `EsnnCNNBetaBuffer`, `build_esnn_cnn_profile`, `dream_episode_cnn_esnn` |
| `SUBSTRATE_NAME` | `"esnn_thalamocortical_richer"` | `"esnn_thalamocortical_spiking_cnn"` |
| `DEFAULT_OUT_JSON / OUT_MD` | `g5-bis-richer-esnn-2026-05-03` | `g5-ter-spiking-cnn-2026-05-03` |
| run-registry profile prefix | `g5-bis/richer/{arm}` | `g5-ter/spiking_cnn/{arm}` |
| verdict key + helper | `_h7a_verdict` → `"h7a_richer_esnn"` | `_h8a_verdict` → `"h8a_spiking_cnn"` |
| CLI flags | `--hidden-1 --hidden-2 --n-steps` | drop `--hidden-*`, keep `--n-steps` (default 20) |

Constants identical to G5-bis :
```python
RESTRUCTURE_FACTOR = 0.05
RECOMBINE_N_SYNTHETIC = 16
RECOMBINE_LR = 0.01
BETA_BUFFER_CAPACITY = 256
BETA_BUFFER_FILL_PER_TASK = 32
C_VERSION = "C-v0.12.0+PARTIAL"
```

The verdict computation, run-registry integration, and `_render_md_report` body are line-for-line copies from G5-bis (rename `H7-A` → `H8-A` in headers and provenance section). Provenance cites `docs/osf-prereg-g5-ter-spiking-cnn.md`, `docs/milestones/g5-bis-aggregate-2026-05-03.md`, `docs/milestones/g4-quinto-step2-2026-05-03.md`.

- [ ] **Step 1: Write the failing smoke test**

```python
"""Smoke test for the G5-ter pilot driver."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.g5_ter_spiking_cnn import run_g5_ter


def _synthetic_cifar_tasks() -> list[dict]:
    rng = np.random.default_rng(0)
    out: list[dict] = []
    for _ in range(5):
        n = 8
        x = rng.standard_normal((n, 32, 32, 3)).astype(np.float32)
        y = rng.integers(0, 2, size=n, dtype=np.int64)
        out.append({
            "x_train": x.reshape(n, -1), "x_train_nhwc": x,
            "y_train": y,
            "x_test": x.reshape(n, -1), "x_test_nhwc": x,
            "y_test": y,
        })
    return out


def test_run_pilot_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "experiments.g5_ter_spiking_cnn.run_g5_ter."
        "load_split_cifar10_5tasks_auto",
        lambda _d: _synthetic_cifar_tasks(),
    )
    out_json = tmp_path / "smoke.json"
    out_md = tmp_path / "smoke.md"
    run_g5_ter.run_pilot(
        data_dir=tmp_path, seeds=(0,),
        out_json=out_json, out_md=out_md,
        registry_db=tmp_path / "reg.sqlite",
        epochs=1, batch_size=4, lr=0.01, n_steps=2,
    )
    body = json.loads(out_json.read_text())
    assert out_md.exists()
    assert len(body["cells"]) == len(run_g5_ter.ARMS)
    assert "h8a_spiking_cnn" in body["verdict"]
    assert "retention_by_arm" in body["verdict"]
```

- [ ] **Step 2: Run — expect ImportError**

`uv run pytest tests/unit/experiments/test_g5_ter_run_smoke.py -v --no-cov`

- [ ] **Step 3: Implement `experiments/g5_ter_spiking_cnn/run_g5_ter.py`**

Copy `experiments/g5_bis_richer_esnn/run_g5_bis.py` byte-for-byte and apply the substitution table above. Drop the `hidden_1` / `hidden_2` argparse flags (and the unused `_run_cell` parameters). Keep `_resolve_commit_sha`, `RunRegistry` integration, the dataclass/typeddict pattern. The renamed `_h8a_verdict` is the only logic change — its body is identical to G5-bis `_h7a_verdict` (Welch one-sided of `baseline` against `P_equ` at α/4 = 0.0125).

- [ ] **Step 4: Run smoke — expect PASS in <60 s**

`uv run pytest tests/unit/experiments/test_g5_ter_run_smoke.py -v --no-cov`

- [ ] **Step 5: Commit**

```bash
git add experiments/g5_ter_spiking_cnn/run_g5_ter.py \
        tests/unit/experiments/test_g5_ter_run_smoke.py
git commit -m "feat(g5ter): pilot driver" -m "run_g5_ter.py — Split-CIFAR-10 sweep, 4 arms x N
seeds, single HP combo C5, writes per-cell milestone
plus H8-A own-substrate verdict. Mirrors run_g5_bis
modulo substrate + dataset loader. EC=PARTIAL."
```

---

## Task 5: Cross-substrate aggregator

**Files:** Create `experiments/g5_ter_spiking_cnn/aggregator.py`. Test: `tests/unit/experiments/test_g5_ter_aggregator.py`.

Reference : `experiments/g5_bis_richer_esnn/aggregator.py`. Substitutions :

- MLX milestone key : `retention_by_arm` (G4-quinto Step 2 schema, NOT `retention_richer_by_arm`).
- E-SNN milestone key : `retention_by_arm` (unchanged).
- Classification labels : `H8-A / H8-B / H8-C`.
- Add two thresholds : `H8A_G_MLX_MINUS_ESNN_MIN = 2.0`, `H8B_G_MLX_MINUS_ESNN_MAX = 1.0`. `H7B_G_THRESHOLD = 0.5` is reused as `g_h8` cutoff.
- Decision rule (in code) :

```python
g_p_equ_cross = per_arm["P_equ"].get("hedges_g_mlx_minus_esnn", 0.0)
if own_insufficient:
    classification = "ambiguous"
elif (
    own_g_h8 < H7B_G_THRESHOLD
    and not welch_reject
    and g_p_equ_cross >= H8A_G_MLX_MINUS_ESNN_MIN
):
    classification = "H8-A"
elif (
    own_g_h8 >= H7B_G_THRESHOLD
    and welch_reject
    and g_p_equ_cross < H8B_G_MLX_MINUS_ESNN_MAX
):
    classification = "H8-B"
else:
    classification = "H8-C"
```

- Output dict keys : `h8_classification`, `g_h8_esnn`, `g_h8_welch_p`, `g_h8_welch_reject_h0`, `alpha_per_arm`, `h8a_g_mlx_minus_esnn_min`, `h8b_g_mlx_minus_esnn_max`, `mlx_milestone`, `esnn_milestone`, `per_arm`.
- Renderer title : `# G5-ter cross-substrate aggregate - H8-A/B/C verdict`. Provenance section cites the new pre-reg + the G5-bis aggregate.

Public API : `aggregate_g5ter_verdict(mlx_milestone, esnn_milestone) -> dict` and `write_aggregate_dump(*, mlx_milestone, esnn_milestone, out_json, out_md) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
"""Unit tests for the G5-ter aggregator."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.g5_ter_spiking_cnn.aggregator import (
    aggregate_g5ter_verdict, write_aggregate_dump,
)


def _write(path: Path, retention: dict) -> None:
    path.write_text(
        json.dumps({"verdict": {"retention_by_arm": retention}})
    )


def _ret(b: float, p: float) -> dict:
    return {
        "baseline": [b] * 10, "P_min": [b + 0.05] * 10,
        "P_equ": [p] * 10, "P_max": [p - 0.05] * 10,
    }


def _make_pair(
    tmp_path: Path, mlx_p: float, esnn_p: float,
) -> tuple[Path, Path]:
    mlx, esnn = tmp_path / "mlx.json", tmp_path / "esnn.json"
    _write(mlx, _ret(0.5, mlx_p))
    _write(esnn, _ret(0.5, esnn_p))
    return mlx, esnn


def test_h8a_lif_washout(tmp_path: Path) -> None:
    """own-fail + huge MLX-minus-ESNN gap -> LIF non-linearity."""
    mlx, esnn = _make_pair(tmp_path, 0.95, 0.52)
    assert aggregate_g5ter_verdict(mlx, esnn)[
        "h8_classification"
    ] == "H8-A"


def test_h8b_architecture_recovery(tmp_path: Path) -> None:
    """own-pass + closed gap -> CNN architecture recovers signal."""
    mlx, esnn = _make_pair(tmp_path, 0.95, 0.93)
    assert aggregate_g5ter_verdict(mlx, esnn)[
        "h8_classification"
    ] == "H8-B"


def test_h8c_partial(tmp_path: Path) -> None:
    """own-pass + persistent gap -> partial."""
    mlx, esnn = _make_pair(tmp_path, 0.95, 0.65)
    assert aggregate_g5ter_verdict(mlx, esnn)[
        "h8_classification"
    ] == "H8-C"


def test_write_aggregate_dump(tmp_path: Path) -> None:
    flat = {
        a: [0.5] * 5 for a in
        ("baseline", "P_min", "P_equ", "P_max")
    }
    mlx, esnn = tmp_path / "mlx.json", tmp_path / "esnn.json"
    _write(mlx, flat)
    _write(esnn, flat)
    out_json, out_md = tmp_path / "agg.json", tmp_path / "agg.md"
    write_aggregate_dump(
        mlx_milestone=mlx, esnn_milestone=esnn,
        out_json=out_json, out_md=out_md,
    )
    assert "H8" in out_md.read_text()
```

- [ ] **Step 2: Run — expect ImportError**

`uv run pytest tests/unit/experiments/test_g5_ter_aggregator.py -v --no-cov`

- [ ] **Step 3: Implement `experiments/g5_ter_spiking_cnn/aggregator.py`**

Copy the G5-bis aggregator body and apply the substitutions above. The per-arm Welch two-sided computation, `_load_retention`, and `_welch_two_sided` helper are unchanged — only the classification block, threshold constants, and key names differ.

- [ ] **Step 4: Run — expect 4/4 PASS**

`uv run pytest tests/unit/experiments/test_g5_ter_aggregator.py -v --no-cov`

- [ ] **Step 5: Commit**

```bash
git add experiments/g5_ter_spiking_cnn/aggregator.py \
        tests/unit/experiments/test_g5_ter_aggregator.py
git commit -m "feat(g5ter): H8 aggregator" -m "Cross-substrate H8-A/B/C verdict — distinguishes
LIF-non-linearity washout from architecture
mismatch via own-substrate g_h8 plus MLX-minus-ESNN
gap. Thresholds 0.5 / 1.0 / 2.0 locked at pre-reg."
```

---

## Task 6: Run the pilot + milestone dump

- [ ] **Step 1: Smoke run (Option C, 4 cells, ~3-5 min)**

```bash
uv run python experiments/g5_ter_spiking_cnn/run_g5_ter.py --smoke \
    --out-json /tmp/g5-ter-smoke.json --out-md /tmp/g5-ter-smoke.md \
    --registry-db /tmp/g5-ter-reg.sqlite
```
Expected : 4 cells, exit 0, no traceback. **Do NOT commit /tmp dumps.**

- [ ] **Step 2: Option B sweep (40 cells, ~30-60 min)**

```bash
uv run python experiments/g5_ter_spiking_cnn/run_g5_ter.py
```
Expected : `docs/milestones/g5-ter-spiking-cnn-2026-05-03.{json,md}` written.

- [ ] **Step 3: Run cross-substrate aggregator**

```bash
uv run python -c "
from pathlib import Path
from experiments.g5_ter_spiking_cnn.aggregator import write_aggregate_dump
write_aggregate_dump(
    mlx_milestone=Path('docs/milestones/g4-quinto-step2-2026-05-03.json'),
    esnn_milestone=Path('docs/milestones/g5-ter-spiking-cnn-2026-05-03.json'),
    out_json=Path('docs/milestones/g5-ter-aggregate-2026-05-03.json'),
    out_md=Path('docs/milestones/g5-ter-aggregate-2026-05-03.md'),
)
"
```

- [ ] **Step 4: Verify verdict**

`python -c "import json; v=json.load(open('docs/milestones/g5-ter-aggregate-2026-05-03.json')); print(v['h8_classification'], v.get('g_h8_esnn'))"`
Expected : one of `{H8-A, H8-B, H8-C, ambiguous}` plus a float.

- [ ] **Step 5: Commit milestones**

```bash
git add docs/milestones/g5-ter-spiking-cnn-2026-05-03.json \
        docs/milestones/g5-ter-spiking-cnn-2026-05-03.md \
        docs/milestones/g5-ter-aggregate-2026-05-03.json \
        docs/milestones/g5-ter-aggregate-2026-05-03.md
git commit -m "milestone(g5ter): H8 spiking-CNN dump" -m "G5-ter pilot milestone (40 cells, Option B) plus
cross-substrate H8-A/B/C verdict. Tests whether
the G4-ter / G4-quinto positive effect transfers
through LIF rate-coding when the architecture is
convolutional rather than fully connected."
```

---

## Task 7: Paper 2 §7.1.10 — EN + FR mirror

**Files:** Modify `docs/papers/paper2/results.md` and `docs/papers/paper2-fr/results.md`.

- [ ] **Step 1: Confirm parents exist**

`test -f docs/papers/paper2/results.md && test -f docs/papers/paper2-fr/results.md && echo OK`

- [ ] **Step 2: Append §7.1.10 (EN)** — substitute `<H8_X>`, `<g_h8>`, `<g_p_equ_cross>` with values from `docs/milestones/g5-ter-aggregate-2026-05-03.json` :

```markdown
### 7.1.10 G5-ter — Spiking-CNN cross-substrate test

To distinguish whether the G5-bis fail-to-reject verdict
(`g_h7a = +0.1043`, MLX-only artefact for the 3-layer LIF MLP)
stems from (i) the LIF rate-coding non-linearity itself washing
out the small dream-arm signal, or (ii) the architectural mismatch
between the MLX dense head and the LIF MLP, we ported the
G4-quinto Step 2 small-CNN architecture onto the E-SNN substrate
as a 4-layer spiking CNN (Conv2d-LIF + Conv2d-LIF + FC-LIF +
Linear, STE backward) and ran the same 4-arm × 10-seed sweep on
Split-CIFAR-10 at HP combo C5.

The cross-substrate verdict is **<H8_X>** (`g_h8 = <g_h8>`,
`g_mlx_minus_esnn(P_equ) = <g_p_equ_cross>`, Bonferroni
α/4 = 0.0125). Under the pre-registered decision rule this means :
<one-sentence interpretation pulled from the threshold table —
"the LIF non-linearity is the load-bearing washout" / "the
convolutional architecture closes the gap" / "both contribute
partially">. Provenance : `docs/osf-prereg-g5-ter-spiking-cnn.md`,
`docs/milestones/g5-ter-spiking-cnn-2026-05-03.md`,
`docs/milestones/g5-ter-aggregate-2026-05-03.md`.
```

- [ ] **Step 3: Append §7.1.10 (FR)** — translate the EN block (same numerical values, same paths). The FR mirror is mandatory per `docs/CLAUDE.md` (EN→FR propagation in same PR).

- [ ] **Step 4: Verify mirror integrity**

`grep -nc "7.1.10" docs/papers/paper2/results.md docs/papers/paper2-fr/results.md`
Expected : both report ≥ 1 hit.

- [ ] **Step 5: Commit**

```bash
git add docs/papers/paper2/results.md docs/papers/paper2-fr/results.md
git commit -m "paper2(g5ter): add 7.1.10 H8 results" -m "Report the H8-A/B/C verdict in Paper 2 §7.1.10
and its FR mirror. Cites the new aggregator
milestone and the OSF pre-reg. EN->FR propagation
in the same PR per docs/CLAUDE.md."
```

---

## Task 8: DR-3 evidence revision

**Files:** Modify `docs/proofs/dr3-substrate-evidence.md` (append-only).

- [ ] **Step 1: Append a dated entry** — wording depends on the H8 outcome :

- **H8-A** : "DR-3 substrate-agnosticity at the positive-effect channel remains exhausted under LIF non-linearity. The G4-ter / G4-quinto `g ≈ +2.77` positive effect does not survive E-SNN rate-coding regardless of architectural depth (MLP : G5-bis H7-B ; CNN : G5-ter H8-A, `g_h8 = <value>`). DR-3 falsified for the cycle-3 positive effect ; reformulation deferred to Paper 3."
- **H8-B** : "DR-3 partially restored at the positive-effect channel : when the architecture is convolutional, the LIF stack recovers the positive dream-arm effect (G5-ter H8-B, `g_h8 = <value>`). DR-3 holds conditionally on architectural inductive bias ; reformulate as DR-3' (architecture-aware substrate-agnosticity) in the next FC bump."
- **H8-C** : "DR-3 ambiguous at the positive-effect channel : architectural inductive bias contributes partially but does not fully close the LIF gap (G5-ter H8-C, `g_h8 = <value>`, `g_mlx_minus_esnn(P_equ) = <value>`). Confirmatory N≥30 follow-up required before DR-3 reformulation."

- [ ] **Step 2: Verify presence**

`uv run python -c "from pathlib import Path; assert Path('docs/proofs/dr3-substrate-evidence.md').exists()"`

- [ ] **Step 3: Commit**

```bash
git add docs/proofs/dr3-substrate-evidence.md
git commit -m "proofs(dr3): G5-ter H8 evidence row" -m "Append a dated DR-3 evidence entry recording the
G5-ter spiking-CNN H8-A/B/C verdict. File is
append-only ; no prior text rewritten."
```

---

## Task 9: CHANGELOG + STATUS (no FC bump)

**Files:** Modify `CHANGELOG.md` and `STATUS.md`.

- [ ] **Step 1: Append to `CHANGELOG.md` `[Unreleased]` `### Added`**

```
G5-ter spiking-CNN cross-substrate test
(`docs/milestones/g5-ter-aggregate-2026-05-03.{json,md}`) — H8-A/B/C
verdict on whether LIF non-linearity vs architecture explains the
G5-bis H7-B washout.
```

- [ ] **Step 2: Update `STATUS.md`**

- Bump `As of` to `2026-05-03`.
- Add gates-table row : `G5-ter | spiking-CNN H8 verdict | <H8-X> | docs/milestones/g5-ter-aggregate-2026-05-03.md`.
- EC stays `PARTIAL`. No FC bump.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md STATUS.md
git commit -m "docs(g5ter): changelog + STATUS row" -m "Record the G5-ter milestone in CHANGELOG and add
the gates-table row in STATUS. EC stays PARTIAL,
no FC bump."
```

---

## Task 10: Self-review + verification gate

- [ ] **Step 1: Test suite**

`uv run pytest tests/unit/experiments/test_g5_ter_*.py -v --no-cov`
Expected : all PASS (≥ 25 tests across 4 files).

- [ ] **Step 2: Lint**

`uv run ruff check experiments/g5_ter_spiking_cnn tests/unit/experiments/test_g5_ter_*.py`
Expected : 0 errors.

- [ ] **Step 3: Project-wide coverage gate**

`uv run pytest`
Expected : suite passes, coverage ≥ 90 % (`experiments/` itself excluded per `pyproject.toml`).

- [ ] **Step 4: Commit-trailer policy**

`git log --since="1 hour ago" --grep="Co-Authored-By" --oneline`
Expected : empty (org policy in `/Users/electron/hypneum-lab/CLAUDE.md` forbids the trailer).

- [ ] **Step 5: Final tree clean**

`git status` should report a clean working tree. If lint/coverage fixups are needed, stage + commit with subject `docs(g5ter): self-review fixups`.

---

## Self-review checklist (executed after writing this plan)

**1. Spec coverage** :

- H8-A/B/C decision rule with locked thresholds 0.5 / 1.0 / 2.0 → Task 1 §1, Task 5, plan header.
- 4-arm × N=10 sweep on HP combo C5 → Task 0 step 4 (verify), Task 4 (driver), Task 6 step 2 (run).
- STE backward on Conv2d → Task 2 (`conv2d_backward`, `_ste_backward`).
- Pure-numpy, no `norse`, no `torch` → Task 2 (only imports `numpy` + `kiki_oniric.substrates.esnn_thalamocortical`).
- HF parquet fallback → reused via `load_split_cifar10_5tasks_auto` (Task 4).
- Pre-reg before run → Task 1 precedes Task 6 (run is gated on the pre-reg commit).
- Cross-substrate aggregator vs G4-quinto Step 2 milestone → Task 5 (key `retention_by_arm` matches Step 2 schema, verified Task 0 step 1).
- EN→FR mirror Paper 2 §7.1.10 → Task 7 step 2 + 3.
- DR-3 evidence revision → Task 8 with three branches per outcome.
- CHANGELOG + STATUS no FC bump → Task 9.
- Smoke variant Option C → Task 6 step 1 (`--smoke`).

**2. Placeholder scan** : no "TBD" / "implement later" / "similar to Task N". The `<H8_X>` / `<g_h8>` placeholders in Tasks 7–9 are values to substitute from the milestone JSON, not undefined types.

**3. Type consistency** :

- `EsnnG5TerSpikingCNN` field names `W_c1, b_c1, W_c2, b_c2, W_fc1, b_fc1, W_out, b_out` — used identically in Tasks 2 + 3 + tests.
- `EsnnCNNBetaBuffer` API matches `EsnnHierBetaBuffer` (Task 3 references the G5-bis sister verbatim).
- `dream_episode_cnn_esnn` kwargs `(beta_buffer, replay_n_records, replay_n_steps, replay_lr, downscale_factor, restructure_factor, recombine_n_synthetic, recombine_lr)` match the call site in Task 4.
- Aggregator returns `h8_classification` — used consistently in Task 5 tests, Task 6 step 4, Task 7 prose, Task 9 STATUS row.
- `train_task` payload key : `task["x_train"]` (NHWC `(N, 32, 32, 3)`) — Task 2 and Task 4 (driver renames `x_train_nhwc` → `x_train` when building the payload). Smoke fixture in Task 4 step 1 carries both keys for monkeypatch convenience.

No issues found.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-03-g5-ter-spiking-cnn-washout-test.md`. Two execution options :

**1. Subagent-Driven (recommended)** — fresh subagent per task with two-stage review. Use `superpowers:subagent-driven-development`.

**2. Inline Execution** — execute tasks in this session with batch checkpoints. Use `superpowers:executing-plans`.

Which approach?
