# G4-quinto CIFAR-10 escalation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the G4-quater H4-C verdict (RECOMBINE empirically empty at FMNIST scale) is **scale-bound** (FMNIST artefact) or **universal** by escalating to Split-CIFAR-10 with both an MLP-on-CIFAR substrate and a small CNN substrate; emit confirmatory verdicts H5-A / H5-B / H5-C and revise DR-4 evidence accordingly.

**Architecture:** Three sequential steps mirroring the G4-quater layout. Step 1 = port the 5-layer deeper MLP onto Split-CIFAR-10 (input dim 3072) — tests whether benchmark scale alone recovers the predicted DR-4 ordering. Step 2 = swap the substrate to a small MLX CNN (`G4SmallCNN`, 2 Conv2d + 2 MaxPool2d + 2 Linear) — tests whether hierarchical conv structure recovers the ordering. Step 3 = replicate G4-quater H4-C placebo design (RECOMBINE strategy ∈ {mog, ae, none}) on the CNN substrate — confirms or refutes universality of the RECOMBINE-empty finding. All three steps reuse the existing pre-reg / Welch / Jonckheere / Bonferroni / R1 run-registry pipeline; no axiom signature changes (no FC bump). EC stays PARTIAL across all outcomes.

**Tech Stack:** Python 3.12, MLX (`mlx.core`, `mlx.nn` — `nn.Conv2d`, `nn.MaxPool2d`), numpy, scipy.stats (Welch), `kiki_oniric.eval.statistics` (Jonckheere, Hedges' g), `harness.storage.run_registry.RunRegistry` (R1 bit-stable run_ids), pure-numpy CIFAR-10 binary loader (~163 MB, SHA-256 pinned).

---

## Read-first context

Before any code, skim signatures and patterns from these files (do **not** rewrite — reuse) :

- `docs/osf-prereg-g4-quater-pilot.md` — pre-reg template (§1-§9 layout, deviation envelopes, honest-reading clause).
- `docs/milestones/g4-quater-step3-2026-05-03.md` — H4-C confirmed verdict (Welch p = 0.9886, RECOMBINE empty at FMNIST scale).
- `docs/proofs/dr4-profile-inclusion.md` §7.1.6 — current G4-quater addendum that this plan **extends** (does not supersede).
- `experiments/g4_quater_test/deeper_classifier.py` — 5-layer MLP pattern with `latent()`, `restructure_step()`, `recombine_step()`, `replay_optimizer_step()`, `downscale_step()`. **Reused verbatim** for MLP-on-CIFAR (Task 3) ; **port-with-substitutions** for the CNN (Task 4).
- `experiments/g4_quater_test/recombine_strategies.py` — the `mog`/`ae`/`none` dispatcher. **Reused as-is** in Step 3 (latent-dim agnostic).
- `experiments/g4_quater_test/run_step1_deeper.py` and `run_step3_recombine_strategies.py` — driver templates (cell loop, beta buffer, dream-episode wrapper, MD/JSON emission, registry registration).
- `experiments/g4_quater_test/aggregator.py` — verdict aggregation pattern.
- `experiments/g4_split_fmnist/dataset.py` — pure-numpy decoder pattern (mirror this style for CIFAR-10).
- `kiki_oniric/eval/statistics.py:37,112,226` — `welch_one_sided`, `jonckheere_trend`, `compute_hedges_g`. We use scipy Welch two-sided directly per the H4-C precedent.
- `harness/storage/run_registry.py` — `RunRegistry.register(c_version, profile, seed, commit_sha) -> run_id`.

**Do NOT** modify `kiki_oniric/dream/episode.py` or any axiom module. This plan is calibration + escalation, not a framework change. FC stays at v0.12.0 across all outcomes.

---

## File structure

| Status | Path | Responsibility |
|--------|------|----------------|
| Create | `experiments/g4_quinto_test/__init__.py` | package marker |
| Create | `experiments/g4_quinto_test/cifar10_dataset.py` | pure-numpy CIFAR-10 binary loader, 5-task class-incremental split |
| Create | `experiments/g4_quinto_test/data/.gitkeep` | data dir for downloaded CIFAR-10 binaries (gitignored payload) |
| Create | `experiments/g4_quinto_test/cifar_mlp_classifier.py` | `G4HierarchicalCIFARClassifier` — port of `G4HierarchicalDeeperClassifier` with `in_dim=3072` |
| Create | `experiments/g4_quinto_test/small_cnn.py` | `G4SmallCNN` — Conv2d×2 + MaxPool2d×2 + Linear×2 MLX CNN substrate |
| Create | `experiments/g4_quinto_test/run_step1_mlp_cifar.py` | Step 1 driver — H5-A test on MLP-on-CIFAR (4 arms × N seeds × 1 HP) |
| Create | `experiments/g4_quinto_test/run_step2_cnn_cifar.py` | Step 2 driver — H5-B test on CNN substrate |
| Create | `experiments/g4_quinto_test/run_step3_cnn_recombine.py` | Step 3 driver — H5-C placebo on CNN (3 strategies × 4 arms × N seeds) |
| Create | `experiments/g4_quinto_test/aggregator.py` | load 3 step JSONs, emit H5-A/B/C aggregate verdict |
| Create | `tests/unit/test_g4_quinto_cifar_loader.py` | unit tests for loader (decode + 5-task split) |
| Create | `tests/unit/test_g4_quinto_cifar_mlp.py` | unit tests for MLP-on-CIFAR substrate |
| Create | `tests/unit/test_g4_quinto_small_cnn.py` | unit tests for CNN substrate |
| Create | `tests/unit/test_g4_quinto_aggregator.py` | unit tests for aggregator verdict |
| Create | `docs/osf-prereg-g4-quinto-pilot.md` | OSF pre-reg G4-quinto, locked **before** Step 1 run |
| Create | `docs/milestones/g4-quinto-step1-2026-05-03.{json,md}` | Step 1 outputs (driver-emitted) |
| Create | `docs/milestones/g4-quinto-step2-2026-05-03.{json,md}` | Step 2 outputs |
| Create | `docs/milestones/g4-quinto-step3-2026-05-03.{json,md}` | Step 3 outputs |
| Create | `docs/milestones/g4-quinto-aggregate-2026-05-03.{json,md}` | aggregator outputs |
| Modify | `docs/papers/paper2/results.md` | add §7.1.7 G4-quinto subsection |
| Modify | `docs/papers/paper2-fr/results.md` | mirror §7.1.7 (FR translation) |
| Modify | `docs/proofs/dr4-profile-inclusion.md` | add §7.1.7 G4-quinto evidence row |
| Modify | `CHANGELOG.md` | Empirical entry under [Unreleased] |
| Modify | `STATUS.md` | append G4-quinto line under Empirical |

**Decomposition rationale**: each step is a self-contained
driver with its own milestone artefact (append-only per
`docs/CLAUDE.md`). The aggregator is the single point that
combines the three verdicts. Substrate code is split between
`cifar_mlp_classifier.py` (for Step 1) and `small_cnn.py` (for
Steps 2 + 3) so Step 1 can land before CNN code exists.

---

## Architecture decisions

### Compute budget — pick **before** Task 1

The driver-default `--n-seeds` flag controls scope. Pick one:

| Option | Steps | Cells | Wall time M1 Max | Use when |
|--------|-------|-------|------------------|----------|
| **A — full (default, pre-reg N=30)** | Step 1 + Step 2 + Step 3 | 120 + 120 + 360 = **600** | ≈ 9–15 h | Overnight run, full scientific verdict |
| **B — reduced** | Step 1 + Step 2 only | 120 + 120 = **240** | ≈ 3–6 h | Compute-pressure session ; defer Step 3 to G4-sexto follow-up |
| **C — smoke** | 1 task × 5 seeds × 4 arms × 1 substrate | **20** | ≈ 30 min | Pipeline validation only — not a science verdict |

**Recommend Option A** if the user accepts an overnight run.
**Default to Option B** if the session is < 4 h and there is no
explicit overnight commitment. Option C is **only** for CI / smoke
gates and must never produce milestone artefacts under
`docs/milestones/` (use `--smoke` flag — driver writes to
`/tmp/...` and the JSON header carries `"smoke": true`).

The **decision is recorded as a Task 0.5 commit** (`docs(g4-quinto):
record compute-option <X>`) before Task 1 starts.

### Dataset

CIFAR-10 binary version, canonical mirror
``https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz`` — 163 MB
gzipped tar. SHA-256 pinned in the loader header
(``c4a38c50a1bc5f3a1c5437f8d2d9f1c9...`` — verified in Task 2 once
the file is on disk; do not hardcode a wrong hash).

Class-incremental 5-task split mirroring Split-FMNIST canonical:

```
task 0 : classes {0, 1}  (airplane, automobile)
task 1 : classes {2, 3}  (bird, cat)
task 2 : classes {4, 5}  (deer, dog)
task 3 : classes {6, 7}  (frog, horse)
task 4 : classes {8, 9}  (ship, truck)
```

Labels remapped to {0, 1} per task (binary head shared). Images
stored as ``np.float32`` in ``[0, 1]``, layout
``(N, 32, 32, 3)`` for CNN consumption and flattened to
``(N, 3072)`` for MLP consumption (the loader returns both).

### Substrates

**MLP-on-CIFAR (Step 1)** — `G4HierarchicalCIFARClassifier`. 5-layer signature copied verbatim from `G4HierarchicalDeeperClassifier` ; only the constructor defaults change (`in_dim=3072`, `hidden=(256,128,64,32)`). RESTRUCTURE site = `_l3` ; RECOMBINE sampling site = `latent()` -> h3 activations (dim 64).

**CNN (Steps 2 + 3)** — `G4SmallCNN`. NHWC layout (MLX convention, verified in Task 0) :

```
(N,32,32,3) -> Conv2d(3,16,3,pad=1) ReLU MaxPool2d(2,2)  -> (N,16,16,16)
            -> Conv2d(16,32,3,pad=1) ReLU MaxPool2d(2,2) -> (N,8,8,32)
            -> Flatten -> Linear(2048,64) ReLU -> Linear(64,2)
```

Op site mapping :
- **REPLAY** — full-model SGD pass on a batch from the beta buffer (buffer stores `(32,32,3)` float32 + 64-dim latent + label).
- **DOWNSCALE** — multiply every weight+bias of {conv1, conv2, fc1, fc2} by `factor`. Bound `(0, 1]` identical.
- **RESTRUCTURE** — perturb `conv2.weight` only (second feature extractor, analogue of the MLP middle layer). Same `factor*sigma*N(0,1)` rule.
- **RECOMBINE** — synthetic latents (dim 64) per the active strategy ∈ {mog, ae, none}, single CE-loss SGD step on `fc2`.

### Statistics & multiplicity

Same machinery as G4-quater:

- H5-A : Jonckheere on retention across (P_min, P_equ, P_max),
  one-sided α = 0.05 / 3 = 0.0167 (Bonferroni for 3 hypotheses).
- H5-B : Jonckheere on the CNN run, same α = 0.0167.
- H5-C : Welch two-sided P_max(mog) vs P_max(none), α = 0.0167.
  **Failing** to reject confirms H5-C (universality of the
  RECOMBINE-empty finding from FMNIST onto CIFAR-CNN).

**Honest reading clause** (must be embedded verbatim in every
milestone MD): *"Welch fail-to-reject = absence of evidence at
this N for a difference between mog and none — under H5-C
specifically, this **is** the predicted positive empirical
claim that RECOMBINE adds nothing measurable beyond
REPLAY+DOWNSCALE on the CNN substrate at CIFAR-10 scale."*

### Pre-reg discipline

OSF pre-reg G4-quinto (Task 1) is locked at the commit *that
introduces it* — **before Task 6 (Step 1 driver run)**. The
pre-reg cites G4-quater verdict (H4-A/B falsified, H4-C
confirmed) explicitly as exploratory baseline ; H5-A/B/C are
the confirmatory hypotheses. Deviation envelope mirrors
G4-quater §9.

---

## Common commands

```bash
# Install (mlx + scipy already in pyproject)
uv sync --all-extras

# Smoke (any step)
uv run python experiments/g4_quinto_test/run_step1_mlp_cifar.py --smoke
uv run python experiments/g4_quinto_test/run_step2_cnn_cifar.py --smoke
uv run python experiments/g4_quinto_test/run_step3_cnn_recombine.py --smoke

# Full (Option A, ~9-15 h overnight)
uv run python experiments/g4_quinto_test/run_step1_mlp_cifar.py --n-seeds 30
uv run python experiments/g4_quinto_test/run_step2_cnn_cifar.py --n-seeds 30
uv run python experiments/g4_quinto_test/run_step3_cnn_recombine.py --n-seeds 30

# Aggregate
uv run python experiments/g4_quinto_test/aggregator.py

# Lint + types + tests (must pass before each commit)
uv run ruff check experiments/g4_quinto_test tests/unit/test_g4_quinto_*.py
uv run mypy experiments/g4_quinto_test
uv run pytest tests/unit/test_g4_quinto_*.py -v
```

---

## Tasks

### Task 0: Investigate prior art + MLX Conv2d API

**Files:**
- Read only: `experiments/g4_quater_test/{deeper_classifier,recombine_strategies,run_step1_deeper,run_step3_recombine_strategies,aggregator}.py`
- Read only: `experiments/g4_split_fmnist/dataset.py`
- Read only: `kiki_oniric/dream/episode.py` (Operation enum + DreamEpisode signature)
- Read only: `harness/storage/run_registry.py`

- [ ] **Step 1: Confirm MLX Conv2d / MaxPool2d API**

```bash
uv run python -c "import mlx.nn as nn; print(nn.Conv2d.__init__.__doc__); print(nn.MaxPool2d.__init__.__doc__)"
```

Expected: signature `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)`, `nn.MaxPool2d(kernel_size, stride=None, padding=0)`. Layout is **NHWC** (channels last).

- [ ] **Step 2: Confirm key reused signatures**

Skim and note:

- `G4HierarchicalDeeperClassifier(in_dim, hidden, n_classes, seed)` — `__post_init__` requires `len(hidden)==4`.
- `recombine_strategies.sample_synthetic_latents(strategy, latents, labels, n_synthetic, seed)` — works on any latent dim.
- `BetaBufferHierFIFO` (in `dream_wrap_hier.py`) and the local `_BetaBufferDeeper` (in `run_step1_deeper.py`) — pick the deeper one as base for CIFAR variants because it already supports per-record latents.

No code changes in Task 0. No commit.

---

### Task 0.5: Record compute option (A / B / C)

**Files:**
- Modify: `docs/osf-prereg-g4-quinto-pilot.md` (does not exist yet — skipped if Task 1 not done)

This is a **decision step**, not code. The decision is logged as
a one-line note in the executor's session log. It will be
embedded in the OSF pre-reg created in Task 1 §3 (Power
analysis) — N depends on option chosen.

- [ ] **Step 1: Decide A / B / C**

Read this plan's "Compute budget" table. Match against:
- Available wall time
- Whether the user accepts an overnight run
- Smoke gate vs science gate

- [ ] **Step 2: Commit the decision marker**

(No file change — the decision is committed implicitly via the
N value chosen in the OSF pre-reg in Task 1.)

---

### Task 1: OSF pre-registration — G4-quinto pilot

**Files:**
- Create: `docs/osf-prereg-g4-quinto-pilot.md`

- [ ] **Step 1: Write the pre-registration**

Mirror the §1-§9 structure of `docs/osf-prereg-g4-quater-pilot.md` verbatim. Replace pilot-specific content as follows :

- **Header**: pilot = G4-quinto ; sister pilot = G4-quater (H4-A/B falsified, H4-C confirmed) ; substrates = MLX 5-layer MLP-on-CIFAR (Step 1) + MLX small CNN (Steps 2+3) ; benchmark = Split-CIFAR-10.
- **§1 Background**: cite G4-quater commit `83a6ae8`, summarise H4-A/B falsified + H4-C confirmed (Welch p = 0.9886, P_max(mog) = 0.7007, P_max(none) = 0.7006) ; cite G4-quater pre-reg §6 "future work CIFAR-10 / ImageNet" clause as the trigger for this escalation.
- **§2 Hypotheses (confirmatory)** :
  - **H5-A (benchmark-scale)** — on Split-CIFAR-10 with MLP-on-CIFAR (`hidden=(256,128,64,32)`, `in_dim=3072`), `P_max ≥ P_equ ≥ P_min` recovers. Jonckheere, α = 0.05/3 = 0.0167.
  - **H5-B (architecture-scale)** — same on `G4SmallCNN` (2 Conv2d + 2 MaxPool2d + 2 Linear, 64-dim latent). Jonckheere, α = 0.0167.
  - **H5-C (universality of RECOMBINE-empty)** — on CNN, Welch two-sided P_max(mog) vs P_max(none) at α = 0.0167 ; **failing** to reject = positive empirical claim that the G4-quater finding generalises.
- **§3 Power**: N = 30 seeds/arm at α = 0.0167 detects |g| ≥ 0.74 at 80 % power.
- **§4 Exclusion**: identical to G4-quater (acc_initial < 0.5 ; non-finite acc_final ; run_id collision).
- **§5 Paths**: drivers = `experiments/g4_quinto_test/run_step{1,2,3}_*.py` ; substrates = `cifar_mlp_classifier.G4HierarchicalCIFARClassifier` + `small_cnn.G4SmallCNN` ; loader = `cifar10_dataset.load_split_cifar10_5tasks`.
- **§6 DualVer outcome matrix** (5 rows) :
  1. H5-A and H5-B confirmed → EC PARTIAL ; scope-bound STABLE for CIFAR-10 (MLP and CNN) ; ImageNet open Q.
  2. H5-A confirmed only → EC PARTIAL ; scope-bound STABLE for MLP-CIFAR only.
  3. H5-B confirmed only → EC PARTIAL ; scope-bound STABLE for CNN-CIFAR only.
  4. H5-C confirmed → EC PARTIAL ; **DR-4 partial refutation universalises across FMNIST + CIFAR-CNN** ; framework C claim "richer ops yield richer consolidation" empirically refuted across 2 benchmarks ; DR-4 evidence file revised.
  5. All three falsified → EC PARTIAL ; G4-quinto reported as null ; G4-quater FMNIST refutation remains dominant evidence.
  6. All three confirmed → EC PARTIAL ; ImageNet escalation prerequisite for any STABLE promotion.

  EC stays PARTIAL across all rows ; FC stays at v0.12.0 across all rows.
- **§7 Reporting**: Welch fail-to-reject = "absence of evidence" — except H5-C, where it **is** the predicted positive claim. Verbatim honest-reading clause from G4-quater §7.
- **§8 Audit trail**: profile keys `g4-quinto/{step1,step2,step3}/<arm>/<combo>[/<strategy>]` ; milestones at `docs/milestones/g4-quinto-step{1,2,3}-2026-05-03.{json,md}` + aggregate.
- **§9 Deviations**: (a) CIFAR-10 download fails → abort + amendment ; (b) CNN acc_initial < 0.5 → epochs 3 → 5, document ; (c) Step 3 wall time > 10 h M1 Max → reduce N 30 → 20 ; (d) Option B chosen → Step 3 deferred, aggregator reports `h5c_deferred = True`.

- [ ] **Step 2: Commit the pre-reg**

```bash
git add docs/osf-prereg-g4-quinto-pilot.md
git commit -m "docs(g4-quinto): lock OSF pre-reg pilot"
```

---

### Task 2: Pure-numpy CIFAR-10 loader + 5-task split

**Files:**
- Create: `experiments/g4_quinto_test/__init__.py` (empty)
- Create: `experiments/g4_quinto_test/cifar10_dataset.py`
- Create: `experiments/g4_quinto_test/data/.gitkeep` (empty)
- Modify: `.gitignore` — add `experiments/g4_quinto_test/data/cifar-10-batches-bin/` and `experiments/g4_quinto_test/data/cifar-10-binary.tar.gz`
- Test: `tests/unit/test_g4_quinto_cifar_loader.py`

CIFAR-10 binary format (per https://www.cs.toronto.edu/~kriz/cifar.html) :
each record = 1 label byte + 3072 image bytes (1024 R + 1024 G +
1024 B, row-major). Five training files
``data_batch_1.bin`` … ``data_batch_5.bin`` (10000 records each)
+ ``test_batch.bin`` (10000 records).

- [ ] **Step 1: Write the failing test**

Build a tiny synthetic CIFAR-10 fixture in `tmp_path` (no 163 MB download needed for unit tests). Three test cases :

```python
# tests/unit/test_g4_quinto_cifar_loader.py
"""Unit tests for G4-quinto CIFAR-10 loader (synthetic tmp_path fixture)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from experiments.g4_quinto_test.cifar10_dataset import (
    CIFAR10_RECORD_SIZE, decode_cifar10_bin, load_split_cifar10_5tasks,
)


def _write_batch(path: Path, labels: list[int], rng: np.random.Generator) -> None:
    rows = [
        bytes([lbl]) + rng.integers(0, 256, size=3072, dtype=np.uint8).tobytes()
        for lbl in labels
    ]
    path.write_bytes(b"".join(rows))


def test_decode_cifar10_bin_shape(tmp_path: Path) -> None:
    f = tmp_path / "batch.bin"
    _write_batch(f, [0, 1, 2, 3], np.random.default_rng(0))
    images, labels = decode_cifar10_bin(f)
    assert images.shape == (4, 32, 32, 3) and images.dtype == np.uint8
    assert labels.tolist() == [0, 1, 2, 3]


def test_decode_cifar10_bin_truncated_raises(tmp_path: Path) -> None:
    f = tmp_path / "bad.bin"
    f.write_bytes(b"\x00" * (CIFAR10_RECORD_SIZE - 1))
    with pytest.raises(ValueError, match="truncated"):
        decode_cifar10_bin(f)


def test_load_split_cifar10_5tasks_split(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    bin_dir = tmp_path / "cifar-10-batches-bin"; bin_dir.mkdir()
    _write_batch(bin_dir / "data_batch_1.bin",
                 [c for c in range(10) for _ in range(2)], rng)
    for k in range(2, 6):
        _write_batch(bin_dir / f"data_batch_{k}.bin", [], rng)
    _write_batch(bin_dir / "test_batch.bin", list(range(10)), rng)
    tasks = load_split_cifar10_5tasks(bin_dir)
    assert len(tasks) == 5
    for task in tasks:
        assert task["x_train"].shape[1] == 3072
        assert task["x_train_nhwc"].shape[1:] == (32, 32, 3)
        assert set(task["y_train"].tolist()) <= {0, 1}
        assert set(task["y_test"].tolist()) <= {0, 1}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/unit/test_g4_quinto_cifar_loader.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.g4_quinto_test.cifar10_dataset'`.

- [ ] **Step 3: Implement the loader**

Create `experiments/g4_quinto_test/__init__.py` (empty), then `cifar10_dataset.py`. Three exported APIs ; structure mirrors `experiments/g4_split_fmnist/dataset.py` (decode helpers + 5-task split) — the IDX-magic / gzip handling is replaced by raw-binary record handling per the CIFAR-10 spec (1 label byte + 3072 image bytes per record).

```python
"""Split-CIFAR-10 5-task loader — pure numpy, no torchvision.

Source : https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
Records : 1 label byte + 3072 image bytes (CHW : 1024 R + 1024 G + 1024 B).
"""
from __future__ import annotations
import hashlib, tarfile, urllib.request
from pathlib import Path
from typing import Final, TypedDict
import numpy as np

CIFAR10_LABEL_BYTES: Final[int] = 1
CIFAR10_IMAGE_BYTES: Final[int] = 32 * 32 * 3
CIFAR10_RECORD_SIZE: Final[int] = CIFAR10_LABEL_BYTES + CIFAR10_IMAGE_BYTES
CIFAR10_URL: Final[str] = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR10_TAR_SHA256: Final[str] = "...replace_in_task9..."  # one-shot pin
SPLIT_CIFAR10_TASKS: Final[tuple[tuple[int, int], ...]] = (
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9),
)


class SplitCIFAR10Task(TypedDict):
    x_train: np.ndarray; x_train_nhwc: np.ndarray; y_train: np.ndarray
    x_test:  np.ndarray; x_test_nhwc:  np.ndarray; y_test:  np.ndarray


def decode_cifar10_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """One binary -> (images uint8 NHWC, labels uint8). Raises ValueError
    if len(raw) is not a multiple of CIFAR10_RECORD_SIZE (truncated)."""
    raw = path.read_bytes()
    n, rem = divmod(len(raw), CIFAR10_RECORD_SIZE)
    if rem != 0:
        raise ValueError(
            f"truncated CIFAR-10 binary in {path} : "
            f"{len(raw)} bytes is not a multiple of {CIFAR10_RECORD_SIZE}"
        )
    if n == 0:
        return (np.zeros((0, 32, 32, 3), np.uint8), np.zeros((0,), np.uint8))
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(n, CIFAR10_RECORD_SIZE)
    labels = arr[:, 0].copy()
    nhwc = np.transpose(arr[:, 1:].reshape(n, 3, 32, 32), (0, 2, 3, 1)).copy()
    return nhwc, labels


def download_if_missing(data_dir: Path) -> Path:
    """Download tar -> verify SHA-256 -> extract. Returns
    cifar-10-batches-bin/ path. Raises FileNotFoundError on network
    failure (pre-reg §9.1 deviation envelope)."""
    bin_dir = data_dir / "cifar-10-batches-bin"
    if bin_dir.exists() and (bin_dir / "test_batch.bin").exists():
        return bin_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = data_dir / "cifar-10-binary.tar.gz"
    if not tar_path.exists():
        try:
            urllib.request.urlretrieve(CIFAR10_URL, tar_path)
        except OSError as exc:
            raise FileNotFoundError(f"CIFAR-10 download failed : {exc}") from exc
    if not CIFAR10_TAR_SHA256.startswith("..."):
        h = hashlib.sha256(tar_path.read_bytes()).hexdigest()
        if h != CIFAR10_TAR_SHA256:
            raise ValueError(f"SHA-256 mismatch : got {h}, expected {CIFAR10_TAR_SHA256}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir)
    return bin_dir


def load_split_cifar10_5tasks(data_dir: Path) -> list[SplitCIFAR10Task]:
    """Load 5 sequential 2-class tasks. Raises FileNotFoundError if any
    expected batch file is absent (caller may invoke download_if_missing)."""
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"CIFAR-10 dir does not exist : {data_dir}")
    test_path = data_dir / "test_batch.bin"
    if not test_path.exists():
        raise FileNotFoundError(f"missing CIFAR-10 test batch : {test_path}")
    train_imgs, train_lbls = [], []
    for k in range(1, 6):
        p = data_dir / f"data_batch_{k}.bin"
        if not p.exists():
            raise FileNotFoundError(f"missing CIFAR-10 train batch : {p}")
        x, y = decode_cifar10_bin(p); train_imgs.append(x); train_lbls.append(y)
    x_train_raw = np.concatenate(train_imgs, axis=0)
    y_train_raw = np.concatenate(train_lbls, axis=0)
    x_test_raw, y_test_raw = decode_cifar10_bin(test_path)
    x_tr_nhwc = x_train_raw.astype(np.float32) / 255.0
    x_te_nhwc = x_test_raw.astype(np.float32) / 255.0
    x_tr_flat = x_tr_nhwc.reshape(x_tr_nhwc.shape[0], -1)
    x_te_flat = x_te_nhwc.reshape(x_te_nhwc.shape[0], -1)
    tasks: list[SplitCIFAR10Task] = []
    for class_a, class_b in SPLIT_CIFAR10_TASKS:
        tr = (y_train_raw == class_a) | (y_train_raw == class_b)
        te = (y_test_raw  == class_a) | (y_test_raw  == class_b)
        y_tr = np.where(y_train_raw[tr] == class_a, 0, 1).astype(np.int64)
        y_te = np.where(y_test_raw[te]  == class_a, 0, 1).astype(np.int64)
        tasks.append(SplitCIFAR10Task(
            x_train=x_tr_flat[tr], x_train_nhwc=x_tr_nhwc[tr], y_train=y_tr,
            x_test =x_te_flat[te], x_test_nhwc =x_te_nhwc[te], y_test =y_te,
        ))
    return tasks
```

- [ ] **Step 4: Verify SHA-256 once at first download (manual)**

After Task 6 (or earlier) actually downloads the file, replace
the placeholder hash:

```bash
sha256sum experiments/g4_quinto_test/data/cifar-10-binary.tar.gz
# replace CIFAR10_TAR_SHA256 with the real hash, commit:
# git commit -m "chore(g4-quinto): pin CIFAR-10 SHA-256"
```

- [ ] **Step 5: Run tests to verify pass**

```bash
uv run pytest tests/unit/test_g4_quinto_cifar_loader.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add experiments/g4_quinto_test/__init__.py experiments/g4_quinto_test/cifar10_dataset.py experiments/g4_quinto_test/data/.gitkeep tests/unit/test_g4_quinto_cifar_loader.py .gitignore
git commit -m "feat(g4-quinto): add CIFAR-10 loader"
```

---

### Task 3: G4HierarchicalCIFARClassifier (MLP-on-CIFAR substrate)

**Files:**
- Create: `experiments/g4_quinto_test/cifar_mlp_classifier.py`
- Test: `tests/unit/test_g4_quinto_cifar_mlp.py`

This substrate is a **straight port** of `G4HierarchicalDeeperClassifier` with `in_dim=3072` and `hidden=(256, 128, 64, 32)`.

- [ ] **Step 1: Write the failing tests**

Five unit tests covering shape, latent dim, RESTRUCTURE noop+bound, DOWNSCALE bound. Pattern mirrors `tests/unit/test_g4_quater_deeper_classifier.py` (existing tests for the parent class) — only the constructor changes (in_dim 3072, hidden (256,128,64,32)). Tests :

```python
# tests/unit/test_g4_quinto_cifar_mlp.py
"""Unit tests for G4HierarchicalCIFARClassifier (port of deeper MLP)."""
from __future__ import annotations
import numpy as np
import pytest
from experiments.g4_quinto_test.cifar_mlp_classifier import (
    G4HierarchicalCIFARClassifier,
)

H = (256, 128, 64, 32)
def _clf() -> G4HierarchicalCIFARClassifier:
    return G4HierarchicalCIFARClassifier(in_dim=3072, hidden=H, n_classes=2, seed=0)


def test_predict_logits_shape() -> None:
    x = np.random.default_rng(0).standard_normal((4, 3072)).astype(np.float32)
    assert _clf().predict_logits(x).shape == (4, 2)


def test_latent_returns_h3_dim() -> None:
    x = np.random.default_rng(0).standard_normal((4, 3072)).astype(np.float32)
    assert _clf().latent(x).shape == (4, 64)  # h3 = third hidden width


def test_restructure_step_zero_factor_is_noop() -> None:
    clf = _clf(); w = np.asarray(clf._l3.weight).copy()
    clf.restructure_step(factor=0.0, seed=0)
    np.testing.assert_array_equal(np.asarray(clf._l3.weight), w)


def test_restructure_step_negative_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _clf().restructure_step(factor=-0.1, seed=0)


def test_downscale_bounds() -> None:
    clf = _clf()
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        clf.downscale_step(factor=0.0)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        clf.downscale_step(factor=1.1)
```

- [ ] **Step 2: Run to confirm it fails**

```bash
uv run pytest tests/unit/test_g4_quinto_cifar_mlp.py -v
```

Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement the substrate**

`cifar_mlp_classifier.py` is a **direct copy** of `experiments/g4_quater_test/deeper_classifier.py` with:

- module docstring updated to reference Split-CIFAR-10
- class renamed from `G4HierarchicalDeeperClassifier` to `G4HierarchicalCIFARClassifier`
- no signature changes (constructor still takes `in_dim, hidden, n_classes, seed`)
- all methods preserved verbatim (`predict_logits`, `latent`,
  `eval_accuracy`, `train_task`, `restructure_step`,
  `downscale_step`, `replay_optimizer_step`, `recombine_step`)

Refer to `deeper_classifier.py` lines 22-249 — copy-paste with
the rename. The only logical change is the docstring header
referencing `in_dim=3072` and `hidden=(256, 128, 64, 32)` as the
default Step 1 sizing.

- [ ] **Step 4: Run + commit**

```bash
uv run pytest tests/unit/test_g4_quinto_cifar_mlp.py -v
uv run mypy experiments/g4_quinto_test/cifar_mlp_classifier.py
git add experiments/g4_quinto_test/cifar_mlp_classifier.py tests/unit/test_g4_quinto_cifar_mlp.py
git commit -m "feat(g4-quinto): MLP-on-CIFAR substrate"
```

---

### Task 4: G4SmallCNN (CNN substrate for Steps 2 + 3)

**Files:**
- Create: `experiments/g4_quinto_test/small_cnn.py`
- Test: `tests/unit/test_g4_quinto_small_cnn.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_g4_quinto_small_cnn.py
"""Unit tests for G4SmallCNN (Conv2d x2 + MaxPool2d x2 + Linear x2)."""
from __future__ import annotations
import numpy as np
import pytest
from experiments.g4_quinto_test.small_cnn import G4SmallCNN


def _cnn() -> G4SmallCNN:
    return G4SmallCNN(latent_dim=64, n_classes=2, seed=0)


def _batch() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((4, 32, 32, 3)).astype(np.float32)


def test_forward_shape_2_classes() -> None:
    assert _cnn().predict_logits(_batch()).shape == (4, 2)


def test_latent_shape() -> None:
    assert _cnn().latent(_batch()).shape == (4, 64)


def test_restructure_step_perturbs_conv2_only() -> None:
    cnn = _cnn()
    w1 = np.asarray(cnn._conv1.weight).copy()
    w2 = np.asarray(cnn._conv2.weight).copy()
    cnn.restructure_step(factor=0.05, seed=42)
    np.testing.assert_array_equal(np.asarray(cnn._conv1.weight), w1)
    assert not np.allclose(np.asarray(cnn._conv2.weight), w2)


def test_downscale_multiplies_all_weights() -> None:
    cnn = _cnn(); w = np.asarray(cnn._fc2.weight).copy()
    cnn.downscale_step(factor=0.5)
    np.testing.assert_allclose(np.asarray(cnn._fc2.weight), 0.5 * w, rtol=1e-6)


def test_downscale_bounds_reject() -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        _cnn().downscale_step(factor=0.0)


def test_recombine_step_with_two_classes_runs() -> None:
    rng = np.random.default_rng(0)
    latents = [
        (rng.standard_normal(64).astype(np.float32).tolist(), c)
        for c in (0, 0, 1, 1)
    ]
    _cnn().recombine_step(latents=latents, n_synthetic=8, lr=0.01, seed=0)


def test_recombine_step_empty_is_noop() -> None:
    cnn = _cnn(); w = np.asarray(cnn._fc2.weight).copy()
    cnn.recombine_step(latents=[], n_synthetic=8, lr=0.01, seed=0)
    np.testing.assert_array_equal(np.asarray(cnn._fc2.weight), w)
```

- [ ] **Step 2: Run to confirm fail**

```bash
uv run pytest tests/unit/test_g4_quinto_small_cnn.py -v
```

Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement G4SmallCNN**

`small_cnn.py` mirrors the dataclass+post-init structure of `experiments/g4_quater_test/deeper_classifier.py`. The architecture skeleton + the four bespoke methods that diverge from the MLP parent :

```python
"""Small MLX CNN substrate for G4-quinto Steps 2 + 3 — NHWC layout.

Architecture : Conv2d(3,16,3,pad=1) ReLU MaxPool2d(2,2)
            -> Conv2d(16,32,3,pad=1) ReLU MaxPool2d(2,2)
            -> Flatten -> Linear(2048, latent_dim) ReLU
            -> Linear(latent_dim, n_classes).
Op sites : RESTRUCTURE -> conv2.weight ; RECOMBINE -> latent() =
post-relu3 fc1 activation, synthetic batch -> CE step on fc2 only.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


@dataclass
class G4SmallCNN:
    latent_dim: int
    n_classes: int
    seed: int
    _conv1: nn.Conv2d = field(init=False, repr=False)
    _conv2: nn.Conv2d = field(init=False, repr=False)
    _pool1: nn.MaxPool2d = field(init=False, repr=False)
    _pool2: nn.MaxPool2d = field(init=False, repr=False)
    _fc1:   nn.Linear   = field(init=False, repr=False)
    _fc2:   nn.Linear   = field(init=False, repr=False)
    _model: nn.Module   = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mx.random.seed(self.seed); np.random.seed(self.seed)
        self._conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self._conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self._pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._fc1   = nn.Linear(2048, self.latent_dim)
        self._fc2   = nn.Linear(self.latent_dim, self.n_classes)
        self._model = nn.Sequential(
            self._conv1, nn.ReLU(), self._pool1,
            self._conv2, nn.ReLU(), self._pool2,
            _Flatten(), self._fc1, nn.ReLU(), self._fc2,
        )
        mx.eval(self._model.parameters())

    def latent(self, x: np.ndarray) -> np.ndarray:
        h = self._pool1(nn.relu(self._conv1(mx.array(x))))
        h = self._pool2(nn.relu(self._conv2(h)))
        h = mx.reshape(h, (h.shape[0], -1))
        h = nn.relu(self._fc1(h))
        mx.eval(h)
        return np.asarray(h)

    def downscale_step(self, *, factor: float) -> None:
        if not (0.0 < factor <= 1.0):
            raise ValueError(f"shrink_factor must be in (0, 1], got {factor}")
        for layer in (self._conv1, self._conv2, self._fc1, self._fc2):
            layer.weight = layer.weight * factor
            if getattr(layer, "bias", None) is not None:
                layer.bias = layer.bias * factor
        mx.eval(self._model.parameters())


class _Flatten(nn.Module):
    """Reshape (N, H, W, C) -> (N, H*W*C). MLX has no built-in nn.Flatten."""

    def __call__(self, x: mx.array) -> mx.array:
        return mx.reshape(x, (x.shape[0], -1))
```

Methods to **port verbatim from `deeper_classifier.py`** with the documented one-line substitutions :

| Method | Source lines | Substitution |
|---|---|---|
| `predict_logits` | 75-79 | none — same `self._model(...)` wrapper |
| `eval_accuracy` | 93-98 | none |
| `train_task` | 100-128 | accept `(task_x, task_y)` instead of `task: dict` (CNN consumes NHWC arrays directly, no flat dict) |
| `restructure_step` | 130-151 | `self._l3.weight` → `self._conv2.weight` (sole change) |
| `replay_optimizer_step` | 168-188 | prepend a reshape guard `if x_np.ndim == 2: x_np = x_np.reshape(-1, 32, 32, 3)` so the buffer can store flat OR NHWC |
| `recombine_step` | 190-248 | drop the `h4 = relu(_l4(x_lat))` intermediate (no l4 here) ; replace `self._l5` target with `self._fc2` ; synthetic latents feed `fc2` directly |

These are mechanical substitutions — preserve all bound checks
(`factor >= 0`, `factor in (0, 1]`), zero-noop fast paths, RNG
seeding offsets, and `mx.eval(...)` flushes verbatim.

- [ ] **Step 4: Run + commit**

```bash
uv run pytest tests/unit/test_g4_quinto_small_cnn.py -v
uv run mypy experiments/g4_quinto_test/small_cnn.py
git add experiments/g4_quinto_test/small_cnn.py tests/unit/test_g4_quinto_small_cnn.py
git commit -m "feat(g4-quinto): add small CNN substrate"
```

---

### Task 5: Step 1 driver — H5-A test (MLP-on-CIFAR)

**Files:**
- Create: `experiments/g4_quinto_test/run_step1_mlp_cifar.py`

Port of `experiments/g4_quater_test/run_step1_deeper.py`. Apply these 8 substitutions :

1. Loader import : `load_split_cifar10_5tasks` (replaces FMNIST loader).
2. Substrate import : `G4HierarchicalCIFARClassifier` (replaces `G4HierarchicalDeeperClassifier`).
3. `HIDDEN = (256, 128, 64, 32)` ; `DEFAULT_N_SEEDS = 30` (Option A).
4. Registry profile key : `g4-quinto/step1/<arm>/<combo>`.
5. Output paths : `docs/milestones/g4-quinto-step1-2026-05-03.{json,md}`.
6. Verdict function rename : `_h5a_verdict` (body identical to `_h4a_verdict`).
7. MD report H1 : "G4-quinto Step 1 — H5-A MLP-on-CIFAR" ; pre-reg path : `docs/osf-prereg-g4-quinto-pilot.md`.
8. `_run_cell` reads `task["x_train"]` (the flat 3072-dim view from the loader).

The cell loop, `_BetaBufferDeeper`, `_dream_episode_deeper` wrapper are copied verbatim — they are substrate-agnostic.

- [ ] **Step 1: Copy + apply substitutions**

```bash
cp experiments/g4_quater_test/run_step1_deeper.py experiments/g4_quinto_test/run_step1_mlp_cifar.py
# apply the 8 substitutions above
```

- [ ] **Step 2: Smoke run** — `uv run python experiments/g4_quinto_test/run_step1_mlp_cifar.py --smoke`. Expected: `/tmp/...` JSON+MD with `"smoke": true`, single seed, < 5 min wall time.
- [ ] **Step 3: Lint + types** — `uv run ruff check experiments/g4_quinto_test/run_step1_mlp_cifar.py && uv run mypy experiments/g4_quinto_test/run_step1_mlp_cifar.py`.
- [ ] **Step 4: Commit** — `git add experiments/g4_quinto_test/run_step1_mlp_cifar.py && git commit -m "feat(g4-quinto): step1 MLP-on-CIFAR driver"`.

---

### Task 6: Step 2 driver — H5-B test (CNN-on-CIFAR)

**Files:**
- Create: `experiments/g4_quinto_test/run_step2_cnn_cifar.py`

Mirrors Step 1 except :

1. Substrate : `G4SmallCNN(latent_dim=64, n_classes=2, seed=seed)`.
2. `_run_cell` passes `task["x_train_nhwc"]` to `train_task` (CNN consumes NHWC). `feat_dim` is no longer needed.
3. Buffer : rename `_BetaBufferDeeper` to `_BetaBufferCNN`, push `(32, 32, 3)` arrays (or accept flat 3072 — `replay_optimizer_step` reshapes).
4. `LATENT_DIM = 64` ; output paths : `docs/milestones/g4-quinto-step2-2026-05-03.{json,md}`.
5. Verdict rename : `_h5b_verdict`. Profile key : `g4-quinto/step2/<arm>/<combo>`.
6. `_dream_episode_*` wrapper calls `cnn.{replay_optimizer_step,downscale_step,restructure_step,recombine_step}` ; same DR-0 spectator runtime path.

- [ ] **Step 1: Implement** — copy `run_step1_mlp_cifar.py` and apply the 6 differences above.
- [ ] **Step 2: Smoke run** — `uv run python experiments/g4_quinto_test/run_step2_cnn_cifar.py --smoke`. Expected ~ 2-3× slower per cell than Step 1.
- [ ] **Step 3: Lint + types + commit** — `uv run ruff check ... && uv run mypy ... && git commit -m "feat(g4-quinto): step2 CNN-on-CIFAR driver"`.

---

### Task 7: Step 3 driver — H5-C test (CNN + RECOMBINE strategies)

**Files:**
- Create: `experiments/g4_quinto_test/run_step3_cnn_recombine.py`

Mirrors `experiments/g4_quater_test/run_step3_recombine_strategies.py` with these substitutions :

1. Substrate : `G4SmallCNN` (Step 2 driver's wrapper) instead of `G4HierarchicalClassifier`.
2. `_strategy_aware_recombine` operates on `cnn._fc2` instead of `clf._l3` — the surgery point shifts from the 3-layer middle (FMNIST) to the CNN head (CIFAR-CNN). Otherwise verbatim.
3. `STRATEGIES = ("mog", "ae", "none")` — unchanged ; `recombine_strategies.sample_synthetic_latents` is latent-dim agnostic and reused as-is.
4. Output paths : `docs/milestones/g4-quinto-step3-2026-05-03.{json,md}` ; profile key `g4-quinto/step3/<arm>/<combo>/<strategy>`.
5. Verdict rename : `_h5c_verdict` (Welch two-sided body identical) ; emits `h5c_recombine_empty_confirmed` flag.
6. Cell counts : 3 × 4 × 30 = 360 (Option A) ; 3 × 4 × 1 = 12 (smoke).

The critical surgery point — `_strategy_aware_recombine` :

```python
def _strategy_aware_recombine(
    cnn: G4SmallCNN, *, strategy: RecombineStrategy,
    beta_buffer: _BetaBufferCNN, n_synthetic: int, lr: float, seed: int,
) -> None:
    populated = beta_buffer.latents()  # list[(latent, label)]
    if not populated:
        return
    latents_arr = np.asarray([lat for lat, _ in populated], dtype=np.float32)
    labels_arr  = np.asarray([lbl for _, lbl in populated], dtype=np.int64)
    batch = sample_synthetic_latents(
        strategy=strategy, latents=latents_arr, labels=labels_arr,
        n_synthetic=n_synthetic, seed=seed,
    )
    if batch is None:
        return
    x = mx.array(batch["x"]); y = mx.array(batch["y"].astype(np.int32))
    opt = optim.SGD(learning_rate=lr)

    def loss_fn(layer: nn.Linear, xb: mx.array, yb: mx.array) -> mx.array:
        return nn.losses.cross_entropy(layer(xb), yb, reduction="mean")

    loss_and_grad = nn.value_and_grad(cnn._fc2, loss_fn)
    _loss, grads = loss_and_grad(cnn._fc2, x, y)
    opt.update(cnn._fc2, grads)
    mx.eval(cnn._fc2.parameters(), opt.state)
```

- [ ] **Step 1: Implement** — copy Step 3 quater driver, apply substitutions 1-6 above + the `_strategy_aware_recombine` body.
- [ ] **Step 2: Smoke run** — `uv run python experiments/g4_quinto_test/run_step3_cnn_recombine.py --smoke`. Expected: 12 cells, < 5 min, smoke flag.
- [ ] **Step 3: Lint + types + commit** — `git commit -m "feat(g4-quinto): step3 CNN+RECOMBINE driver"`.

---

### Task 8: Aggregator + verdict

**Files:**
- Create: `experiments/g4_quinto_test/aggregator.py`
- Test: `tests/unit/test_g4_quinto_aggregator.py`

Mirrors `experiments/g4_quater_test/aggregator.py`. Key
differences :

1. Loads `g4-quinto-step{1,2,3}-2026-05-03.json`.
2. Renamed verdict keys : `h5a_benchmark_scale` (from step1
   `verdict.h5a_mlp_cifar`), `h5b_architecture_scale` (from step2
   `verdict.h5b_cnn_cifar`), `h5c_universality_recombine_empty`
   (from step3 `verdict.h5c_recombine_strategy`).
3. Adds a **summary universality block** : flags whether the
   G4-quater H4-C finding generalises (`h4c_to_h5c_universality`
   field — True iff both H4-C confirmed at FMNIST and H5-C
   confirmed at CIFAR-CNN).
4. Output paths : `docs/milestones/g4-quinto-aggregate-2026-05-03.{json,md}`.

If the user chose Option B, Step 3 milestone is missing —
aggregator must handle the absent file gracefully (`step3_path: Path | None`):

```python
def aggregate_g4_quinto_verdict(
    step1_path: Path, step2_path: Path, step3_path: Path | None,
) -> dict[str, Any]:
    s1 = json.loads(step1_path.read_text())
    s2 = json.loads(step2_path.read_text())
    s3 = (
        json.loads(step3_path.read_text())
        if step3_path is not None and step3_path.exists()
        else None
    )

    h5a = s1["verdict"]["h5a_mlp_cifar"]
    h5b = s2["verdict"]["h5b_cnn_cifar"]
    h5a_confirmed = (
        not h5a.get("insufficient_samples", False)
        and bool(h5a.get("reject_h0"))
        and bool(h5a.get("monotonic_observed"))
    )
    h5b_confirmed = (
        not h5b.get("insufficient_samples", False)
        and bool(h5b.get("reject_h0"))
        and bool(h5b.get("monotonic_observed"))
    )
    if s3 is None:
        h5c_block: dict[str, Any] = {"deferred": True, "confirmed": False}
    else:
        h5c = s3["verdict"]["h5c_recombine_strategy"]
        h5c_confirmed = (
            not h5c.get("insufficient_samples", False)
            and bool(h5c.get("h5c_recombine_empty_confirmed"))
        )
        h5c_block = {**h5c, "deferred": False, "confirmed": h5c_confirmed}

    return {
        "h5a_benchmark_scale": {**h5a, "confirmed": h5a_confirmed},
        "h5b_architecture_scale": {**h5b, "confirmed": h5b_confirmed},
        "h5c_universality_recombine_empty": h5c_block,
        "summary": {
            "h5a_confirmed": h5a_confirmed,
            "h5b_confirmed": h5b_confirmed,
            "h5c_confirmed": h5c_block["confirmed"],
            "h5c_deferred": h5c_block.get("deferred", False),
            "h4c_to_h5c_universality": h5c_block.get("confirmed") is True,
            "any_confirmed": h5a_confirmed or h5b_confirmed or h5c_block["confirmed"],
        },
    }
```

The MD renderer mirrors `experiments/g4_quater_test/aggregator.py:_render_md` with H5-A/B/C section headings + a universality block ; reuse the same column structure and the `*Honest reading*` clause verbatim from G4-quater for the H5-C section.

- [ ] **Step 1: Test (JSON-fixture based, no driver run)**

```python
# tests/unit/test_g4_quinto_aggregator.py
"""Aggregator unit tests using JSON fixtures (no real driver run)."""
from __future__ import annotations
import json
from pathlib import Path
from experiments.g4_quinto_test.aggregator import aggregate_g4_quinto_verdict


def _h5_jonckheere_payload(*, reject: bool, monotonic: bool) -> dict:
    return {
        "reject_h0": reject, "monotonic_observed": monotonic,
        "j_statistic": 100.0 if reject else 1.0,
        "p_value": 0.001 if reject else 0.5,
        "alpha_per_test": 0.0167,
        "mean_p_min": 0.5 if monotonic else 0.7,
        "mean_p_equ": 0.6 if monotonic else 0.7,
        "mean_p_max": 0.7,
    }


def _h5c_payload(*, confirmed: bool) -> dict:
    return {
        "h5c_recombine_empty_confirmed": confirmed,
        "fail_to_reject_h0": confirmed,
        "welch_t": 0.01, "welch_p_two_sided": 0.99 if confirmed else 0.001,
        "alpha_per_test": 0.0167,
        "mean_p_max_mog": 0.7, "mean_p_max_none": 0.7,
        "hedges_g_mog_vs_none": 0.001,
        "n_p_max_mog": 30, "n_p_max_none": 30,
    }


def _write(path: Path, key: str, payload: dict) -> None:
    path.write_text(json.dumps({"verdict": {key: payload}}))


def test_aggregator_all_confirmed(tmp_path: Path) -> None:
    s1, s2, s3 = (tmp_path / f"s{k}.json" for k in (1, 2, 3))
    _write(s1, "h5a_mlp_cifar", _h5_jonckheere_payload(reject=True, monotonic=True))
    _write(s2, "h5b_cnn_cifar", _h5_jonckheere_payload(reject=True, monotonic=True))
    _write(s3, "h5c_recombine_strategy", _h5c_payload(confirmed=True))
    v = aggregate_g4_quinto_verdict(s1, s2, s3)
    s = v["summary"]
    assert s["h5a_confirmed"] is True and s["h5b_confirmed"] is True
    assert s["h5c_confirmed"] is True and s["h4c_to_h5c_universality"] is True


def test_aggregator_step3_deferred(tmp_path: Path) -> None:
    s1, s2 = (tmp_path / f"s{k}.json" for k in (1, 2))
    _write(s1, "h5a_mlp_cifar", _h5_jonckheere_payload(reject=False, monotonic=False))
    _write(s2, "h5b_cnn_cifar", _h5_jonckheere_payload(reject=False, monotonic=False))
    v = aggregate_g4_quinto_verdict(s1, s2, None)
    s = v["summary"]
    assert s["h5c_deferred"] is True and s["h5c_confirmed"] is False
    assert s["h4c_to_h5c_universality"] is False
```

- [ ] **Step 2: Run + fail**

```bash
uv run pytest tests/unit/test_g4_quinto_aggregator.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement aggregator**

Use the function above. The MD renderer mirrors
`experiments/g4_quater_test/aggregator.py:_render_md` with the
H5-A/B/C section headings and the universality block.

- [ ] **Step 4: Run pass + commit**

```bash
uv run pytest tests/unit/test_g4_quinto_aggregator.py -v
uv run ruff check experiments/g4_quinto_test/aggregator.py
uv run mypy experiments/g4_quinto_test/aggregator.py
git add experiments/g4_quinto_test/aggregator.py tests/unit/test_g4_quinto_aggregator.py
git commit -m "feat(g4-quinto): aggregate H5-A/B/C verdict"
```

---

### Task 9: Run pilot — produce milestones (gated by Task 0.5 option)

**Files (driver-emitted):**
- Create: `docs/milestones/g4-quinto-step1-2026-05-03.{json,md}`
- Create: `docs/milestones/g4-quinto-step2-2026-05-03.{json,md}`
- Create (Option A only): `docs/milestones/g4-quinto-step3-2026-05-03.{json,md}`
- Create: `docs/milestones/g4-quinto-aggregate-2026-05-03.{json,md}`

- [ ] **Step 1: Confirm CIFAR-10 download**

```bash
uv run python -c "
from pathlib import Path
from experiments.g4_quinto_test.cifar10_dataset import download_if_missing
p = download_if_missing(Path('experiments/g4_quinto_test/data'))
print('OK', p)
"
```

If this raises `FileNotFoundError`, abort the pilot and document a §9.1 deviation (network unavailable).

- [ ] **Step 2: Pin SHA-256 (one-shot)**

```bash
sha256sum experiments/g4_quinto_test/data/cifar-10-binary.tar.gz
# update CIFAR10_TAR_SHA256 in cifar10_dataset.py with the value
git add experiments/g4_quinto_test/cifar10_dataset.py
git commit -m "chore(g4-quinto): pin CIFAR-10 SHA-256"
```

- [ ] **Step 3: Run Step 1 (Option A: ~3-5 h ; Option B: same)**

```bash
uv run python experiments/g4_quinto_test/run_step1_mlp_cifar.py --n-seeds 30
```

Expected: writes `docs/milestones/g4-quinto-step1-2026-05-03.{json,md}`. Console prints `H5-A : monotonic=<bool> J=<float> p=<float>`.

- [ ] **Step 4: Run Step 2 (~3-6 h)**

```bash
uv run python experiments/g4_quinto_test/run_step2_cnn_cifar.py --n-seeds 30
```

Expected: `g4-quinto-step2-2026-05-03.{json,md}` written.

- [ ] **Step 5: Run Step 3 (Option A only ; ~3-9 h)**

If Option B was chosen, **skip this step** and document the deferral in CHANGELOG (Task 12).

```bash
uv run python experiments/g4_quinto_test/run_step3_cnn_recombine.py --n-seeds 30
```

Expected: `g4-quinto-step3-2026-05-03.{json,md}` written.

- [ ] **Step 6: Run aggregator**

```bash
uv run python experiments/g4_quinto_test/aggregator.py
# Option B: pass --step3 /dev/null or omit ; aggregator handles missing file.
uv run python experiments/g4_quinto_test/aggregator.py --step3 /dev/null  # if Option B
```

- [ ] **Step 7: Commit milestones**

```bash
git add docs/milestones/g4-quinto-*-2026-05-03.{json,md}
git commit -m "feat(g4-quinto): emit pilot milestones"
```

---

### Task 10: Paper 2 §7.1.7 (EN + FR, lockstep)

**Files:**
- Modify: `docs/papers/paper2/results.md` — append §7.1.7 G4-quinto (after §7.1.6).
- Modify: `docs/papers/paper2-fr/results.md` — mirror §7.1.7 in French.

§7.1.7 must contain : (a) cite G4-quater verdict as exploratory baseline (H4-A/B falsified, H4-C confirmed at FMNIST) ; (b) cite OSF pre-reg `docs/osf-prereg-g4-quinto-pilot.md` lock commit ; (c) tabulate H5-A/B/C with the §7.1.6 column structure (Hypothesis | Test | Result | Verdict) ; (d) embed the honest-reading clause ; (e) cite milestone artefacts (driver paths + JSON paths) ; (f) close with the universality verdict (`h4c_to_h5c_universality`) and a DR-4 evidence revision pointer to §7.1.7 of `dr4-profile-inclusion.md`.

- [ ] **Step 1: Read aggregate numbers** — `cat docs/milestones/g4-quinto-aggregate-2026-05-03.json`. Note per-H5 stats (means / J / p / Welch t / Hedges' g).
- [ ] **Step 2: Append §7.1.7 to EN paper** — populate the table with Step 1 numbers ; verdict matrix matches the OSF pre-reg §6 row that fired.
- [ ] **Step 3: Mirror to FR paper** — translate, append after §7.1.6 in `paper2-fr/results.md`. Use canonical glossary terms (`docs/glossary.md`) — no local synonyms.
- [ ] **Step 4: Commit** — `git commit -m "docs(paper2): add 7.1.7 G4-quinto results"`.

---

### Task 11: DR-4 evidence revision

**Files:**
- Modify: `docs/proofs/dr4-profile-inclusion.md` — append §7.1.7 G4-quinto evidence row.

- [ ] **Step 1: Append §7.1.7** mirroring the §7.1.6 G4-quater addendum. Record : (a) pre-reg lock commit + pilot date ; (b) the H5-A/B/C verdict tuple ; (c) updated partial-refutation status — H5-C confirmed → "DR-4 partial refutation universalises across FMNIST + CIFAR-CNN ; framework C claim 'richer ops yield richer consolidation' empirically refuted across 2 benchmarks × 3 substrates" ; H5-C falsified → "refutation FMNIST-bound ; CIFAR-CNN preserves RECOMBINE contribution ; framework C scope-bound STABLE for CIFAR-CNN" ; Option B → "verdict deferred to G4-sexto follow-up".
- [ ] **Step 2: Bump proof header version** per `docs/proofs/CLAUDE.md` (append `v0.4 (2026-05-03 G4-quinto addendum)`, update `Status:` line).
- [ ] **Step 3: Commit** — `git commit -m "docs(dr4): G4-quinto evidence row"`.

---

### Task 12: CHANGELOG + STATUS

**Files:** Modify `CHANGELOG.md` and `STATUS.md`.

- [ ] **Step 1: CHANGELOG** — under `## [Unreleased]`, add `### Empirical (G4-quinto, 2026-05-03)` block after the G5 entry. Document : pre-reg path + lock commit ; cell counts per step (120 / 120 / 360 — or "deferred" for Option B) ; wall-time totals ; H5-A/B/C verdict + Welch/Jonckheere stats ; universality verdict (`h4c_to_h5c_universality`) ; DR-4 revision pointer ; run-registry profile keys. **No EC bump** (stays PARTIAL per OSF pre-reg §6) ; **no FC bump** (stays v0.12.0).
- [ ] **Step 2: STATUS** — append under existing `### Empirical (...)` :

```markdown
### Empirical (G4-quinto CIFAR-10 escalation, 2026-05-03)

- G4-quinto pilot — Split-CIFAR-10 mid-scale escalation of the
  G4-quater RECOMBINE-empty finding. <600 | 240> cells across
  {MLP-on-CIFAR, CNN}. Verdict : H5-A=<bool>, H5-B=<bool>,
  H5-C=<bool|deferred>. EC stays PARTIAL. Pre-reg
  `docs/osf-prereg-g4-quinto-pilot.md`, milestones
  `docs/milestones/g4-quinto-{step1,step2,step3,aggregate}-2026-05-03.{json,md}`.
```

- [ ] **Step 3: Commit** — `git commit -m "docs(changelog): record G4-quinto outcome"`.

---

### Task 13: Final verification

- [ ] **Step 1: Full test suite** — `uv run pytest -v`. Expected: all passing, coverage ≥ 90 %.
- [ ] **Step 2: Lint + types** — `uv run ruff check experiments/g4_quinto_test tests/unit/test_g4_quinto_* && uv run mypy harness tests`. Clean.
- [ ] **Step 3: R1 reproducibility (Apple Silicon)** — `uv run pytest tests/reproducibility/ --no-cov`. New G4-quinto run_ids are bit-stable per `(c_version, profile, seed, commit_sha)`.
- [ ] **Step 4: Cross-doc citation audit** — `grep -rn "G4-quinto" docs/papers/paper2*/results.md docs/proofs/dr4-*.md docs/milestones/g4-quinto-* CHANGELOG.md STATUS.md`. Every paper/proof reference must point to a milestone artefact ; no `HEAD` citations.
- [ ] **Step 5: EN-FR mirror check** — `diff <(grep -E "^## 7\.1\." docs/papers/paper2/results.md) <(grep -E "^## 7\.1\." docs/papers/paper2-fr/results.md)`. Section numbers must match exactly.
- [ ] **Step 6: No FC bump audit** — `grep -E "C-v0\.[0-9]+\.[0-9]+" CHANGELOG.md | head -3`. Still `C-v0.12.0+PARTIAL`.
- [ ] **Step 7: If Step 4 surfaced drift, fix and commit** — `git commit -m "docs(g4-quinto): cite-pin <area>"`. Otherwise, no final commit.

---

## Self-Review

**1. Spec coverage** : every spec item maps to ≥ 1 task — sequential 3-step (Tasks 5/6/7) ; MLP-on-CIFAR substrate (Task 3) ; CNN substrate (Task 4) ; RECOMBINE strategies on CNN (Task 4 `recombine_step` + Task 7 driver wiring) ; ~600 cells / ~9-15 h (Task 0.5 + Task 9) ; Compute Options A/B/C (Task 0.5) ; OSF pre-reg before pilot (Task 1, locked before Task 9) ; CIFAR-10 numpy loader fail-graceful (Task 2) ; aggregator (Task 8) ; Paper 2 §7.1.7 EN+FR mirror (Task 10) ; DR-4 evidence revision (Task 11) ; CHANGELOG + STATUS, EC stays PARTIAL (Task 12) ; final verification (Task 13). Caveats 1-7 from the spec all anchor on dedicated tasks (compute budget → Task 0.5 ; download fail → Task 2 + Task 9 Step 1 ; MLX CNN perf NHWC verified → Task 0 Step 1 + Task 4 ; honest reporting → Task 1 §7 + Task 10 + Task 11 ; no FC bump → Task 13 Step 6 ; EN→FR → Task 10 + Task 13 Step 5 ; pre-reg fidelity citing G4-quater → Task 1 §1+§2).

**2. Placeholder scan** : no "TBD", "implement later", or "add appropriate error handling". Every code step contains real code or an explicit copy-with-substitutions table (Tasks 3, 4, 5, 6, 7). The CIFAR-10 SHA-256 placeholder is explicitly marked as a one-shot pin step (Task 2 Step 4 + Task 9 Step 2) — unavoidable real-data dependency.

**3. Type consistency** : `G4HierarchicalCIFARClassifier(in_dim, hidden, n_classes, seed)` — consistent across Tasks 3, 5, 13. `G4SmallCNN(latent_dim, n_classes, seed)` — consistent across Tasks 4, 6, 7. `sample_synthetic_latents(strategy, latents, labels, n_synthetic, seed)` — reused unchanged from G4-quater. `aggregate_g4_quinto_verdict(step1_path, step2_path, step3_path: Path | None)` — Task 8, Option-B-aware. Verdict keys aligned : `h5a_mlp_cifar`, `h5b_cnn_cifar`, `h5c_recombine_strategy`. Profile-registry keys aligned : `g4-quinto/step{1,2,3}/<arm>/<combo>[/<strategy>]`. No drift.

---

<!-- buddy: *narrows yellow eyes* universality test or nothing -->

