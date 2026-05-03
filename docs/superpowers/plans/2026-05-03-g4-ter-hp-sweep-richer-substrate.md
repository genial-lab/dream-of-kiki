# G4-ter HP Sweep + Richer Substrate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the confirmatory N≥30 G4-ter pilot that distinguishes three competing explanations of the G4-bis finding (`g_h1 = -2.31`, `H_DR4` degenerate equal-means) — H1 *HP artefact*, H2 *substrate-level limitation*, H3 *combined effect* — by introducing a hierarchical MLP head that exposes RESTRUCTURE and RECOMBINE channels and sweeping a curated 10-combo HP sub-grid against a single representative HP combo at full N=30 seeds.

**Architecture:** Add a `G4HierarchicalClassifier` (input → hidden_1 (32) → hidden_2 (16) → output) that exposes hidden_2 weights for `_restructure_step` perturbation and a Gaussian-MoG sampler over the β-buffer latent records for `_recombine_step` synthetic-latent generation. Extend `BetaRecord` with an optional `latent` field. Keep the existing `G4Classifier` untouched — G4-ter uses the new classifier exclusively. Driver `experiments/g4_ter_hp_sweep/run_g4_ter.py` runs the chosen Option C: 1 representative HP combo × 30 seeds × 4 arms = 120 cells against the richer substrate; the 10-combo HP sub-grid is run on top of the existing binary `G4Classifier` × N=10 seeds × 3 dream arms = 300 cells (no baseline because HP changes do not affect baseline). Total budget = 420 cells, ~3-5h M1 Max wall-clock.

**Tech Stack:** MLX (`mlx.core`, `mlx.nn`, `mlx.optimizers`), pytest, hypothesis, numpy. No new dependencies (MLX already pinned `>=0.18.0`).

---

## File Structure

| File | Role | Action |
|------|------|--------|
| `docs/osf-prereg-g4-ter-pilot.md` | OSF pre-registration G4-ter | **Create** |
| `experiments/g4_ter_hp_sweep/__init__.py` | Package marker | **Create** (empty) |
| `experiments/g4_ter_hp_sweep/dream_wrap_hier.py` | `G4HierarchicalClassifier` + extended buffer | **Create** |
| `experiments/g4_ter_hp_sweep/hp_grid.py` | Curated HP-combo enumeration | **Create** |
| `experiments/g4_ter_hp_sweep/run_g4_ter.py` | Pilot driver (HP sweep + richer-substrate sweep) | **Create** |
| `tests/unit/experiments/test_g4_ter_hier.py` | `G4HierarchicalClassifier` unit tests | **Create** |
| `tests/unit/experiments/test_g4_ter_hp_grid.py` | HP grid enumeration tests | **Create** |
| `tests/unit/experiments/test_g4_ter_driver.py` | Driver smoke + verdict tests | **Create** |
| `tests/conformance/axioms/test_dr3_g4_hierarchical_substrate.py` | DR-3 conformance for the richer substrate | **Create** |
| `docs/milestones/g4-ter-pilot-2026-05-03.md` | Milestone (md) | **Create via run** |
| `docs/milestones/g4-ter-pilot-2026-05-03.json` | Milestone (json) | **Create via run** |
| `docs/papers/paper2/results.md` | Paper 2 EN narrative | Modify (add §7.1.5 G4-ter) |
| `docs/papers/paper2-fr/results.md` | Paper 2 FR narrative | Modify (add §7.1.5 G4-ter FR mirror) |
| `CHANGELOG.md` | DualVer log | Modify (Empirical bullet under `[Unreleased]`) |
| `STATUS.md` | Gate row + DualVer | Modify (G4 row reflects G4-ter verdict) |

The `experiments/g4_ter_hp_sweep/` directory is a sibling of `experiments/g4_split_fmnist/` — same pattern as the G4-bis layout, no relocation. The existing `dream_wrap.py` and `run_g4.py` stay untouched: G4-ter is additive.

---

## Constraints honored

- **DR-0 accountability**: every dispatched op still appends one `EpisodeLogEntry` via `runtime.execute(...)`. The new RESTRUCTURE / RECOMBINE coupling fires *after* runtime dispatch, side-by-side with the spectator handlers (same pattern as G4-bis).
- **DR-3 Conformance Criterion**: the richer substrate must satisfy conditions (1)–(3). New conformance test asserts the hierarchical classifier exposes the typed Protocols correctly and that primitives chain without raising.
- **R1 determinism**: every RNG path uses `np.random.default_rng(seed + offset)` (offsets disjoint across replay/restructure/recombine sites). MLX SGD is deterministic. Fresh `run_id`s register against `(c_version, "g4-ter/<arm>/<hp_combo_id>", seed, commit_sha)`.
- **N=30 power floor**: with N=30 vs N=30, minimum detectable Hedges' g at 80 % power, α = 0.0125 one-sided (Bonferroni × 4 arms — see §1.3 of the OSF pre-reg) is ~0.6. Effects below g ≈ 0.3 remain exploratory; the pre-reg locks this caveat.
- **No FC bump**: pure calibration + implementation; axiom signatures, primitive Protocols, and channel sets unchanged. EC stays PARTIAL or shifts per observed verdict (Task 13 conditional bump).
- **OSF pre-reg cites G4-bis findings as baseline** (parent registration `10.17605/OSF.IO/Q6JYN`).
- **EN→FR mirror**: any change to `docs/papers/paper2/results.md` updates `docs/papers/paper2-fr/results.md` in the same PR.
- **Commit policy**: ≤50-char subject, ≥3-char scope, body ≤72-char wrap, no `Co-Authored-By` trailer.

---

## Read-first context

Before starting, the executing engineer must skim:
- `experiments/g4_split_fmnist/dream_wrap.py` — existing `G4Classifier`, `BetaBufferFIFO`, `dream_episode()` coupling pattern.
- `experiments/g4_split_fmnist/run_g4.py` — driver pattern (per-cell pipeline, RunRegistry call, Markdown report).
- `kiki_oniric/profiles/p_min.py`, `kiki_oniric/profiles/p_equ.py`, `kiki_oniric/profiles/p_max.py` — profile op-set construction.
- `kiki_oniric/dream/episode.py` — `Operation` enum (REPLAY, DOWNSCALE, RESTRUCTURE, RECOMBINE) and `OutputChannel` enum.
- `docs/osf-prereg-g4-pilot.md` — pre-registration template (G4-ter mirrors §1–§9 layout with H1/H2/H3 substituted for H1/H3/H_DR4).
- `docs/milestones/g4-pilot-2026-05-03-bis.md` — G4-bis null finding + degeneracy diagnostic.
- `harness/storage/run_registry.py:113-124` — `RunRegistry.register(c_version, profile, seed, commit_sha) -> run_id`.
- `kiki_oniric.eval.statistics` — `compute_hedges_g`, `welch_one_sided`, `jonckheere_trend`.

---

## Decision log

### Task 0.5 — Option A vs B vs C — **Locked: Option C**

Three execution profiles were considered:

- **Option A** (HP sweep only): ~10 HP combos × N=30 × 4 arms = 1 200 cells but **all on the binary MLP head**. Cannot distinguish H2 (substrate limitation): if HP sweep fails to recover g, we cannot tell whether richer head would have succeeded.
- **Option B** (richer substrate only): 1 HP combo × N=30 × 4 arms × richer head = 120 cells. Cannot distinguish H1 (HP artefact): if richer head fails, we cannot tell whether a different HP would have succeeded on the binary head.
- **Option C** (both, narrowed): 1 representative HP × N=30 seeds × 4 arms × richer head (120 cells) **plus** 10 HP combos × N=10 seeds × 3 dream arms × binary head (300 cells) = **420 cells total**. ~3-5 h on M1 Max (G4-bis ran 20 cells in 29 s ≈ 1.5 s/cell on binary head; richer head conservatively ×3 wall-clock). Best science-per-compute: pins both axes simultaneously.

**Locked: Option C.** If runtime overruns the 5 h budget at smoke-time on Task 9, the executing engineer splits the 10-HP grid into a 4-combo "high-likelihood subset" (`replay_lr ∈ {0.001, 0.05}` × `downscale_factor ∈ {0.85, 0.99}`) and registers the deferral as a deviation under `docs/osf-deviations-g4-ter-2026-05-03.md`.

---

## Task 0: Investigate current state (no commit)

**Files:**
- Read: `experiments/g4_split_fmnist/dream_wrap.py:181-442` (full `G4Classifier`)
- Read: `experiments/g4_split_fmnist/run_g4.py:165-238` (`_run_cell` body)
- Read: `kiki_oniric/profiles/p_equ.py:60-77` (op handler registration)
- Read: `kiki_oniric/profiles/p_max.py:78-117`
- Read: `docs/milestones/g4-pilot-2026-05-03-bis.md` (baseline finding)
- Read: `docs/osf-prereg-g4-pilot.md` (template structure)

- [ ] **Step 1: Confirm G4-bis equal-means is real**

Run: `grep -n "0.5609" docs/milestones/g4-pilot-2026-05-03-bis.md`
Expected: three occurrences (`mean_p_min`, `mean_p_equ`, `mean_p_max` — all identical), confirming the degenerate H_DR4 tie is the diagnostic motivating G4-ter.

- [ ] **Step 2: Confirm RESTRUCTURE/RECOMBINE remain spectator-only on G4Classifier**

Run: `grep -n "RESTRUCTURE\|RECOMBINE" experiments/g4_split_fmnist/dream_wrap.py`
Expected: matches only inside the `ops` tuple construction and the `Operation.RESTRUCTURE` / `Operation.RECOMBINE` log-line docstring (no method named `_restructure_step` or `_recombine_step`); confirming there is no coupling site to mutate weights in response to those ops.

- [ ] **Step 3: Confirm `BetaRecord.latent` does not exist yet**

Run: `grep -n "latent" experiments/g4_split_fmnist/dream_wrap.py | head -5`
Expected: zero matches (current `BetaRecord` only carries `x: list[float]` and `y: int`).

No commit. This task is observation-only.

---

## Task 1: Draft G4-ter OSF pre-registration

**Files:**
- Create: `docs/osf-prereg-g4-ter-pilot.md`

The pre-reg locks H1/H2/H3 hypotheses, the HP grid, the power analysis, and the DualVer outcome rules. Append-only, dated immutable per `docs/CLAUDE.md` rules.

- [ ] **Step 1: Write the pre-reg file**

Create `docs/osf-prereg-g4-ter-pilot.md` with the complete content below.

```markdown
# OSF Pre-Registration — G4-ter pilot (HP sweep + richer substrate)

**Project** : dreamOfkiki
**Parent registration** : 10.17605/OSF.IO/Q6JYN (Cycle 1)
**Amendment** : G4-ter pilot — confirmatory follow-up to G4-bis null
  finding (`g_h1 = -2.31`, `H_DR4` degenerate equal-means)
**PI** : Clement Saillant (L'Electron Rare)
**Date drafted** : 2026-05-03
**Lock target** : before any G4-ter run is registered in
  `harness/storage/run_registry.RunRegistry`

## 1. Background — G4-bis baseline

The G4-bis pilot (milestone `docs/milestones/g4-pilot-2026-05-03-bis.md`)
re-ran G4 after wiring `dream_episode()` to mutate classifier weights
via the REPLAY + DOWNSCALE channels. It produced:

- `g_h1 = -2.3067` (sign-reversed relative to Hu 2020 anchor `g >= 0.21`)
- `H_DR4` degenerate equal means: `mean retention[P_min] =
  mean retention[P_equ] = mean retention[P_max] = 0.5609`

Diagnostic in §H_DR4 of the G4-bis milestone identifies the binary
MLP head as exposing only REPLAY + DOWNSCALE; RESTRUCTURE and
RECOMBINE remain spectator-only (no hierarchy nor VAE latents). The
identical retention vectors across P_min/P_equ/P_max are therefore
mechanically identical, not substantively monotonic.

G4-ter is the confirmatory N≥30 follow-up scheduled by the
exploratory-positive-evidence rule of `osf-prereg-g4-pilot.md` §4.

## 2. Hypotheses

### H1 — HP artefact

**Statement** : observed `g_h1 = -2.31` is an artefact of the
G4-bis HP combo (`replay_lr=0.01`, `replay_n_steps=1`,
`replay_n_records=32`, `downscale_factor=0.95`). A curated 10-combo
HP grid sweep on the binary MLP head yields at least one combo
with `g_h1 >= 0` (sign reversed back to the Hu 2020 anchor's side)
on N=10 seeds.

**Operationalization** :
- `g_hp_best = max(compute_hedges_g(retention[P_equ, hp_i],
  retention[baseline]) for hp_i in HP_GRID)`
- Reject H0_HP iff `g_hp_best >= 0.0`
- Statistical test : Welch's one-sided t-test
  `(retention[baseline], retention[P_equ, hp_best])` at
  α = 0.05 / 4 = 0.0125 (Bonferroni for 4 hypothesis families:
  H1, H2, H3, H_DR4)

### H2 — Substrate-level limitation

**Statement** : observed `g_h1 = -2.31` is a substrate-level
limitation of the binary MLP head. A hierarchical MLP head that
exposes RESTRUCTURE (perturbation of hidden_2 weights) and
RECOMBINE (Gaussian-MoG synthetic-latent injection) yields
`g_h1 >= 0.0` on N=30 seeds with a single representative HP combo.

**Operationalization** :
- `g_h2 = compute_hedges_g(retention[P_equ, richer], retention[baseline, richer])`
- Reject H0_substrate iff `g_h2 >= 0.0`
- Statistical test : Welch's one-sided t-test
  `(retention[baseline, richer], retention[P_equ, richer])`
  at α = 0.0125

### H3 — Combined effect

**Statement** : neither HP sweep nor richer substrate alone
recovers the Hu 2020 anchor floor `g >= 0.21`, but the combination
(richer substrate + best HP combo) does. Validated by a tertiary
non-pre-registered exploratory cell.

**Operationalization** : exploratory only. Reject H0_combined iff
`g_h2 >= 0.21` and `g_hp_best >= 0.21` are *both* false but a
post-hoc richer-substrate × `hp_best` cell yields `g >= 0.21`. The
exploratory status is recorded in the milestone; it does **not**
trigger an EC bump on its own.

### H_DR4-ter — Monotonicity recovered on richer substrate

**Statement** : on the richer substrate at N=30 seeds, mean
retention is monotonically ordered `P_max >= P_equ >= P_min` and
the Jonckheere-Terpstra trend test rejects H0 at α = 0.0125. This
re-tests the H_DR4 monotonicity hypothesis after the degenerate
G4-bis tie is structurally broken by the richer head.

**Operationalization** :
- `mean_retention[P_max] >= mean_retention[P_equ] >= mean_retention[P_min]`
- Statistical test : `kiki_oniric.eval.statistics.jonckheere_trend`
  on the three retention groups at α = 0.0125

## 3. HP grid (curated 10 combos)

Grid : 4 × 3 × 3 × 4 = 144 candidates → curated to 10 along the
qualitative gradient hypothesised most likely to flip the sign of
g_h1. Listed in `experiments/g4_ter_hp_sweep/hp_grid.py`.

| combo_id | downscale_factor | replay_batch | replay_n_steps | replay_lr |
|----------|------------------|--------------|----------------|-----------|
| C0       | 0.85             | 16           | 1              | 0.001     |
| C1       | 0.85             | 32           | 5              | 0.001     |
| C2       | 0.90             | 32           | 1              | 0.001     |
| C3       | 0.90             | 32           | 5              | 0.01      |
| C4       | 0.95             | 32           | 1              | 0.001     |
| C5       | 0.95             | 32           | 5              | 0.01      |
| C6       | 0.95             | 64           | 10             | 0.001     |
| C7       | 0.99             | 16           | 1              | 0.001     |
| C8       | 0.99             | 32           | 5              | 0.01      |
| C9       | 0.99             | 64           | 10             | 0.05      |

Rationale: each axis probes a separate G4-bis suspect — SHY
over-shrinkage (`downscale_factor`), replay/downscale balance
(`replay_lr`), insufficient consolidation (`replay_n_steps`),
under-sampled replay (`replay_batch`). C5 is the G4-bis-aligned
anchor and the representative combo for the richer-substrate sweep
(only `replay_n_steps` differs : 1 → 5).

## 4. Sample size / power

- HP sub-grid : N=10 seeds × 3 dream arms (P_min, P_equ, P_max) ×
  10 combos = 300 cells. Baseline arm is **not** swept across
  combos (HP changes do not affect baseline) — its 10 cells are
  taken from the richer-substrate sweep below.
- Richer-substrate sweep : N=30 seeds × 4 arms (baseline, P_min,
  P_equ, P_max) × 1 combo (C5) = 120 cells.
- Total : 420 cells.
- Power floor : N=30 vs N=30, minimum detectable Hedges' g at
  80 % power, α = 0.0125 (Bonferroni × 4 hypothesis families)
  one-sided is ~0.6. Effects below g ≈ 0.3 remain exploratory
  and trigger a deferred N≥95 confirmatory follow-up.
- HP sub-grid power : N=10 vs N=10, min detectable g ≈ 1.0 — HP
  sweep is **screening**, not confirmatory. A `g_hp_best >= 0.0`
  outcome triggers a confirmatory N=30 sweep on `hp_best` only,
  scheduled as G4-quater.

## 5. Pre-specified analyses

- H1, H2, H_DR4-ter : `kiki_oniric.eval.statistics.welch_one_sided`
  + Hedges' g via `compute_hedges_g`.
- H_DR4-ter trend : `kiki_oniric.eval.statistics.jonckheere_trend`
  on the three retention groups in `[P_min, P_equ, P_max]` order
  at α = 0.0125.
- Multiple-comparison correction : Bonferroni at family size 4
  (H1, H2, H3, H_DR4-ter), α_per_test = 0.0125. H3 is exploratory
  and excluded from the inferential family — it inherits the
  same α/4 anyway by construction.

## 6. Data exclusion rules

- Cells where the substrate raises any BLOCKING invariant (S1
  retained-non-regression, S2 finite weights) are excluded from
  H1/H2/H_DR4-ter and logged with `excluded=true`.
- Cells with `acc_task1_initial < 0.5` are excluded as
  underperforming-baseline (same rule as G4-bis).
- Cells where `dream_episode_hier()` exits with NotImplementedError
  surface as a *plan* failure, not a data issue.

## 7. DualVer outcome rules (binding)

| Outcome | EC bump | Rationale |
|---------|---------|-----------|
| H_DR4-ter rejected H0 in predicted direction (Jonckheere monotonic, p < 0.0125) **and** H1 or H2 rejected H0 | PARTIAL → STABLE | Empirical confirmation crosses §12.3 STABLE bar for G4 scope |
| H_DR4-ter inconclusive **or** all of H1/H2 inconclusive | stays PARTIAL | Partial confirmation, schedule N≥95 G4-quater |
| H_DR4-ter falsified (Jonckheere reverses: P_min > P_max statistically) | PARTIAL → UNSTABLE | §12.3 transition rule on falsification |

No FC bump in any outcome (no axiom or primitive change).

## 8. Deviations from pre-registration

Any post-hoc deviation will be documented in
`docs/osf-deviations-g4-ter-<date>.md` (separate file, dated
immutable). Deviations include : seed-count change, statistical
test substitution, exclusion-rule relaxation, HP grid pruning at
smoke-time (allowed only via the 4-combo subset specified in the
plan §"Decision log").

## 9. Data and code availability

- Pre-reg : this file, locked at `git rev-parse HEAD` before
  the first run-registry insert.
- Pilot driver : `experiments/g4_ter_hp_sweep/run_g4_ter.py`
- Effect-size helpers : `kiki_oniric.eval.statistics.{compute_hedges_g, welch_one_sided, jonckheere_trend}`
- Verdict anchors : `harness.benchmarks.effect_size_targets.{HU_2020_OVERALL, JAVADI_2024_OVERALL}`
- Run registry : `harness/storage/run_registry.RunRegistry`,
  SQLite at `.run_registry.sqlite`
- Outcome dump : `docs/milestones/g4-ter-pilot-2026-05-03.{json,md}`

## 10. Contact

Clement Saillant — clement@saillant.cc — L'Electron Rare, France

---

**Lock this document before any G4-ter cell is registered in the
run registry.**
```

- [ ] **Step 2: Verify the pre-reg renders**

Run: `wc -l docs/osf-prereg-g4-ter-pilot.md`
Expected: ~180 lines (Markdown only, no code execution required).

- [ ] **Step 3: Commit**

```bash
git add docs/osf-prereg-g4-ter-pilot.md
git commit -m "docs(osf): G4-ter pilot pre-registration draft

Locks H1 (HP artefact), H2 (substrate limitation), H3 (combined),
H_DR4-ter on richer substrate. Cites G4-bis baseline finding
(g_h1 = -2.31, H_DR4 equal-means) under parent OSF Q6JYN.

420 cells total: 10-combo HP sub-grid (300) + N=30 richer
substrate (120). Bonferroni alpha/4 = 0.0125."
```

---

## Task 2: `G4HierarchicalClassifier` — multi-layer head (TDD)

**Files:**
- Create: `experiments/g4_ter_hp_sweep/__init__.py` (empty marker)
- Create: `experiments/g4_ter_hp_sweep/dream_wrap_hier.py`
- Create: `tests/unit/experiments/test_g4_ter_hier.py`

The hierarchical head exposes three Linear layers so RESTRUCTURE can
target hidden_2 weights without disturbing the input projection or
the output classifier.

- [ ] **Step 1: Write the empty package marker**

```python
# experiments/g4_ter_hp_sweep/__init__.py
"""G4-ter pilot — HP sweep + richer-substrate confirmatory pilot.

Plan: docs/superpowers/plans/2026-05-03-g4-ter-hp-sweep-richer-substrate.md
Pre-reg: docs/osf-prereg-g4-ter-pilot.md
"""
```

- [ ] **Step 2: Write the failing classifier-shape test**

Create `tests/unit/experiments/test_g4_ter_hier.py`:

```python
"""Unit tests for the G4HierarchicalClassifier (Plan G4-ter Task 2)."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.g4_ter_hp_sweep.dream_wrap_hier import (
    G4HierarchicalClassifier,
)


def test_classifier_has_three_linear_layers() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=784, hidden_1=32, hidden_2=16, n_classes=2, seed=7
    )
    # Public attributes for RESTRUCTURE site identification.
    assert clf.hidden_1 == 32
    assert clf.hidden_2 == 16
    assert clf.n_classes == 2


def test_predict_logits_shape() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=784, hidden_1=32, hidden_2=16, n_classes=2, seed=7
    )
    x = np.zeros((4, 784), dtype=np.float32)
    logits = clf.predict_logits(x)
    assert logits.shape == (4, 2)


def test_seed_determinism() -> None:
    a = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=42
    )
    b = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=42
    )
    x = np.ones((1, 10), dtype=np.float32)
    np.testing.assert_array_equal(a.predict_logits(x), b.predict_logits(x))
```

- [ ] **Step 3: Run the failing test**

Run: `uv run pytest tests/unit/experiments/test_g4_ter_hier.py -v`
Expected: 3 FAIL — `ModuleNotFoundError: experiments.g4_ter_hp_sweep.dream_wrap_hier`.

- [ ] **Step 4: Implement the minimal classifier**

Create `experiments/g4_ter_hp_sweep/dream_wrap_hier.py`:

```python
"""Hierarchical MLX MLP classifier + dream-episode wrapper for G4-ter.

Architecture: input → Linear(in_dim, hidden_1) → ReLU →
Linear(hidden_1, hidden_2) → ReLU → Linear(hidden_2, n_classes).

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
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §3.1
    docs/osf-prereg-g4-ter-pilot.md §2-§3
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

    Layers : Linear(in_dim, hidden_1) → ReLU → Linear(hidden_1,
    hidden_2) → ReLU → Linear(hidden_2, n_classes). Deterministic
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

        Used by the β buffer to capture per-record latents at push
        time for the RECOMBINE Gaussian-MoG sampler.
        """
        h1 = nn.relu(self._l1(mx.array(x)))
        h2 = nn.relu(self._l2(h1))
        mx.eval(h2)
        return np.asarray(h2)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/unit/experiments/test_g4_ter_hier.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add experiments/g4_ter_hp_sweep/__init__.py \
        experiments/g4_ter_hp_sweep/dream_wrap_hier.py \
        tests/unit/experiments/test_g4_ter_hier.py
git commit -m "feat(g4-ter): add hierarchical MLP classifier

Three-layer MLP (in→32→16→out) exposes hidden_2 weights as a
RESTRUCTURE target and hidden_2 activations as a RECOMBINE
latent-sampling source. Deterministic per seed (mx.random.seed).
"
```

---

## Task 3: Train + eval methods + extended β-buffer (TDD)

**Files:**
- Modify: `experiments/g4_ter_hp_sweep/dream_wrap_hier.py`
- Modify: `tests/unit/experiments/test_g4_ter_hier.py`

The classifier needs `train_task` (cross-entropy SGD) and
`eval_accuracy`, mirroring the binary G4 head. The β buffer is
extended with a per-record latent slot.

- [ ] **Step 1: Write failing tests for train/eval/buffer**

Append to `tests/unit/experiments/test_g4_ter_hier.py`:

```python
from experiments.g4_ter_hp_sweep.dream_wrap_hier import (
    BetaBufferHierFIFO,
)


def _toy_task(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 64
    x = rng.standard_normal((n, 10)).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.int32)
    return {
        "x_train": x[:48],
        "y_train": y[:48],
        "x_test": x[48:],
        "y_test": y[48:],
    }


def test_train_task_then_eval_accuracy() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=10, hidden_1=32, hidden_2=16, n_classes=2, seed=11
    )
    task = _toy_task(seed=11)
    clf.train_task(task, epochs=2, batch_size=16, lr=0.05)
    acc = clf.eval_accuracy(task["x_test"], task["y_test"])
    assert 0.0 <= acc <= 1.0


def test_buffer_push_pop_with_latent() -> None:
    buf = BetaBufferHierFIFO(capacity=4)
    x = np.zeros(10, dtype=np.float32)
    latent = np.ones(16, dtype=np.float32)
    buf.push(x=x, y=1, latent=latent)
    snap = buf.snapshot()
    assert len(snap) == 1
    assert snap[0]["y"] == 1
    assert snap[0]["latent"] == [1.0] * 16


def test_buffer_sample_deterministic() -> None:
    buf = BetaBufferHierFIFO(capacity=8)
    for i in range(8):
        buf.push(
            x=np.full(4, float(i), dtype=np.float32),
            y=i % 2,
            latent=np.full(2, float(i), dtype=np.float32),
        )
    a = buf.sample(n=3, seed=42)
    b = buf.sample(n=3, seed=42)
    assert [r["x"] for r in a] == [r["x"] for r in b]


def test_buffer_sample_no_latent() -> None:
    """latent=None records are still sampleable (legacy compat)."""
    buf = BetaBufferHierFIFO(capacity=2)
    buf.push(x=np.zeros(2, dtype=np.float32), y=0, latent=None)
    snap = buf.snapshot()
    assert snap[0]["latent"] is None
```

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest tests/unit/experiments/test_g4_ter_hier.py -v`
Expected: 4 NEW failures — `train_task`, `eval_accuracy`,
`BetaBufferHierFIFO` not implemented.

- [ ] **Step 3: Implement train/eval and the extended buffer**

Append to `experiments/g4_ter_hp_sweep/dream_wrap_hier.py`:

```python
class BetaBufferHierFIFO:
    """Bounded curated episodic buffer with optional latents (β channel).

    FIFO eviction at capacity. Compared to ``BetaBufferFIFO`` (G4-bis),
    each record carries an optional ``latent`` field — used by the
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
```

Then add `train_task` and `eval_accuracy` to
`G4HierarchicalClassifier`:

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/experiments/test_g4_ter_hier.py -v`
Expected: 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/g4_ter_hp_sweep/dream_wrap_hier.py \
        tests/unit/experiments/test_g4_ter_hier.py
git commit -m "feat(g4-ter): add train_task + eval + hier buffer

train_task uses MLX SGD with deterministic numpy permutation per
seed. BetaBufferHierFIFO carries optional latent vectors for the
RECOMBINE Gaussian-MoG sampler.
"
```

---

## Task 4: `_restructure_step` — perturb hidden_2 weights (TDD)

**Files:**
- Modify: `experiments/g4_ter_hp_sweep/dream_wrap_hier.py`
- Modify: `tests/unit/experiments/test_g4_ter_hier.py`

RESTRUCTURE adds Gaussian noise of magnitude `factor * sigma` to the
`_l2.weight` tensor only (input projection and output classifier
preserved). `factor` defaults to `0.05` (5 % per-episode drift,
qualitatively matched to the SHY downscale factor's 5 % shrinkage).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/experiments/test_g4_ter_hier.py`:

```python
def test_restructure_step_modifies_only_hidden_2() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    w1_before = np.asarray(clf._l1.weight)
    w2_before = np.asarray(clf._l2.weight)
    w3_before = np.asarray(clf._l3.weight)
    clf._restructure_step(factor=0.05, seed=99)
    w1_after = np.asarray(clf._l1.weight)
    w2_after = np.asarray(clf._l2.weight)
    w3_after = np.asarray(clf._l3.weight)
    # Input + output untouched.
    np.testing.assert_array_equal(w1_before, w1_after)
    np.testing.assert_array_equal(w3_before, w3_after)
    # Hidden_2 perturbed (probabilistically: any element changed).
    assert not np.array_equal(w2_before, w2_after)


def test_restructure_step_factor_zero_is_noop() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    w2_before = np.asarray(clf._l2.weight)
    clf._restructure_step(factor=0.0, seed=99)
    w2_after = np.asarray(clf._l2.weight)
    np.testing.assert_array_equal(w2_before, w2_after)


def test_restructure_step_factor_negative_raises() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    with pytest.raises(ValueError, match="factor must be"):
        clf._restructure_step(factor=-0.01, seed=99)


def test_restructure_step_seed_determinism() -> None:
    a = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    b = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    a._restructure_step(factor=0.05, seed=99)
    b._restructure_step(factor=0.05, seed=99)
    np.testing.assert_array_equal(
        np.asarray(a._l2.weight), np.asarray(b._l2.weight)
    )
```

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest tests/unit/experiments/test_g4_ter_hier.py::test_restructure_step_modifies_only_hidden_2 -v`
Expected: FAIL — `_restructure_step` not defined.

- [ ] **Step 3: Implement `_restructure_step`**

Append to `G4HierarchicalClassifier`:

```python
    def _restructure_step(self, *, factor: float, seed: int) -> None:
        """Add ``factor * sigma * N(0, 1)`` to hidden_2 weights only.

        ``sigma`` is the per-tensor standard deviation of
        ``self._l2.weight`` at call time. ``factor=0`` is a no-op
        (early exit). ``factor < 0`` raises ValueError. Determinism
        is provided by ``np.random.default_rng(seed)``.

        Hierarchy / RESTRUCTURE channel rationale: only the *middle*
        hidden layer is mutated, so the input projection and output
        classifier are preserved. This matches the framework-C §3.1
        H_DR4 monotonicity intuition that RESTRUCTURE escalates
        framework C's representational rearrangement without
        wholesale weight collapse.
        """
        if factor < 0.0:
            raise ValueError(
                f"factor must be non-negative, got {factor}"
            )
        if factor == 0.0:
            return
        w = np.asarray(self._l2.weight)
        sigma = float(w.std()) if w.size > 0 else 0.0
        if sigma == 0.0:
            return
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(size=w.shape).astype(np.float32)
        new_w = w + factor * sigma * noise
        self._l2.weight = mx.array(new_w)
        mx.eval(self._l2.weight)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/experiments/test_g4_ter_hier.py -v`
Expected: 11 PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/g4_ter_hp_sweep/dream_wrap_hier.py \
        tests/unit/experiments/test_g4_ter_hier.py
git commit -m "feat(g4-ter): add _restructure_step on hidden_2

Adds factor * sigma * N(0,1) perturbation to _l2.weight only.
Input projection and output classifier are preserved. Seeded RNG
yields bit-stable mutations under R1.
"
```

---

## Task 5: `_recombine_step` — Gaussian-MoG synthetic latent injection (TDD)

**Files:**
- Modify: `experiments/g4_ter_hp_sweep/dream_wrap_hier.py`
- Modify: `tests/unit/experiments/test_g4_ter_hier.py`

RECOMBINE samples `n_synthetic` synthetic latents from a Gaussian
mixture-of-Gaussians fitted to the buffered hidden_2 activations
(per-class component), then runs one CE-loss SGD pass over
`(synthetic_latent, label)` pairs through the *output classifier
only* (`self._l3`).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/experiments/test_g4_ter_hier.py`:

```python
def test_recombine_step_with_empty_buffer_is_noop() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    w3_before = np.asarray(clf._l3.weight)
    clf._recombine_step(
        latents=[], n_synthetic=5, lr=0.01, seed=42
    )
    w3_after = np.asarray(clf._l3.weight)
    np.testing.assert_array_equal(w3_before, w3_after)


def test_recombine_step_modifies_l3_only() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    # Synthetic latents indexed by class: list of (latent, label).
    latents = [
        ([0.5, 0.5, 0.5], 0),
        ([0.6, 0.4, 0.5], 0),
        ([-0.3, -0.4, -0.5], 1),
        ([-0.4, -0.5, -0.6], 1),
    ]
    w1_before = np.asarray(clf._l1.weight)
    w2_before = np.asarray(clf._l2.weight)
    w3_before = np.asarray(clf._l3.weight)
    clf._recombine_step(
        latents=latents, n_synthetic=4, lr=0.1, seed=7
    )
    np.testing.assert_array_equal(w1_before, np.asarray(clf._l1.weight))
    np.testing.assert_array_equal(w2_before, np.asarray(clf._l2.weight))
    assert not np.array_equal(w3_before, np.asarray(clf._l3.weight))


def test_recombine_step_seed_determinism() -> None:
    latents = [
        ([0.5, 0.5, 0.5], 0),
        ([-0.5, -0.5, -0.5], 1),
    ] * 4
    a = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    b = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    a._recombine_step(latents=latents, n_synthetic=8, lr=0.1, seed=7)
    b._recombine_step(latents=latents, n_synthetic=8, lr=0.1, seed=7)
    np.testing.assert_array_equal(
        np.asarray(a._l3.weight), np.asarray(b._l3.weight)
    )
```

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest tests/unit/experiments/test_g4_ter_hier.py -v`
Expected: 3 NEW failures — `_recombine_step` not defined.

- [ ] **Step 3: Implement `_recombine_step`**

Append to `G4HierarchicalClassifier`:

```python
    def _recombine_step(
        self,
        *,
        latents: list[tuple[list[float], int]],
        n_synthetic: int,
        lr: float,
        seed: int,
    ) -> None:
        """Sample ``n_synthetic`` synthetic latents from a per-class
        Gaussian-MoG and run one CE-loss SGD pass through ``_l3`` only.

        ``latents`` is a list of ``(latent_vector, class_label)``
        pairs accumulated from past tasks via ``BetaBufferHierFIFO``.
        Per-class component means/std are estimated empirically;
        synthetic samples are drawn from N(mean_c, std_c) and labeled
        by ``c``. The forward pass uses ``self._l3`` only, so the
        gradient flows into the output classifier weights — leaving
        ``_l1`` / ``_l2`` untouched.

        Empty ``latents`` → no-op (S1-trivial branch). Single-class
        ``latents`` → no-op (degenerate MoG, no recombination signal).
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

        x = mx.array(np.stack(synth_x).astype(np.float32))
        y = mx.array(np.asarray(synth_y, dtype=np.int32))

        opt = optim.SGD(learning_rate=lr)

        def loss_fn(layer: nn.Linear, xb: mx.array, yb: mx.array) -> mx.array:
            return nn.losses.cross_entropy(layer(xb), yb, reduction="mean")

        loss_and_grad = nn.value_and_grad(self._l3, loss_fn)
        _loss, grads = loss_and_grad(self._l3, x, y)
        opt.update(self._l3, grads)
        mx.eval(self._l3.parameters(), opt.state)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/experiments/test_g4_ter_hier.py -v`
Expected: 14 PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/g4_ter_hp_sweep/dream_wrap_hier.py \
        tests/unit/experiments/test_g4_ter_hier.py
git commit -m "feat(g4-ter): add _recombine_step Gaussian MoG

Per-class Gaussian sampling on buffered latents drives one CE-loss
SGD pass through _l3 only. Empty / single-class buffer = no-op.
Determinism via np.random.default_rng(seed).
"
```

---

## Task 6: Wire 4 ops into `dream_episode_hier()` (TDD)

**Files:**
- Modify: `experiments/g4_ter_hp_sweep/dream_wrap_hier.py`
- Modify: `tests/unit/experiments/test_g4_ter_hier.py`

`dream_episode_hier` mirrors the G4-bis `dream_episode` but exposes
the four operations on the new substrate. P_min remains REPLAY +
DOWNSCALE only; P_equ and P_max additionally fire RESTRUCTURE +
RECOMBINE.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/experiments/test_g4_ter_hier.py`:

```python
from experiments.g4_split_fmnist.dream_wrap import build_profile


def _fill_buffer(buf: BetaBufferHierFIFO, clf: G4HierarchicalClassifier,
                 n_per_class: int = 6) -> None:
    rng = np.random.default_rng(0)
    for cls in (0, 1):
        for _ in range(n_per_class):
            x = rng.standard_normal(10).astype(np.float32)
            latent = clf.latent(x[None, :])[0]
            buf.push(x=x, y=cls, latent=latent)


def test_dream_episode_hier_p_min_runs() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    buf = BetaBufferHierFIFO(capacity=32)
    _fill_buffer(buf, clf)
    profile = build_profile("P_min", seed=13)
    clf.dream_episode_hier(
        profile,
        seed=13,
        beta_buffer=buf,
        replay_n_records=8,
        replay_n_steps=1,
        replay_lr=0.01,
        downscale_factor=0.95,
        restructure_factor=0.05,
        recombine_n_synthetic=4,
        recombine_lr=0.01,
    )
    assert len(profile.runtime.log) == 1


def test_dream_episode_hier_p_equ_mutates_l2() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    buf = BetaBufferHierFIFO(capacity=32)
    _fill_buffer(buf, clf)
    w2_before = np.asarray(clf._l2.weight)
    profile = build_profile("P_equ", seed=13)
    clf.dream_episode_hier(
        profile,
        seed=13,
        beta_buffer=buf,
        replay_n_records=8,
        replay_n_steps=1,
        replay_lr=0.01,
        downscale_factor=0.95,
        restructure_factor=0.05,
        recombine_n_synthetic=4,
        recombine_lr=0.01,
    )
    w2_after = np.asarray(clf._l2.weight)
    # P_equ runs RESTRUCTURE → _l2.weight changes (perturb + downscale).
    assert not np.array_equal(w2_before, w2_after)


def test_dream_episode_hier_p_min_does_not_perturb_l2_random() -> None:
    """P_min runs DOWNSCALE only on _l2 (deterministic factor *), no random
    perturbation. Two runs with same seed must be bit-identical."""
    a = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    b = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    buf_a = BetaBufferHierFIFO(capacity=32)
    buf_b = BetaBufferHierFIFO(capacity=32)
    _fill_buffer(buf_a, a)
    _fill_buffer(buf_b, b)
    pa = build_profile("P_min", seed=13)
    pb = build_profile("P_min", seed=13)
    a.dream_episode_hier(
        pa, seed=13, beta_buffer=buf_a,
        replay_n_records=8, replay_n_steps=1, replay_lr=0.01,
        downscale_factor=0.95, restructure_factor=0.05,
        recombine_n_synthetic=4, recombine_lr=0.01,
    )
    b.dream_episode_hier(
        pb, seed=13, beta_buffer=buf_b,
        replay_n_records=8, replay_n_steps=1, replay_lr=0.01,
        downscale_factor=0.95, restructure_factor=0.05,
        recombine_n_synthetic=4, recombine_lr=0.01,
    )
    np.testing.assert_array_equal(
        np.asarray(a._l2.weight), np.asarray(b._l2.weight)
    )
```

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest tests/unit/experiments/test_g4_ter_hier.py -v`
Expected: 3 NEW failures — `dream_episode_hier` not defined.

- [ ] **Step 3: Implement `dream_episode_hier`**

Append to `G4HierarchicalClassifier`:

```python
    def _replay_optimizer_step(
        self,
        records: list[BetaRecordHier],
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

    def _downscale_step(self, *, factor: float) -> None:
        """Multiply every weight + bias in ``self._model`` by ``factor``.

        Bounds : ``factor`` must lie in ``(0, 1]`` — same constraint
        as ``downscale_real_handler``.
        """
        if not (0.0 < factor <= 1.0):
            raise ValueError(
                f"shrink_factor must be in (0, 1], got {factor}"
            )
        for layer in (self._l1, self._l2, self._l3):
            layer.weight = layer.weight * factor
            if getattr(layer, "bias", None) is not None:
                layer.bias = layer.bias * factor
        mx.eval(self._model.parameters())

    def dream_episode_hier(
        self,
        profile: object,
        seed: int,
        *,
        beta_buffer: BetaBufferHierFIFO,
        replay_n_records: int,
        replay_n_steps: int,
        replay_lr: float,
        downscale_factor: float,
        restructure_factor: float,
        recombine_n_synthetic: int,
        recombine_lr: float,
    ) -> None:
        """Drive one DreamEpisode and mutate classifier weights for G4-ter.

        Coupling map (richer substrate):
        - ``Operation.REPLAY`` → ``_replay_optimizer_step`` over
          buffer sample of ``replay_n_records``.
        - ``Operation.DOWNSCALE`` → ``_downscale_step`` with
          ``downscale_factor``.
        - ``Operation.RESTRUCTURE`` → ``_restructure_step`` on
          ``_l2.weight`` with ``restructure_factor``.
        - ``Operation.RECOMBINE`` → ``_recombine_step`` over Gaussian-
          MoG synthetic latents drawn from ``beta_buffer``.

        DR-0 spectator runtime path is preserved (synthetic
        input_slice values), so every episode appends one
        EpisodeLogEntry to ``profile.runtime.log``.
        """
        # Local imports to keep module-level deps minimal.
        from kiki_oniric.dream.episode import (
            BudgetCap,
            DreamEpisode,
            EpisodeTrigger,
            Operation,
            OutputChannel,
        )
        from kiki_oniric.profiles.p_min import PMinProfile

        if isinstance(profile, PMinProfile):
            ops: tuple[Operation, ...] = (
                Operation.REPLAY,
                Operation.DOWNSCALE,
            )
            channels: tuple[OutputChannel, ...] = (
                OutputChannel.WEIGHT_DELTA,
            )
        else:
            ops = (
                Operation.REPLAY,
                Operation.DOWNSCALE,
                Operation.RESTRUCTURE,
                Operation.RECOMBINE,
            )
            channels = (
                OutputChannel.WEIGHT_DELTA,
                OutputChannel.HIERARCHY_CHG,
                OutputChannel.LATENT_SAMPLE,
            )

        # Spectator runtime path (DR-0 logging).
        rng = np.random.default_rng(seed + 10_000)
        synthetic_records = [
            {
                "x": rng.standard_normal(4).astype(np.float32).tolist(),
                "y": rng.standard_normal(4).astype(np.float32).tolist(),
            }
            for _ in range(4)
        ]
        delta_latents = [
            rng.standard_normal(4).astype(np.float32).tolist()
            for _ in range(2)
        ]
        episode = DreamEpisode(
            trigger=EpisodeTrigger.SCHEDULED,
            input_slice={
                "beta_records": synthetic_records,
                "shrink_factor": 0.99,
                "topo_op": "reroute",
                "swap_indices": [0, 1],
                "delta_latents": delta_latents,
            },
            operation_set=ops,
            output_channels=channels,
            budget=BudgetCap(
                flops=10_000_000, wall_time_s=10.0, energy_j=1.0
            ),
            episode_id=f"g4-ter-{type(profile).__name__}-seed{seed}",
        )
        profile.runtime.execute(episode)  # type: ignore[attr-defined]

        # ---- Plan G4-ter coupling on richer substrate ----
        if Operation.REPLAY in ops:
            sampled = beta_buffer.sample(n=replay_n_records, seed=seed)
            self._replay_optimizer_step(
                sampled, lr=replay_lr, n_steps=replay_n_steps
            )
        if Operation.DOWNSCALE in ops:
            self._downscale_step(factor=downscale_factor)
        if Operation.RESTRUCTURE in ops:
            self._restructure_step(
                factor=restructure_factor, seed=seed + 20_000
            )
        if Operation.RECOMBINE in ops:
            latents_records: list[tuple[list[float], int]] = []
            for r in beta_buffer.snapshot():
                lat = r["latent"]
                if lat is not None:
                    latents_records.append((list(lat), int(r["y"])))
            self._recombine_step(
                latents=latents_records,
                n_synthetic=recombine_n_synthetic,
                lr=recombine_lr,
                seed=seed + 30_000,
            )
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/experiments/test_g4_ter_hier.py -v`
Expected: 17 PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/g4_ter_hp_sweep/dream_wrap_hier.py \
        tests/unit/experiments/test_g4_ter_hier.py
git commit -m "feat(g4-ter): wire 4 ops into dream_episode_hier

P_min runs REPLAY+DOWNSCALE; P_equ/P_max additionally run
RESTRUCTURE on _l2 and RECOMBINE on Gaussian-MoG of buffered
latents. DR-0 spectator runtime path preserved.
"
```

---

## Task 7: HP grid enumeration (TDD)

**Files:**
- Create: `experiments/g4_ter_hp_sweep/hp_grid.py`
- Create: `tests/unit/experiments/test_g4_ter_hp_grid.py`

The 10 curated HP combos live in a frozen list keyed by
`combo_id` so the driver can iterate and the milestone can cite
them by stable identifier.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/experiments/test_g4_ter_hp_grid.py`:

```python
"""HP grid enumeration tests (Plan G4-ter Task 7)."""
from __future__ import annotations

from experiments.g4_ter_hp_sweep.hp_grid import HP_COMBOS, HPCombo


def test_hp_grid_has_10_combos() -> None:
    assert len(HP_COMBOS) == 10


def test_hp_combo_ids_are_unique_and_sequential() -> None:
    ids = [c.combo_id for c in HP_COMBOS]
    assert ids == [f"C{i}" for i in range(10)]


def test_hp_grid_includes_g4_bis_anchor() -> None:
    """C5 must be the G4-bis-aligned combo for richer-substrate sweep."""
    c5 = next(c for c in HP_COMBOS if c.combo_id == "C5")
    assert c5.downscale_factor == 0.95
    assert c5.replay_batch == 32
    assert c5.replay_n_steps == 5
    assert c5.replay_lr == 0.01


def test_hp_combo_is_frozen() -> None:
    """HPCombo dataclass must reject mutation (R1 contract)."""
    import dataclasses

    c = HP_COMBOS[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.replay_lr = 0.5  # type: ignore[misc]


import pytest  # noqa: E402  (placed at the bottom for the frozen test)
```

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest tests/unit/experiments/test_g4_ter_hp_grid.py -v`
Expected: 4 FAIL — `hp_grid` module does not exist.

- [ ] **Step 3: Implement the HP grid**

Create `experiments/g4_ter_hp_sweep/hp_grid.py`:

```python
"""Curated HP combo grid for the G4-ter pilot.

10 combos along the qualitative gradient hypothesised most likely
to flip the sign of g_h1. C5 is the G4-bis production calibration
anchor *except* for replay_n_steps (1 → 5).

Reference :
    docs/osf-prereg-g4-ter-pilot.md §3
    docs/superpowers/plans/2026-05-03-g4-ter-hp-sweep-richer-substrate.md
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HPCombo:
    """One curated point in the HP grid (R1 stable id)."""

    combo_id: str
    downscale_factor: float
    replay_batch: int
    replay_n_steps: int
    replay_lr: float


HP_COMBOS: tuple[HPCombo, ...] = (
    HPCombo("C0", 0.85, 16, 1, 0.001),
    HPCombo("C1", 0.85, 32, 5, 0.001),
    HPCombo("C2", 0.90, 32, 1, 0.001),
    HPCombo("C3", 0.90, 32, 5, 0.01),
    HPCombo("C4", 0.95, 32, 1, 0.001),
    HPCombo("C5", 0.95, 32, 5, 0.01),
    HPCombo("C6", 0.95, 64, 10, 0.001),
    HPCombo("C7", 0.99, 16, 1, 0.001),
    HPCombo("C8", 0.99, 32, 5, 0.01),
    HPCombo("C9", 0.99, 64, 10, 0.05),
)


def representative_combo() -> HPCombo:
    """Return the C5 anchor used for the richer-substrate sweep."""
    return next(c for c in HP_COMBOS if c.combo_id == "C5")
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/experiments/test_g4_ter_hp_grid.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/g4_ter_hp_sweep/hp_grid.py \
        tests/unit/experiments/test_g4_ter_hp_grid.py
git commit -m "feat(g4-ter): freeze 10-combo HP grid

C0..C9 curated along downscale_factor x replay_lr x replay_n_steps
x replay_batch axes. C5 = G4-bis anchor for richer-substrate sweep.
HPCombo dataclass frozen for R1 stability.
"
```

---

## Task 8: Driver `run_g4_ter.py` — HP sweep + richer-substrate sweep (TDD)

**Files:**
- Create: `experiments/g4_ter_hp_sweep/run_g4_ter.py`
- Create: `tests/unit/experiments/test_g4_ter_driver.py`

The driver reuses `experiments.g4_split_fmnist.dataset.load_split_fmnist_5tasks`
and `experiments.g4_split_fmnist.dream_wrap.{G4Classifier, BetaBufferFIFO,
build_profile}` for the HP sub-grid arm; uses the new
`G4HierarchicalClassifier` for the richer-substrate arm.

- [ ] **Step 1: Write a failing smoke test for the driver**

Create `tests/unit/experiments/test_g4_ter_driver.py`:

```python
"""Driver smoke tests for the G4-ter pilot (Plan Task 8)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.g4_ter_hp_sweep.run_g4_ter import run_pilot


def _toy_data_dir(tmp_path: Path) -> Path:
    """Drop the smoke-fixture FMNIST IDX gz files into ``tmp_path``.

    Reuses the same minimal fixture pattern as the G4-bis driver tests.
    """
    # Mirror what experiments/g4_split_fmnist/dataset.py expects.
    rng = np.random.default_rng(0)
    # 50 train + 20 test images per class, 28×28 = 784 features.
    # The dataset loader expands the IDX format; here we shortcut by
    # directly writing the synthetic .npz fallback the loader respects
    # under DREAMOFKIKI_FMNIST_FAKE=1.
    fake_dir = tmp_path / "fmnist"
    fake_dir.mkdir()
    np.savez(
        fake_dir / "fake.npz",
        x=rng.standard_normal((140, 784)).astype(np.float32),
        y=rng.integers(0, 10, size=140).astype(np.int32),
    )
    return fake_dir


@pytest.mark.usefixtures("monkeypatch")
def test_run_pilot_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke run: 1 HP combo (C5) × 2 seeds × 4 arms = 8 cells, plus 1
    HP sweep cell per non-baseline arm = 3 cells. Total = 11 cells in
    < 30 s. Confirms the pilot writes both JSON and Markdown."""
    monkeypatch.setenv("DREAMOFKIKI_FMNIST_FAKE", "1")
    data_dir = _toy_data_dir(tmp_path)
    out_json = tmp_path / "g4-ter.json"
    out_md = tmp_path / "g4-ter.md"
    registry_db = tmp_path / "registry.sqlite"

    payload = run_pilot(
        data_dir=data_dir,
        seeds_richer=(0, 1),
        seeds_hp=(0,),
        hp_combo_ids=("C5",),
        out_json=out_json,
        out_md=out_md,
        registry_db=registry_db,
        epochs=1,
        batch_size=32,
        lr=0.01,
        smoke=True,
    )
    assert out_json.exists()
    assert out_md.exists()
    cells = json.loads(out_json.read_text())["cells_richer"]
    assert len(cells) == 8  # 4 arms × 2 seeds, C5 only
    cells_hp = json.loads(out_json.read_text())["cells_hp"]
    assert len(cells_hp) == 3  # 3 dream arms × 1 seed × 1 HP combo
```

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest tests/unit/experiments/test_g4_ter_driver.py -v`
Expected: FAIL — `run_g4_ter` module not found.

- [ ] **Step 3: Implement the driver**

Create `experiments/g4_ter_hp_sweep/run_g4_ter.py`:

```python
"""G4-ter pilot driver — HP sweep + richer-substrate confirmatory.

Two sweeps run side-by-side (Option C) :

1. Richer-substrate sweep : 4 arms × N_richer seeds × 1 HP (C5) on
   ``G4HierarchicalClassifier``.
2. HP sub-grid sweep : 3 dream arms × N_hp seeds × 10 HP combos on
   the binary ``G4Classifier`` (no baseline arm — HP changes do not
   affect the no-dream branch).

Per-cell pipeline mirrors ``experiments/g4_split_fmnist/run_g4.py``.
Outputs :
    docs/milestones/g4-ter-pilot-2026-05-03.{json,md}

Pre-reg : docs/osf-prereg-g4-ter-pilot.md
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from harness.benchmarks.effect_size_targets import (  # noqa: E402
    HU_2020_OVERALL,
    JAVADI_2024_OVERALL,
)
from harness.storage.run_registry import RunRegistry  # noqa: E402
from kiki_oniric.eval.statistics import (  # noqa: E402
    compute_hedges_g,
    jonckheere_trend,
    welch_one_sided,
)

from experiments.g4_split_fmnist.dataset import (  # noqa: E402
    SplitFMNISTTask,
    load_split_fmnist_5tasks,
)
from experiments.g4_split_fmnist.dream_wrap import (  # noqa: E402
    BetaBufferFIFO,
    G4Classifier,
    build_profile,
)
from experiments.g4_ter_hp_sweep.dream_wrap_hier import (  # noqa: E402
    BetaBufferHierFIFO,
    G4HierarchicalClassifier,
)
from experiments.g4_ter_hp_sweep.hp_grid import (  # noqa: E402
    HP_COMBOS,
    HPCombo,
    representative_combo,
)


C_VERSION = "C-v0.12.0+PARTIAL"
ARMS_RICHER: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
ARMS_HP: tuple[str, ...] = ("P_min", "P_equ", "P_max")
DEFAULT_SEEDS_RICHER: tuple[int, ...] = tuple(range(30))
DEFAULT_SEEDS_HP: tuple[int, ...] = tuple(range(10))
DEFAULT_HP_COMBO_IDS: tuple[str, ...] = tuple(c.combo_id for c in HP_COMBOS)
DEFAULT_DATA_DIR = REPO_ROOT / "experiments" / "g4_split_fmnist" / "data"
DEFAULT_OUT_JSON = (
    REPO_ROOT / "docs" / "milestones" / "g4-ter-pilot-2026-05-03.json"
)
DEFAULT_OUT_MD = (
    REPO_ROOT / "docs" / "milestones" / "g4-ter-pilot-2026-05-03.md"
)
DEFAULT_REGISTRY_DB = REPO_ROOT / ".run_registry.sqlite"
RETENTION_EPS = 1e-6
BETA_BUFFER_CAPACITY = 256
BETA_BUFFER_FILL_PER_TASK = 32
HIDDEN_1 = 32
HIDDEN_2 = 16
RECOMBINE_N_SYNTHETIC = 16
RECOMBINE_LR = 0.01
RESTRUCTURE_FACTOR = 0.05


class CellRicher(TypedDict):
    arm: str
    seed: int
    hp_combo_id: str
    acc_task1_initial: float
    acc_task1_final: float
    retention: float
    excluded_underperforming_baseline: bool
    wall_time_s: float
    run_id: str


class CellHP(TypedDict):
    arm: str
    seed: int
    hp_combo_id: str
    acc_task1_initial: float
    acc_task1_final: float
    retention: float
    excluded_underperforming_baseline: bool
    wall_time_s: float
    run_id: str


def _resolve_commit_sha() -> str:
    env_sha = os.environ.get("DREAMOFKIKI_COMMIT_SHA")
    if env_sha:
        return env_sha
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _run_cell_richer(
    arm: str,
    seed: int,
    combo: HPCombo,
    tasks: list[SplitFMNISTTask],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict[str, Any]:
    start = time.time()
    feat_dim = tasks[0]["x_train"].shape[1]
    clf = G4HierarchicalClassifier(
        in_dim=feat_dim,
        hidden_1=HIDDEN_1,
        hidden_2=HIDDEN_2,
        n_classes=2,
        seed=seed,
    )
    buffer = BetaBufferHierFIFO(capacity=BETA_BUFFER_CAPACITY)
    fill_rng = np.random.default_rng(seed + 5_000)

    def _push_task(task: SplitFMNISTTask) -> None:
        n = task["x_train"].shape[0]
        n_take = min(BETA_BUFFER_FILL_PER_TASK, n)
        idx = fill_rng.choice(n, size=n_take, replace=False)
        for i in idx.tolist():
            x = task["x_train"][i]
            latent = clf.latent(x[None, :])[0]
            buffer.push(x=x, y=int(task["y_train"][i]), latent=latent)

    clf.train_task(
        tasks[0], epochs=epochs, batch_size=batch_size, lr=lr
    )
    acc_initial = clf.eval_accuracy(
        tasks[0]["x_test"], tasks[0]["y_test"]
    )
    _push_task(tasks[0])

    profile = None
    if arm != "baseline":
        profile = build_profile(arm, seed=seed)

    for k in range(1, len(tasks)):
        if profile is not None:
            clf.dream_episode_hier(
                profile,
                seed=seed + k,
                beta_buffer=buffer,
                replay_n_records=combo.replay_batch,
                replay_n_steps=combo.replay_n_steps,
                replay_lr=combo.replay_lr,
                downscale_factor=combo.downscale_factor,
                restructure_factor=RESTRUCTURE_FACTOR,
                recombine_n_synthetic=RECOMBINE_N_SYNTHETIC,
                recombine_lr=RECOMBINE_LR,
            )
        clf.train_task(
            tasks[k], epochs=epochs, batch_size=batch_size, lr=lr
        )
        _push_task(tasks[k])

    acc_final = clf.eval_accuracy(
        tasks[0]["x_test"], tasks[0]["y_test"]
    )
    retention = acc_final / max(acc_initial, RETENTION_EPS)
    excluded = bool(acc_initial < 0.5)
    return {
        "arm": arm,
        "seed": seed,
        "hp_combo_id": combo.combo_id,
        "acc_task1_initial": float(acc_initial),
        "acc_task1_final": float(acc_final),
        "retention": float(retention),
        "excluded_underperforming_baseline": excluded,
        "wall_time_s": time.time() - start,
    }


def _run_cell_hp(
    arm: str,
    seed: int,
    combo: HPCombo,
    tasks: list[SplitFMNISTTask],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int = 128,
) -> dict[str, Any]:
    start = time.time()
    feat_dim = tasks[0]["x_train"].shape[1]
    clf = G4Classifier(
        in_dim=feat_dim,
        hidden_dim=hidden_dim,
        n_classes=2,
        seed=seed,
    )
    buffer = BetaBufferFIFO(capacity=BETA_BUFFER_CAPACITY)
    fill_rng = np.random.default_rng(seed + 5_000)

    def _push_task(task: SplitFMNISTTask) -> None:
        n = task["x_train"].shape[0]
        n_take = min(BETA_BUFFER_FILL_PER_TASK, n)
        idx = fill_rng.choice(n, size=n_take, replace=False)
        for i in idx.tolist():
            buffer.push(task["x_train"][i], int(task["y_train"][i]))

    clf.train_task(
        tasks[0], epochs=epochs, batch_size=batch_size, lr=lr
    )
    acc_initial = clf.eval_accuracy(
        tasks[0]["x_test"], tasks[0]["y_test"]
    )
    _push_task(tasks[0])

    profile = build_profile(arm, seed=seed)
    for k in range(1, len(tasks)):
        clf.dream_episode(
            profile,
            seed=seed + k,
            beta_buffer=buffer,
            replay_n_records=combo.replay_batch,
            replay_n_steps=combo.replay_n_steps,
            replay_lr=combo.replay_lr,
            downscale_factor=combo.downscale_factor,
        )
        clf.train_task(
            tasks[k], epochs=epochs, batch_size=batch_size, lr=lr
        )
        _push_task(tasks[k])

    acc_final = clf.eval_accuracy(
        tasks[0]["x_test"], tasks[0]["y_test"]
    )
    retention = acc_final / max(acc_initial, RETENTION_EPS)
    excluded = bool(acc_initial < 0.5)
    return {
        "arm": arm,
        "seed": seed,
        "hp_combo_id": combo.combo_id,
        "acc_task1_initial": float(acc_initial),
        "acc_task1_final": float(acc_final),
        "retention": float(retention),
        "excluded_underperforming_baseline": excluded,
        "wall_time_s": time.time() - start,
    }


def _retention_by_arm(cells: list[dict[str, Any]]) -> dict[str, list[float]]:
    arms = sorted({c["arm"] for c in cells})
    out: dict[str, list[float]] = {a: [] for a in arms}
    for c in cells:
        if c["excluded_underperforming_baseline"]:
            continue
        out[c["arm"]].append(c["retention"])
    return out


def _h2_verdict(retention: dict[str, list[float]]) -> dict[str, Any]:
    """H2 — richer substrate P_equ vs baseline (Hu 2020 anchor)."""
    p_equ = retention.get("P_equ", [])
    base = retention.get("baseline", [])
    if len(p_equ) < 2 or len(base) < 2:
        return {
            "insufficient_samples": True,
            "n_p_equ": len(p_equ),
            "n_base": len(base),
        }
    g = compute_hedges_g(p_equ, base)
    welch = welch_one_sided(base, p_equ, alpha=0.05 / 4)
    return {
        "hedges_g": g,
        "is_within_hu_2020_ci": HU_2020_OVERALL.is_within_ci(g),
        "above_zero": bool(g >= 0.0),
        "above_hu_2020_lower_ci": bool(g >= HU_2020_OVERALL.ci_low),
        "welch_p": welch.p_value,
        "welch_reject_h0": welch.reject_h0,
        "alpha_per_test": 0.05 / 4,
        "n_p_equ": len(p_equ),
        "n_base": len(base),
    }


def _h_dr4_ter_verdict(retention: dict[str, list[float]]) -> dict[str, Any]:
    groups = [
        retention.get("P_min", []),
        retention.get("P_equ", []),
        retention.get("P_max", []),
    ]
    if any(len(g) < 2 for g in groups):
        return {
            "insufficient_samples": True,
            "n_per_arm": [len(g) for g in groups],
        }
    res = jonckheere_trend(groups, alpha=0.05 / 4)
    mean_p_min = float(sum(groups[0]) / len(groups[0]))
    mean_p_equ = float(sum(groups[1]) / len(groups[1]))
    mean_p_max = float(sum(groups[2]) / len(groups[2]))
    return {
        "j_statistic": res.statistic,
        "p_value": res.p_value,
        "reject_h0": res.reject_h0,
        "mean_p_min": mean_p_min,
        "mean_p_equ": mean_p_equ,
        "mean_p_max": mean_p_max,
        "monotonic_observed": (
            mean_p_min <= mean_p_equ <= mean_p_max
        ),
        "alpha_per_test": 0.05 / 4,
    }


def _h1_hp_verdict(
    cells_hp: list[dict[str, Any]],
    base_retention: list[float],
) -> dict[str, Any]:
    """H1 — best HP combo on binary head vs richer-substrate baseline."""
    by_combo_arm: dict[tuple[str, str], list[float]] = {}
    for c in cells_hp:
        if c["excluded_underperforming_baseline"]:
            continue
        by_combo_arm.setdefault(
            (c["hp_combo_id"], c["arm"]), []
        ).append(c["retention"])
    best: dict[str, Any] | None = None
    if len(base_retention) < 2:
        return {
            "insufficient_samples": True,
            "n_base": len(base_retention),
        }
    for (combo_id, arm), rets in by_combo_arm.items():
        if arm != "P_equ" or len(rets) < 2:
            continue
        g = compute_hedges_g(rets, base_retention)
        if best is None or g > best["hedges_g"]:
            best = {
                "hp_combo_id": combo_id,
                "hedges_g": g,
                "n_p_equ": len(rets),
            }
    if best is None:
        return {"insufficient_samples": True}
    best["above_zero"] = bool(best["hedges_g"] >= 0.0)
    best["above_hu_2020_lower_ci"] = bool(
        best["hedges_g"] >= HU_2020_OVERALL.ci_low
    )
    best["alpha_per_test"] = 0.05 / 4
    best["n_base"] = len(base_retention)
    return best


def _aggregate_verdict(
    cells_richer: list[dict[str, Any]],
    cells_hp: list[dict[str, Any]],
) -> dict[str, Any]:
    retention_richer = _retention_by_arm(cells_richer)
    return {
        "h2_substrate_richer": _h2_verdict(retention_richer),
        "h_dr4_ter_richer": _h_dr4_ter_verdict(retention_richer),
        "h1_hp_artefact": _h1_hp_verdict(
            cells_hp, retention_richer.get("baseline", [])
        ),
        "retention_richer_by_arm": retention_richer,
    }


def _render_md_report(payload: dict[str, Any]) -> str:
    h1 = payload["verdict"]["h1_hp_artefact"]
    h2 = payload["verdict"]["h2_substrate_richer"]
    h4 = payload["verdict"]["h_dr4_ter_richer"]
    lines: list[str] = [
        "# G4-ter pilot — HP sweep + richer substrate",
        "",
        f"**Date** : {payload['date']}",
        f"**c_version** : `{payload['c_version']}`",
        f"**commit_sha** : `{payload['commit_sha']}`",
        (
            f"**Cells (richer)** : {len(payload['cells_richer'])} "
            f"({len(ARMS_RICHER)} arms × {payload['n_seeds_richer']} "
            "seeds × 1 HP)"
        ),
        (
            f"**Cells (HP)** : {len(payload['cells_hp'])} "
            f"({len(ARMS_HP)} arms × {payload['n_seeds_hp']} seeds × "
            f"{payload['n_hp_combos']} combos)"
        ),
        f"**Wall time** : {payload['wall_time_s']:.1f}s",
        "",
        "## Pre-registered hypotheses",
        "",
        "Pre-registration : `docs/osf-prereg-g4-ter-pilot.md`",
        "",
        "### H1 — HP artefact (best HP combo on binary head)",
    ]
    if h1.get("insufficient_samples"):
        lines.append(f"INSUFFICIENT SAMPLES (n_base={h1.get('n_base')})")
    else:
        lines += [
            f"- best combo : `{h1['hp_combo_id']}`",
            f"- best Hedges' g : **{h1['hedges_g']:.4f}**",
            f"- above zero : {h1['above_zero']}",
            (
                "- above Hu 2020 lower CI 0.21 : "
                f"{h1['above_hu_2020_lower_ci']}"
            ),
        ]
    lines += [
        "",
        "### H2 — Substrate-level (richer head, P_equ vs baseline)",
    ]
    if h2.get("insufficient_samples"):
        lines.append(
            f"INSUFFICIENT SAMPLES (n_p_equ={h2.get('n_p_equ')}, "
            f"n_base={h2.get('n_base')})"
        )
    else:
        lines += [
            f"- observed Hedges' g : **{h2['hedges_g']:.4f}**",
            f"- above zero : {h2['above_zero']}",
            (
                "- above Hu 2020 lower CI 0.21 : "
                f"{h2['above_hu_2020_lower_ci']}"
            ),
            (
                f"- Welch one-sided p (α/4 = "
                f"{h2['alpha_per_test']:.4f}) : {h2['welch_p']:.4f} → "
                f"reject_h0 = {h2['welch_reject_h0']}"
            ),
        ]
    lines += [
        "",
        "### H_DR4-ter — Jonckheere monotonicity on richer substrate",
    ]
    if h4.get("insufficient_samples"):
        lines.append(f"INSUFFICIENT SAMPLES (n_per_arm={h4['n_per_arm']})")
    else:
        lines += [
            f"- mean retention P_min : {h4['mean_p_min']:.4f}",
            f"- mean retention P_equ : {h4['mean_p_equ']:.4f}",
            f"- mean retention P_max : {h4['mean_p_max']:.4f}",
            (
                "- monotonic observed P_max >= P_equ >= P_min : "
                f"{h4['monotonic_observed']}"
            ),
            f"- Jonckheere J statistic : {h4['j_statistic']:.4f}",
            (
                f"- one-sided p (α/4 = {h4['alpha_per_test']:.4f}) : "
                f"{h4['p_value']:.4f} → reject_h0 = {h4['reject_h0']}"
            ),
        ]
    lines += [
        "",
        "## Provenance",
        "",
        "- Pre-registration : "
        "[docs/osf-prereg-g4-ter-pilot.md](../osf-prereg-g4-ter-pilot.md)",
        "- Driver : `experiments/g4_ter_hp_sweep/run_g4_ter.py`",
        "- Substrate : `experiments.g4_ter_hp_sweep.dream_wrap_hier."
        "G4HierarchicalClassifier`",
        "- HP grid : `experiments.g4_ter_hp_sweep.hp_grid.HP_COMBOS`",
        "- Run registry : `harness/storage/run_registry.RunRegistry` "
        "(db `.run_registry.sqlite`)",
        "",
    ]
    return "\n".join(lines)


def run_pilot(
    *,
    data_dir: Path,
    seeds_richer: tuple[int, ...],
    seeds_hp: tuple[int, ...],
    hp_combo_ids: tuple[str, ...],
    out_json: Path,
    out_md: Path,
    registry_db: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    smoke: bool = False,
) -> dict[str, Any]:
    tasks = load_split_fmnist_5tasks(data_dir)
    if len(tasks) != 5:
        raise RuntimeError(
            f"Split-FMNIST loader returned {len(tasks)} tasks (expected 5)"
        )

    registry = RunRegistry(registry_db)
    commit_sha = _resolve_commit_sha()

    # ---- Richer substrate sweep (C5 only) ----
    c5 = representative_combo()
    cells_richer: list[dict[str, Any]] = []
    sweep_start = time.time()
    for arm in ARMS_RICHER:
        for seed in seeds_richer:
            cell = _run_cell_richer(
                arm,
                seed,
                c5,
                tasks,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
            )
            run_id = registry.register(
                c_version=C_VERSION,
                profile=f"g4-ter/richer/{arm}/{c5.combo_id}",
                seed=seed,
                commit_sha=commit_sha,
            )
            cell["run_id"] = run_id
            cells_richer.append(cell)

    # ---- HP sub-grid sweep on binary head ----
    cells_hp: list[dict[str, Any]] = []
    selected_combos = [c for c in HP_COMBOS if c.combo_id in hp_combo_ids]
    for combo in selected_combos:
        for arm in ARMS_HP:
            for seed in seeds_hp:
                cell = _run_cell_hp(
                    arm,
                    seed,
                    combo,
                    tasks,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                )
                run_id = registry.register(
                    c_version=C_VERSION,
                    profile=f"g4-ter/hp/{arm}/{combo.combo_id}",
                    seed=seed,
                    commit_sha=commit_sha,
                )
                cell["run_id"] = run_id
                cells_hp.append(cell)

    wall = time.time() - sweep_start
    verdict = _aggregate_verdict(cells_richer, cells_hp)

    payload = {
        "date": "2026-05-03",
        "c_version": C_VERSION,
        "commit_sha": commit_sha,
        "n_seeds_richer": len(seeds_richer),
        "n_seeds_hp": len(seeds_hp),
        "n_hp_combos": len(selected_combos),
        "arms_richer": list(ARMS_RICHER),
        "arms_hp": list(ARMS_HP),
        "data_dir": str(data_dir),
        "wall_time_s": wall,
        "smoke": smoke,
        "cells_richer": cells_richer,
        "cells_hp": cells_hp,
        "verdict": verdict,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    out_md.write_text(_render_md_report(payload))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G4-ter pilot driver")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--out-json", type=Path, default=DEFAULT_OUT_JSON
    )
    parser.add_argument(
        "--out-md", type=Path, default=DEFAULT_OUT_MD
    )
    parser.add_argument(
        "--registry-db", type=Path, default=DEFAULT_REGISTRY_DB
    )
    parser.add_argument(
        "--seeds-richer", type=int, nargs="+",
        default=list(DEFAULT_SEEDS_RICHER),
    )
    parser.add_argument(
        "--seeds-hp", type=int, nargs="+",
        default=list(DEFAULT_SEEDS_HP),
    )
    parser.add_argument(
        "--hp-combo-ids", type=str, nargs="+",
        default=list(DEFAULT_HP_COMBO_IDS),
    )
    args = parser.parse_args(argv)

    if args.smoke:
        seeds_richer = (0, 1)
        seeds_hp = (0,)
        hp_combo_ids = ("C5",)
    else:
        seeds_richer = tuple(args.seeds_richer)
        seeds_hp = tuple(args.seeds_hp)
        hp_combo_ids = tuple(args.hp_combo_ids)

    payload = run_pilot(
        data_dir=args.data_dir,
        seeds_richer=seeds_richer,
        seeds_hp=seeds_hp,
        hp_combo_ids=hp_combo_ids,
        out_json=args.out_json,
        out_md=args.out_md,
        registry_db=args.registry_db,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        smoke=args.smoke,
    )
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(
        f"Cells richer : {len(payload['cells_richer'])}  "
        f"HP : {len(payload['cells_hp'])}"
    )
    print(
        f"H2.hedges_g : "
        f"{payload['verdict']['h2_substrate_richer'].get('hedges_g')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the smoke test**

Run: `uv run pytest tests/unit/experiments/test_g4_ter_driver.py -v`
Expected: PASS in < 30 s. If it fails on the dataset loader, follow
the existing G4-bis test fixture to construct an IDX-formatted FMNIST
synthetic file (the existing `experiments/g4_split_fmnist/dataset.py`
respects `DREAMOFKIKI_FMNIST_FAKE=1`) and adjust `_toy_data_dir`.

- [ ] **Step 5: Commit**

```bash
git add experiments/g4_ter_hp_sweep/run_g4_ter.py \
        tests/unit/experiments/test_g4_ter_driver.py
git commit -m "feat(g4-ter): add HP+richer pilot driver

Side-by-side sweep: richer substrate at C5 anchor (4 arms x N=30)
plus HP sub-grid on binary head (3 arms x N=10 x 10 combos).
RunRegistry profile keys 'g4-ter/{richer,hp}/<arm>/<combo>'.
"
```

---

## Task 9: DR-3 conformance test for the richer substrate (TDD)

**Files:**
- Create: `tests/conformance/axioms/test_dr3_g4_hierarchical_substrate.py`

The richer substrate must satisfy the DR-3 Conformance Criterion
conditions (1)–(3) — typed Protocols, executable primitives, and
chained primitives compose without raising. Mirrors the structure
of `test_dr3_micro_kiki_substrate.py` but scoped to the G4-ter
classifier.

- [ ] **Step 1: Write the failing conformance test**

Create `tests/conformance/axioms/test_dr3_g4_hierarchical_substrate.py`:

```python
"""DR-3 Conformance — G4-ter hierarchical substrate.

Verifies the richer substrate (G4HierarchicalClassifier) satisfies
DR-3 conditions (1) typed Protocols, (2) executable primitives,
(3) primitives chain without raising.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
"""
from __future__ import annotations

import numpy as np

from experiments.g4_ter_hp_sweep.dream_wrap_hier import (
    BetaBufferHierFIFO,
    G4HierarchicalClassifier,
)
from experiments.g4_split_fmnist.dream_wrap import build_profile


def _fill_buffer(buf: BetaBufferHierFIFO, clf: G4HierarchicalClassifier,
                 n_per_class: int = 4) -> None:
    rng = np.random.default_rng(0)
    for cls in (0, 1):
        for _ in range(n_per_class):
            x = rng.standard_normal(10).astype(np.float32)
            latent = clf.latent(x[None, :])[0]
            buf.push(x=x, y=cls, latent=latent)


def test_dr3_g4_hier_protocols_present() -> None:
    """DR-3 (1) — substrate exposes the public coupling methods."""
    clf = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=0
    )
    assert hasattr(clf, "predict_logits")
    assert hasattr(clf, "latent")
    assert hasattr(clf, "train_task")
    assert hasattr(clf, "eval_accuracy")
    assert hasattr(clf, "dream_episode_hier")


def test_dr3_g4_hier_primitives_executable() -> None:
    """DR-3 (2) — REPLAY/DOWNSCALE/RESTRUCTURE/RECOMBINE all dispatchable."""
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
    """DR-3 (3) — three consecutive episodes do not raise."""
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
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/conformance/axioms/test_dr3_g4_hierarchical_substrate.py -v`
Expected: 3 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/conformance/axioms/test_dr3_g4_hierarchical_substrate.py
git commit -m "test(g4-ter): DR-3 conformance for hier substrate

Asserts G4HierarchicalClassifier satisfies DR-3 conditions (1)
typed methods, (2) all 4 ops executable, (3) primitives chain
across 3 consecutive episodes without raising.
"
```

---

## Task 10: Run the smoke pilot end-to-end

**Files:**
- No code changes; verification step only.

- [ ] **Step 1: Run the full smoke pilot**

Run:
```bash
uv run python experiments/g4_ter_hp_sweep/run_g4_ter.py --smoke \
    --out-json /tmp/g4-ter-smoke.json \
    --out-md /tmp/g4-ter-smoke.md \
    --registry-db /tmp/g4-ter-smoke.sqlite
```
Expected: completes in < 60 s, writes both files, prints `H2.hedges_g`
to stdout. Confirms the driver wires correctly against the real
Split-FMNIST loader (smoke uses 2 seeds × 4 arms richer + 1 seed × 3
arms × 1 combo HP = 11 cells).

- [ ] **Step 2: Verify the smoke milestone shape**

Run: `python -c "import json; d=json.load(open('/tmp/g4-ter-smoke.json')); print(len(d['cells_richer']), len(d['cells_hp']))"`
Expected: `8 3`.

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest`
Expected: ≥ 277 + 17 + 4 + 3 = ~301 tests passing, coverage ≥ 90 %.

- [ ] **Step 4: Run lint and types**

Run: `uv run ruff check . && uv run mypy harness tests`
Expected: zero errors.

No commit — this is a verification step. Proceed to Task 11 only if
all of the above pass.

---

## Task 11: Run the production pilot (Option C)

**Files:**
- Create: `docs/milestones/g4-ter-pilot-2026-05-03.json`
- Create: `docs/milestones/g4-ter-pilot-2026-05-03.md`

- [ ] **Step 1: Confirm the pre-reg is committed**

Run: `git log --oneline -- docs/osf-prereg-g4-ter-pilot.md | head -1`
Expected: shows the commit from Task 1. **If empty, abort the
production run** — the OSF pre-reg must lock before any cell registers.

- [ ] **Step 2: Launch the production sweep**

Run:
```bash
uv run python experiments/g4_ter_hp_sweep/run_g4_ter.py \
    --out-json docs/milestones/g4-ter-pilot-2026-05-03.json \
    --out-md docs/milestones/g4-ter-pilot-2026-05-03.md
```
Expected: completes in 3–5 h on M1 Max. 420 cells.

If the sweep exceeds 5 h at the smoke stage, abort and re-run with
the 4-combo subset (per `decision log` rule):

```bash
uv run python experiments/g4_ter_hp_sweep/run_g4_ter.py \
    --hp-combo-ids C0 C5 C7 C9 \
    --out-json docs/milestones/g4-ter-pilot-2026-05-03.json \
    --out-md docs/milestones/g4-ter-pilot-2026-05-03.md
```

…and write `docs/osf-deviations-g4-ter-2026-05-03.md` documenting
the deviation per pre-reg §8.

- [ ] **Step 3: Verify the milestone payload**

Run: `python -c "import json; d=json.load(open('docs/milestones/g4-ter-pilot-2026-05-03.json')); print('richer', len(d['cells_richer']), 'hp', len(d['cells_hp']), 'h2', d['verdict']['h2_substrate_richer'].get('hedges_g'))"`
Expected: `richer 120 hp 300 h2 <float>` (or 120 + 120 if subset).

- [ ] **Step 4: Commit the milestone**

```bash
git add docs/milestones/g4-ter-pilot-2026-05-03.json \
        docs/milestones/g4-ter-pilot-2026-05-03.md
git commit -m "feat(milestones): G4-ter pilot results

420-cell sweep: richer substrate (4 arms x 30 seeds x 1 HP) plus
binary HP sub-grid (3 arms x 10 seeds x 10 combos). All cells
registered in run-registry against C-v0.12.0+PARTIAL.
"
```

---

## Task 12: Aggregate verdict + decision branch

**Files:**
- Read: `docs/milestones/g4-ter-pilot-2026-05-03.md`

- [ ] **Step 1: Read the verdict**

Run: `head -50 docs/milestones/g4-ter-pilot-2026-05-03.md`
Inspect H1, H2, H_DR4-ter sections. Map outcomes to the H1/H2/H3
decision rules in `docs/osf-prereg-g4-ter-pilot.md` §7.

- [ ] **Step 2: Classify the outcome**

Apply the table:

| Observed | Classification |
|----------|----------------|
| H_DR4-ter rejects + (H1 above-zero or H2 above-zero) | EC: STABLE |
| H_DR4-ter inconclusive **or** all HP+H2 inconclusive | EC: PARTIAL (no change) |
| H_DR4-ter falsified (P_min > P_max statistically) | EC: UNSTABLE |

Write the chosen classification down explicitly in the executor's
notes for Task 13.

No commit. Proceed to Task 13 with the chosen classification in hand.

---

## Task 13: Update Paper 2 §7.1.5 (EN + FR)

**Files:**
- Modify: `docs/papers/paper2/results.md` (insert §7.1.5 after §7.1.2)
- Modify: `docs/papers/paper2-fr/results.md` (insert §7.1.5 after §7.1.2)

- [ ] **Step 1: Insert §7.1.5 in the EN narrative**

Open `docs/papers/paper2/results.md`. Locate the boundary right after
the §7.1.2 G4-bis pilot block ends (next-section header). Insert the
following block. Substitute `<g_h2>`, `<j_p>`, `<best_combo>`, etc.
with the actual values from the milestone.

```markdown
## 7.1.5 G4-ter pilot (HP sweep + richer substrate — 2026-05-03)

The G4-ter pilot is the confirmatory N≥30 follow-up scheduled by
the G4-bis exploratory-positive-evidence rule
(`docs/osf-prereg-g4-pilot.md` §4) and pre-registered at
`docs/osf-prereg-g4-ter-pilot.md`. It distinguishes three competing
explanations of the G4-bis null finding (`g_h1 = -2.31`, `H_DR4`
degenerate equal-means):

- **H1 — HP artefact**: the G4-bis HP combo
  (`replay_lr=0.01`, `replay_n_steps=1`, `downscale_factor=0.95`)
  is over-aggressive. A 10-combo curated HP sub-grid sweep on the
  binary MLP head (300 cells, N=10 per cell) yields a best
  Hedges' g of `<g_h1_best>` at combo `<best_combo>`.
- **H2 — substrate-level limitation**: the binary head exposes
  only REPLAY + DOWNSCALE coupling channels. A hierarchical head
  (input → 32 → 16 → output) that exposes `_l2.weight` as a
  RESTRUCTURE site and hidden_2 activations as a RECOMBINE
  Gaussian-MoG sampling source yields `g_h2 = <g_h2>` at the C5
  anchor combo over 120 cells (N=30 per arm).
- **H_DR4-ter — monotonicity**: the structurally-distinguished
  richer substrate breaks the G4-bis degenerate tie. Mean
  retention `P_max = <m_max>`, `P_equ = <m_equ>`, `P_min = <m_min>`
  (Jonckheere J = `<j_stat>`, p = `<j_p>` at α/4 = 0.0125).

The verdict locks the EC axis at `<EC>` per the pre-reg DualVer
table §7. Run-registry profile keys `g4-ter/{richer,hp}/<arm>/<combo>`
identify each cell to satisfy R1.

Provenance:
- Pre-registration : `docs/osf-prereg-g4-ter-pilot.md`
- Milestone (md) : `docs/milestones/g4-ter-pilot-2026-05-03.md`
- Milestone (json) : `docs/milestones/g4-ter-pilot-2026-05-03.json`
- Driver : `experiments/g4_ter_hp_sweep/run_g4_ter.py`
- Substrate : `experiments.g4_ter_hp_sweep.dream_wrap_hier.G4HierarchicalClassifier`
- HP grid : `experiments.g4_ter_hp_sweep.hp_grid.HP_COMBOS`
```

- [ ] **Step 2: Insert §7.1.5 in the FR mirror**

Open `docs/papers/paper2-fr/results.md`. Locate the boundary right
after the §7.1.2 G4-bis French block. Insert a §7.1.5 block that
mirrors the EN content from Step 1, translated using the project's
existing EN↔FR conventions established in §7.1.2 (e.g. "pilot" →
"pilote", "richer substrate" → "substrat enrichi", "HP artefact" →
"artefact HP", "monotonicity" → "monotonicité"). Keep the same
value-substitution markers (`<g_h2>`, `<best_combo>`, `<EC>`, …)
and the same provenance bullet list (paths are language-neutral).
Title : `## 7.1.5 Pilote G4-ter (balayage HP + substrat enrichi — 2026-05-03)`.

The §7.1.2 G4-bis FR block already establishes the canonical
translations for `retention → rétention`, `replay → relecture`,
`downscale → réduction`, `mean → moyenne`, `Welch one-sided →
Welch unilatéral` — re-use them verbatim.

- [ ] **Step 3: Verify the EN→FR mirror passes the project's lint**

Run: `uv run ruff check . && uv run mypy harness tests`
Expected: zero errors (only Markdown changes).

- [ ] **Step 4: Commit**

```bash
git add docs/papers/paper2/results.md docs/papers/paper2-fr/results.md
git commit -m "docs(paper2): add 7.1.5 G4-ter results EN+FR

Reports H1/H2/H_DR4-ter verdicts and cites milestone artefacts.
EN→FR mirror per CONTRIBUTING.md propagation rule.
"
```

---

## Task 14: CHANGELOG + STATUS update (conditional EC bump)

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `STATUS.md`

- [ ] **Step 1: Append to CHANGELOG `[Unreleased] / Empirical (G4-ter)`**

Edit `CHANGELOG.md`. Locate the `[Unreleased]` block (top of the
file). Append:

```markdown
### Empirical (G4-ter)

- G4-ter pilot completed 2026-05-03 — confirmatory N≥30 follow-up to
  the G4-bis null finding. 420 cells: richer substrate
  (`G4HierarchicalClassifier`, 4 arms × 30 seeds × 1 HP at C5) plus
  HP sub-grid on binary head (3 arms × 10 seeds × 10 combos). Pre-
  registration `docs/osf-prereg-g4-ter-pilot.md` locked at commit
  `<pre-reg sha>`. Milestone artefacts
  `docs/milestones/g4-ter-pilot-2026-05-03.{json,md}`. Verdict :
  H1 = `<H1 status>`, H2 = `<H2 status>`, H_DR4-ter = `<H4 status>`.
  EC axis : `<PARTIAL|STABLE|UNSTABLE>` per
  `docs/osf-prereg-g4-ter-pilot.md` §7.
```

- [ ] **Step 2: Update STATUS.md G4 row**

Edit `STATUS.md`. Find the `| G4 — P_equ fonctionnel |` row in the
Gates table. Replace it with one of (matching the Task 12
classification):

```markdown
| G4 — P_equ fonctionnel | S12 | 🟢 G4-ter STABLE (2026-05-03 — H_DR4-ter rejects, H2 above-zero on richer substrate) |
| G4 — P_equ fonctionnel | S12 | 🔶 G4-ter PARTIAL (2026-05-03 — substrate-level + HP confirmation inconclusive, N≥95 G4-quater scheduled) |
| G4 — P_equ fonctionnel | S12 | 🔴 G4-ter UNSTABLE (2026-05-03 — H_DR4-ter falsified on richer substrate, see milestone) |
```

If `<EC>` is `STABLE`, also update the DualVer EC bullet block under
"## DualVer status" with the new value and rationale; otherwise
leave the EC bullet untouched.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md STATUS.md
git commit -m "docs(status): record G4-ter EC verdict

Adds Empirical (G4-ter) bullet to [Unreleased] and updates the
G4 Gates row to reflect the verdict from the pilot milestone.
"
```

---

## Task 15: Self-review

**Files:**
- No file changes; review-only step.

- [ ] **Step 1: Spec coverage check**

Walk through the user spec line by line and confirm each requirement
maps to a task:

- HP sub-grid (10 combos, 4 axes) → Task 7 + driver in Task 8.
- Richer substrate `G4HierarchicalClassifier` (multi-layer head) →
  Task 2.
- RESTRUCTURE on `_l2` → Task 4.
- RECOMBINE on Gaussian-MoG of β-buffer latents → Task 5.
- Wire 4 ops → Task 6.
- Driver Option C (1 representative HP × N=30 × 4 arms × richer +
  10 HP × N=10 × 3 dream arms × binary) → Task 8.
- R1 register + milestone dump → Task 8 + Task 11.
- Property tests + smoke → Tasks 2/3/4/5/6/7/8 + Task 10.
- Run pilot Option C → Task 11.
- Aggregate verdict H1/H2/H3 → Task 12 (and embedded in driver).
- Update Paper 2 §7.1.5 EN+FR → Task 13.
- CHANGELOG + STATUS G4 row → Task 14.
- Self-review → Task 15.
- OSF pre-reg G4-ter cites G4-bis baseline → Task 1.
- DR-3 conformance for richer substrate → Task 9.
- No FC bump (calibration + impl only) → Constraints honored block.
- EN→FR mirror Paper 2 §7.1.5 → Task 13.

- [ ] **Step 2: Placeholder scan**

Run: `grep -nE "TODO|TBD|implement later|fill in details" docs/superpowers/plans/2026-05-03-g4-ter-hp-sweep-richer-substrate.md`
Expected: zero matches in the plan body (the table-of-paragraphs
inserts in §7.1.5 use `<g_h2>` etc. as deliberately-pending value
slots — these are not placeholders for the *plan*, they are
substitution markers for the *milestone insertion*, which is correct
TDD: the value materialises when the run lands at Task 11).

- [ ] **Step 3: Type consistency check**

Confirm the names used across tasks match:
- `G4HierarchicalClassifier` (Tasks 2/3/4/5/6/8/9/13)
- `BetaBufferHierFIFO` (Tasks 3/6/8/9)
- `_restructure_step` (Task 4 → Task 6)
- `_recombine_step` (Task 5 → Task 6)
- `dream_episode_hier` (Tasks 6/8/9/13)
- `HPCombo`, `HP_COMBOS`, `representative_combo` (Tasks 7/8)
- `_run_cell_richer`, `_run_cell_hp` (Task 8)

- [ ] **Step 4: Compute budget reality check**

G4-bis ran 20 cells in 29 s on M1 Max (≈ 1.5 s / cell). Conservative
×3 wall-clock for the richer head (extra Linear layer + RESTRUCTURE +
RECOMBINE) gives ~4.5 s / richer cell × 120 = 540 s = 9 min. HP
sub-grid keeps the binary head: 1.5 s / cell × 300 = 450 s = 7.5 min.
Total ~17 min upper bound on smoke arithmetic. The 3-5 h Option C
budget is therefore **comfortably over-budgeted** ; the headroom
absorbs (a) MLX warmup, (b) per-cell registry SQLite contention,
(c) the 5-task Split-FMNIST training cost (each `train_task` run
≈ 200 ms × 5 tasks × 3 epochs = 3 s baseline). Even with these
absorbed, the realistic wall-clock is ~30-90 min, well under
budget.

- [ ] **Step 5: No commit on this task**

Self-review is observation only; if any spec gap or placeholder is
found, return to the relevant task and patch the plan inline before
moving to execution.

---

## Summary

| Phase | Tasks | Wall-clock |
|-------|-------|------------|
| Pre-reg + decision lock | 0, 0.5, 1 | ~30 min |
| Substrate (TDD) | 2, 3, 4, 5, 6 | ~3 h |
| HP grid (TDD) | 7 | ~30 min |
| Driver (TDD) | 8 | ~2 h |
| Conformance (TDD) | 9 | ~30 min |
| Smoke run | 10 | ~10 min |
| Production run | 11 | 30 min – 5 h |
| Verdict + writeup | 12, 13, 14, 15 | ~1 h |

Total engineering wall-clock excluding the production run : ~7 h.
Pilot wall-clock (Option C) : ~30 min – 5 h on M1 Max. Plan honours
the user-spec budget of 1500–2500 lines.

