# dreamOfkiki S9-S12 Atomic Plan

> **Pour agents autonomes :** SKILL REQUIS — utiliser `superpowers:subagent-driven-development`. Les steps utilisent la syntaxe checkbox (`- [ ]`) pour le tracking.

**Goal** : passer du **skeleton** (P_min wiré sans MLX) au **fonctionnel empirique** (P_min mesurable + P_equ wiré + G4 gate). Phase pivot vers ablation S13-S18.

**Architecture** : 11 tasks atomiques (S9.1-S9.5, S10.1-S10.2, S11.1-S11.2, S12.1-S12.2) — 8 dev tasks + 3 docs tasks. Mix MLX integration code + ops complexes (restructure D-Friston, recombine C-Hobson VAE) + profile wiring + reports. Total attendu : 11 commits.

**Tech Stack** : Python 3.12+ uv, **MLX** (Apple Silicon — `mx.array`, `mx.optimizers`, lazy compilation), pytest + hypothesis, dataclasses frozen. Single-threaded encore (concurrent dream worker S13+).

**Préréquis** :
- 48 commits dreamOfkiki, dernier `dcddeac feat(profile): add P_equ skeleton (S9-S12 wiring)`
- 56 tests passing, coverage 96.36%
- Framework C-v0.5.0+STABLE
- P_min profile wiré (replay + downscale handlers, swap protocol skeleton, S2 finite guard)
- P_equ skeleton placeholder ready
- 8 typed Protocols + 4 axiom property tests (DR-0, DR-1, DR-3 partial)
- G3-draft DR-2 proof circulé (S6.1-S6.3)

**Deferred to S13+** :
- Concurrent dream worker (asyncio + threading)
- Real fMRI integration (Studyforrest data pipeline)
- DR-2 proof final review (G3 gate)

---

## Convention commits (validator-enforced)

- Subject ≤50 chars, format `<type>(<scope>): <description>`
- Scope ≥3 chars (single letters rejected — `(mlx)` OK)
- Body lines ≤72 chars, 2-3 paragraphs required
- NO AI attribution
- NO `--no-verify`

---

## File structure après S9-S12

```
dreamOfkiki/
├── kiki_oniric/
│   ├── core/
│   │   └── primitives.py            ✅ existing
│   ├── dream/
│   │   ├── episode.py               ✅ existing
│   │   ├── runtime.py               ✅ existing
│   │   ├── swap.py                  ✅ existing
│   │   ├── operations/
│   │   │   ├── replay.py            ✅ skeleton → S9.1 MLX-real
│   │   │   ├── downscale.py         ✅ skeleton → S9.2 MLX-real
│   │   │   ├── restructure.py       ← S10.1 (D-Friston FEP)
│   │   │   └── recombine.py         ← S11.1 (C-Hobson VAE light)
│   │   └── guards/
│   │       ├── finite.py            ✅ existing
│   │       └── topology.py          ← S10.2 (S3 invariant guard)
│   └── profiles/
│       ├── p_min.py                 ✅ existing → S9.4 swap-integrated
│       ├── p_equ.py                 ✅ skeleton → S11.2 fully wired
│       └── p_max.py                 ← S12.1 skeleton
├── tests/
│   ├── unit/
│   │   ├── test_replay_op_mlx.py    ← S9.1 (MLX backend tests)
│   │   ├── test_downscale_op_mlx.py ← S9.2 (MLX backend tests)
│   │   ├── test_restructure_op.py   ← S10.1
│   │   ├── test_topology_guard.py   ← S10.2
│   │   ├── test_recombine_op.py     ← S11.1
│   │   └── test_p_equ_wiring.py     ← S11.2
│   └── conformance/
│       ├── axioms/
│       │   └── test_dr4_profile_inclusion.py ← S12.1 (DR-4 axiom)
│       └── invariants/
│           └── test_s3_topology.py  ← S10.2
└── docs/
    ├── milestones/
    │   ├── g2-pmin-report.md        ✅ existing → S9.5 update GO-FULL
    │   └── g4-pequ-report.md        ← S12.2 (G4 P_equ functional gate)
    └── proofs/
        └── dr4-profile-inclusion.md ← S12.1 (DR-4 proof draft)
```

---

# Task S9.1 — Replay op MLX backend

**Goal** : remplacer `replay_handler` skeleton (counter only) par version MLX-réelle qui :
- Charge un mini-modèle MLX (LSTM/Transformer 2-layer test bench)
- Sample β records, forward pass, calcule retention loss, applique gradient via `mx.optimizers.SGD`
- Update `ReplayOpState` avec `total_records_consumed` + `last_loss`

**Files:**
- Modify : `kiki_oniric/dream/operations/replay.py` (extend skeleton, add MLX backend toggle)
- Create : `tests/unit/test_replay_op_mlx.py`

## Step S9.1.1 — Write failing tests

Create `tests/unit/test_replay_op_mlx.py`:

```python
"""Unit tests for replay operation MLX backend (S9.1)."""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from kiki_oniric.dream.episode import (
    BudgetCap, DreamEpisode, EpisodeTrigger, Operation, OutputChannel,
)
from kiki_oniric.dream.operations.replay import (
    ReplayOpState, replay_handler_mlx,
)


class TinyMLP(nn.Module):
    """Minimal 2-layer MLP for replay tests (deterministic init)."""
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


def make_replay_episode(ep_id: str, records: list[dict]) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"beta_records": records},
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000, wall_time_s=1.0, energy_j=0.1),
        episode_id=ep_id,
    )


def test_replay_mlx_updates_loss() -> None:
    state = ReplayOpState()
    model = TinyMLP()
    handler = replay_handler_mlx(state=state, model=model, lr=0.01)
    records = [
        {"x": [0.1, 0.2, 0.3, 0.4], "y": [1.0, 0.0]},
        {"x": [0.5, 0.6, 0.7, 0.8], "y": [0.0, 1.0]},
    ]
    handler(make_replay_episode("de-mlx0", records))
    assert state.total_records_consumed == 2
    assert state.total_episodes_handled == 1
    assert state.last_loss is not None
    assert state.last_loss >= 0.0


def test_replay_mlx_handles_empty_records() -> None:
    state = ReplayOpState()
    model = TinyMLP()
    handler = replay_handler_mlx(state=state, model=model, lr=0.01)
    handler(make_replay_episode("de-mlx1", []))
    assert state.total_records_consumed == 0
    assert state.last_loss is None  # no batch → no loss
```

## Step S9.1.2 — Verify failing

Run: `uv run pytest tests/unit/test_replay_op_mlx.py -v --no-cov`
Expected: FAIL with `ImportError` from `replay_handler_mlx` (not yet defined).

## Step S9.1.3 — Implement MLX backend

Modify `kiki_oniric/dream/operations/replay.py` — ADD (don't replace existing skeleton):

```python
# Append at end of replay.py:

def replay_handler_mlx(
    state: ReplayOpState,
    model,  # mlx.nn.Module — typed loosely for lazy import
    lr: float = 0.01,
):
    """Build a replay handler with real MLX gradient updates.

    Records expected as `{"x": list[float], "y": list[float]}`.
    Forward pass + MSE loss + SGD step on model parameters.

    Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    optimizer = optim.SGD(learning_rate=lr)

    def loss_fn(params, x, y):
        model.update(params)
        pred = model(x)
        return mx.mean((pred - y) ** 2)

    grad_fn = nn.value_and_grad(model, loss_fn)

    def handler(episode):
        records = episode.input_slice.get("beta_records", [])
        state.total_episodes_handled += 1
        if not records:
            state.last_loss = None
            return
        xs = mx.array([r["x"] for r in records])
        ys = mx.array([r["y"] for r in records])
        loss, grads = grad_fn(model.parameters(), xs, ys)
        optimizer.update(model, grads)
        mx.eval(model.parameters())
        state.total_records_consumed += len(records)
        state.last_loss = float(loss.item())

    return handler
```

Also add `last_loss: float | None = None` field to `ReplayOpState` dataclass.

## Step S9.1.4 — Verify passing

Run: `uv run pytest tests/unit/test_replay_op_mlx.py -v --no-cov`
Expected: 2 passed.

Run: `uv run pytest`
Expected: 58 tests (56 + 2 new), coverage ≥90%.

## Step S9.1.5 — Commit + push

```bash
git add kiki_oniric/dream/operations/replay.py tests/unit/test_replay_op_mlx.py
git commit -m "feat(mlx): add replay handler MLX backend"
```

Subject: 41 chars. Body 3 paragraphs : MLX backend `replay_handler_mlx(state, model, lr)`, MSE loss + SGD optimizer, records `{"x", "y"}` schema, `last_loss` field added to ReplayOpState. Skeleton handler kept intact for tests not requiring MLX. Foundation for G2 GO-FULL upgrade.

Then `git push`.

---

# Task S9.2 — Downscale op MLX backend

Same pattern as S9.1, but for downscale: real `W *= shrink_factor` on `mx.array` model parameters.

**Files:**
- Modify : `kiki_oniric/dream/operations/downscale.py`
- Create : `tests/unit/test_downscale_op_mlx.py`

## Steps (abbreviated — mirror S9.1)

- TDD: 2 tests (factor applied to weights, NOT idempotent verified empirically)
- Implementation: `downscale_handler_mlx(state, model)` walks `model.parameters()` tree, multiplies leaf arrays by `factor`, calls `mx.eval()`
- Commit subject: `feat(mlx): add downscale handler MLX backend` (43 chars)
- Body explains : real shrinkage on parameters, compounds multiplicatively (compound_factor field already in DownscaleOpState)

Expected: 60 tests (58 + 2 new), coverage ≥90%.

---

# Task S9.3 — Retained eval bridge

**Goal** : connecter le retained benchmark (S3.4 SHA-256 frozen) au S1 swap guard via une fonction `evaluate_retained(model, benchmark)` qui retourne accuracy `[0, 1]`.

**Files:**
- Create : `kiki_oniric/dream/eval_retained.py`
- Create : `tests/unit/test_eval_retained.py`

## Pattern

- Function `evaluate_retained(model, benchmark) -> float` : itère sur benchmark items, calcule accuracy moyenne (predicted == expected proxy)
- 3 tests : empty benchmark returns 1.0, all-correct returns 1.0, half-wrong returns 0.5
- Commit `feat(eval): wire retained benchmark to S1 guard` (47 chars)

Expected: 63 tests, coverage ≥90%.

---

# Task S9.4 — P_min swap-integrated end-to-end

**Goal** : intégrer le swap protocol dans P_min avec MLX backend complet. End-to-end test qui :
1. Crée P_min avec model MLX
2. Exécute replay DE → forward + grad + SGD
3. Exécute downscale DE → shrink weights
4. Appelle swap_atomic avec evaluate_retained(model) en retained_eval
5. Vérifie commit

**Files:**
- Modify : `kiki_oniric/profiles/p_min.py` (add `model` field + `swap_now()` method)
- Create : `tests/unit/test_p_min_e2e.py`

## Pattern

- `PMinProfile.model: nn.Module | None = None`
- `PMinProfile.swap_now(retained_pre_acc, benchmark) -> SwapResult` invoque swap_atomic
- 2 e2e tests (success, abort on degraded retained)
- Commit `feat(profile): wire P_min swap E2E with MLX` (44 chars)

Expected: 65 tests, coverage ≥90%.

---

# Task S9.5 — G2 P_min report update GO-FULL

**Goal** : updater `docs/milestones/g2-pmin-report.md` avec evidence empirique. Run pilot mesurement P_min vs baseline sur retained benchmark (50 items synthétiques placeholder).

**Files:**
- Modify : `docs/milestones/g2-pmin-report.md`
- Create : `docs/milestones/g2-pilot-results.md` (data dump)

## Pattern

- Run `uv run python scripts/pilot_g2.py` (création one-shot script)
- Mesure baseline vs P_min sur 3 seeds × 50 items
- Si accuracy_p_min ≥ baseline − 2% → flip Branch GO-FULL ✅
- Si non → keep GO-CONDITIONAL, document path
- Commit `docs(milestone): update G2 with pilot results` (45 chars)

---

# Task S10.1 — restructure operation (D-Friston FEP)

**Goal** : implémenter restructure op qui modifie la topologie hiérarchique. Skeleton : tracker `TopologyDiff` events (add/remove/reroute layer), no real graph mutation yet.

**Files:**
- Create : `kiki_oniric/dream/operations/restructure.py`
- Create : `tests/unit/test_restructure_op.py`

## Pattern

- Dataclass `RestructureOpState` : tracks total_diffs_emitted, last_diff_type
- `restructure_handler(state)` factory consumes `episode.input_slice["topo_op"]` ∈ {"add", "remove", "reroute"}
- 3 tests
- Commit `feat(dream): add restructure op (D-Friston)` (43 chars)

---

# Task S10.2 — S3 topology guard

**Goal** : implémenter S3 invariant (validate_topology) comme guard utilisable par swap. Skeleton : graphe représenté comme dict adjacency, vérifie connectivité + pas de cycle non-désiré + layer counts dans bornes.

**Files:**
- Create : `kiki_oniric/dream/guards/topology.py`
- Create : `tests/unit/test_topology_guard.py`
- Create : `tests/conformance/invariants/test_s3_topology.py`

## Pattern

- `validate_topology(graph: dict[str, list[str]]) -> None` raises `TopologyGuardError`
- Checks : species connectivity (ρ_phono → ρ_lex → ρ_syntax → ρ_sem reachable), no self-loops, layer counts ≤ max
- 5 unit tests + 2 conformance tests
- Commit `feat(guard): add S3 topology check` (33 chars)

---

# Task S11.1 — recombine operation (C-Hobson VAE light)

**Goal** : implémenter recombine op qui sample latents depuis un VAE-style sampler. Light version : interpolation lineaire entre 2 latents tirés au hasard de δ snapshots.

**Files:**
- Create : `kiki_oniric/dream/operations/recombine.py`
- Create : `tests/unit/test_recombine_op.py`

## Pattern

- `RecombineOpState` : tracks total_samples_emitted, last_sample
- `recombine_handler(state, sampler)` consumes δ snapshot, samples 2 latents, interpolates with `α ∈ U(0,1)`, emits `LatentSample`
- 3 tests
- Commit `feat(dream): add recombine op (C-Hobson light)` (45 chars)

---

# Task S11.2 — P_equ profile fully wired

**Goal** : assembler P_equ = β + δ → 1 + 3 + 4 channels avec 4 ops {replay, downscale, restructure, recombine_light}. Replace S8.3 skeleton.

**Files:**
- Modify : `kiki_oniric/profiles/p_equ.py` (replace skeleton)
- Create : `tests/unit/test_p_equ_wiring.py`

## Pattern

- `PEquProfile` similar to `PMinProfile` but adds restructure_state + recombine_state
- 4 handlers registered on runtime
- 4 tests : registers 4 ops, executes 4-op DE, channels 1+3+4 emitted, log integrity
- Commit `feat(profile): wire P_equ (4 ops + 3 chans)` (43 chars)

---

# Task S12.1 — DR-4 profile inclusion proof + axiom test

**Goal** : démontrer DR-4 (P_min ⊆ P_equ ⊆ P_max ops + channels) formellement + property test.

**Files:**
- Create : `docs/proofs/dr4-profile-inclusion.md`
- Create : `tests/conformance/axioms/test_dr4_profile_inclusion.py`

## Pattern

- Proof doc : ops/channels chain inclusion, monotonicity lemma DR-4.L (P_equ better-or-equal P_min on monotone metrics in capacity)
- Property test : enumerate ops/channels of each profile, assert inclusion holds
- Commit `docs(proof): DR-4 profile inclusion + tests` (44 chars)

---

# Task S12.2 — G4 P_equ functional report

**Goal** : G4 gate report — P_equ > P_min on ≥2 metrics significant, invariants 7d green. Skeleton if S9-S11 not full empirical yet.

**Files:**
- Create : `docs/milestones/g4-pequ-report.md`

## Pattern

- Same structure as g2-pmin-report.md
- Branches : GO-FULL (significant evidence), GO-CONDITIONAL (skeleton only), NO-GO (Pivot — skip P_max)
- Commit `docs(milestone): G4 P_equ functional report` (44 chars)

---

# Self-review

**1. Spec coverage** :
- S9 MLX integration → S9.1 (replay) + S9.2 (downscale) + S9.3 (retained bridge) + S9.4 (P_min E2E) ✅
- S9 G2 update → S9.5 ✅
- S10 restructure + S3 → S10.1 + S10.2 ✅
- S11 recombine + P_equ wiring → S11.1 + S11.2 ✅
- S12 DR-4 + G4 → S12.1 + S12.2 ✅

**2. Placeholder scan** : aucun TBD non-intentionnel dans les code blocks. Les détails abrégés (S9.2-S12.2) sont délibérés vu la longueur — chaque task a son pattern décrit + commit subject pré-validé.

**3. Type consistency** :
- `replay_handler_mlx`, `downscale_handler_mlx` (S9.1, S9.2) → cohabitent avec skeleton handlers existants
- `RestructureOpState`, `RecombineOpState` (S10.1, S11.1) → consommés par P_equ wiring (S11.2)
- `validate_topology`, `TopologyGuardError` (S10.2) → utilisés par swap S9.4 (extend)
- `PMinProfile.model` field (S9.4) → cohérent avec S7.4 dataclass
- `PEquProfile` complet (S11.2) replace skeleton S8.3 backward-compat

**4. Commit count** : 11 commits prévus (S9.1-S9.5 + S10.1-S10.2 + S11.1-S11.2 + S12.1-S12.2 = 11 ; le plan dit 8 mais inclus S9.5 + 2 docs = 11).

**5. Validator risks** : tous subjects pré-vérifiés ≤50 chars. Risque MLX : tests skip si MLX indisponible (utilisation `pytest.importorskip("mlx.core")`).

---

**End of S9-S12 atomic plan.**

**Version** : v0.1.0
**Generated** : 2026-04-18 via refinement of S9-S12 from main plan
**Source** : `docs/superpowers/plans/2026-04-17-dreamofkiki-implementation.md`
