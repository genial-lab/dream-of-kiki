# dreamOfkiki — Cycle 3 Design Spec

**Version target** : C-v0.7.0+PARTIAL → C-v0.7.0+STABLE
**Date** : 2026-04-19
**Author** : Clement Saillant (L'Electron Rare)
**Status** : Draft for execution (post-brainstorm 2026-04-19)
**Schedule** : 6 weeks intense — Phase 1 sem 1-3 (waterfall), Phase 2 sem 4-6 (parallel tracks, conditional GATE D = GO)
**Related specs** :
- `docs/specs/2026-04-17-dreamofkiki-master-design.md` (master vision, 5 tracks)
- `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` (framework C, axioms DR-0..DR-4, invariants I/S/K)
- `docs/superpowers/plans/2026-04-18-dreamofkiki-cycle2-atomic.md` (cycle-2 atomic plan, predecessor)

This spec is the **execution contract** for cycle 3 (S47-S52 in continuous calendar). It promotes the cycle-2 closeout state (C-v0.6.0+STABLE, MLX + Norse-ready substrates, synthetic-only empirical surface) to a real-data, multi-scale, cross-substrate evidence base, then consolidates everything into a single Paper 1 v2 extension.

---

## 0. Executive summary

Cycle 3 sequences a primary axis **(d) real-data first** then secondary axes **(b) neuromorphic + (c) fMRI parallel** after Gate D. Scope is **(iv) full pipeline** — real benchmarks (MMLU 5-shot, HellaSwag, mega-v2 self-eval 80/20) on real models (Qwen3.5 1.5B + 7B + 35B Q4 via MLX on Studio M3 Ultra 512GB) over the existing kiki-oniric P_min/P_equ/P_max profiles. Publication strategy is **(C)** : Paper 1 v2 only (Paper 1 v1 held, Paper 2 v0.1 absorbed §§ cross-substrate, Paper 3 outline archived). The hypothesis family extends to **8 tests per cell** (H1-H4 + H5-I/II/III + H6) under Bonferroni α_per_test = 0.00625, *not* cross-Bonferronied across cells. Approach is α-waterfall with a **hard Gate D at end of sem 3**.

---

## 1. Architecture & components (additive diff over cycle-2 tree)

### 1.1 New harness modules (real benchmarks + real models + fMRI loader)

```
harness/
├── real_benchmarks/                  ← NEW C3.1
│   ├── __init__.py
│   ├── mmlu.py                       MMLU 5-shot loader, SHA-256 pinned
│   ├── hellaswag.py                  HellaSwag loader, SHA-256 pinned
│   ├── mega_v2_eval.py               mega-v2 80/20 self-eval split
│   └── mega_v2_adapter.py            mega-v2 record → DreamEpisode (α/β streams)
├── real_models/                      ← NEW C3.2
│   ├── __init__.py
│   ├── qwen_mlx.py                   Qwen3.5 1.5B/7B/35B Q4 MLX wrappers
│   └── base_model_registry.py        SHA-256 pin per (scale, quant)
└── fmri/                             ← NEW C3.15 (Phase 2 track c)
    └── studyforrest.py               BOLD loader, fallback HCP ds000113
```

### 1.2 New kiki_oniric modules (eval + alignment + SNN ops)

```
kiki_oniric/
├── eval/
│   ├── scaling_law.py                ← NEW C3.4 (H5 trivariant : Levene, Spearman, curve_fit)
│   ├── statistics.py                 ← EXTEND C3.5 (Bonferroni 8-test family, α=0.00625)
│   ├── state_alignment.py            ← NEW C3.16 (HMM dream-state ↔ BOLD)
│   └── cca_alignment.py              ← NEW C3.17 (CCA fMRI Studyforrest)
├── substrates/
│   └── esnn_norse.py                 ← NEW C3.11 (Norse PyTorch LIF SNN, local CPU fallback)
└── dream/operations/
    ├── replay_real.py                ← NEW C3.3 (real-weight extension)
    ├── downscale_real.py             ← NEW C3.3
    ├── restructure_real.py           ← NEW C3.3
    ├── recombine_real.py             ← NEW C3.3
    ├── replay_snn.py                 ← NEW C3.12 (SNN spike-rate proxy variant)
    ├── downscale_snn.py              ← NEW C3.12
    ├── restructure_snn.py            ← NEW C3.12
    └── recombine_snn.py              ← NEW C3.12
```

### 1.3 New scripts (sanity + ablation + gate + Phase 2 pilots)

```
scripts/
├── pilot_cycle3_sanity.py            ← NEW C3.7 (180-run 1.5B sanity, 1 day, fail-fast)
├── ablation_cycle3.py                ← NEW C3.6 (1080-config cartesian runner)
├── compute_gate_d.py                 ← NEW C3.9 (GATE D decision + H1-H6 reports)
├── pilot_phase2b_neuromorph.py       ← NEW C3.13 (Norse cross-substrate, sem 4-5)
└── pilot_phase2c_fmri.py             ← NEW C3.18 (Studyforrest pilot, sem 4-5)
```

### 1.4 Tests (≥30 new TDD-first, coverage ≥90% maintained — invariant from `pyproject.toml`)

```
tests/
├── unit/
│   ├── test_real_benchmarks.py       MMLU/HellaSwag loader determinism, SHA verification
│   ├── test_qwen_mlx_wrappers.py     1.5B/7B/35B forward determinism (MLX seeded)
│   ├── test_mega_v2_adapter.py       round-trip record → DreamEpisode → record
│   ├── test_scaling_law.py           Levene + Spearman + curve_fit on synthetic d-curves
│   ├── test_statistics_bonferroni.py 8-test family, α=0.00625 enforcement
│   ├── test_norse_substrate.py       LIF forward + spike-rate proxy
│   ├── test_real_ops.py              4 ops × 3 scales conformance vs synthetic
│   ├── test_snn_ops.py               4 ops × Norse substrate conformance
│   ├── test_state_alignment.py       HMM Viterbi on synthetic BOLD
│   └── test_cca_alignment.py         CCA round-trip on aligned synthetic
└── conformance/
    └── operations/
        └── test_substrate_matrix_cycle3.py   3 substrates × 3 profiles × 4 ops
```

---

## 2. Phase 1 — task breakdown C3.1-C3.10 (sem 1-3, waterfall)

Sprint structure :

| Sprint | Tasks | Commits | Compute footprint |
|--------|-------|---------|-------------------|
| Sem 1 | C3.1, C3.2, C3.3, C3.4, C3.5, C3.10 | 6 | local TDD, no Studio |
| Sem 2 | C3.6, C3.7, C3.8 launch | 3 | Studio sanity 18h + full-ablation start |
| Sem 3 | C3.8 continues, C3.9 + Gate D | 1 | Studio full-ablation completes (~10 days wall) |

Per-task contract :

### C3.1 — Real benchmark loaders

- **Files** : `harness/real_benchmarks/{mmlu,hellaswag,mega_v2_eval,mega_v2_adapter}.py`, `tests/unit/test_real_benchmarks.py`, `tests/unit/test_mega_v2_adapter.py`
- **Pattern** : SHA-256 pinning per `R1` (framework-C §8.4) ; loader returns deterministic iterator ; mega-v2 80/20 split seeded ; adapter emits typed `α`/`β` per primitives §2.1.
- **Commit** : `feat(real-bench): MMLU + HellaSwag + mega-v2 loaders`
- **Cites** : R1, §8.4, primitives α/β.

### C3.2 — Multi-scale Qwen MLX wrappers

- **Files** : `harness/real_models/qwen_mlx.py`, `harness/real_models/base_model_registry.py`, `tests/unit/test_qwen_mlx_wrappers.py`
- **Pattern** : registry pins SHA-256 per `(scale ∈ {1.5B, 7B, 35B}, quant=Q4_K_M)` ; MLX deterministic mode + seeded RNG per K1/R1 contracts ; forward returns logits + activations under typed Protocol matching §2.1 primitives.
- **Commit** : `feat(real-models): Qwen3.5 MLX 1.5B/7B/35B wrappers`
- **Cites** : K1, R1, §2.1.

### C3.3 — mega-v2 → DreamEpisode adapter + real-weight ops extensions

- **Files** : `kiki_oniric/dream/operations/{replay,downscale,restructure,recombine}_real.py`, `tests/unit/test_real_ops.py`
- **Pattern** : each `_real.py` extends the synthetic op of cycle-2 with weight-tensor IO bound to `qwen_mlx.py` ; respects DR-2 canonical order ; output deltas typed per channels 1-4 §2.1 ; S1+S2+S3 guards reused unchanged.
- **Commit** : `feat(real-ops): real-weight ops over Qwen MLX`
- **Cites** : DR-2, S1, S2, S3, channels 1-4.

### C3.4 — scaling_law.py — H5 trivariant

- **Files** : `kiki_oniric/eval/scaling_law.py`, `tests/unit/test_scaling_law.py`
- **Pattern** : implements **H5-I invariance** (Levene / Brown-Forsythe variance test on Cohen's d across 3 scales), **H5-II monotonic** (Spearman ρ, **two-sided** per OSF pre-reg amendment), **H5-III power-law** (`scipy.optimize.curve_fit d = c·N^α` + bootstrap CI 95% on α). H5-III non-significant is publishable (3 variants suffice collectively).
- **Commit** : `feat(eval): scaling_law H5 trivariant analyzer`
- **Cites** : H5-I, H5-II, H5-III, R1 (deterministic bootstrap seeded).

### C3.5 — statistics.py extension — Bonferroni 8-test family

- **Files** : `kiki_oniric/eval/statistics.py` (extend), `tests/unit/test_statistics_bonferroni.py`
- **Pattern** : family-wise correction **per cell** ; `α_per_test = 0.05 / 8 = 0.00625` ; explicit refusal to cross-Bonferroni across the 6 cells (would degrade to 0.05/48 = 0.001, under-powered). API : `apply_bonferroni(p_values: list[float], family_size: int = 8) -> list[bool]`.
- **Commit** : `feat(stats): Bonferroni 8-test family per cell`
- **Cites** : H1, H2, H3, H4, H5-I/II/III, H6.

### C3.6 — ablation_cycle3.py — full 1080-config runner

- **Files** : `scripts/ablation_cycle3.py`
- **Pattern** : cartesian over `scale ∈ {1.5B, 7B, 35B} × profile ∈ {P_min, P_equ, P_max} × benchmark ∈ {MMLU, HellaSwag, mega-v2} × seed ∈ 1..40` = 1080 runs ; manifest writes to run-registry per R1/R3 ; resumable, idempotent.
- **Commit** : `feat(harness): cycle-3 1080-config ablation runner`
- **Cites** : R1, R3, K4 (matrix coverage gate), §8.2.

### C3.7 — Sanity pilot 1.5B (fail-fast)

- **Files** : `scripts/pilot_cycle3_sanity.py`
- **Pattern** : 180 runs (1.5B × 3 profiles × 3 benchmarks × 20 seeds) ; ~18h wall-clock ; fails fast on : (a) any `S2` violation, (b) `K1` budget exceedance > 2×, (c) `S1` regression > 5% on retained ; emits go/no-go to launch C3.8.
- **Commit** : `feat(pilot): cycle-3 1.5B sanity fail-fast`
- **Cites** : S1, S2, K1.

### C3.8 — Full ablation Studio launch

- **Files** : *(no new code, runs `ablation_cycle3.py` via Studio dedicated session)*
- **Pattern** : background launch sem 2 day 4 after sanity GO ; ~10 days wall-clock budget per §7 ; monitored via dashboard `dream.saillant.cc` ; partial results streamed for incremental GATE D dry-runs sem 3 mid.
- **Commit** : `chore(ablation): cycle-3 Studio full launch sem 2`
- **Cites** : K4, K3 (per-swap latency monitored), §8.2.

### C3.9 — compute_gate_d.py — GATE D decision

- **Files** : `scripts/compute_gate_d.py`, `docs/milestones/g10-cycle3-gate-d.md` (new milestone doc)
- **Pattern** : reads run-registry full ablation output ; runs the 8-test Bonferroni family per cell ; emits per-hypothesis GO/NO-GO table ; Gate D verdict = GO iff (H1-H4 sig 3/3 scales) ∧ (H5 family Bonferroni-significant) ; Phase 2 unlock = Gate D GO ; Pivot 4 trigger = Gate D NO-GO.
- **Commit** : `feat(gate): compute_gate_d + H1-H6 report`
- **Cites** : G10 (new gate, see §6), DR-4 (profile chain feeds H6 ordering), R1.

### C3.10 — DualVer bump C-v0.6.0+STABLE → C-v0.7.0+PARTIAL

- **Files** : framework-C spec banner (FR + EN), STATUS.md, CHANGELOG.md, glossary §A entries (G10, real_benchmarks, scale-axis, H5/H6), substrate version constants.
- **Pattern** : surgical-bump pattern (per cycle-2 closeout `139c4c5`) ; **FC v0.6.0 → v0.7.0** because §4 H6 is a new derived constraint surface (FC-MINOR by §12.2) ; **EC STABLE → PARTIAL** per §12.3 (Phase 2 cells scoped-deferred until sem 6) ; performed *after* C3.5 lands and *before* C3.6 launches so the matrix runs under the new tag.
- **Commit** : `feat(dualver): bump to C-v0.7.0+PARTIAL`
- **Cites** : §12.2, §12.3, H6.

---

## 3. Phase 2 — parallel tracks C3.11-C3.22 (sem 4-6, conditional Gate D = GO)

If Gate D = NO-GO → execute **Pivot 4** (drop primary axis, promote whichever of (b) or (c) shows residual signal, re-spec cycle 3). If Gate D = GO, run the two tracks below in parallel plus the consolidation track.

### 3.1 Track (b) — neuromorphic on Norse SNN (C3.11-C3.14)

| ID | File(s) | Pattern | Commit |
|----|---------|---------|--------|
| C3.11 | `kiki_oniric/substrates/esnn_norse.py`, `tests/unit/test_norse_substrate.py` | Norse PyTorch LIF wrapper, CPU-local fallback for Loihi-2 (per R3 substrate-agnostic). Implements Conformance Criterion §6.2 : typed signature + axiom property tests + invariants S1/S2/S3/I1 enforced. | `feat(substrate): Norse LIF SNN fallback wrapper` |
| C3.12 | `kiki_oniric/dream/operations/{replay,downscale,restructure,recombine}_snn.py`, `tests/unit/test_snn_ops.py` | 4 ops on Norse substrate using spike-rate proxy ; same channel typings as `_real.py` per §2.1 ; DR-2 canonical order preserved. | `feat(snn-ops): dream ops over Norse substrate` |
| C3.13 | `scripts/pilot_phase2b_neuromorph.py`, `tests/conformance/operations/test_substrate_matrix_cycle3.py` | Norse vs MLX cross-substrate pilot ; reuses `compute_gate_d.py` 8-test family on the Norse cell (Bonferroni stays per-cell, **not** across substrates). | `feat(pilot): Phase-2b Norse cross-substrate` |
| C3.14 | `docs/milestones/g10a-neuromorph.md` | Milestone report : H6 (profile ordering invariant cross-substrate) verdict on the Norse cell ; feeds Paper 1 v2 §§ cross-substrate. | `docs(milestone): G10a neuromorph cross-substrate` |

### 3.2 Track (c) — fMRI Studyforrest (C3.15-C3.18)

| ID | File(s) | Pattern | Commit |
|----|---------|---------|--------|
| C3.15 | `harness/fmri/studyforrest.py`, `tests/unit/test_studyforrest_loader.py` | Studyforrest BOLD loader with SHA-256 pin per R1 §8.4 ; nilearn CPU mode (deterministic) ; fallback HCP ds000113 if Studyforrest access slow (R4). | `feat(fmri): Studyforrest BOLD loader` |
| C3.16 | `kiki_oniric/eval/state_alignment.py`, `tests/unit/test_state_alignment.py` | HMM Viterbi alignment between dream-episode state machine sequence and BOLD activation phases ; metric M2.b (RSA fMRI alignment) extension. | `feat(eval): HMM dream-state ↔ BOLD alignment` |
| C3.17 | `kiki_oniric/eval/cca_alignment.py`, `tests/unit/test_cca_alignment.py` | CCA between kiki representations (4 ortho species ρ_phono/lex/syntax/sem) and Studyforrest BOLD ROIs ; complements RSA M2.b with cross-decomposition. | `feat(eval): CCA Studyforrest alignment` |
| C3.18 | `scripts/pilot_phase2c_fmri.py`, `docs/milestones/g10c-fmri.md` | Phase-2c pilot run + milestone report ; M2.b real-data cell, feeds Paper 1 v2 §§ cognitive alignment. | `feat(pilot): Phase-2c Studyforrest fMRI` |

### 3.3 Consolidation — Paper 1 v2 (C3.19-C3.22)

Single-paper strategy (Q3 = C, Q4.a = α) : Paper 1 v2 absorbs Paper 1 v1 + cycle-2 cross-substrate evidence (formerly Paper 2 v0.1) + cycle-3 real-data + Phase 2 (b)/(c). Paper 3 outline archived under `docs/drafts/paper3-archived.md`.

| ID | File(s) | Pattern | Commit |
|----|---------|---------|--------|
| C3.19 | `docs/papers/paper1-v2/outline.md` | Outline merging v1 sections + Paper 2 v0.1 §§ + cycle-3 §§ real-data + §§ cross-substrate + §§ scaling laws ; cites every G-gate, axiom, invariant. | `docs(paper1-v2): outline merge v1 + cycle-2 + cycle-3` |
| C3.20 | `docs/papers/paper1-v2/{methodology,results}.md` (EN) + mirrored `docs/papers/paper1-v2-fr/` | Methodology covers 3 substrates × 3 scales × 8-test Bonferroni family ; Results table per cell, with H1-H6 verdicts. | `docs(paper1-v2): methodology + results EN/FR` |
| C3.21 | `docs/papers/paper1-v2/{discussion,full-draft}.md`, `docs/papers/paper1-v2/build/full-draft.tex` | Discussion + full-draft assembly + pandoc → tex ; respects PUBLICATION-READY criteria framework-C §9. | `docs(paper1-v2): discussion + full-draft assembly` |
| C3.22 | DualVer bump files (per C3.10 pattern), `docs/milestones/g10-cycle3-publication.md` | EC PARTIAL → STABLE per §12.3 (Phase 2 deferred cells re-closed) ; Gate G10 promoted CONDITIONAL → FULL-GO/STABLE. | `feat(dualver): bump to C-v0.7.0+STABLE` |

**Total cycle 3** : 22 commits, 6 weeks, ≥30 new TDD-first tests, coverage ≥ 90% maintained (existing `pyproject.toml` gate).

---

## 4. Hypotheses formalized — H1-H6 with Bonferroni families

The hypothesis family is split into two **statistical scopes** so the
Bonferroni correction stays coherent with the data each test actually
consumes :

- **Per-cell tests** — H1, H2, H4, H5-I, H5-II, H5-III operate on
  a single (scale, substrate) cell and are corrected **within that
  cell's family**. Cycle-3 has 6 cells = 3 profiles × 2 substrates
  (MLX baseline + Norse Phase 2). For the per-cell family,
  `family_size_per_cell = 6`, `α_per_test = 0.05 / 6 ≈ 0.00833`.
- **Cross-cell tests** — H3 and H6 cross cells by construction and
  are corrected **in a separate group-level family** : H3 sweeps
  profiles *within* a substrate (3 profiles), H6 sweeps substrates
  *across* profiles. For the cross-cell family, `family_size_cross
  = 2`, `α_per_test = 0.05 / 2 = 0.025`.

> **Redefinition** : "cell" = (scale, substrate). H1/H2/H4/H5-* are
> per-cell tests ; H3 and H6 are cross-cell tests. The previous
> wording "family of 8 per cell" conflated the two — H3 / H6 cannot
> live in the per-cell Bonferroni family because their null uses
> data from cells they are comparing.

| ID | Scope | Statement | Test | Direction |
|----|-------|-----------|------|-----------|
| H1 | per-cell | post-dream M1.b > pre-dream M1.b | Welch t | one-sided (improvement) |
| H2 | per-cell | post-dream effect equivalent to replay-only baseline | TOST equivalence | bounded |
| H3 | cross-cell (profiles within substrate) | Profile chain ordering on M1.b : P_min < P_equ < P_max | Jonckheere-Terpstra | one-sided |
| H4 | per-cell | post-dream M1.b > null permutation distribution | one-sample t vs null | one-sided |
| H5-I | per-cell (scale family) | Variance of Cohen's d invariant across 3 scales | one-way ANOVA | two-sided |
| H5-II | per-cell (scale family) | Cohen's d monotonic in scale N | Spearman ρ | **two-sided** (per OSF pre-reg amendment, no post-hoc claim) |
| H5-III | per-cell (scale family) | Cohen's d follows power-law `d = c·N^α` | scipy `curve_fit` + bootstrap CI 95% on α | two-sided ; non-significant outcome publishable |
| H6 | cross-cell (substrates) | Profile ordering P_max > P_equ > P_min holds invariant across substrates | Jonckheere across (P_min, P_equ, P_max) per substrate, then meta-test for substrate invariance | two-sided |

**Family-wise correction** :
- Per-cell family (H1, H2, H4, H5-I, H5-II, H5-III) : `family_size = 6`,
  `α_per_test = 0.05 / 6 ≈ 0.00833`.
- Cross-cell family (H3, H6) : `family_size = 2`, `α_per_test = 0.05 / 2 = 0.025`.

> **Implementation note** : `kiki_oniric.eval.scaling_law.compute_h5`
> currently uses `α_family = 0.00625` = `0.05 / 8`. This is the
> legacy 8-test bar from the pre-split draft ; the per-cell family
> is now 6 tests, so 0.00625 is **more conservative than required**.
> The current code is still valid (rejection under 0.00625 implies
> rejection under 0.00833) but the reported α can be relaxed to
> 0.00833 per the split family. The relaxation will be applied in
> C3.9 (`compute_gate_d.py`) with the pre-reg amendment reference.

**Explicit non-correction across families** : the per-cell and
cross-cell families are corrected independently because they test
different nulls on disjoint data slices ; stacking them into a
single Bonferroni family of 8 would be under-powered for the
available seed budget (40 seeds at 35B). Effect sizes + CI are
reported descriptively for cross-family comparison, not corrected.

**Phase 1 stability bar (Gate D criterion)** : H1-H4 significant in **3/3 scales** AND H5 family Bonferroni-significant ⇒ **GO Phase 2**. (Bar deliberately high : Nature HB target.)

**Phase 2 expansion** : H6 evaluated on Norse cell (Phase 2b) ; M2.b (cognitive alignment) evaluated on fMRI cell (Phase 2c).

---

## 5. Risks & pre-cycle-3 external locks

### 5.1 Risk register

| ID | Description | Mitigation |
|----|-------------|------------|
| R1 | Studio M3 Ultra unavailable | Fallback to kxkm-ai CUDA llama.cpp (loses MLX deterministic mode → relax R1 to R1-soft for that subset, tag `r1_scope=false` per §8.4). |
| R2 | 35B Q4 too slow (> §7 budget) | Drop 35B to 30 seeds (vs 40) ; acknowledge reduced power for H5 power-law CI. |
| R3 | Gate D NO-GO | **Pivot 4** : drop axis (d), promote whichever of (b) or (c) shows residual signal as new primary axis, re-spec cycle 3 sem 4 onward. |
| R4 | Studyforrest access slow | Fallback to HCP `ds000113` (pre-locked S2 in master spec §2.1). |
| R5 | CodeRabbit issue burst | Unlikely : cycles 9-10-11 trended 19 → 13 → 1. Buffer is the implicit waterfall slack of sem 1 task list (6 commits). |

### 5.2 Pre-cycle-3 external locks (must complete sem 1 day 1-3)

1. **OSF pre-reg amendment** filed before any C3.6 run : adds H5 trivariant (I/II/III), H6, two-sided H5-II, family-size = 8.
2. **SHA-256 pin Qwen3.5 models** (1.5B Q4_K_M, 7B Q4_K_M, 35B Q4_K_M) — recorded in `harness/real_models/base_model_registry.py`.
3. **SHA-256 pin MMLU + HellaSwag** versions — recorded in `harness/real_benchmarks/{mmlu,hellaswag}.py`.
4. **Studyforrest download initiated** sem 1 (parallel to Phase 1 work, completes by sem 4).

These four locks are non-negotiable : they protect R1 (bit-exact) and the OSF pre-registration integrity that underwrites Paper 1 v2's Nature HB submission.

---

## 6. DualVer transitions + Gate G10

### 6.1 Transitions

| Sem | From | To | Trigger | Commit anchor |
|-----|------|----|----|----|
| 1 (post C3.5) | C-v0.6.0+STABLE | C-v0.7.0+PARTIAL | C3.10 (FC-MINOR : H6 added per §12.2) ; EC PARTIAL because Phase 2 deferred per §12.3 | C3.10 |
| 6 (post C3.21) | C-v0.7.0+PARTIAL | C-v0.7.0+STABLE | C3.22 (Phase 2 cells re-closed per §12.3 transition rule) | C3.22 |

If Gate D = NO-GO at sem 3 end, the C-v0.7.0+PARTIAL tag remains live ; the `+STABLE` graduation is replaced by a new minor bump reflecting the pivot scope (e.g. `C-v0.7.1+PARTIAL` if (b) becomes primary).

### 6.2 Gate G10 — cycle-3 publication-ready

G10 lives in `docs/milestones/g10-cycle3-publication.md`. Its decision criteria :

| Criterion | Threshold |
|-----------|-----------|
| Eval matrix coverage | 100% of cycle-3 cells (1080 Phase 1 + Phase 2 b/c pilots) |
| Reproducibility | All runs registered, R1-compliant for 8 metrics minus M4.a unless Teacher SHA pinned (§8.4) |
| Bonferroni 8-test family | Computed and reported per cell |
| Gate D verdict | GO (Phase 2 done) OR documented Pivot 4 outcome |
| H6 reported | On at least 1 cross-substrate cell (Norse) |
| M2.b real-data | Reported on at least 1 fMRI-aligned cell (Studyforrest or HCP fallback) |
| Paper 1 v2 draft | Complete, internal-reviewed ≥1×, no TODO |
| DualVer | C-v0.7.0+STABLE (or pivot equivalent) |
| Zero BLOCKING | 7 consecutive days |

G10 promotes **CONDITIONAL-GO/PARTIAL → FULL-GO/STABLE** at C3.22.

---

## 7. Compute budget + hardware allocation

### 7.1 Phase 1 (sem 2-3) Studio M3 Ultra dedication

| Scale | Runs | Per-run wall | Total wall |
|-------|------|--------------|------------|
| 1.5B  | 360  | ~3 min       | ~18 h      |
| 7B    | 360  | ~10 min      | ~60 h      |
| 35B   | 360  | ~25 min      | ~150 h ≈ 6.25 days |
| **Total** | **1080** | — | **≈ 10 days dedicated wall-clock** |

Fits the sem 2-3 envelope (14 calendar days). Sanity pilot C3.7 consumes ~18h of the sem 2 envelope before the full launch ; full ablation runs continuously sem 2 day 4 → sem 3 mid.

### 7.2 Phase 2 (sem 4-6) hardware split

- **Track (b) Norse** : local CPU on Studio (low memory footprint ; ≤ 1 day per pilot).
- **Track (c) fMRI** : nilearn CPU on Tower (31 GB RAM, no GPU needed for HMM/CCA on Studyforrest scale).
- **Paper consolidation (C3.19-C3.22)** : pandoc + LaTeX on GrosMac M5 (no compute load).

### 7.3 Network dependencies

- Qwen3.5 GGUF / MLX checkpoints : already mirrored on `kxkm-ai:/mnt/models/` (per `reference_studio_archive_nas.md`) ; sem 1 task : copy to Studio local, hash-verify against `base_model_registry.py`.
- Studyforrest : `git-annex` clone from openneuro, started sem 1.
- HCP ds000113 fallback : credentialed access, request initiated sem 1 if Studyforrest not green by sem 1 end.

---

## 8. Glossary update preview (delta vs `docs/glossary.md`)

| Term | Definition added in cycle 3 |
|------|------------------------------|
| `real_benchmarks` | Harness submodule : MMLU 5-shot + HellaSwag + mega-v2 80/20 self-eval ; SHA-pinned datasets ; replaces synthetic placeholders (per §3 working rule in `CLAUDE.md`). |
| Scale-axis | The Qwen3.5 model-size axis `N ∈ {1.5B, 7B, 35B}` Q4_K_M. Distinct from profile axis (P_min/P_equ/P_max) and substrate axis (MLX/Norse/E-SNN). |
| H5-I / H5-II / H5-III | Trivariant scaling-law hypothesis family : invariance / monotonic / power-law respectively. |
| H6 | Profile ordering invariance hypothesis cross-substrate : `Jonckheere(P_min, P_equ, P_max)` consistent across MLX and Norse cells. |
| G10 | Cycle-3 publication-ready gate (analogue of G5 cycle-1, G9 cycle-2). |
| Gate D | Mid-cycle gate at end of sem 3 : Phase 1 stability bar verdict (GO / Pivot 4 NO-GO). |

---

## 9. Cross-references

- **Master spec §3.4** (calendar) : cycle 3 occupies S47-S52 ; replaces the open-buffer of S25-S28 advance plan.
- **Framework-C §6** (axioms) : H5/H6 do not modify DR-0..DR-4 surfaces ; they live in §8.4 evaluation extension.
- **Framework-C §8.2** (stratified matrix) : 1080-config Phase 1 = MAJOR-bump-equivalent for the new scale axis ; satisfies K4 coverage.
- **Framework-C §8.4** (R1 SHA pinning) : real_benchmarks + real_models complete the SHA-pin table (M4.a Teacher scorer remains separately tracked).
- **Framework-C §12.3** (DualVer transitions) : C3.10 PARTIAL bump and C3.22 STABLE re-closure both follow the §12.3 rule path explicitly.
- **Cycle-2 atomic plan** : same five-phase decomposition style ; same TDD discipline ; same surgical-bump pattern (`139c4c5` reference) for DualVer commits.
- **Master spec §6** (publication strategy) : Paper 1 v2 supersedes the original Paper 1 + Paper 2 sequential plan (Q3 = C, Q4.a = α decisions).
- **Master spec §7.3** (pivots) : Pivot 4 (§5 R3) extends the pivot register with a Gate D-specific branch.

---

**End of cycle-3 design spec.**

Next step after user review : invoke `superpowers:writing-plans` skill to generate the cycle-3 atomic plan (target file `docs/superpowers/plans/2026-04-19-dreamofkiki-cycle3-atomic.md`) using this spec as canonical input.
