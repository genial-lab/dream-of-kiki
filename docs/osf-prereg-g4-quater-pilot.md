# G4-quater pilot pre-registration

**Date:** 2026-05-03
**Parent OSF:** 10.17605/OSF.IO/Q6JYN
**Sister pilot:** G4-ter (g_h2 = +2.77, H_DR4 inverted)
**Substrate:** MLX hierarchical (G4-ter 3-layer + new 5-layer)
**Benchmark:** Split-FMNIST 5 binary tasks
**Lock commit:** `e7232b9` (this file's first commit)
**Lock timestamp:** 2026-05-03 18:52:41 CEST (verified `e7232b9` < first pilot run `e7d74e8` at 19:11:30, ~19 min gap)

## §1 Background

G4-ter (commit `98b1305`) reported g_h2 = +2.77 (Welch
p = 4.9e-14, N = 30) under the hierarchical substrate, but
observed retention `P_min = 0.7065, P_equ = 0.7046,
P_max = 0.7046` is the inverse of the DR-4 prediction
`P_max ≥ P_equ ≥ P_min`. Reframe (commit `751370d`) treats
this as a **partial refutation** of the framework-C claim that
richer op-sets yield richer consolidation.

The Hu 2020 anchor (g = 0.29, human sleep-dependent memory
consolidation) is used here strictly as a **directional**
reference — sign of the alternative hypothesis — not as a
magnitude calibrator. A cross-class comparison between a
biological cohort and a seeded numerical pilot would be a
category error.

## §2 Hypotheses (confirmatory)

- **H4-A (substrate-depth)** : on a deeper hierarchical head
  (5 layers, hidden 64-32-16-8), the predicted profile
  ordering `P_max ≥ P_equ ≥ P_min` recovers. Test :
  Jonckheere trend on retention across the three profiles,
  one-sided alpha = 0.05 / 3 = 0.0167 (Bonferroni for 3
  hypotheses).

- **H4-B (HP-calibration)** : on the existing 3-layer head,
  sweeping RESTRUCTURE factor over {0.85, 0.95, 0.99}
  reveals at least one factor where the predicted ordering
  recovers. Test : Jonckheere on retention per factor cell,
  multiplicity-adjusted alpha = 0.05 / 9 (3 factors × 3
  hypotheses) = 0.0056.

- **H4-C (theoretical-emptiness)** : RECOMBINE channel is
  structurally empty at this scale. Operationalised as :
  `retention(P_max with RECOMBINE=none)` is not statistically
  distinguishable from `retention(P_max with RECOMBINE=mog)`
  (Welch two-sided fails to reject H0 at alpha = 0.05 / 3
  = 0.0167).

## §3 Power analysis

N = 95 seeds per arm at alpha = 0.0167 detects g ≥ 0.40 at
80 % power (Welch one-sided, equal variance approximation).
Effect sizes below 0.40 remain exploratory.

## §4 Exclusion criteria

- `acc_initial < 0.5` (random chance) — exclude cell.
- `acc_final` non-finite — exclude cell.
- any seed reproducing run_id collision with prior pilot's
  registry — abort and amend pre-reg.

## §5 Substrate / driver paths

- Step 1 : `experiments/g4_quater_test/run_step1_deeper.py`
- Step 2 : `experiments/g4_quater_test/run_step2_restructure_sweep.py`
- Step 3 : `experiments/g4_quater_test/run_step3_recombine_strategies.py`
- Substrates :
  - 5-layer : `experiments.g4_quater_test.deeper_classifier.G4HierarchicalDeeperClassifier`
  - 3-layer : `experiments.g4_ter_hp_sweep.dream_wrap_hier.G4HierarchicalClassifier`

## §6 DualVer outcome rules

| Outcome | EC bump |
|---|---|
| All H4-* rejected H0 (DR-4 ordering recovers under any condition) | EC stays PARTIAL ; FC unchanged ; G4-quater scope STABLE row in STATUS |
| H4-A confirmed only | EC stays PARTIAL ; flag substrate-depth scope-bound STABLE for 5-layer ; document open Q for shallower substrates |
| H4-B confirmed only | EC stays PARTIAL ; flag HP-window scope-bound STABLE ; document HP-pin amendment |
| H4-C confirmed (RECOMBINE empty at this scale) | EC stays PARTIAL ; **DR-4 partial refutation strengthens** ; framework C claim "richer ops yield richer consolidation at this scale" formally refuted ; DR-4 evidence file revised |
| All three confirmed | EC stays PARTIAL ; full empirical revision of DR-4 + paper section ; future work : test deeper benchmarks (CIFAR-10, ImageNet) before any STABLE promotion |

EC stays PARTIAL across **all** outcomes. FC stays at v0.12.0
across all outcomes (no formal-axis bump scheduled by this
pilot).

## §7 Reporting commitment

Honest reporting of all observed scalars regardless of
outcome ; if Welch tests do not reject H0, the verdict is
"no evidence for the hypothesis at this N", not "hypothesis
falsified" (absence of evidence vs evidence of absence).
H4-C confirmation specifically requires Welch failing to
reject difference between RECOMBINE=mog and RECOMBINE=none —
a positive empirical claim that RECOMBINE adds nothing at
this scale.

If H4-C is confirmed, the partial refutation of DR-4
established by G4-ter is **strengthened**, not weakened, and
the DR-4 empirical evidence file (`docs/proofs/dr4-profile-
inclusion.md`) is amended accordingly.

## §8 Audit trail

Every cell registered via `harness/storage/run_registry.py`
with profile keys
`g4-quater/{step1,step2,step3}/<arm>/<combo>/<seed>` and R1
bit-stable run_ids. Milestone artefacts under
`docs/milestones/g4-quater-step{1,2,3}-2026-05-03.{json,md}`
plus aggregate.

## §9 Deviations

Any deviation from this pre-reg requires (a) a separate
amendment commit before the affected cell runs, or (b) a
post-hoc honest disclosure in the paper section
acknowledging the deviation and its impact on the
confirmatory status of the hypothesis.

Pre-known deviation envelopes (executor open questions) :

1. If the Step 3 `ae` strategy fails to converge in one MSE
   pass, drop `ae` and run only `{mog, none}` ; H4-C verdict
   still holds at reduced power.
2. If the Step 1 5-layer head shows `acc_initial < 0.5`,
   raise per-task epochs from 3 to 5 ; document in step1
   milestone.
3. If Step 3 wall time exceeds 60 min on M1 Max, reduce N
   from 95 to 60 for Step 3 only ; H4-C verdict tagged
   exploratory at reduced N.
