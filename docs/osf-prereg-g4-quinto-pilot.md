# G4-quinto pilot pre-registration

**Date:** 2026-05-03
**Parent OSF:** 10.17605/OSF.IO/Q6JYN
**Sister pilot:** G4-quater (H4-A/B falsified, H4-C confirmed —
Welch p = 0.9886, P_max(mog) = 0.7007, P_max(none) = 0.7006,
N = 95, commit `83a6ae8`)
**Substrates:**
- Step 1 — MLX 5-layer MLP-on-CIFAR (`G4HierarchicalCIFARClassifier`,
  `in_dim=3072`, `hidden=(256, 128, 64, 32)`).
- Steps 2 + 3 — MLX small CNN (`G4SmallCNN`, 2 Conv2d + 2 MaxPool2d
  + 2 Linear, `latent_dim=64`).
**Benchmark:** Split-CIFAR-10 5 binary tasks (canonical class-incremental
split: {0,1}, {2,3}, {4,5}, {6,7}, {8,9}).
**Compute option:** A (full 3-step, N = 30 seeds/arm, ~600 cells,
~9-15 h overnight on M1 Max).
**Lock commit:** *(to be filled by the introducing commit hash)*
**Lock timestamp:** 2026-05-03 (pre-Step 1 driver run).

## §1 Background

G4-quater (commit `83a6ae8`, milestone
`docs/milestones/g4-quater-step3-2026-05-03.{json,md}`) reported
**H4-A False, H4-B False, H4-C Confirmed** on Split-FMNIST
across a 3-step sequential pipeline (1880 cells, ~58 min M1
Max). The H4-C confirmed verdict — Welch two-sided p = 0.9886,
Hedges' g = 0.002, N = 95, P_max(mog) = 0.7007 vs
P_max(none) = 0.7006 — establishes that, *at the Split-FMNIST
3-layer scale*, the RECOMBINE channel is empirically empty :
the predicted positive empirical claim that RECOMBINE adds
nothing measurable beyond REPLAY+DOWNSCALE on this substrate.

The G4-quater pre-reg §6 row 5 explicitly flags
"future work : test deeper benchmarks (CIFAR-10, ImageNet)
before any STABLE promotion". G4-quinto fires that follow-up
on the next benchmark in the escalation ladder
(Split-CIFAR-10) with two substrates (MLP-on-CIFAR and a small
CNN) to test whether the H4-C verdict is **scale-bound**
(FMNIST-bound artefact recoverable by larger benchmark or
hierarchical conv structure) or **universal** (RECOMBINE remains
empty across substrates and benchmarks).

The Hu 2020 anchor (g = 0.29, human sleep-dependent memory
consolidation) is used here strictly as a **directional**
reference — sign of the alternative hypothesis — not a magnitude
calibrator. A cross-class comparison between a biological cohort
and a seeded numerical pilot would be a category error.

## §2 Hypotheses (confirmatory)

- **H5-A (benchmark-scale)** — on Split-CIFAR-10 with the
  MLP-on-CIFAR substrate (`G4HierarchicalCIFARClassifier`,
  `in_dim=3072`, `hidden=(256, 128, 64, 32)`), the predicted
  profile ordering `P_max ≥ P_equ ≥ P_min` recovers. Test :
  Jonckheere trend on retention across the three profiles,
  one-sided alpha = 0.05 / 3 = 0.0167 (Bonferroni for 3
  hypotheses).

- **H5-B (architecture-scale)** — on Split-CIFAR-10 with the
  small CNN substrate (`G4SmallCNN`, `latent_dim=64`,
  Conv2d×2 + MaxPool2d×2 + Linear×2), the predicted profile
  ordering `P_max ≥ P_equ ≥ P_min` recovers. Test : Jonckheere
  trend on retention across the three profiles, one-sided
  alpha = 0.05 / 3 = 0.0167.

- **H5-C (universality of RECOMBINE-empty)** — on the CNN
  substrate, `retention(P_max with RECOMBINE=mog)` is not
  statistically distinguishable from
  `retention(P_max with RECOMBINE=none)`. Test : Welch two-sided
  fails to reject H0 at alpha = 0.05 / 3 = 0.0167. **Failing**
  to reject **is** the predicted positive empirical claim that
  the G4-quater H4-C finding generalises across benchmarks and
  substrates.

## §3 Power analysis

N = 30 seeds per arm at alpha = 0.0167 detects |g| ≥ 0.74 at
80 % power (Welch two-sided, equal-variance approximation,
`scipy.stats.power.tt_ind_solve_power`). Effect sizes below
|g| ≈ 0.74 remain exploratory at this N. The lower N (30 vs
G4-quater's 95) is dictated by the CIFAR-10 / CNN compute
envelope — full Option A run already targets ~9-15 h on
M1 Max ; N = 95 would push wall time beyond a single
overnight session.

## §4 Exclusion criteria

- `acc_initial < 0.5` (random chance) — exclude cell.
- `acc_final` non-finite — exclude cell.
- any seed reproducing run_id collision with prior pilot's
  registry — abort and amend pre-reg.

## §5 Substrate / driver paths

- Step 1 driver : `experiments/g4_quinto_test/run_step1_mlp_cifar.py`
- Step 2 driver : `experiments/g4_quinto_test/run_step2_cnn_cifar.py`
- Step 3 driver : `experiments/g4_quinto_test/run_step3_cnn_recombine.py`
- Substrates :
  - 5-layer MLP-on-CIFAR : `experiments.g4_quinto_test.cifar_mlp_classifier.G4HierarchicalCIFARClassifier`
  - small CNN : `experiments.g4_quinto_test.small_cnn.G4SmallCNN`
- Loader : `experiments.g4_quinto_test.cifar10_dataset.load_split_cifar10_5tasks`
- Source : https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
  (~163 MB, SHA-256 pinned in `cifar10_dataset.CIFAR10_TAR_SHA256`
  at first-download lock commit).

## §6 DualVer outcome rules

| Outcome | EC bump |
|---|---|
| H5-A and H5-B confirmed | EC stays PARTIAL ; scope-bound STABLE for CIFAR-10 (MLP and CNN) ; ImageNet open Q |
| H5-A confirmed only | EC stays PARTIAL ; scope-bound STABLE for MLP-CIFAR only ; CNN recovery still open |
| H5-B confirmed only | EC stays PARTIAL ; scope-bound STABLE for CNN-CIFAR only ; MLP-on-CIFAR recovery still open |
| H5-C confirmed | EC stays PARTIAL ; **DR-4 partial refutation universalises across FMNIST + CIFAR-CNN** ; framework C claim "richer ops yield richer consolidation" empirically refuted across 2 benchmarks ; DR-4 evidence file revised |
| All three falsified | EC stays PARTIAL ; G4-quinto reported as null ; G4-quater FMNIST refutation remains dominant evidence |
| All three confirmed | EC stays PARTIAL ; ImageNet escalation prerequisite for any STABLE promotion |

EC stays PARTIAL across **all** rows. FC stays at v0.12.0
across all rows (no formal-axis bump scheduled by this pilot).

## §7 Reporting commitment

Honest reporting of all observed scalars regardless of
outcome ; if Welch tests do not reject H0, the verdict is
"no evidence for the hypothesis at this N", not "hypothesis
falsified" (absence of evidence vs evidence of absence).

H5-C confirmation specifically requires Welch failing to
reject difference between RECOMBINE=mog and RECOMBINE=none —
*"Welch fail-to-reject = absence of evidence at this N for a
difference between mog and none — under H5-C specifically,
this **is** the predicted positive empirical claim that
RECOMBINE adds nothing measurable beyond REPLAY+DOWNSCALE on
the CNN substrate at CIFAR-10 scale."* (Verbatim honest-reading
clause from G4-quater §7, embedded in every G4-quinto milestone
MD.)

If H5-C is confirmed, the partial refutation of DR-4
established by G4-ter and strengthened by G4-quater is
**universalised** across FMNIST + CIFAR-CNN, and the DR-4
empirical evidence file (`docs/proofs/dr4-profile-inclusion.md`)
is amended accordingly with a §7.1.7 G4-quinto evidence row.

## §8 Audit trail

Every cell registered via `harness/storage/run_registry.py`
with profile keys
`g4-quinto/{step1,step2,step3}/<arm>/<combo>[/<strategy>]`
and R1 bit-stable run_ids. Milestone artefacts under
`docs/milestones/g4-quinto-step{1,2,3}-2026-05-03.{json,md}`
plus aggregate
`docs/milestones/g4-quinto-aggregate-2026-05-03.{json,md}`.

## §9 Deviations

Any deviation from this pre-reg requires (a) a separate
amendment commit before the affected cell runs, or (b) a
post-hoc honest disclosure in the paper section
acknowledging the deviation and its impact on the
confirmatory status of the hypothesis.

Pre-known deviation envelopes (executor open questions) :

1. CIFAR-10 download fails (network unavailable / SHA-256
   mismatch on canonical mirror) — abort the pilot and file
   a §9.1 amendment ; do not proceed with cells.
2. Step 1 / Step 2 `acc_initial < 0.5` for a majority of
   seeds — raise per-task epochs from 3 to 5, document in the
   step milestone header.
3. Step 3 wall time > 10 h on M1 Max — reduce N from 30 to
   20 for Step 3 only ; H5-C verdict tagged exploratory at
   reduced N.
4. Compute Option B chosen instead of Option A — Step 3 is
   deferred to a G4-sexto follow-up ; aggregator reports
   `h5c_deferred = True`, no H5-C verdict in this pilot ;
   document deferral in CHANGELOG entry.
