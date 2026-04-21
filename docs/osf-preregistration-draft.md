# OSF Pre-Registration — dreamOfkiki Cycle 1

**Project** : dreamOfkiki
**PI** : Clement Saillant (L'Electron Rare)
**Date drafted** : 2026-04-17 (S3)
**Lock date target** : S3 before experiments begin
**OSF URL** : https://osf.io/q6jyn
**DOI** : 10.17605/OSF.IO/Q6JYN (auto-minted by DataCite 2026-04-19 on OSF publish)

## 1. Study design

A within-architecture ablation comparing three dream-consolidation
profiles (P_min, P_equ, P_max) of a kiki-oniric linguistic model
against a no-dream baseline, on mega-v2 continual learning tasks
and Studyforrest fMRI representational alignment.

## 2. Hypotheses

### H1 — P_equ reduces catastrophic forgetting

**Statement** : P_equ reduces forgetting rate (M1.a) by ≥10% vs
no-dream baseline on mega-v2 continual learning benchmark.

**Operationalization** :
- M1.a = (acc_task_1_initial − acc_task_1_after_task_N) /
        acc_task_1_initial
- Measured across 25 domains of mega-v2
- Statistical test : paired t-test, one-sided,
  α = 0.05, n = 3 seeds minimum
- Reject H0 if p < 0.05 AND effect size ≥ 10% relative improvement

### H2 — P_max shows diminishing returns vs P_equ

**Statement** : P_max provides marginal improvement ≤5% over P_equ
on M1.a (law of diminishing returns).

**Operationalization** :
- Compare M1.a(P_max) vs M1.a(P_equ)
- Statistical test : equivalence test (TOST), ε_equivalence = 5%
- Conclude H2 supported if CI falls within [−5%, +5%]

### H3 — Monotonic RSA fMRI alignment

**Statement** : M2.b representational alignment increases
monotonically across profiles : P_min < P_equ < P_max.

**Operationalization** :
- M2.b = Pearson correlation between RDM(kiki activations) and
        RDM(Studyforrest fMRI ROIs STG/IFG/AG)
- Data : Studyforrest (Branch A locked S2.1 G1)
- Statistical test : Jonckheere-Terpstra monotonic trend test,
  α = 0.05
- Reject H0 if p < 0.05 AND pairwise P_min < P_equ and
  P_equ < P_max both significant

### H4 — Dream compute overhead is bounded

**Statement** : M3.c energy proxy ratio stays ≤2× awake for P_equ
(deployment viability).

**Operationalization** :
- M3.c = FLOPs-derived energy proxy per dream-episode
- Ratio = energy(dream) / energy(awake) over equivalent wall-clock
- Reject H4 violation if 95% CI upper bound exceeds 2.0

## 3. Pre-specified analyses

### Primary analyses

- H1 : paired t-test, one-sided
- H2 : TOST equivalence test, bidirectional
- H3 : Jonckheere-Terpstra monotonic trend test
- H4 : one-sample t-test against ratio = 2.0

### Sample size / power

- n = 3 seeds per (profile, metric) cell (minimum for PUBLICATION-
  READY gate)
- Power analysis based on pilot runs S5-S7 — if power < 0.8 for
  any hypothesis, increase to 5 seeds

### Multiple comparison correction

- Bonferroni correction across H1-H4 family : α_per_hypothesis =
  0.05 / 4 = 0.0125

## 4. Data exclusion rules

- Runs that violate any BLOCKING invariant (I1, I2, S1, S2, S3, K1,
  K4) are excluded and logged to `aborted-swaps/` for forensic
  review
- Runs with `EC_state != STABLE` at time of measurement are
  excluded from hypothesis testing (they can be re-run after
  EC stabilization)
- Seeds that fail repro smoke test (R1 contract) are excluded and
  re-run with deterministic seed fix

## 5. Deviations from pre-registration

Any post-hoc deviations from this analysis plan will be :
1. Documented in a separate `docs/osf-deviations-<date>.md`
2. Reported in paper 1 as exploratory (not confirmatory)
3. Linked from OSF project page

## 6. Data and code availability

- Code : https://github.com/electron-rare/dreamOfkiki (MIT)
- Models : https://huggingface.co/clemsail/kiki-oniric-*
- Raw runs : Zenodo DOI (at S20-S22)
- Harness : pip-installable `dreamOfkiki.harness`

## 7. Contact

Clement Saillant — clement@saillant.cc — L'Electron Rare, France

---

**Lock this document before any experiment begins.**
