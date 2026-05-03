# G4-quinto aggregate verdict

**Date** : 2026-05-03
**Pre-registration** : [docs/osf-prereg-g4-quinto-pilot.md](../osf-prereg-g4-quinto-pilot.md)

## Summary

- H5-A (benchmark-scale) confirmed : **False**
- H5-B (architecture-scale) confirmed : **False**
- H5-C (universality of RECOMBINE-empty) confirmed : **True**
- H4-C -> H5-C universality (FMNIST + CIFAR-CNN both empty) : **True**
- All three confirmed : False

## H5-A — benchmark-scale (MLP-on-CIFAR)

- mean P_min : 0.8713
- mean P_equ : 0.8754
- mean P_max : 0.8754
- monotonic_observed : True
- Jonckheere J : 1362.0000
- one-sided p (alpha = 0.0167) : 0.4646
- reject_h0 : False
- **H5-A confirmed** : False

## H5-B — architecture-scale (CNN substrate)

- mean P_min : 0.9841
- mean P_equ : 0.9842
- mean P_max : 0.9842
- monotonic_observed : True
- Jonckheere J : 1356.0000
- one-sided p (alpha = 0.0167) : 0.4823
- reject_h0 : False
- **H5-B confirmed** : False

## H5-C — universality of RECOMBINE-empty (CNN substrate)

- mean P_max (mog) : 0.9842
- mean P_max (none) : 0.9845
- Hedges' g (mog vs none) : -0.0026
- Welch t : -0.0104
- Welch p two-sided (alpha = 0.0167) : 0.9918
- fail_to_reject_h0 : True -> H5-C confirmed = True

*Honest reading* : Welch fail-to-reject = absence of evidence at this N for a difference between mog and none — under H5-C specifically, this **is** the predicted positive empirical claim that RECOMBINE adds nothing measurable beyond REPLAY+DOWNSCALE on the CNN substrate at CIFAR-10 scale.

### Secondary observation — AE strategy

- mean P_max (ae) : 0.9840
- mean P_max (none) : 0.9845
- Welch p two-sided : 0.9857

## Verdict — DR-4 evidence

Per pre-reg §6 : EC stays PARTIAL across all outcomes ; FC stays at C-v0.12.0. If H5-C is confirmed, the partial refutation of DR-4 established by G4-ter and strengthened by G4-quater is **universalised** across FMNIST + CIFAR-CNN ; framework C claim 'richer ops yield richer consolidation' empirically refuted across two benchmarks. If H5-C is falsified, the refutation remains FMNIST-bound and CIFAR-CNN preserves the RECOMBINE contribution (scope-bound STABLE candidate).
