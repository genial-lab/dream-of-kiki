# G4-quinto Step 2 — H5-B CNN-on-CIFAR

**Date** : 2026-05-03
**c_version** : `C-v0.12.0+PARTIAL`
**commit_sha** : `cb0da1838baa047bf052a32e7e16d5fa5a2ac799`
**Cells** : 120 (4 arms x 30 seeds x 1 HP)
**Latent dim** : 64
**Wall time** : 1037.8s

## Pre-registered hypothesis

Pre-registration : `docs/osf-prereg-g4-quinto-pilot.md`

### H5-B — architecture-scale (small CNN substrate)
- mean retention P_min : 0.9841
- mean retention P_equ : 0.9842
- mean retention P_max : 0.9842
- monotonic observed P_max >= P_equ >= P_min : True
- Jonckheere J statistic : 1356.0000
- one-sided p (alpha = 0.0167) : 0.4823 -> reject_h0 = False

*Honest reading* : reject_h0 means there is evidence for the predicted ordering at this N ; failure to reject means no evidence at this N (absence of evidence vs evidence of absence).

## Provenance

- Pre-registration : [docs/osf-prereg-g4-quinto-pilot.md](../osf-prereg-g4-quinto-pilot.md)
- Driver : `experiments/g4_quinto_test/run_step2_cnn_cifar.py`
- Substrate : `experiments.g4_quinto_test.small_cnn.G4SmallCNN`
- Run registry : `harness/storage/run_registry.RunRegistry` (db `.run_registry.sqlite`)
