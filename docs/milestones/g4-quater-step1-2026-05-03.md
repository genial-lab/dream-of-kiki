# G4-quater Step 1 — H4-A deeper substrate

**Date** : 2026-05-03
**c_version** : `C-v0.12.0+PARTIAL`
**commit_sha** : `ac7ea7d0cfd1fb80813623754c7cc5ab02e7fbdb`
**Cells** : 380 (4 arms x 95 seeds x 1 HP)
**Hidden** : (64, 32, 16, 8)
**Wall time** : 791.3s

## Pre-registered hypothesis

Pre-registration : `docs/osf-prereg-g4-quater-pilot.md`

### H4-A — substrate-depth (5-layer deeper hierarchical head)
- mean retention P_min : 0.5959
- mean retention P_equ : 0.5958
- mean retention P_max : 0.5958
- monotonic observed P_max >= P_equ >= P_min : False
- Jonckheere J statistic : 13511.5000
- one-sided p (alpha = 0.0167) : 0.5137 -> reject_h0 = False

*Honest reading* : reject_h0 means there is evidence for the predicted ordering at this N ; failure to reject means no evidence at this N (absence of evidence vs evidence of absence).

## Provenance

- Pre-registration : [docs/osf-prereg-g4-quater-pilot.md](../osf-prereg-g4-quater-pilot.md)
- Driver : `experiments/g4_quater_test/run_step1_deeper.py`
- Substrate : `experiments.g4_quater_test.deeper_classifier.G4HierarchicalDeeperClassifier`
- Run registry : `harness/storage/run_registry.RunRegistry` (db `.run_registry.sqlite`)
