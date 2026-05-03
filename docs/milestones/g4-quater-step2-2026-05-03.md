# G4-quater Step 2 — H4-B RESTRUCTURE factor sweep

**Date** : 2026-05-03
**c_version** : `C-v0.12.0+PARTIAL`
**commit_sha** : `e7d74e8d796f2a8c9cf2f286490dcb58ef3bb246`
**Cells** : 360 (3 factors x 4 arms x 30 seeds)
**Wall time** : 647.6s

## Pre-registered hypothesis

Pre-registration : `docs/osf-prereg-g4-quater-pilot.md`

### H4-B — HP-calibration (RESTRUCTURE factor sweep)

Per-factor Jonckheere on retention across (P_min, P_equ, P_max). Multiplicity-adjusted alpha = 0.05 / 9 = 0.0056.

#### factor = 0.85
- mean retention P_min : 0.7065
- mean retention P_equ : 0.6554
- mean retention P_max : 0.6554
- monotonic P_max >= P_equ >= P_min : False
- Jonckheere J : 1034.0000
- one-sided p (alpha = 0.0056) : 0.9904 -> reject_h0 = False

#### factor = 0.95
- mean retention P_min : 0.7065
- mean retention P_equ : 0.6582
- mean retention P_max : 0.6582
- monotonic P_max >= P_equ >= P_min : False
- Jonckheere J : 1076.0000
- one-sided p (alpha = 0.0056) : 0.9788 -> reject_h0 = False

#### factor = 0.99
- mean retention P_min : 0.7065
- mean retention P_equ : 0.6589
- mean retention P_max : 0.6589
- monotonic P_max >= P_equ >= P_min : False
- Jonckheere J : 1094.0000
- one-sided p (alpha = 0.0056) : 0.9710 -> reject_h0 = False

*Honest reading* : even one factor cell with reject_h0=True and monotonic_observed=True is sufficient to confirm H4-B (at the multiplicity-adjusted alpha). All factors failing to reject is consistent with H4-B refutation at this N.

## Provenance

- Pre-registration : [docs/osf-prereg-g4-quater-pilot.md](../osf-prereg-g4-quater-pilot.md)
- Driver : `experiments/g4_quater_test/run_step2_restructure_sweep.py`
- Substrate : `experiments.g4_ter_hp_sweep.dream_wrap_hier.G4HierarchicalClassifier`
- Run registry : `harness/storage/run_registry.RunRegistry` (db `.run_registry.sqlite`)
