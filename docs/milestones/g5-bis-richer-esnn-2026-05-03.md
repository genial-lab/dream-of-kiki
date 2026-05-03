# G5-bis pilot - richer head ported to E-SNN substrate

**Date** : 2026-05-03
**c_version** : `C-v0.12.0+PARTIAL`
**commit_sha** : `69fd86fc66fc23ab094e0ff708f0e1bf4df7f7f8`
**Substrate** : esnn_thalamocortical_richer
**Cells** : 40 (4 arms x 10 seeds x 1 HP)
**Wall time** : 957.6s

## Pre-registered hypothesis (own-substrate H7-A)

Pre-registration : `docs/osf-prereg-g5-bis-richer-esnn.md`

### H7-A - E-SNN richer (P_equ vs baseline retention)
- observed Hedges' g_h7a : **0.1043**
- above zero : True
- above Hu 2020 lower CI 0.21 : False
- Welch one-sided p (alpha/4 = 0.0125) : 0.4052 -> reject_h0 = False

## Cells (per arm x seed)

| arm | seed | hp | acc_task1_initial | acc_task1_final | retention | excluded |
|-----|------|----|--------------------|------------------|-----------|----------|
| baseline | 0 | C5 | 0.9750 | 0.5000 | 0.5128 | False |
| baseline | 1 | C5 | 0.9785 | 0.5000 | 0.5110 | False |
| baseline | 2 | C5 | 0.9780 | 0.5000 | 0.5112 | False |
| baseline | 3 | C5 | 0.9785 | 0.4995 | 0.5105 | False |
| baseline | 4 | C5 | 0.9715 | 0.4995 | 0.5142 | False |
| baseline | 5 | C5 | 0.9745 | 0.5000 | 0.5131 | False |
| baseline | 6 | C5 | 0.9725 | 0.4995 | 0.5136 | False |
| baseline | 7 | C5 | 0.9770 | 0.4995 | 0.5113 | False |
| baseline | 8 | C5 | 0.9730 | 0.4990 | 0.5128 | False |
| baseline | 9 | C5 | 0.9665 | 0.4990 | 0.5163 | False |
| P_min | 0 | C5 | 0.9750 | 0.5000 | 0.5128 | False |
| P_min | 1 | C5 | 0.9785 | 0.5000 | 0.5110 | False |
| P_min | 2 | C5 | 0.9780 | 0.5000 | 0.5112 | False |
| P_min | 3 | C5 | 0.9785 | 0.5000 | 0.5110 | False |
| P_min | 4 | C5 | 0.9715 | 0.4995 | 0.5142 | False |
| P_min | 5 | C5 | 0.9745 | 0.5000 | 0.5131 | False |
| P_min | 6 | C5 | 0.9725 | 0.5000 | 0.5141 | False |
| P_min | 7 | C5 | 0.9770 | 0.4995 | 0.5113 | False |
| P_min | 8 | C5 | 0.9730 | 0.5000 | 0.5139 | False |
| P_min | 9 | C5 | 0.9665 | 0.4990 | 0.5163 | False |
| P_equ | 0 | C5 | 0.9750 | 0.5000 | 0.5128 | False |
| P_equ | 1 | C5 | 0.9785 | 0.5000 | 0.5110 | False |
| P_equ | 2 | C5 | 0.9780 | 0.5000 | 0.5112 | False |
| P_equ | 3 | C5 | 0.9785 | 0.4995 | 0.5105 | False |
| P_equ | 4 | C5 | 0.9715 | 0.4995 | 0.5142 | False |
| P_equ | 5 | C5 | 0.9745 | 0.5000 | 0.5131 | False |
| P_equ | 6 | C5 | 0.9725 | 0.5000 | 0.5141 | False |
| P_equ | 7 | C5 | 0.9770 | 0.5000 | 0.5118 | False |
| P_equ | 8 | C5 | 0.9730 | 0.4990 | 0.5128 | False |
| P_equ | 9 | C5 | 0.9665 | 0.5000 | 0.5173 | False |
| P_max | 0 | C5 | 0.9750 | 0.5000 | 0.5128 | False |
| P_max | 1 | C5 | 0.9785 | 0.5000 | 0.5110 | False |
| P_max | 2 | C5 | 0.9780 | 0.5000 | 0.5112 | False |
| P_max | 3 | C5 | 0.9785 | 0.4995 | 0.5105 | False |
| P_max | 4 | C5 | 0.9715 | 0.4995 | 0.5142 | False |
| P_max | 5 | C5 | 0.9745 | 0.5000 | 0.5131 | False |
| P_max | 6 | C5 | 0.9725 | 0.5000 | 0.5141 | False |
| P_max | 7 | C5 | 0.9770 | 0.5000 | 0.5118 | False |
| P_max | 8 | C5 | 0.9730 | 0.4990 | 0.5128 | False |
| P_max | 9 | C5 | 0.9665 | 0.5000 | 0.5173 | False |

## Provenance

- Pre-registration : [docs/osf-prereg-g5-bis-richer-esnn.md](../osf-prereg-g5-bis-richer-esnn.md)
- Sister G4-ter MLX milestone : [g4-ter-pilot-2026-05-03.md](g4-ter-pilot-2026-05-03.md)
- Sister G5 binary-head milestone : [g5-cross-substrate-2026-05-03.md](g5-cross-substrate-2026-05-03.md)
- Cross-substrate aggregate : [g5-bis-aggregate-2026-05-03.md](g5-bis-aggregate-2026-05-03.md)
- Driver : `experiments/g5_bis_richer_esnn/run_g5_bis.py`
- Substrate : `experiments.g5_bis_richer_esnn.esnn_hier_classifier.EsnnG5BisHierarchicalClassifier`
- HP combo : C5 (`representative_combo()`)
- Run registry : `harness/storage/run_registry.RunRegistry` (db `.run_registry.sqlite`)
