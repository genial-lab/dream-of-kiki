# G5-bis cross-substrate aggregate - H7-A/B/C verdict

**Date** : 2026-05-03
**MLX milestone** : `docs/milestones/g4-ter-pilot-2026-05-03.json`
**E-SNN milestone** : `docs/milestones/g5-bis-richer-esnn-2026-05-03.json`
**Bonferroni alpha / 4** : 0.0125
**H7-B g threshold** : 0.50

## Verdict : H7-B

- Observed E-SNN g_h7a (P_equ vs baseline) : **+0.1043**
- Welch one-sided p (alpha/4 = 0.0125) : 0.4052
- reject_h0 (own-substrate) : False

## Per-arm cross-substrate Welch consistency

| arm | g (MLX - E-SNN) | Welch p (two-sided) | reject H0 | consistent |
|-----|------------------|----------------------|-----------|------------|
| baseline | +3.2261 | 0.0000 | True | False |
| P_min | +4.1975 | 0.0000 | True | False |
| P_equ | +4.0211 | 0.0000 | True | False |
| P_max | +4.0211 | 0.0000 | True | False |

## Provenance

- DR-3 spec : `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §6.2
- DR-3 evidence record : `docs/proofs/dr3-substrate-evidence.md`
- Aggregator : `experiments/g5_bis_richer_esnn/aggregator.py`
- Pre-registration : `docs/osf-prereg-g5-bis-richer-esnn.md`
