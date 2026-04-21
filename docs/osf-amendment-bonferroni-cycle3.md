# OSF Pre-registration Amendment — 2026-04-19

**Amendment #1 OSF DOI** : `10.17605/OSF.IO/TPM5S` (DataCite-minted 2026-04-21, filed as Open-Ended Registration linked to parent Q6JYN)
**Amendment URL** : https://osf.io/tpm5s/
**Published** : 2026-04-21

Cycle-3 Bonferroni family correction.

## Paste into OSF amendment field

---

**Title** : Bonferroni family-wise alpha correction for cycle-3 multi-scale hypotheses

**Amendment summary** :

The cycle-3 pre-registration (2026-04-19) specified a Bonferroni
family of 8 tests per evaluation cell : {H1, H2, H3, H4, H5-I,
H5-II, H5-III, H6}, with per-test α = 0.05 / 8 = 0.00625.

Review of the hypothesis structure (CodeRabbit cycle-12 finding
on `docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md`)
identified that **H3** (Jonckheere profile-chain effects) and
**H6** (profile ordering cross-substrate) are *cross-cell*
hypotheses that aggregate across multiple (profile, substrate)
cells rather than applying independently per cell.

**Corrected family structure** :

| Hypothesis | Scope | Family |
|---|---|---|
| H1 | per cell | Per-cell (size 6) |
| H2 | per cell | Per-cell (size 6) |
| H4 | per cell | Per-cell (size 6) |
| H5-I | per cell | Per-cell (size 6) |
| H5-II | per cell | Per-cell (size 6) |
| H5-III | per cell | Per-cell (size 6) |
| H3 | cross-cell (6 cells) | Cross-cell (size 2) |
| H6 | cross-cell (6 cells) | Cross-cell (size 2) |

- Per-cell family α = 0.05 / 6 = **0.00833**
- Cross-cell family α = 0.05 / 2 = **0.025**

**Rationale** : H3 and H6 pool effect sizes across cells to test
trend claims ; treating them as per-cell hypotheses would
double-count them and over-correct. Separating them preserves
Nature HB statistical discipline without unnecessary power loss.

**Direction locks** (unchanged since 2026-04-19 primary pre-reg) :
- H5-II : two-sided (no post-hoc direction claim)
- H3 : monotone ordering P_max > P_equ > P_min
- H6 : same ordering, substrate-invariant

**Code impact** :

- `kiki_oniric/eval/scaling_law.py` currently uses α = 0.00625
  (conservative, pre-correction). Will be relaxed to α = 0.00833
  per-cell + α = 0.025 cross-cell in C3.9 `compute_gate_d.py`
  before GATE D evaluation sem 3 end.
- Claims made under α = 0.00625 remain valid (strictly more
  conservative than α = 0.00833).

**References** :
- Primary pre-reg : `10.17605/OSF.IO/Q6JYN` (DataCite-minted on OSF publish 2026-04-19T00:28:05Z ; this amendment set #1 drafted same-day).
  **Filed** : 2026-04-21 as Open-Ended Registration at DOI `10.17605/OSF.IO/TPM5S` (https://osf.io/tpm5s/), linked to the original Q6JYN per COS guidance. A reviewer opening Q6JYN will see the amendment via OSF's Related Registrations link.
- Spec : `docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md` §4 (updated in commit `85e62fc`)
- Finding trail : CodeRabbit cycle-12 finding #26 on `docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md`

**Date of amendment** : 2026-04-19

---

## Timeline note

This amendment lands **before** any real-data compute runs (C3.8
full ablation starts sem 2 earliest). No results have been seen
under either family size, so this is a pre-data specification
correction, not a post-hoc rationalization. The conservative 0.00625
code default is preserved until C3.9 lands.
