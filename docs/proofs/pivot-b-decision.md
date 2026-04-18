<!--
SPDX-License-Identifier: CC-BY-4.0
Authorship byline : dreamOfkiki project contributors
License : Creative Commons Attribution 4.0 International (CC-BY-4.0)
-->

**Authorship byline** : *dreamOfkiki project contributors*
**License** : CC-BY-4.0

# Pivot B Decision Tree

**Trigger** : G5 PUBLICATION-READY gate cannot be reached by S22
(end of cycle 1 buffer)

**Source** : master spec §7.3 Pivot B contingency

---

## Branch B-EXTEND : Extend cycle-1 timeline

**Conditions for selection** :
- DR-2 reviewer feedback delayed but reviewer engaged
- Real mega-v2 access in progress but not yet usable
- Paper 1 draft progress steady, ~80% complete
- No fundamental architectural blocker

**Actions** :
- Extend cycle 1 by 4-8 weeks (S22 → S30)
- Defer Paper 2 outline (cycle 2 start S30+)
- Maintain Nature Human Behaviour as primary target
- Framework version stays C-v0.5.0+STABLE through extension ;
  bump to C-v0.7.0+STABLE only at G3 close

**Pros** : preserves scientific rigor, no journal downgrade
**Cons** : delays cycle 2 (E-SNN), risks reviewer attention drift

---

## Branch B-DOWNGRADE : Downgrade journal target

**Conditions for selection** :
- DR-2 reviewer cannot confirm strict proof, fallback DR-2'
  adopted
- Real mega-v2 + MLX inference results show smaller effects than
  pre-registered (synthetic was optimistic)
- Cycle-1 calendar pressure to close S22

**Actions** :
- Adopt DR-2' (canonical order semigroup) per
  `docs/proofs/dr2-compositionality.md` fallback section
- Tag framework C-v0.7.0-PRIME+STABLE
- Pivot Paper 1 target to **PLoS Computational Biology** (open
  access, broader scope) or **Cognitive Science** (formal
  cognitive modeling)
- Adjust Paper 1 framing : "formal-leaning" rather than "formally
  proven"
- Cycle 2 planning unchanged

**Pros** : closes cycle 1 on time, still publishes peer-reviewed
**Cons** : lower impact factor, narrower audience

---

## Branch B-SCOPE-DOWN : Pivot A (single-paper TMLR/ICLR)

**Conditions for selection** :
- DR-2 review reveals fundamental gap, DR-2' insufficient
- Real ablation shows P_equ ≤ P_min on most metrics (architectural
  hypothesis falsified)
- G2 NO-GO triggered (P_min retained accuracy degraded > 2%)

**Actions** :
- Defer **framework paper** entirely to cycle 2
- Repurpose cycle-1 work as **engineering paper** for **TMLR**
  (Transactions on Machine Learning Research) or **ICLR Workshop**
- Focus Paper 1 narrative on : MLX-native dream-based ablation
  pipeline, swap protocol, S1-S3 invariant guards, OSF pre-reg
  methodology
- Frame as "reproducible engineering contribution" not "formal
  framework contribution"
- Cycle 2 = redo framework paper with stronger formal foundation
  + cycle-1 engineering as supplementary

**Pros** : closes cycle 1 with publishable artifact, salvages
infrastructure work
**Cons** : major scope reduction, framework paper deferred 12+
months

---

## Selection matrix

| Condition | B-EXTEND | B-DOWNGRADE | B-SCOPE-DOWN |
|-----------|----------|-------------|--------------|
| DR-2 reviewer engaged + slow | ✅ | — | — |
| DR-2 fallback DR-2' adopted | — | ✅ | — |
| DR-2 fundamental gap | — | — | ✅ |
| Real ablation strong (≥3 hypotheses) | ✅ | ✅ | — |
| Real ablation weak (≤1 hypothesis) | — | ✅ | ✅ |
| P_equ falsified vs P_min | — | — | ✅ |
| Calendar pressure high | — | ✅ | ✅ |
| Calendar pressure low | ✅ | — | — |

## Outcome (to be filled at S22 if Pivot B activated)

- **Branch chosen** : TBD
- **Date** : TBD
- **Framework version** : TBD
- **Paper 1 journal target** : TBD
- **Justification** (3-5 sentences) : TBD

## Lessons learned (post-Pivot B if activated)

(populated after Pivot B execution if triggered)
