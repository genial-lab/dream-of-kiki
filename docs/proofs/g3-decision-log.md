# G3 Decision Log — DR-2 Compositionality Proof Gate

**Gate** : G3 (DR-2 proof peer-reviewed)
**Target week** : S8
**Status** : **PENDING reviewer feedback (S6-S8 circulation window)**

## Reviewer status

| Item | Status | Notes |
|------|--------|-------|
| Reviewer recruited (Q_CR.1 b) | TODO | See `ops/formal-reviewer-recruitment.md` |
| Draft v0.1 sent | TODO | Pending recruitment |
| Feedback received | TODO | Pending review |
| Revision v0.2 produced | TODO | Pending feedback |
| Final review approval | TODO | Pending revision |

## Decision branches (from S6.2 circulation log)

### Branch DR-2-STRICT (default — happy path)
- Reviewer confirms strict DR-2 proof (closure + budget additivity +
  functional composition + associativity)
- **Action** : tag framework C-v0.7.0+STABLE
- **Paper 1 target** : Nature Human Behaviour
- **Status flag** : `+STABLE`

### Branch DR-2-PRIME (fallback — reviewer flags gap)
- Reviewer identifies issue in strict proof (e.g., free-semigroup
  vs primitive-set distinction needs more rigor)
- Adopt DR-2' (canonical order only) per
  `dr2-compositionality.md` fallback section
- **Action** : tag framework C-v0.7.0-PRIME+STABLE
- **Paper 1 target** : PLoS Computational Biology / Cognitive Science
- **Status flag** : `+STABLE` (different ID)

### Branch G3-FAIL (emergency — no reviewer + sub-agent flags issues)
- No human reviewer confirmed by S8 AND sub-agent `critic` flags
  proof issues
- **Action** : Pivot A activated per master spec §7.3
- **Scope reduction** : single-paper TMLR/ICLR workshop, framework
  paper deferred cycle 2
- **Paper 2 re-positioned** as primary deliverable

## Outcome (to be filled at S8 end)

- **Branch chosen** : TBD
- **Date** : TBD
- **Framework version tagged** : TBD
- **Paper 1 journal target** : TBD
- **Justification** (3-5 sentences) : TBD

## Lessons learned (post-G3)

(populated after gate decision)
