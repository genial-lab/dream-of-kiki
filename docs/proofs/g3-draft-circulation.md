# G3-Draft Circulation Log — DR-2 Compositionality

**Target milestone** : G3-draft (S6) → G3 final (S8)
**Reviewer** : external formal reviewer (recruited via
`ops/formal-reviewer-recruitment.md`)
**Fallback** : sub-agent `critic` + `validator` if no human confirmed
by S6 end

## Circulation timeline

| Date | Event | Actor | Notes |
|------|-------|-------|-------|
| 2026-04-17 | Draft v0.1 written | author | `dr2-compositionality.md` |
| TBD S6 | Sent to reviewer | author | Email via template |
| TBD S7 | Feedback received | reviewer | Log below |
| TBD S7 | Revision v0.2 | author | Address feedback |
| TBD S8 | Final review | reviewer | G3 gate decision |
| TBD S8 | G3 locked | author | `g3-decision-log.md` |

## Review feedback log

(populated during S7-S8)

### Iteration 1 — v0.1 review

- Date : TBD
- Reviewer feedback : TBD
- Revision items : TBD

## Decision tree at G3 gate (S8)

### Branch DR-2-STRICT (default)
- Reviewer confirms proof rigor → paper 1 cites strict DR-2
- Framework v0.7.0+STABLE tag at G3 gate
- Paper 1 target : Nature HB

### Branch DR-2-PRIME (fallback)
- Reviewer identifies unresolved gap in strict proof
- Adopt DR-2' (canonical order) per `dr2-compositionality.md` fallback
- Paper 1 framed as "formal-leaning" (Pivot B partial)
- Paper 1 target : PLoS Comp Bio / Cognitive Science
- Framework v0.7.0-PRIME+STABLE tag

### Branch G3-FAIL (emergency)
- No reviewer confirmed AND sub-agent critic flags issues
- Pivot A activated : single-paper focused on kiki-oniric
  engineering results (P_min empirical)
- Framework paper deferred to cycle 2
- Paper 2 re-positioned as primary deliverable

## Next action

User action S6 day 1 : finalize reviewer identity via
`ops/formal-reviewer-recruitment.md`, send email via
`ops/formal-reviewer-email-template.md`, update this log.
