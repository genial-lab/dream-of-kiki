# C-v0.5.0+STABLE Changelog

**Bump type** : MINOR (FC axis)
**From** : C-v0.3.1+STABLE
**To** : C-v0.5.0+STABLE
**EC status** : STABLE (all property tests pass, no empirical
invalidation)
**Date** : 2026-04-17 (S4.2)

## Changes

### Added (MINOR)

- **Conformance Criterion fully documented** (§6.2) — signature
  typing + axiom tests + BLOCKING invariants enforceable,
  resolving critic finding #3.
- **Typed Protocols exposed** (`kiki_oniric.core.primitives`) —
  8 Protocols satisfying Conformance Criterion condition (1) for
  kiki-oniric substrate.
- **Interface contracts locked** — `docs/interfaces/primitives.md`
  (human-readable) and `docs/interfaces/eval-matrix.yaml` (machine
  readable), tested via `harness.config.eval_matrix.EvalMatrix`.
- **fMRI schema locked** — `docs/interfaces/fmri-schema.yaml`
  pinned to Studyforrest Branch A from G1 decision.
- **Retained benchmark infrastructure** — SHA-256 integrity check
  loader at `harness.benchmarks.retained`, 50-item synthetic
  placeholder, 500-item real set deferred to S5+.
- **OSF pre-registration drafted** — H1-H4 operationalized with
  statistical tests (paired t, TOST, Jonckheere-Terpstra), upload
  checklist separate.
- **T-Col tracking** — fMRI labs + formal reviewer candidates
  registered with weekly status log protocol.
- **G1 gate locked Branch A** — feasibility note with 4 passed
  checks.

### Changed (no FC impact)

- Eval matrix now references `studyforrest` data source for M2.b
  explicitly (was `TBD` before Branch A lock).
- Reviewer fallback protocol documented if Q_CR.1 b human reviewer
  unavailable by S6.

### Not changed

- DR-0..DR-4 axiom statements (unchanged from C-v0.3.1).
- Invariants I/S/K statements (unchanged).
- Swap worktree protocol (unchanged, implementation S7).
- Profile inclusion chain (unchanged).

## Empirical consistency

- No experiments run yet (pre-S5). Contract tests passing :
  - Registry + conformance + eval-matrix + retained benchmark
  - 16 tests passing at time of tag, coverage ≥90%

- EC status : **STABLE** — no invalidation of prior results
  possible since no empirical results exist yet.

## Compat suite results

T-Ops compat suite stratified matrix : N/A at this bump (no MAJOR,
no prior experimental data to re-verify). EC stays STABLE.

## Next bump target

- C-v0.7.0+STABLE at S6 (DR-2 proof draft G3-draft)
- C-v1.0.0+STABLE at S12 (P_equ fonctionnel G4, full stratified
  matrix MAJOR tag)
- C-v1.0.x patches through S18 (PUBLICATION-READY gate G5)
