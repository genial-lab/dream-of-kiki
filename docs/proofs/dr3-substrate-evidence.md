# DR-3 Conformance Criterion — substrate evidence (C2.10)

**Status** : two-substrate evidence, synthetic substitute.

DR-3 claims substrate-agnosticism : any substrate that
satisfies the three conditions below inherits the framework's
guarantees. This document records the **evidence per
substrate** for each condition, with pointers to the test
files and source modules that back the verdicts.

## Conformance conditions (spec §6.2)

1. **Signature typing** — the 8 primitives are declared as
   typed `Protocol`s (awake→dream α/β/γ/δ + 4 channels
   dream→awake). A substrate conforms to C1 by exposing
   handlers / factories compatible with these signatures.
2. **Axiom property tests** — DR-0..DR-4 property tests pass
   on the substrate's state representation.
3. **BLOCKING invariants enforceable** — the S2 finite and
   S3 topology guards can be applied to the substrate's
   state and refuse ill-formed values.

## Evidence per substrate

### `mlx_kiki_oniric`

Evidence summary :

- **C1 — signature typing (typed Protocols)** : `PASS` — all 8 primitives declared as Protocols + registry complete
  - evidence : `tests/conformance/axioms/test_dr3_substrate.py`
- **C2 — axiom property tests pass** : `PASS` — DR-0, DR-1, DR-3, DR-4 axiom suites pass on MLX
  - evidence : `tests/conformance/axioms/`
- **C3 — BLOCKING invariants enforceable** : `PASS` — S2 finite + S3 topology guards enforceable on MLX
  - evidence : `tests/conformance/invariants/`

### `esnn_thalamocortical`

Evidence summary *(synthetic substitute — numpy LIF skeleton)* :

- **C1 — signature typing (typed Protocols)** : `PASS` — 4 op factories callable + core registry shared with MLX (spike-rate numpy LIF skeleton, synthetic substitute)
  - evidence : `tests/conformance/axioms/test_dr3_esnn_substrate.py`
- **C2 — axiom property tests pass** : `PASS` — DR-3 E-SNN conformance suite passes on numpy LIF skeleton (synthetic substitute — no Loihi-2 HW)
  - evidence : `tests/conformance/axioms/test_dr3_esnn_substrate.py`
- **C3 — BLOCKING invariants enforceable** : `PASS` — S2 finite + S3 topology guards enforceable on LIFState (synthetic substitute — spike-rate numpy LIF)
  - evidence : `tests/conformance/axioms/test_dr3_esnn_substrate.py`

### `hypothetical_cycle3` (placeholder — not yet implemented)

This row is reserved for a future (cycle-3) substrate. All cells are `N/A — placeholder for cycle 3`. The DR-3 evidence set is *two* substrates in cycle 2.

## Synthetic-data caveat

The E-SNN substrate rows are backed by a numpy LIF spike-
rate skeleton, not by real Loihi-2 hardware. No fMRI or
behavioural cohort is involved in either substrate's C2
axiom tests. DR-3 two-substrate replication therefore
strengthens the *architectural* claim (the framework's
Conformance Criterion is operational across two
independent implementations of the 8 primitives) — it does
not yet carry a cross-substrate empirical claim on real
biological data.

## Cross-references

- Spec §6.2 : `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`
- Matrix dump : `docs/milestones/conformance-matrix.md`
- JSON dump : `docs/milestones/conformance-matrix.json`
- MLX substrate : `kiki_oniric/substrates/mlx_kiki_oniric.py`
- E-SNN substrate : `kiki_oniric/substrates/esnn_thalamocortical.py`
- C2 axiom suites : `tests/conformance/axioms/`
- C3 invariant suites : `tests/conformance/invariants/`
