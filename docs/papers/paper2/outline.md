# Paper 2 — Outline (cycle 2 amorçage S27.1)

**Target journal** : NeurIPS (primary) / ICML / TMLR (fallbacks)
**Format** : 9 pages main + supplementary unbounded
**Word target** : ~5500 words main

---

## 1. Abstract (200 words target)

Engineering contribution : substrate-agnostic dream-based
consolidation framework with executable Conformance Criterion,
demonstrated across MLX (Apple Silicon) and E-SNN
(Loihi-2 thalamocortical) substrates. Key engineering
contributions : (1) MLX-native operations with deterministic
reproducibility, (2) swap protocol with S1-S3 invariant guards,
(3) OSF pre-registered ablation methodology, (4)
cross-substrate Conformance Criterion validation matrix.
Empirical results : real mega-v2 ablation (cycle-1 closeout
data) + E-SNN replication (cycle-2 fresh data) showing
consistent profile chain effects.

## 2. Introduction (~1 page)

- Engineering problem : reproducible AI consolidation across
  hardware substrates
- Practical motivation : enable researchers to validate
  framework claims on their own substrates without re-deriving
  the theory
- Contribution roadmap (4 numbered items)

## 3. Background (brief, ~0.5 pages)

- Reference Paper 1 for theoretical foundation
- Cite SHY (Tononi), FEP (Friston), CLS (McClelland), brain-
  inspired replay (van de Ven)
- Establish that this paper focuses on the *engineering*
  realization, not the theory

## 4. Conformance Criterion in Practice (~1.5 pages)

- 4.1 Three conditions recap (signature typing, axiom property
  tests, BLOCKING invariants enforceable)
- 4.2 MLX substrate conformance verification (kiki-oniric)
- 4.3 E-SNN substrate conformance verification (cycle 2)
- 4.4 Conformance test suite (`tests/conformance/`) as a
  reusable artifact

## 5. Engineering Architecture (~1.5 pages)

- 5.1 Operations as composable handlers (replay, downscale,
  restructure, recombine — MLX + skeleton variants for testing)
- 5.2 Swap protocol with S1 retained-eval gating, S2 finite
  guard, S3 topology guard
- 5.3 Concurrent dream worker (Future-API skeleton in cycle 1,
  real asyncio in cycle 2)
- 5.4 Run registry with deterministic R1 contract (32-hex
  SHA-256 prefix)

## 6. Methodology (~1 page)

- 6.1 OSF pre-registration (cycle 1 hypotheses H1-H4 reused)
- 6.2 Statistical pipeline (Welch / TOST / Jonckheere /
  one-sample t with Bonferroni)
- 6.3 mega-v2 stratified retained benchmark (500 items, SHA-256
  frozen)
- 6.4 Cross-substrate measurement protocol

## 7. Results (~2 pages)

- 7.1 MLX substrate ablation (cycle 1 real ablation closeout
  data, replacing synthetic placeholders)
- 7.2 E-SNN substrate ablation (cycle 2 fresh data)
- 7.3 Cross-substrate comparison (consistency of profile chain
  effects)
- 7.4 Statistical significance + Bonferroni-corrected H1-H4
  outcomes

## 8. Discussion (~1 page)

- 8.1 Reproducibility validated across substrates
- 8.2 Engineering trade-offs (MLX speed vs E-SNN energy)
- 8.3 Limitations (only 2 substrates ; transformer or other
  architectures pending cycle 3)
- 8.4 Comparison with Paper 1 theoretical claims

## 9. Future Work (~0.5 pages)

- 9.1 Additional substrates (transformer, RWKV, state-space)
- 9.2 Dynamic profile selection at runtime
- 9.3 Production deployment patterns

## 10. References

Reuse most of Paper 1's references.bib + add engineering
citations (MLX, scipy, Loihi-2, asyncio patterns).

---

## Differentiation from Paper 1

| Aspect | Paper 1 | Paper 2 |
|--------|---------|---------|
| Scope | Theoretical framework | Engineering implementation |
| Target | Nature HB / PLoS CB | NeurIPS / ICML / TMLR |
| Audience | Cognitive scientists + theorists | ML engineers + systems researchers |
| Substrates covered | 1 (kiki-oniric MLX) | 2+ (MLX + E-SNN) |
| Length | ~5000 words | ~5500 words |
| Formal proofs | DR-0..DR-4 in main | Reference Paper 1 |
| Reproducibility focus | Pre-registration + R1 contract | Cross-substrate validation matrix |
| Open-source emphasis | Conceptual | Operational (run on your substrate) |

---

## Cycle-2 dependencies

- E-SNN substrate (Loihi-2 access via Intel NRC partnership —
  pursue post-G6 decision)
- Real mega-v2 dataset access (closeout cycle 1 S20+ if not
  done by then)
- Real fMRI lab partnership (T-Col extension cycle 2)
- Paper 1 acceptance / preprint citation for cross-reference

---

## Cycle-2 timeline (rough estimate)

- S29-S32 : E-SNN substrate wiring + conformance verification
- S33-S36 : E-SNN ablation runs + cross-substrate comparison
- S37-S40 : Paper 2 draft sections
- S41-S42 : pre-submission review
- S43-S44 : NeurIPS submission window
- S45+ : reviewer rounds + revisions

---

## Notes

- This outline is amorçage only ; full draft begins post-cycle-1
  closeout (S28+) once Paper 1 is submitted
- Outline subject to revision based on Paper 1 reviewer feedback
  and G6 cycle-2 scope decision
- Cycle-2 scope might exclude E-SNN if Loihi-2 access not
  granted ; in that case, fall back to MLX + simulation-based
  E-SNN as a second substrate with clear caveats
