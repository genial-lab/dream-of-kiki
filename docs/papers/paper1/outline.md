# Paper 1 — Outline (cycle 1 draft)

**Target journal** : Nature Human Behaviour (primary)
**Format** : 8-10 pages main + 30-50 pages supplementary
**Word target** : ~5000 words main + supp unbounded

---

## 1. Abstract (250 words)

→ `abstract.md`

Structure :
- Sentence 1 : context / open problem (catastrophic forgetting,
  consolidation gap in AI)
- Sentence 2 : contribution claim (formal framework with executable
  axioms DR-0..DR-4 + Conformance Criterion)
- Sentence 3 : empirical validation summary (kiki-oniric ablation,
  P_min vs P_equ on retained benchmark + RSA fMRI on Studyforrest)
- Sentence 4 : significance (substrate-agnostic, open-science
  reproducible, OSF pre-registered)
- 1-2 sentences : key result + future work pointer (E SNN cycle 2)

---

## 2. Introduction (~1.5 pages)

→ `introduction.md`

Subsections :
- 2.1 Catastrophic forgetting and the consolidation gap in AI
- 2.2 Sleep-inspired consolidation : prior art (van de Ven 2020,
  Kirkpatrick 2017 EWC, Tononi SHY, Friston FEP, Hobson/Solms)
- 2.3 The four pillars (A consolidation, B homeostasis, C creative,
  D predictive) and why a unified formal framework is missing
- 2.4 Contribution roadmap : framework C with executable axioms +
  Conformance Criterion + ablation on kiki-oniric

---

## 3. Background — four theoretical pillars

Subsections :
- 3.1 Pillar A : Walker/Stickgold consolidation
- 3.2 Pillar B : Tononi SHY synaptic homeostasis
- 3.3 Pillar C : Hobson/Solms creative dreaming
- 3.4 Pillar D : Friston Free Energy Principle
- 3.5 The compositional gap : why piecewise integration fails

---

## 4. Framework C

Subsections :
- 4.1 Primitives : 8 typed Protocols (α, β, γ, δ, 1-4 channels)
- 4.2 Profiles : P_min, P_equ, P_max with chain inclusion DR-4
- 4.3 Dream-episode 5-tuple ontology
- 4.4 Operations : replay, downscale, restructure, recombine
- 4.5 Axioms DR-0..DR-4
  - 4.5.1 DR-0 Accountability
  - 4.5.2 DR-1 Episodic conservation
  - 4.5.3 DR-2 Compositionality (free semigroup, see proofs)
  - 4.5.4 DR-3 Substrate-agnosticism + Conformance Criterion
  - 4.5.5 DR-4 Profile chain inclusion
- 4.6 Invariants I/S/K with enforcement matrix
- 4.7 DualVer formal+empirical versioning

---

## 5. Implementation : kiki-oniric

Subsections :
- 5.1 Substrate choice : MLX on Apple Silicon
- 5.2 Runtime : DreamRuntime with DR-0 log guarantee + swap protocol
- 5.3 Profiles wired : P_min (replay+downscale), P_equ (4 ops + 3
  channels)
- 5.4 Concurrent worker skeleton (cycle-2 forward-compat API)
- 5.5 Open-source : github.com/electron-rare/dream-of-kiki (MIT)

---

## 6. Methodology

Subsections :
- 6.1 Hypotheses H1-H4 (OSF pre-registration DOI cited)
- 6.2 Statistical tests : Welch (H1), TOST (H2), Jonckheere (H3),
  one-sample t (H4) ; Bonferroni α = 0.0125
- 6.3 Benchmark : mega-v2 (498K examples, 25 domains, stratified
  500-item retained set with SHA-256 integrity)
- 6.4 RSA fMRI : Studyforrest (Branch A locked G1, FreeSurfer
  parcellation STG/IFG/AG)
- 6.5 Reproducibility : R1 contract (deterministic run_id), Zenodo
  DOI artifact pinning

---

## 7. Results

→ `results-section.md` (S18.2 draft)

Subsections :
- 7.1 P_min viability (G2 GO-CONDITIONAL → real GO-FULL evidence)
- 7.2 P_equ functional ablation (G4 ablation results)
- 7.3 H1 forgetting reduction
- 7.4 H3 monotonic representational alignment
- 7.5 H4 energy budget compliance
- 7.6 H2 P_max equivalence (cycle 2 deferred, partial smoke test)

---

## 8. Discussion

Subsections :
- 8.1 Theoretical contribution : first executable formal framework
  for dream-based consolidation
- 8.2 Empirical contribution : ablation evidence on real linguistic
  hierarchical model (kiki-oniric)
- 8.3 Limitations : synthetic-data caveats, single-substrate
  validation cycle 1, P_max skeleton only
- 8.4 Comparison with prior art (van de Ven, EWC, etc.)

---

## 9. Future Work — Cycle 2

Subsections :
- 9.1 E-SNN substrate (Loihi-2 thalamocortical, deferred from
  cycle 1 SCOPE-DOWN)
- 9.2 P_max full wiring + α-stream + ATTENTION_PRIOR canal-4
- 9.3 Real fMRI lab partnership (T-Col reviewer outreach S3-S5)
- 9.4 Multi-substrate Conformance Criterion validation

---

## 10. References (placeholder)

Key citations to populate :
- Walker MP, Stickgold R (2004). Sleep-dependent learning.
- Kirkpatrick J et al (2017). EWC.
- Tononi G, Cirelli C (2014). SHY.
- Friston K (2010). FEP.
- Hobson JA (2009). REM dreaming.
- van de Ven GM, Tuytelaars T, Tolias AS (2020). Brain-inspired
  replay.
- McClelland JL et al (1995). Complementary learning systems.

---

## Artifacts

- Code : github.com/electron-rare/dream-of-kiki (MIT, frozen at
  arXiv submission tag)
- Models : huggingface.co/clemsail/kiki-oniric-{P_min,P_equ}
- Data : Zenodo DOI (post-S18 finalization)
- Pre-registration : OSF DOI (TBD post-upload)
- Dashboard : dream.saillant.cc (public read-only)
