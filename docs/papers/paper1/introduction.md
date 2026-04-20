<!--
SPDX-License-Identifier: CC-BY-4.0
Authorship byline : Saillant, Clément
License : Creative Commons Attribution 4.0 International (CC-BY-4.0)
-->

# Introduction (Paper 1, draft)

**Authorship byline** : *Saillant, Clément*
**License** : CC-BY-4.0

**Target length** : ~1.5 pages markdown (≈ 1200 words)

---

## 1. Catastrophic forgetting and the consolidation gap

Modern artificial cognitive systems excel at single-task learning
but degrade rapidly when trained sequentially across tasks — a
phenomenon known as **catastrophic forgetting** [McCloskey & Cohen
1989, French 1999]. Despite two decades of mitigation strategies
(elastic weight consolidation [Kirkpatrick et al. 2017], generative
replay [Shin et al. 2017], rehearsal-based memory [Rebuffi et al.
2017]), the field still lacks a *unified theoretical account* of
why these mechanisms work and when they should compose.

Biological cognition solves this problem during **sleep**.
Hippocampal replay during NREM, synaptic downscaling, predictive
restructuring of cortical representations, and creative
recombination during REM together form a multi-stage
consolidation pipeline [Diekelmann & Born 2010, Tononi & Cirelli
2014]. Yet existing AI work has integrated only fragments of this
biology, typically focusing on a single mechanism (e.g., replay
alone) without a principled account of how mechanisms interact.

## 2. Four pillars of dream-based consolidation

We identify four theoretical pillars that any complete
dream-inspired AI consolidation framework must address :

- **A — Walker/Stickgold consolidation** : episodic-to-semantic
  transfer via replay [Walker & Stickgold 2004, Stickgold 2005].
- **B — Tononi SHY** : synaptic homeostasis renormalizing weights
  during sleep [Tononi & Cirelli 2014].
- **C — Hobson/Solms creative dreaming** : recombination and
  abstraction during REM [Hobson 2009, Solms 2021].
- **D — Friston FEP** : minimization of free energy as a unifying
  account of inference and consolidation [Friston 2010].

Prior AI work has implemented A (van de Ven et al. 2020), B
(Kirkpatrick et al. 2017 as a SHY-adjacent regularization), and
elements of D (Rao & Ballard 1999, Whittington & Bogacz 2017),
but **no work has formalized how the four pillars compose** in a
substrate-agnostic manner amenable to ablation and proof.

## 3. The compositional gap

Why does composition matter ? Empirically, the order in which
consolidation operations apply changes the resulting cognitive
state — replay before downscaling preserves episodic specificity,
while downscaling before restructuring may erase the very
representations that restructuring is meant to refine. Our
analysis (`docs/proofs/op-pair-analysis.md`) enumerates the 16
op-pairs and finds 12 cross-pairs are non-commutative, reinforcing
that *order is part of the framework*, not an implementation
detail.

A proper formal framework must therefore (i) specify the
operations as composable primitives with well-defined types, (ii)
make explicit which compositions are valid, (iii) provide an
**executable** account that any compliant substrate can implement,
and (iv) support empirical ablation comparing different operation
profiles. None of the prior art does all four.

## 4. Contribution roadmap

In this paper we present **dreamOfkiki**, the first formal
framework for dream-based consolidation in artificial cognitive
systems with the following contributions :

1. **Framework C-v0.5.0+STABLE** : 8 typed primitives, 4 canonical
   operations forming a free semigroup, 4 OutputChannels, 5-tuple
   Dream Episode ontology, axioms DR-0..DR-4 with executable
   Conformance Criterion (§4). Items 2–4 below are reported in
   Paper 2 (empirical companion) ; Paper 1 confines itself to the
   formal contributions and the conformance roadmap.
2. **Roadmap** to substrate generalization (additional
   substrates beyond cycle-1's reference implementation) and
   real fMRI representational alignment (real lab partnership
   pursued via T-Col outreach).

The remainder of the paper is organized as follows : §3 reviews
the four pillars in depth ; §4 develops Framework
C-v0.5.0+STABLE with axioms and proofs ; §5 sketches the
Conformance Criterion validation approach (per-substrate
empirical results live in Paper 2) ; §6 details the methodology ;
§7 reports the synthetic pipeline-validation results ; §8
discusses implications and limitations ; §9 outlines cycle-2
future work.

---

## Notes for revision

- Insert proper bibtex citations once reference manager is set up
- Cross-reference §3-§9 line numbers once full paper is laid out
  in target journal template
- Tighten to ≤1500 words for Nature HB main text discipline
