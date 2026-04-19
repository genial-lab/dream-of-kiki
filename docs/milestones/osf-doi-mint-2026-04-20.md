# OSF DOI Mint Checklist — Paper 1 v0.2 PLOS CB pivot

**Date** : 2026-04-20
**Cycle** : 1 closeout / cycle 3 setup
**Status** : **PENDING — checklist below, mint after arXiv ID lock**

This file tracks the OSF (Open Science Framework) project setup
and DOI mint workflow for Paper 1 v0.2 under the PLOS
Computational Biology submission target. The OSF DOI is the
durable preregistration handle cited in §1 Abstract and §6.1
Methodology of `docs/papers/paper1/full-draft.md` and is the
peer-review-grade artifact pointer Paper 2 will cross-cite.

## Project setup checklist

- [ ] Create OSF project under namespace `dreamofkiki` (or attach
      to existing `genial-lab` group if one is set up at OSF)
- [ ] Project title : `dreamOfkiki — substrate-agnostic formal
      framework for dream-based knowledge consolidation`
- [ ] Description : 2-paragraph abstract from
      `docs/papers/paper1/abstract.md` (≤ 250 words)
- [ ] Tags : `continual-learning`, `synaptic-homeostasis`,
      `dream-replay`, `formal-framework`, `conformance-criterion`,
      `MLX`, `spiking-neural-network`, `preregistration`
- [ ] License : MIT for code attachments, CC-BY-4.0 for the
      preregistration document and the full draft
- [ ] Contributors : `dreamOfkiki project contributors` (group
      byline ; corresponding author Clement Saillant). No AI
      attribution. CONTRIBUTORS.md frozen at commit `22784f8`.
- [ ] Linked GitHub repo : `https://github.com/genial-lab/dream-of-kiki`
- [ ] Project visibility : public

## Preregistration components to attach

- [ ] Primary preregistration draft :
      `docs/osf-preregistration-draft.md`
- [ ] Bonferroni cycle-3 amendment :
      `docs/osf-amendment-bonferroni-cycle3.md`
- [ ] Upload-package manifest : `docs/osf-upload-package.md`
- [ ] Upload checklist : `docs/osf-upload-checklist.md`
- [ ] Paper 1 v0.2 PDF (frozen) :
      `docs/papers/paper1/build/full-draft.pdf` (22 pages,
      296 KB)
- [ ] Paper 1 v0.2 LaTeX source :
      `docs/papers/paper1/build/full-draft.tex`
- [ ] Bibliography : `docs/papers/paper1/references.bib`
- [ ] Conformance matrix snapshot :
      `docs/milestones/conformance-matrix.md`
- [ ] G7 LOCKED report : `docs/milestones/g7-esnn-conformance.md`

## DOI mint trigger

The DOI is minted **after** both of the following are true :

1. **arXiv ID locked.** The arXiv preprint of Paper 1 v0.2 has
   been assigned an ID (format `2604.XXXXX`) and is publicly
   announced. See `docs/milestones/arxiv-submit-log.md`
   2026-04-20 entry for the corresponding arXiv tracker.
2. **Paper 1 v0.2 frozen.** The repo is tagged `arxiv-v0.2` at
   the commit that produced
   `docs/papers/paper1/build/full-draft.pdf` (currently
   `22784f8` ; will be re-tagged if a final stylistic pass lands
   before arXiv deposit).

Once both conditions hold, the OSF "Mint DOI" workflow is
triggered through the OSF web UI (Project Settings →
Identifiers → Mint DOI). The minted DOI is then :

- **Inserted** into `docs/papers/paper1/full-draft.md` §1
  Abstract and §6.1 Methodology, replacing the preregistration
  DOI placeholder `10.17605/OSF.IO/Q6JYN` if a fresh DOI is
  issued (the project already has a slot reserved at this
  identifier).
- **Cross-cited** in Paper 2 at
  `docs/papers/paper2/full-draft.md` (preregistration handle for
  the H1–H4 evaluation).
- **Recorded** in this file under "Minted DOI" below.

## Minted DOI (fill after mint)

- **OSF DOI** : TBD (currently reserved as
  `10.17605/OSF.IO/Q6JYN`)
- **Mint date** : TBD
- **OSF project URL** : TBD
- **Linked arXiv ID** : TBD (cross-reference to
  `docs/milestones/arxiv-submit-log.md`)
- **Linked Zenodo DOI (artifacts)** : TBD (separate mint for the
  code + benchmark snapshot, planned post-G5)

## Cross-references

- `docs/milestones/arxiv-submit-log.md` — sibling tracker for
  arXiv preprint deposit
- `docs/papers/paper1/cover-letter-plos-cb.md` — PLOS CB cover
  letter draft for v0.2 submission
- `docs/osf-preregistration-draft.md` — primary preregistration
- `docs/osf-amendment-bonferroni-cycle3.md` — α-correction
  amendment landed before any cycle-3 real-data compute
- `docs/milestones/g5-publication-ready.md` — gate G5 with PLOS
  CB target + DR-2 proved reference (updated 2026-04-19)
- `docs/milestones/g7-esnn-conformance.md` — gate G7 LOCKED for
  cross-substrate conformance
