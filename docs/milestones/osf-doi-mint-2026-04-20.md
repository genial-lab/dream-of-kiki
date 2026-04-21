# OSF DOI Mint Checklist — Paper 1 v0.2 PLOS CB pivot

**Date** : 2026-04-20
**Cycle** : 1 closeout / cycle 3 setup
**Status** : **MINTED 2026-04-19T00:28:05Z** — DataCite DOI
`10.17605/OSF.IO/Q6JYN` auto-registered at the moment the OSF
registration was made public (OSF convention : DOI string is a
deterministic derivation of the registration GUID, prefix
`10.17605/OSF.IO/` + uppercased GUID, so paper drafts could
cite it pre-publish without re-render). Confirmed via
`api.osf.io/v2/registrations/q6jyn/identifiers/`. Note : OSF
does **not** offer "reserve-then-activate" DOI semantics — this
file's earlier "Confirm path" / "reserved slot" language has
been corrected post-audit (2026-04-21).

This file tracks the OSF (Open Science Framework) project setup
and DOI mint workflow for Paper 1 v0.2 under the PLOS
Computational Biology submission target. The OSF DOI is the
durable preregistration handle cited in §1 Abstract and §6.1
Methodology of `docs/papers/paper1/full-draft.md` and is the
peer-review-grade artifact pointer Paper 2 will cross-cite.

## Project setup checklist

- [ ] Create OSF project under namespace `dreamofkiki`. Decision
      rule: if the `hypneum-lab` group already exists on OSF at
      project-creation time, attach the project to that group
      instead of creating a standalone namespace project. Default
      to the standalone `dreamofkiki` project if no `hypneum-lab`
      group is present.
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
- [ ] Linked GitHub repo : `https://github.com/hypneum-lab/dream-of-kiki`
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
   `docs/papers/paper1/build/full-draft.pdf` (provisional anchor
   `22784f8`, subject to final freeze: if a stylistic pass lands
   before arXiv deposit the tag is moved to that commit and this
   line is updated with the final hash).

Once both conditions hold, the OSF "Mint DOI" workflow is
triggered through the OSF web UI (Project Settings →
Identifiers → Mint DOI). The minted DOI is then :

- **Inserted** into `docs/papers/paper1/full-draft.md` §1
  Abstract and §6.1 Methodology. Decision workflow performed by
  the corresponding author at mint time:
  - **Mint path**: a fresh DOI is issued by OSF. The placeholder
    `10.17605/OSF.IO/Q6JYN` is *replaced* in the paper sources
    and re-rendered before arXiv v2 upload.
  - **Confirm path**: the reserved slot at
    `10.17605/OSF.IO/Q6JYN` is the one that gets activated. The
    placeholder is *kept verbatim*; only the "Minted DOI" block
    below is updated to record the confirmation (no source-text
    replacement, no re-render needed).
- **Cross-cited** in Paper 2 at
  `docs/papers/paper2/full-draft.md` (preregistration handle for
  the H1–H4 evaluation).
- **Recorded** in this file under "Minted DOI" below.

## Minted DOI (filled 2026-04-19, corrected 2026-04-21 post-audit)

- **OSF DOI** : `10.17605/OSF.IO/Q6JYN` (auto-minted by DataCite
  when OSF registration was made public ; DOI string is a
  deterministic derivation of the registration GUID — this is
  standard OSF behaviour, not a "reserve-then-activate" workflow)
- **Mint date** : 2026-04-19T00:28:05Z (OSF API `date_registered`)
- **OSF project URL** : https://osf.io/q6jyn (resolves via
  `https://doi.org/10.17605/OSF.IO/Q6JYN` 302 redirect)
- **Registration immutability** : per OSF policy, this
  registration cannot be edited. Any amendment (e.g.,
  `docs/osf-amendment-bonferroni-cycle3.md`) must be filed as a
  new linked OSF registration before it is reviewer-visible.
- **Linked arXiv ID** : TBD (filled after arXiv v0.2 deposit ;
  cross-reference to `docs/milestones/arxiv-submit-log.md`)
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
