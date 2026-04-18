# dreamOfkiki S19-S28 Atomic Plan

> **Pour agents autonomes :** SKILL REQUIS — utiliser `superpowers:subagent-driven-development`. Les steps utilisent la syntaxe checkbox (`- [ ]`).

**Goal** : phase finale cycle 1 — Paper 1 finalization + arXiv preprint + Nature HB submission + buffer + cycle-2 decision G6. Convergence vers livraison scientifique publique.

**Architecture** : 11 tasks atomiques (S19.1-S19.3, S20.1-S20.3, S21.1, S22.1, S25.1, S26.1, S27.1, S28.1) — 4 docs/papier + 4 ops externes + 3 cycle-2 préparation. Total attendu : 11 commits.

**Tech Stack** : Markdown + Quarto/LaTeX (paper rendering, future), arXiv submit (manual), Nature HB submission portal (manual user action), Zenodo DOI minting (manual user action), Git tags pour preprint freeze.

**Préréquis** :
- 81 commits dreamOfkiki, dernier `372624e docs(proof): add Pivot B decision tree`
- 114 tests passing, coverage 91.96%
- Framework C-v0.5.0+STABLE (target bump → C-v0.7.0+STABLE post-G3)
- Paper 1 outline + abstract + introduction + results-section drafts ready
- G2/G4 GO-CONDITIONAL, G5 DEFER, G3 PENDING reviewer
- 5 atomic plans antérieurs livrés et exécutés

**Deferred to cycle 2** :
- Paper 2 (ablation engineering paper) — outline initiation S28.1, full draft cycle 2
- E-SNN substrate (Loihi-2 thalamocortical)
- P_max real wiring + α-stream + ATTENTION_PRIOR canal-4
- Real fMRI lab partnership (T-Col extension cycle 2)

---

## Convention commits (validator-enforced)

- Subject ≤50 chars, format `<type>(<scope>): <description>`
- Scope ≥3 chars
- Body lines ≤72 chars, 2-3 paragraphs required
- NO AI attribution
- NO `--no-verify`

---

## File structure après S19-S28

```
dreamOfkiki/
├── docs/papers/paper1/
│   ├── outline.md                ✅ existing
│   ├── abstract.md               ✅ existing
│   ├── introduction.md           ✅ existing
│   ├── results-section.md        ✅ existing
│   ├── discussion.md             ← S19.1
│   ├── future-work.md            ← S19.2
│   ├── references.bib            ← S19.3
│   ├── methodology.md            ← S20.1
│   ├── background.md             ← S20.2
│   └── full-draft.md             ← S20.3 (assembly)
├── docs/milestones/
│   ├── g5-publication-ready.md   ✅ existing → S22.1 update GO-FULL or Pivot B
│   ├── arxiv-submit-log.md       ← S21.1
│   └── g6-cycle2-decision.md     ← S28.1
├── docs/papers/paper2/
│   └── outline.md                ← S27.1 (cycle 2 amorçage)
└── ops/
    └── nature-hb-submit-tracker.md ← S22.1 (manual submission tracker)
```

---

# Task S19.1 — Paper 1 Discussion section draft

**Goal** : draft §8 Discussion subsections (theoretical contribution, empirical contribution, limitations, comparison with prior art).

**Files:**
- Create : `docs/papers/paper1/discussion.md`

## Pattern

- 4 subsections (§8.1-8.4 per outline.md)
- Length target ~1.5 pages markdown
- Synthetic results caveats explicitly flagged
- Cross-reference §7 Results values + §3 Background pillars
- Commit subject: `docs(paper1): add discussion draft` (33 chars)

---

# Task S19.2 — Paper 1 Future Work section draft

**Goal** : draft §9 Future Work — cycle 2 E-SNN substrate, P_max wiring, real fMRI partnership, multi-substrate Conformance Criterion validation.

**Files:**
- Create : `docs/papers/paper1/future-work.md`

## Pattern

- 4 subsections (§9.1-9.4 per outline.md)
- Length target ~0.5-1 page markdown
- Cross-reference cycle-2 plans (deferred specs)
- Commit subject: `docs(paper1): add future work draft` (35 chars)

---

# Task S19.3 — Paper 1 references.bib stub

**Goal** : create BibTeX file with the key citations from outline §10 (Walker, Kirkpatrick, Tononi, Friston, Hobson, van de Ven, McClelland) + 5-10 supporting refs (McCloskey & Cohen 1989 forgetting, French 1999, Diekelmann & Born 2010, Stickgold 2005, Solms 2021, Whittington & Bogacz 2017, Rao & Ballard 1999, Shin 2017, Rebuffi 2017).

**Files:**
- Create : `docs/papers/paper1/references.bib`

## Pattern

- 15-20 BibTeX entries with proper IDs (e.g., `walker2004sleep`)
- Order: alphabetical by first author
- Commit subject: `docs(paper1): add references.bib stub` (37 chars)

---

# Task S20.1 — Paper 1 Methodology section draft

**Goal** : draft §6 Methodology — hypotheses H1-H4 with OSF DOI placeholder, statistical tests + Bonferroni, mega-v2 benchmark, RSA fMRI Studyforrest, R1 reproducibility contract.

**Files:**
- Create : `docs/papers/paper1/methodology.md`

## Pattern

- 5 subsections (§6.1-6.5 per outline.md)
- Length target ~1.5 pages markdown
- Cross-reference OSF pre-reg, statistics module, mega-v2 adapter, fmri-schema.yaml
- Commit subject: `docs(paper1): add methodology draft` (35 chars)

---

# Task S20.2 — Paper 1 Background section draft

**Goal** : draft §3 Background — four pillars A/B/D/C in depth + compositional gap argument.

**Files:**
- Create : `docs/papers/paper1/background.md`

## Pattern

- 5 subsections (§3.1-3.5 per outline.md)
- Length target ~1.5 pages markdown
- Cross-reference op-pair-analysis.md for compositional gap
- Cite all 4 pillar founders with proper bibtex IDs from references.bib
- Commit subject: `docs(paper1): add background draft` (34 chars)

---

# Task S20.3 — Paper 1 full-draft assembly

**Goal** : assemble all sections (abstract, introduction, background, framework, implementation, methodology, results, discussion, future-work, references) into single full-draft.md ready for arXiv preprint generation.

**Files:**
- Create : `docs/papers/paper1/full-draft.md`

## Pattern

- Concatenate all section files with proper headers (§1-§10)
- Add front matter (title, authors, affiliations, contact)
- Verify length target (8-10 pages main, supp unbounded)
- Mark synthetic placeholders explicitly for tracker
- Commit subject: `docs(paper1): assemble full draft v0.1` (38 chars)

---

# Task S21.1 — arXiv preprint submission log

**Goal** : prepare arXiv preprint submission. **Action externe utilisateur** : convert Markdown to arXiv-compatible LaTeX (Quarto / pandoc), upload to arXiv, capture preprint ID. Skeleton tracker.

**Files:**
- Create : `docs/milestones/arxiv-submit-log.md`

## Pattern

- Submission checklist (manual user action)
- arXiv ID placeholder (filled after submission)
- Tag commit `arxiv-v0.1` for freeze
- Commit subject: `docs(milestone): arXiv submit prep` (35 chars)

---

# Task S22.1 — G5 update + Nature HB submission tracker

**Goal** : update G5 report with submission decision (GO-FULL → Nature HB, or Pivot B activated). Create Nature HB submission tracker.

**Files:**
- Modify : `docs/milestones/g5-publication-ready.md` (add decision section)
- Create : `ops/nature-hb-submit-tracker.md`

## Pattern

- G5 outcome filled : GO-FULL or Pivot B branch chosen
- Nature HB tracker : portal URL, submission ID placeholder, cover letter draft, recommended reviewers list (T-Col network)
- Commit subject: `docs(milestone): G5 outcome + Nat HB tracker` (44 chars)

---

# Task S25.1 — Reviewer feedback handling skeleton

**Goal** : skeleton document for handling pre-submission reviewer feedback (T-Col.4 network) and post-submission reviewer responses.

**Files:**
- Create : `docs/papers/paper1/reviewer-feedback.md`

## Pattern

- Pre-submission feedback log (T-Col.4 reviewers)
- Post-submission reviewer round 1 responses (placeholder)
- Action items tracker
- Commit subject: `docs(paper1): add reviewer feedback skel` (39 chars)

---

# Task S26.1 — Buffer maintenance + bug-fix sweep

**Goal** : during S25-S28 buffer, address any deferred small fixes, refactor opportunities, doc polish. Skeleton placeholder commit if no specific items.

**Files:**
- Various (depends on findings)

## Pattern

- Run final CodeRabbit review on full repo
- Address any remaining MINOR findings
- Update CHANGELOG with cycle-1 close notes
- Commit subject: `chore: cycle-1 close maintenance sweep` (37 chars)

---

# Task S27.1 — Paper 2 outline (cycle 2 amorçage)

**Goal** : initiate Paper 2 outline (engineering ablation paper, target NeurIPS/ICML/TMLR per master spec §6).

**Files:**
- Create : `docs/papers/paper2/__init__.md`
- Create : `docs/papers/paper2/outline.md`

## Pattern

- Paper 2 structure : focus on engineering contributions (MLX-native ops, swap protocol, S1-S3 invariants, OSF reproducibility)
- Differentiate from Paper 1 (theory vs engineering)
- Cross-reference shared infrastructure
- Commit subject: `docs(paper2): add cycle-2 outline` (34 chars)

---

# Task S28.1 — G6 cycle-2 decision report

**Goal** : G6 gate report — cycle-2 amorçage decision based on cycle-1 outcome.

**Files:**
- Create : `docs/milestones/g6-cycle2-decision.md`

## Pattern

- Cycle-1 retrospective summary
- Cycle-2 scope decision : E-SNN substrate priority, P_max real wiring, real fMRI lab partnership, Paper 2 timeline
- Resource budget for cycle 2
- Commit subject: `docs(milestone): G6 cycle-2 decision` (37 chars)

---

# Self-review

**1. Spec coverage** :
- S19 Discussion + Future Work + References → S19.1 + S19.2 + S19.3 ✅
- S20 Methodology + Background + Full draft → S20.1 + S20.2 + S20.3 ✅
- S21 arXiv submit prep → S21.1 ✅
- S22 G5 outcome + Nature HB tracker → S22.1 ✅
- S25-S28 buffer / reviewer feedback / Paper 2 / G6 → S25.1 + S26.1 + S27.1 + S28.1 ✅

**2. Placeholder scan** : Pattern abrégé délibéré (5 plans atomiques antérieurs ont éprouvé le format). Chaque task a son contrat (files + commit subject pré-validé).

**3. Type consistency** : pas d'API code dans cette phase (pure documentation + manuel submission). Cohérence interne via cross-références aux files existants.

**4. Commit count** : 11 commits.

**5. Validator risks** : tous subjects pré-vérifiés ≤50 chars.

**6. Manual user actions critical** :
- arXiv preprint submission (S21.1)
- Nature HB submission portal (S22.1)
- Pre-submission reviewer outreach + feedback collection (S25.1, T-Col.4)
- OSF DOI lock (action externe en attente depuis S3)
- Reviewer DR-2 recruitment + feedback (G3 gate, action externe en attente depuis S3)

These actions are **not delegable to subagents** — require user account + judgment + identity. Plan documents them as critical-path items in S21.1 / S22.1 / S25.1 trackers.

---

**End of S19-S28 atomic plan.**

**Version** : v0.1.0
**Generated** : 2026-04-18 via refinement of S19-S28 from main plan
**Source** : `docs/superpowers/plans/2026-04-17-dreamofkiki-implementation.md`
