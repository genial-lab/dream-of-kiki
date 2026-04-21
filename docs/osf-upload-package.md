# OSF Upload Package — Ready to Paste

**Status** : EXECUTED 2026-04-19 — OSF project created,
pre-registration made public (immutable after publish per OSF
semantics), DataCite DOI `10.17605/OSF.IO/Q6JYN` auto-registered
on publish, project URL `https://osf.io/q6jyn`. This file is
retained as historical record of the upload payload.

Use this file during the OSF upload procedure.
Follow `docs/osf-upload-checklist.md` step-by-step.
Copy-paste content below into OSF web forms.

---

## Project creation (OSF dashboard)

**Title** :
```
dreamOfkiki: A Substrate-Agnostic Formal Framework for Dream-Based Knowledge Consolidation in Artificial Cognitive Systems
```

**Description** :
```
Research program investigating how artificial cognitive systems can
learn, memorize, and organize knowledge through dream-inspired offline
consolidation. Produces a formal framework with executable axioms
DR-0..DR-4 (substrate-agnostic) and an empirical ablation on
kiki-oniric (a hierarchical linguistic model with four ortho species:
phonological, lexical, syntactic, semantic). Three profiles tested:
P_min (minimal), P_equ (balanced), P_max (maximalist), against a
no-dream baseline.

Theoretical pillars (serial): A Walker/Stickgold consolidation, B
Tononi SHY homeostasis, D Friston Free Energy. In parallel: C
Hobson/Solms creative recombination.

Code: https://github.com/electron-rare/dream-of-kiki (MIT)
Contact: Clement Saillant — clement@saillant.cc — L'Electron Rare
```

**Category** : Project
**License** : CC-BY 4.0

---

## Pre-registration form (Standard Pre-Data Collection template)

### Study information

**Title** :
```
Dream-Based Knowledge Consolidation in Artificial Cognitive Systems: A Pre-Registered Ablation of Three Consolidation Profiles
```

**Authors** :
```
Clement Saillant — L'Electron Rare, France
(dreamOfkiki project contributors, see CONTRIBUTORS.md at project repo)
```

**Research questions / Hypotheses** — PASTE CONTENT FROM :
`docs/osf-preregistration-draft.md` section 2 (H1-H4 complete)

### Design plan

**Study type** : Observational (ablation of dream-consolidation
profiles on pre-existing architecture kiki-oniric)

**Blinding** : Not applicable (algorithmic comparison, no human
participants)

**Study design** :
```
Within-architecture ablation comparing 3 dream-consolidation profiles
(P_min, P_equ, P_max) against a no-dream baseline, on:
- mega-v2 continual learning benchmark (25 domains, 498K examples)
- Studyforrest fMRI representational alignment (STG, IFG, AG ROIs)

Each profile evaluated with 3 seeds minimum, 5 if power < 0.8.
Pre-registered analyses use paired t-test (H1), TOST (H2),
Jonckheere-Terpstra (H3), one-sample t-test (H4).
Bonferroni correction α = 0.0125 per hypothesis.
```

### Sampling plan

**Existing data** : No (experiments begin S5+)
**Data collection procedures** :
```
Runs generated via dreamOfkiki.harness (open-source, MIT).
3 seeds per (profile, metric) cell minimum.
Power analysis on pilot runs S5-S7 to determine final seed count.
```

### Variables

**Independent variables** :
```
- Profile (categorical): P_min, P_equ, P_max, baseline (no-dream)
- C-version (covariate, expected STABLE throughout)
```

**Dependent variables** :
```
- M1.a forgetting rate (continual learning)
- M1.b average accuracy cross-tasks
- M2.b RSA fMRI alignment (Pearson correlation)
- M3.a FLOPs ratio dream/awake
- M3.b offline gain (pivot)
- M3.c energy per episode (proxy)
- M4.a recombination quality (teacher scorer gelé)
- M4.b structure discovery (permutation test)
```

### Analysis plan

PASTE CONTENT FROM :
`docs/osf-preregistration-draft.md` sections 3, 4, 5 verbatim
(primary analyses, sample size, multiple comparison correction,
exclusion rules, deviations handling).

### Other

**Existing analysis** : No (primary analyses confirmatory per
pre-registration).

**Other comments** :
```
Framework formal definition at:
https://github.com/electron-rare/dream-of-kiki/blob/main/docs/specs/2026-04-17-dreamofkiki-framework-C-design.md

Canonical glossary at:
https://github.com/electron-rare/dream-of-kiki/blob/main/docs/glossary.md

Cycle 2 (SNN substrate E) documented as Future Work in paper 1,
not part of this pre-registration.
```

---

## After lock — commit template

Once OSF registration is locked and you have the URL + DOI:

```bash
cd /Users/electron/Documents/Projets/dreamOfkiki

# Update pre-registration draft header
# Replace "https://osf.io/XXXX" with actual URL
# Replace "10.17605/OSF.IO/XXXX" with actual DOI
$EDITOR docs/osf-preregistration-draft.md

git add docs/osf-preregistration-draft.md
git commit -m "docs(osf): lock H1-H4 registration"

git push
```

Subject length check: `docs(osf): lock H1-H4 registration` = 35 chars ≤50 ✓

---

## Verification after lock

- [ ] https://osf.io/{your-id}/ resolves
- [ ] https://doi.org/10.17605/OSF.IO/{your-id} resolves
- [ ] Registration timestamp is shown on OSF page
- [ ] H1-H4 text matches `docs/osf-preregistration-draft.md` section 2
- [ ] Update `STATUS.md` and `ops/tcol-outreach-plan.md` with lock
  confirmation + DOI in follow-up commit

---

## Total time estimate

- Account login : 5 min
- Project creation + description : 10 min
- Pre-registration form fill : 15 min
- Lock + DOI copy : 5 min
- **Total** : ~35 min human time

---

## Blocking conditions

- OSF service down → defer next business day, do not start S5
  experiments
- Template doesn't fit → use "Open-Ended Registration" template,
  attach `docs/osf-preregistration-draft.md` as file
- Account issue → create new OSF account (uses your email)
