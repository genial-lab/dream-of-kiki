# OSF Amendment #1 — Submission Package

**Parent registration** : https://osf.io/q6jyn
(DOI `10.17605/OSF.IO/Q6JYN`, dreamOfkiki Cycle 1, publicly
registered 2026-04-19T00:28:05Z per OSF API)

**Amendment scope** : Bonferroni family restructure (8-test family
split into per-cell 6-test + cross-cell 2-test families).

**Why a new linked registration** : per OSF semantics,
registrations are immutable after publish. Storing the amendment
only as a repo markdown (`docs/osf-amendment-bonferroni-cycle3.md`)
would make it invisible to any reviewer who opens the Q6JYN
registration page. COS guidance ("Preregistration: A Plan, Not a
Prison") recommends filing amendments as a new OSF registration
in the same project, linked to the original.

---

## Steps on osf.io

1. Log in to osf.io as the corresponding author.
2. Open the dreamOfkiki project page (parent of registration
   Q6JYN).
3. Navigate to *Registrations* → *Create a new registration*.
4. Choose template : **Open-Ended Registration** (the existing
   preregistration used *Standard Pre-Data Collection* ; open-ended
   is the OSF-sanctioned container for amendments and linked
   registrations).
5. Paste the payload below into the form fields.
6. In *Supplementary files*, attach the full text of
   `docs/osf-amendment-bonferroni-cycle3.md` (75 lines) as a
   supplementary file.
7. Publish.
8. Copy the minted DOI of the new amendment registration.

---

## Payload to paste

### Title

```
dreamOfkiki Cycle 1 — Amendment #1 : Bonferroni Family Restructure
(pre-data, linked to 10.17605/OSF.IO/Q6JYN)
```

### Summary

```
This is Amendment #1 to the dreamOfkiki Cycle 1 pre-registration
(OSF DOI 10.17605/OSF.IO/Q6JYN, registered 2026-04-19). It is
filed as a separate linked registration because OSF registrations
are immutable after publish.

Scope: restructure of the Bonferroni family correction for cycle-3
multi-scale hypotheses. The original family of 8 tests per
evaluation cell {H1, H2, H3, H4, H5-I, H5-II, H5-III, H6} with
alpha = 0.05/8 = 0.00625 is split into:
- a per-cell family of 6 tests at alpha = 0.05/6 = 0.00833 ;
- a cross-cell family of 2 tests (H3, H6) at alpha = 0.05/2 = 0.025.

Rationale: H3 and H6 aggregate effects across cells and should
not be double-counted by per-cell correction.

Timing: filed before any real-data compute (C3.8 full ablation
starts sem 2 earliest). This is a pre-data specification
correction, not a post-hoc rationalization.

Reporting: for transparency, the primary analysis reported in
Paper 1 (PLOS Computational Biology submission) uses the original
alpha = 0.00625 as the strictly more conservative bound; the
restructured family is reported as a secondary analysis.

Full amendment document (75 lines) attached as
osf-amendment-bonferroni-cycle3.md.
```

### Linked DOIs

```
Parent preregistration: 10.17605/OSF.IO/Q6JYN
```

### Tags

```
amendment, bonferroni, cycle-3, statistical-correction,
preregistration, dreamOfkiki
```

### Contributors

```
Clément Saillant (corresponding, L'Electron Rare, France)
```

### License

```
CC-BY-4.0
```

---

## After publish — repo updates to land

1. Copy the new OSF amendment DOI (format
   `10.17605/OSF.IO/XXXXX`).
2. Edit `docs/osf-amendment-bonferroni-cycle3.md` :
   - Add header with the new DOI and publish timestamp.
   - Remove the "Important open issue" note that was added on
     2026-04-21 (now resolved).
3. Edit `docs/papers/paper1/full-draft.md` §6.1 (Methodology /
   Pre-registered hypotheses) : add a sentence linking the
   amendment DOI alongside the primary pre-registration DOI.
4. Edit `docs/papers/paper1/cover-letter-plos-cb.md` : add the
   amendment DOI in the sign-off *Preregistration* line.
5. Edit `STATUS.md` : remove the "Open issue" about amendment
   invisibility from the Outstanding human actions section.
6. Commit (single PR, scope `osf`) :
   ```
   docs(osf): link amendment #1 DOI across paper, cover, status
   ```

---

## Transparency boilerplate for the PLOS CB Methods section

Insert near the Bonferroni paragraph in §6.1 :

> Pre-registration was filed at OSF DOI `10.17605/OSF.IO/Q6JYN`
> on 2026-04-19. A single amendment (Amendment #1, Bonferroni
> family restructure) was filed the same day, before any data
> collection, at OSF DOI `[to fill post-publish]`. The primary
> analysis reported here uses the original α = 0.00625 as the
> more conservative of the two bounds ; the restructured family
> (α = 0.00833 per-cell, 0.025 cross-cell) is provided as a
> secondary analysis. Both registrations are accessible via the
> parent OSF project page https://osf.io/q6jyn.

---

## Estimated time

- Login + navigate to registrations : 3 min
- Paste payload + attach file : 5 min
- Publish + copy DOI : 2 min
- Repo updates (5 file edits + commit) : 10 min
- **Total** : ~20 min
