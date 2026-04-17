# OSF Pre-Registration Upload Checklist

This is a **human action checklist** for the S3 OSF pre-registration.

Source document : `docs/osf-preregistration-draft.md`

## Checklist

- [ ] Login to https://osf.io with account (or create if needed)
- [ ] Create project "dreamOfkiki" with description from README.md
- [ ] Add wiki section "Pre-registration H1-H4"
- [ ] Upload `docs/osf-preregistration-draft.md` content to OSF
  pre-registration form using the "Standard Pre-Data Collection"
  template
- [ ] Lock the pre-registration (timestamped registration)
- [ ] Copy OSF project URL to `docs/osf-preregistration-draft.md`
  header
- [ ] Copy OSF DOI to `docs/osf-preregistration-draft.md` header
- [ ] Commit the URL+DOI update :

      ```bash
      git add docs/osf-preregistration-draft.md
      git commit -m "docs(osf): lock H1-H4 registration [DOI: ...]"
      ```

- [ ] Share OSF project link internally with T-Col.4 pre-submission
  reviewers network
- [ ] Update `ops/tcol-outreach-plan.md` status log with lock
  confirmation

## Estimated time

- OSF account setup : 10 min (one-time)
- Project creation + upload : 20 min
- Lock + URL/DOI update : 5 min
- **Total** : ~35 min human time

## Verification

After lock, verify :
- OSF project URL resolves
- DOI resolves via https://doi.org/10.17605/OSF.IO/XXXX
- Registration is timestamped and immutable
- H1-H4 hypotheses present and match `draft.md`

## Blocking conditions

- If OSF service is down : defer to next business day, do not
  proceed with experiments until pre-reg locked
- If template doesn't fit : use the generic "Open-Ended
  Registration" and attach the draft.md as supplementary file
