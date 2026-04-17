# Studyforrest RSA Feasibility for kiki-oniric M2.b

**Date** : 2026-04-17 (S2)
**Author** : Clement Saillant
**Status** : Decision locked (G1 gate)

## Question

Can Studyforrest fMRI dataset support RSA alignment against
kiki-oniric linguistic embeddings (M2.b metric)?

## Check list (populated by web research)

1. **Linguistic ROIs coverage** — Studyforrest provides STG, IFG, AG
   ROIs ?
   - [x] Verification : Studyforrest ships FreeSurfer cortical surface
     reconstructions for every participant (superdataset component
     `studyforrest-data-freesurfer`). FreeSurfer's default recon-all
     output includes the Desikan-Killiany (`aparc`) and Destrieux
     (`aparc.a2009s`) atlases, which label STG (bankssts /
     superiortemporal), IFG subregions (parsopercularis,
     parstriangularis, parsorbitalis), and inferior parietal (which
     contains AG / supramarginal). In addition, aggregate ROI
     timeseries under the Shen et al. (2013) 268-node functional
     parcellation are released for the 3T audiovisual movie run,
     covering the full cortex including language-lateralised regions.
     No dedicated "language localiser" ROI masks are shipped, but
     these can be derived on the fly from the FreeSurfer aparc labels.
   - Sources :
     - https://www.studyforrest.org/data.html
     - https://github.com/psychoinformatics-de/studyforrest-data-freesurfer
     - https://github.com/psychoinformatics-de/studyforrest-data
   - Verdict : PARTIAL-to-YES (ROIs not pre-packaged as language
     masks, but trivially derivable from bundled FreeSurfer aparc +
     Shen parcellation — no extra scanning needed).

2. **Stimulus-embedding mappability** — Forrest Gump narrative
   projectable to kiki ortho species ?
   - [x] Verification : The `studyforrest-data-annotations`
     superdataset extension ships a multi-level, time-aligned speech
     annotation of the German-dubbed movie and its English
     audio-description track. Per the F1000Research paper (Häusler &
     Hanke 2021, PMC7921887): 2,528 sentences, 16,187 words (incl.
     202 non-speech vocalisations), 66,611 phonemes — all with exact
     onset/offset timings. For every word the annotation provides
     lemma, 15-class and 43-class POS tags, syntactic dependencies
     with arc labels, and a 300-dim pre-computed word embedding. This
     maps directly onto the four kiki ortho-species layers :
     rho_phono (phoneme timings + ARPAbet/Prosodylab), rho_lex
     (lemma + POS), rho_syntax (dependency arcs), rho_sem (300-d
     vectors, trivially replaceable by kiki-oniric's own semantic
     embeddings). Nothing needs to be re-annotated.
   - Sources :
     - https://pmc.ncbi.nlm.nih.gov/articles/PMC7921887/
     - https://f1000research.com/articles/10-54
     - https://github.com/psychoinformatics-de/studyforrest-data-annotations
   - Verdict : word-level + phoneme-level + sentence-level ALL
     available — full four-layer mappability.

3. **RDM dimensionality** — enough stimuli for robust RDM (≥50
   conditions) ?
   - [x] Estimation : 16,187 timed word events and 2,528 sentence
     events across the 2-hour film. Even after aggregating rare
     tokens and restricting to content words, the unique-word
     vocabulary is ~3-5k types, and unique sentence-length bins
     trivially exceed the ≥50 threshold. For event-related RSA one
     can bin stimuli into ~100-500 meaningful conditions (POS bin
     x semantic cluster, or K-means clusters in the 300-d space),
     well above the Kriegeskorte robustness floor of 50.
   - Sources : direct counts from Häusler & Hanke 2021
     (https://pmc.ncbi.nlm.nih.gov/articles/PMC7921887/) ; TR=2s on
     3T run gives ~3600 volumes over the 2h film, providing ample
     trial-level degrees of freedom.
   - Verdict : SUFFICIENT (2-3 orders of magnitude above the ≥50
     floor).

4. **Access & licensing** — open access, permissive license ?
   - [x] Verification : All Studyforrest data are released under the
     ODC Public Domain Dedication and Licence (PDDL — equivalent to
     CC0 for data). No registration, no IRB gatekeeper, no paywall.
     Access is unrestricted via DataLad: `datalad clone
     https://github.com/psychoinformatics-de/studyforrest-data` gives
     the superdataset, then `datalad get <path>` pulls only the files
     needed (fine-grained, git-annex-backed). Data are also mirrored
     on OpenNeuro and GIN. Phase-2 audiovisual data is explicitly
     BIDS-compliant ([BIDS] tag in repo name). Only request is to
     cite the Studyforrest website.
   - Sources :
     - https://www.studyforrest.org/access.html
     - https://github.com/psychoinformatics-de/studyforrest-data
     - https://github.com/psychoinformatics-de/studyforrest-data-phase2
     - https://www.re3data.org/repository/r3d100011071
   - Verdict : OPEN (PDDL, no auth, BIDS, DataLad — best-case
     scenario).

## Decision tree

### Branch A — GO-STUDYFORREST (default if 4 checks pass)
- Adopt Studyforrest as M2.b fallback data source
- Lock `docs/interfaces/fmri-schema.yaml` at S4 referencing Studyforrest
- Proceed with H3 hypothesis pre-registration unchanged :
  "M2.b RSA alignment increases monotonically P_min < P_equ < P_max"

### Branch B — PIVOT-SYNTHETIC (if ROIs/stimuli inadequate)
- Create synthetic perceptual benchmark as proxy RSA
- Use controlled linguistic stimuli (e.g., existing EEG linguistic
  corpora, LibriSpeech transcripts paired with LLM activations)
- Adapt H3 hypothesis : "representational alignment with synthetic
  benchmark"
- Paper 1 framing shift : less "cognitive alignment" more
  "representational hierarchy"

### Branch C — DOWNGRADE-M2.b (if compute/data infeasible entirely)
- Move M2.b to Paper 1 Future Work section
- Remove M2.b from PUBLICATION-READY gate criteria (§9 framework spec
  amendment required)
- Paper 1 focus shifts to formal framework + ablation on other 7
  metrics
- Requires updating OSF pre-registration hypotheses : drop H3

## Decision taken

- **Branch** : A (GO-STUDYFORREST) — all four feasibility checks pass,
  and the four-layer linguistic annotation (phoneme / word / syntax /
  300-d semantic) maps one-to-one onto the kiki ortho species.
- **Rationale** : Studyforrest is the rare natural-stimulus fMRI
  corpus that is simultaneously (i) permissively licensed (PDDL,
  BIDS-compliant, DataLad-streamable so only needed subsets hit disk),
  (ii) equipped with whole-brain BOLD and FreeSurfer recons from which
  STG / IFG / AG masks are trivially derivable via the shipped
  Desikan-Killiany aparc and Shen-268 parcellations, (iii)
  stimulus-rich enough (16k timed words, 2.5k sentences, 66k
  phonemes) to support robust RDM estimation at every kiki layer
  without re-annotation, and (iv) already validated by prior
  language-fMRI work. The critic's caveat "narrative fMRI ≠ matched
  kiki stimuli" is neutralised by computing kiki-oniric embeddings on
  the exact transcript timestamps Studyforrest provides: the model
  sees the same linguistic input the subject heard, so RSA is a
  legitimate alignment test rather than a domain-transfer fudge.
  Compute is Apple-Silicon-feasible: ~3600 TR volumes x ~300k
  cortical vertices fits comfortably in unified memory on GrosMac
  M5 (16 GB) once restricted to the three language ROIs.
- **Next actions** :
  - [ ] Update `ops/tcol-outreach-plan.md` priority list if relevant
  - [ ] Queue `docs/interfaces/fmri-schema.yaml` draft (S4 task)
  - [ ] If Branch B or C: flag OSF pre-registration needs adjustment
    (S3 task) — NOT triggered, Branch A retained.
