# dreamOfkiki — Master Design Spec

**Version** : v0.1.0-draft
**Date** : 2026-04-17
**Author** : Clement Saillant (L'Electron Rare)
**Status** : Draft for user review
**Related specs** :
- `2026-04-17-dreamofkiki-framework-C-design.md` (framework formel)
**Related projects** :
- `kiki-flow-research` (3 stories en cours — indépendantes de ce programme)
- `kiki-flow-core` (source de la fork `kiki-oniric`)

---

## 0. Abstract

dreamOfkiki est un programme de recherche de 6-7 mois (cycle 1) visant à concevoir une architecture cognitive inspirée du sommeil humain pour des systèmes d'IA. Le programme produit deux papers complémentaires — un paper théorique (framework formel C, cible *Nature Human Behaviour*) et un paper empirique (ablation sur substrat kiki-oniric, cible NeurIPS/ICML/TMLR) — avec publication séquentielle stricte.

L'architecture repose sur quatre piliers théoriques (consolidation Walker/Stickgold, homéostasie synaptique Tononi SHY, prédictive Friston FEP en série + recombinaison créative Hobson/Solms en parallèle), instantié sur kiki-oniric (fork dédiée de kiki-flow-core) via un runtime asynchrone à swap worktree atomique, évalué sur une matrice stratifiée de 8 métriques (E3 cognitive + E4 engineering) avec reproductibilité bit-exact.

Le programme adopte une **approche 5-tracks** (C framework, A kiki implementation, T-Ops infrastructure, T-Col external collaboration, plus un track E SNN thalamocortical **différé à cycle 2**), un versionning hybride **DualVer** (formel + empirique), et un modèle de livraison scientifique **open-by-default** (code MIT, pre-registration OSF, DOI Zenodo, dashboard public read-only).

---

## 1. Motivation & positionnement scientifique

### 1.1 Question de recherche centrale

*Comment une IA peut-elle apprendre, mémoriser et organiser son savoir par le rêve ?*

Le rêve n'est pas traité comme métaphore mais comme **fonction computationnelle** : un processus offline/asynchrone dédié à consolider, réguler et restructurer le savoir acquis pendant l'éveil. Le programme formalise cette intuition en un framework substrat-agnostique et le valide empiriquement sur un système linguistique hiérarchique existant (kiki).

### 1.2 Piliers théoriques

**En série (séquence canonique du dream-episode)** :
- **A — Consolidation mnésique** *(Walker, Stickgold, McClelland 1995)* : replay hippocampique transfère l'épisodique vers le sémantique
- **B — Homéostasie synaptique SHY** *(Tononi, Cirelli)* : downscaling réduit le bruit, économise l'énergie, améliore signal/bruit
- **D — Restructuration prédictive** *(Friston free energy, Clark predictive processing)* : minimisation d'énergie libre globale, abstraction d'invariants

**En parallèle (branche créative)** :
- **C — Recombinaison & simulation** *(Hobson, Solms, Revonsuo)* : génération de scénarios nouveaux, exploration de l'espace des possibles

L'ordre A→B→D est thermodynamiquement justifié (dissipation progressive : consolider d'abord, dégrader ensuite, restructurer enfin). C parallèle capture la branche REM créative qui coexiste avec NREM consolidant.

### 1.3 Positionnement par rapport à la littérature

| Travaux | Contribution | dreamOfkiki addition |
|---------|--------------|----------------------|
| van de Ven 2020 (brain-inspired replay) | Generative replay via VAE latent | Framework formel + ablation 3 profils + alignement fMRI |
| Kirkpatrick 2017 (EWC) / OPLoRA | Consolidation synaptique Fisher | Intégration comme instance de B-Tononi dans taxonomie unifiée |
| Complementary Learning Systems | Hippocampe + néocortex analogie | Opérationnalisation en dream-episode comptable |
| Diba & Buzsáki 2007 | Replay en veille calme | Justifie choix Q6-C interleaved continu |
| Tononi SHY | Downscaling pendant sommeil | Axiomatisé dans framework C (opération downscale) |
| Friston FEP | Minimisation énergie libre | Opération restructure |

**Différentiation** : le framework C est le premier (à notre connaissance) à **axiomatiser** formellement l'ensemble des opérations oniriques comme un **semi-groupe non-commutatif** d'opérations composables avec budget borné (DR-2 compositionnalité affaiblie 2026-04-21, preuve v0.2 sous précondition `¬(RESTRUCTURE ≺ REPLAY)` ; irréductibilité sémantique du 4ᵉ générateur `recombine` établie §7 de `docs/proofs/dr2-compositionality.md`).

---

## 2. Scope & non-scope

### 2.1 Scope cycle 1 (6-7 mois, S1-S28)

- **Track C** : framework formel (40-80 pages) avec axiomes DR-0..DR-4 formalisés strictement, preuve compositionnalité DR-2 affaiblie (précondition empirique 2026-04-21, v0.2 rédigée 2026-04-21) avec fallback DR-2' canonique en réserve
- **Track A** : implémentation 3 profils (P_min, P_equ, P_max) sur fork `kiki-oniric` avec runtime async swap worktree
- **Track T-Ops** : infrastructure CI/CD, eval harness stratifié, dashboard public, scorer teacher gelé, reproducibilité bit-exact
- **Track T-Col** : outreach fMRI labs (Gallant/Norman/Huth), fallback Studyforrest pré-locké, pre-submission network
- **2 papers** : Paper 1 (Nature HB) soumis S20, Paper 2 (NeurIPS/ICML/TMLR) draft complet S24, soumis **après acceptation Paper 1** (S1-séquentielle stricte)
- **Infrastructure ouverte** : GitHub public, HuggingFace models, Zenodo DOI, OSF pre-registration

### 2.2 Non-scope cycle 1 (déplacé à cycle 2)

- **Track E — SNN thalamocortical substrat** : co-design clean-room sur Loihi-2 ou MLX-SNN custom. Positionné comme **Future Work** dans Paper 1, avec amorce outreach Intel NRC T-Col.5 dès S9+
- **Intégrations downstream** : mascarade consolidation, Zacus Professor dreams, Factory 4 Life agents — tout cela relève de projets séparés
- **Applications grand public / commerciales** : formations Moodle, consulting, levier business L'Electron Rare — gérés hors du programme scientifique
- **Extensions multimodales** : perception + action embodied — réservé à cycles ultérieurs

---

## 3. Architecture d'ensemble (5 tracks)

```
┌──────────────────────────────────────────────────────────┐
│           META-COORDINATION LAYER                         │
│  Invariants I/S/K · Glossary · DualVer · Dream-sync HITL │
└────┬────────┬────────────┬───────────┬───────────────────┘
     │        │            │           │
  ┌──▼──┐  ┌──▼──┐      ┌──▼───┐   ┌───▼──┐
  │  C  │  │  A  │      │T-Ops │   │T-Col │
  │Frmwk│  │kiki-│      │infra │   │ext   │
  │     │  │oniric│      │trans │   │trans │
  │ 8s  │  │ 12s │      │ 28s  │   │ 28s  │
  │     │  │     │      │      │   │      │
  │P1   │  │P2   │      │(tech │   │(fMRI │
  │NatHB│  │NeuR │      │report│   │ lab) │
  │     │  │IPS  │      │      │   │      │
  └─────┘  └─────┘      └──────┘   └──────┘
     ▲        ▲
     └────────┴────┐
                   ▼
         SHARED EVAL HARNESS
         (stratified matrix, bit-exact repro)

ROADMAP cycle 2 (2027+) :
└─ E — SNN thalamocortical co-design (Loihi-2 ou MLX-SNN)
   Documented as Future Work Section 7 de Paper 1.
```

### 3.1 Meta-coordination

- **Invariants registry** : `invariants.md` numéroté (I1..In information, S1..Sn safety, K1..Kn compute, DR-0..DR-4 axiomes théoriques). Chaque track référence explicitement.
- **Glossaire canonique** : `glossary.md`, termes définis une fois. Mention du nom logique `dreamOfkiki` (camelCase) vs nom repo technique `dream-of-kiki` (kebab-case).
- **Sync HITL** : "Dream-sync Monday" chaque lundi, 1h chrono, sync pack par track, décisions prises par vous. Budget attention ≤15h/sem avec cut-gate 2 semaines.
- **DualVer** : format `C-v<FC-MAJOR>.<FC-MINOR>.<FC-PATCH>+<EC-STATE>`. FC = formal consistency (SemVer), EC = empirical consistency {STABLE, DIRTY, INVALIDATED}.

### 3.2 Tracks scientifiques

| Track | Durée | Livrables clés | Owner responsabilité |
|-------|-------|---------------|---------------------|
| **C** (framework) | S1-S8 | `framework-C-v1.0.md`, axiomes DR formalisés, preuve DR-2 affaiblie (v0.2 2026-04-21, précondition empirique `¬(RESTRUCTURE ≺ REPLAY)`), eval protocol | Vous |
| **A** (kiki-oniric) | S1-S12 | Fork `kiki-oniric`, Story 0 expose-primitives, 3 profils P_min/P_equ/P_max, runtime swap worktree | Vous + sub-agents |

### 3.3 Tracks services transverses

| Track | Durée | Livrables clés |
|-------|-------|----------------|
| **T-Ops** | S1-S28 | Monorepo CI/CD, eval harness stratifié, dashboard public `dream.saillant.cc`, bit-exact repro enforcement, teacher scorer gelé (Qwen3.5-9B Q4_K_M SHA pinned) |
| **T-Col** | S1-S28 (front-loaded S1-S8) | Fallback Studyforrest locké S2, proposals 3 labs fMRI S4, pre-submission network S14-S18, cycle-2 Intel NRC seeding S9+ |

### 3.4 Calendrier 6-7 mois (28 semaines)

| Phase | Semaines | Activités dominantes |
|-------|----------|----------------------|
| **Setup** | S1-S2 | T-Col fallback lock, T-Ops infra, C kickoff, A Story 0 expose-primitives |
| **Formalization + Foundation** | S3-S6 | C v0.3→v0.5, A P_min fonctionnel, OSF pre-registration H1-H4 |
| **Core implementation** | S7-S12 | C v0.7→v0.9, A P_equ, milestone G2 (P_min viable S8), G3 (DR-2 preuve sous précondition — v0.2 disponible 2026-04-21, external review S6-S8), G4 (P_equ fonctionnel S12) |
| **Ablation + Experiments** | S13-S18 | C v1.0, A P_max, full E3+E4 eval, gate G5 PUBLICATION-READY S18 |
| **Paper 1 submission** | S18-S20 | Paper 1 arXiv S18, pre-submission review T-Col.4, submit Nature HB S20 |
| **Paper 2 draft + buffer** | S20-S24 | Paper 2 draft complet S24 (gelé en attente acceptation Paper 1), business L'Electron Rare launch |
| **Buffer** | S25-S28 | Maintenance, response aux review Paper 1 si rapide, gate G6 cycle-2 decision S28 |

---

## 4. Runtime architecture (éveil + rêve asynchrones)

### 4.1 Topologie variable par profil

| Profil | Awake | Dream | Shared state |
|--------|-------|-------|--------------|
| **P_min** (β → 1) | Studio M3 Ultra | Studio M3 Ultra (worker thread + worktree isolé) | mmap local Arrow IPC |
| **P_equ** (β+δ → 1+3+4) | Studio M3 Ultra | KXKM-AI RTX 4090 (GPU dédié dream) | NFS `/mnt/shared` + sync batch |
| **P_max** (α+β+δ → 1+2+3+4) | Studio M3 Ultra | KXKM-AI + GrosMac M5 (recombination VAE) | 3-machine pipeline + cache LRU |

### 4.2 Swap worktree atomique (pas de merge actif)

Dream process tient `W_scratch` (copie travail modifiable) tandis qu'awake opère sur `W_awake`. `W_dream` est un snapshot frozen. Quand le dream-episode signale `ready_to_commit` (budget atteint / N DE complétés / validation interne OK) :

1. Awake pause ~500ms
2. Guard test S2 (NaN/Inf) sur W_scratch
3. Guard test S1 (retained benchmark non-régression) sur W_scratch
4. Guard test S3 (topology valid) si canal 3 présent
5. Si tous OK → **swap atomique** : `W_awake ← W_scratch`, re-baseline `W_dream ← W_awake`, new `W_scratch ← W_awake`
6. Awake resume

Avantages vs EASGD : conflits éliminés par construction, rollback trivial (swap avorté), reproducibilité préservée.

### 4.3 Canaux d'information (profils)

**Éveil → Rêve** :
- α raw traces (P_max only) : ring buffer disque 10GB max
- β episodic curated : SQLite append-log avec signature saillance
- δ hierarchical latents : ring 256 snapshots × 5 niveaux ortho species
- γ weights-only : pointeur checkpoint courant

**Rêve → Éveil** :
- 1 weight_delta : via swap
- 2 latent_samples : queue FIFO consommée par awake data augmenter
- 3 hierarchy_chg : appliqué atomiquement au swap, avec guard S3
- 4 attention_prior : copié au swap ou live read-only

---

## 5. Évaluation (E3 cognitive + E4 engineering)

### 5.1 Métriques (8 total, M3.b pivot)

| Code | Nom | Famille | Substrat cible |
|------|-----|---------|----------------|
| **M1.a** | Forgetting rate | Continual learning | kiki sur SplitNLP + mega-v2 |
| **M1.b** | Average accuracy cross-tasks | Continual learning | kiki |
| **M2.b** | RSA fMRI alignment | Representational | kiki + fMRI (Studyforrest fallback ou lab) |
| **M3.a** | FLOPs ratio dream/awake | Engineering | kiki MLX profile |
| **M3.b** ★ | Offline gain (FLOPs-equiv wall-clock) | Engineering (pivot E3/E4) | kiki |
| **M3.c** | Energy per episode (proxy MLX) | Engineering | kiki (Loihi natif en cycle 2) |
| **M4.a** | Recombination quality (teacher scorer gelé) | Emergence | kiki + Qwen3.5-9B Q4_K_M SHA-pinned |
| **M4.b** | Structure discovery (permutation test) | Emergence | kiki |

### 5.2 Stratification eval matrix

| Bump type | Cellules obligatoires |
|-----------|-----------------------|
| **PATCH** | Métrique(s) de l'axe touché × P_equ × 1 seed |
| **MINOR** | 8 métriques × P_equ × 3 seeds |
| **MAJOR** | **Full grid** : 8 × 3 profils × 3 seeds = 72 runs |
| **EC change** | Re-run métriques publiées seulement |

### 5.3 Reproducibilité bit-exact (8 métriques, contrat R1 étendu)

Tous résultats bit-identiques pour même `(c_version, profile, seed, run_id, commit_sha, benchmark_version)`. Stratégies par métrique : RNG seeded, MLX deterministic mode ou CPU fallback, FLOPs-equivalent wall-clock (pas wall-clock réel), teacher scorer Qwen3.5-9B Q4_K_M gelé par SHA256.

### 5.4 Hypothèses pre-registrées (OSF, S3)

- **H1** : P_equ améliore M1.a de ≥10% vs baseline sans-dream sur mega-v2
- **H2** : P_max marginal ≤5% vs P_equ (rendements décroissants)
- **H3** : M2.b augmente monotoniquement P_min < P_equ < P_max
- **H4** : M3.c proxy ≤2× awake pour P_equ (déployement viable)

---

## 6. Publication strategy (S1-séquentielle stricte)

### 6.1 Paper 1 — Nature Human Behaviour / PLoS Comp Bio

*"dreamOfkiki: A Substrate-Agnostic Formal Framework for Dream-Based Knowledge Consolidation in Artificial Cognitive Systems"*

- Contribution théorique (framework formel) + validation empirique (kiki-oniric ablation) + alignement cognitif (RSA fMRI)
- Main : 8-10 pages ; Supplementary : 30-50 pages (formal proofs, details)
- Milestone : arXiv S18, submit S20

### 6.2 Paper 2 — NeurIPS / ICML / TMLR

*"Ablating Dream Channels: Engineering Trade-offs of Memory Consolidation in Large Language Models"*

- Contribution ingénierie (swap worktree, runtime async) + ablation empirique 3 profils + open-source release
- Main : 9 pages ; Supp : unlimited
- Milestone : draft complet S24, submit **après acceptation Paper 1** (likely 2027)

### 6.3 Authorship & affiliation

**Byline** : "dreamOfkiki project contributors"

```
Clement Saillant¹ (corresponding),
[fMRI lab collaborator]² (courtesy affiliation, if applicable),
and AI Collaborators³ (acknowledged separately)

¹ L'Electron Rare, France
² [Lab name], [University]
³ See CONTRIBUTORS.md for full list
```

À vérifier avant S18 : Nature HB submission guidelines. Fallback : single author "Saillant, C. on behalf of dreamOfkiki project contributors".

### 6.4 Open science engagements

| Artéfact | Plateforme | Timing |
|----------|-----------|--------|
| Code | `github.com/electron-rare/dreamOfkiki` (MIT) | S1 |
| Models | `huggingface.co/clemsail/kiki-oniric-{P_min,P_equ,P_max}` | S22 |
| Dataset eval | retained benchmark 500 items | S20 |
| Harness | `dreamOfkiki.harness` pip-installable | S22 |
| Dashboard | `dream.saillant.cc` public read-only | S1 (growing) |
| Pre-registration | OSF | S3 |
| DOI artefacts | Zenodo | S20-S22 |
| FAIR compliance | Paper 1 supp | S18 |

---

## 7. Risques & gates go/no-go

### 7.1 Risques HIGH consolidés

| ID | Description | Mitigation |
|----|-------------|------------|
| R-EXT-01 | fMRI lab outreach échoue | Fallback Studyforrest pré-locké S2 |
| R-CHA-01 | Charge cognitive > 15h/sem | Cut-gate 2 sem, délégation agressive |
| R-FRM-01 | DR-2 preuve v0.2 rejetée par reviewers externes | Fallback DR-2' ordre canonique (activation critères dans `docs/milestones/g3-decision-log.md`) |
| R-IMP-01 | Swap guards trop stricts | Seuils configurables, permissif au début |
| R-CAL-01 | Paper 1 reject chronique | Fallback PLoS CB / Cognitive Science |

### 7.2 Gates go/no-go

| Gate | S | Décision | Critère go |
|------|---|----------|------------|
| G1 | S2 | T-Col fallback locked | Adapter Studyforrest testé |
| G2 | S8 | P_min viable | Accuracy ≥ baseline − 2%, runtime stable 48h |
| G3 | S8 | DR-2 preuve OK | Preuve v0.2 (affaiblie avec précondition empirique) complétée et peer-reviewée externe |
| G4 | S12 | P_equ fonctionnel | > P_min sur ≥2 métriques significatives, invariants green 7j |
| G5 | S18 | PUBLICATION-READY | Tous critères Section 4.6 spec framework |
| G6 | S28 | Amorcer cycle 2 | Paper 1 submitted, bandwidth dispo |

### 7.3 Pivots préparés

**Pivot A** (si G2 ou G4 fail) : single-paper TMLR/ICLR workshop, durée 4-5 mois, budget libéré → L'Electron Rare
**Pivot B** (si G3 fail) : Paper 1 semi-formel, cible Neural Computation / Cognitive Systems Research, calendrier S18 préservé

### 7.4 Exit criteria cycle 1 (seuil 5/8 succès complet)

1. Paper 1 soumis
2. Paper 2 draft complet
3. Harness open-source publié (GitHub + Zenodo DOI)
4. 3 model snapshots HuggingFace publics
5. Dashboard `dream.saillant.cc` live
6. OSF pre-registration H1-H4 fermé avec résultats
7. `CONTRIBUTORS.md` à jour
8. Zero violation BLOCKING non résolue

---

## 8. Ressources & dépendances

### 8.1 Compute

- **Studio M3 Ultra 512GB** (MLX) : primary awake, peut absorber dream P_min
- **KXKM-AI RTX 4090 24GB** : GPU dream dédié P_equ/P_max
- **GrosMac M5 16GB** : recombination VAE P_max uniquement
- **Tower 31GB** : services (Grafana mirror, Langfuse, Piper TTS non-critical)

### 8.2 Data

- **mega-v2** : 498,723 examples, 25 domaines (existant)
- **retained benchmark** : 500 items gelés (à créer S1-S3)
- **Studyforrest fMRI** (fallback) ou **lab partner data** (target S8)

### 8.3 External dependencies

- fMRI lab partner (Gallant UC Berkeley / Norman Princeton / Huth UT Austin) — T-Col outreach S3-S4
- Intel NRC cycle-2 outreach (T-Col.5 S9+, not blocking cycle 1)
- OSF pre-registration (S3)
- Zenodo DOIs (S20-S22)
- HuggingFace model hosting (existant, compte `clemsail`)

### 8.4 Infrastructure existante réutilisée

- Plugins Claude Code (superpowers, OMC, oh-my-claude statusline)
- Sub-agents : `general-purpose`, `critic`, `validator`, `explore`, `planner`, `oh-my-claude:librarian`
- OPLoRA consolidation baseline (instance de B-Tononi dans C)
- `kiki-flow-core` source (forké en `kiki-oniric`)
- Grist + Keycloak (tracking tasks optionnel)
- autossh tunnels cross-machines

---

## 9. Relations avec les projets existants

| Projet | Relation | Impact |
|--------|----------|--------|
| `kiki-flow-research` (3 stories lancées 2026-04-16) | **Indépendant** — Q2.3 (c) : fork dédiée `kiki-oniric`, no blocking | None sur kiki-flow-research |
| `kiki-flow-core` | Source de la fork, rebase jalonné (r3) S1/S8/S18 | Rebase consomme ~1 semaine chaque fois |
| `micro-kiki` (triple-hybrid SNN+quantique+classique) | **Précurseur conceptuel** pour E cycle 2 | Acquis SNN réutilisés en cycle 2 |
| OPLoRA | Instance B-Tononi (downscale operation) dans framework C | Cité formellement dans framework §4.2 operations |
| mascarade | Potentielle consommatrice cycle 3+ | Hors scope cycle 1 |
| Factory 4 Life | Potentielle consommatrice cycle 3+ | Hors scope cycle 1 |
| L'Electron Rare launch mai 2026 | **Coïncide S4-S8** cycle 1 | Budget attention 15h/sem inclus |

---

## 10. Open questions & follow-ups

Questions non bloquantes pour la writing-plans phase, à traiter en S1-S3 :

1. **Byline vérification** : Nature HB guidelines autorisent-elles "project contributors" style ? (T-Col vérifie S2)
2. **Benchmark format standardisé** : continual learning community a-t-elle une convention pour ablation multi-profils à adopter ? (T-Ops review S2)
3. **Teacher scorer validation** : Qwen3.5-9B Q4_K_M gelé par SHA produit-il des scores RSA-alignables ? (A.5 test S3)
4. **fMRI lab préférentiel** : ordre de préférence à affiner avec critères (data compatibility, réponse reviewer potentielle, co-author willingness) (T-Col S3)
5. **OSF pre-registration template** : template à adapter à ML/cognitive hybrid plutôt que cognitive-only (S3)
6. **DualVer tooling** : script de bump semi-automatique dans T-Ops.1 (S2-S3)

---

## 11. Appendices (placeholder — renvoi au framework spec)

Les appendices détaillés (invariants formels, axiomes DR, testing plan, interfaces cross-track) sont dans le spec framework C : `2026-04-17-dreamofkiki-framework-C-design.md`.

---

**End of master design spec.**

Next step after user review : invoke `writing-plans` skill to generate implementation plan.
