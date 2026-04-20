<!--
SPDX-License-Identifier: CC-BY-4.0
Authorship byline : Saillant, Clément
License : Creative Commons Attribution 4.0 International (CC-BY-4.0)
-->

# §7 Résultats (Article 2, brouillon C2.15)

**Signataires** : *Saillant, Clément*
**Licence** : CC-BY-4.0

**Cible de longueur** : ~2 pages markdown (≈ 1700 mots)

---

> ⚠️ **Précaution (substitution synthétique — pas de
> revendication empirique).**  Chaque revendication
> quantitative dans cette section est produite par un
> prédicteur mock Python partagé, canalisé à travers deux
> enregistrements de substrats (MLX + squelette numpy LIF
> E-SNN). L'accord inter-substrats est trivialement OUI
> **par construction** ; les valeurs valident le pipeline
> de réplication inter-substrats, pas l'efficacité
> empirique du framework sur des données réelles. Le vrai
> câblage vers l'inférence spécifique au substrat est un
> livrable du cycle 3. Source :
> `docs/milestones/cross-substrate-results.md` (commit
> `fa0f26e`), régénérée par
> `scripts/ablation_cycle2.py`.

---

## 7.1 Provenance (substitution synthétique — pas de revendication empirique)

La matrice inter-substrats cycle-2 est enregistrée sous les
identifiants suivants (verbatim de
`docs/milestones/cross-substrate-results.md`) :

| champ | valeur |
|-------|--------|
| harness_version | `C-v0.6.0+PARTIAL` |
| cycle2_batch_id | `3a94254190224ca82c70586e1f00d845` |
| ablation_runner_run_id | `45eccc12953e758440fca182244ddba2` |
| benchmark | mega-v2 stratifié, 500 items synthétiques |
| benchmark_hash | `synthetic:c8a0712000b641...` |
| seeds | `[42, 123, 7]` |
| substrats | `mlx_kiki_oniric`, `esnn_thalamocortical` |
| data_provenance | synthétique — prédicteurs mock partagés entre étiquettes de substrats |

Reproductibilité : exécuter
`uv run python scripts/ablation_cycle2.py` depuis
`C-v0.6.0+PARTIAL` doit régénérer un JSON identique
(byte-stable) et un Markdown identique (modulo la
sérialisation déterministe) sous la grille de seeds
fixée.

## 7.2 Table comparative inter-substrats H1-H4 (substitution synthétique — pas de revendication empirique)

**Table 7.2 — MLX vs E-SNN hypothèses à Bonferroni
α = 0,0125 (substitution synthétique — pas de
revendication empirique).**

| hypothèse | test | p-value MLX | verdict MLX | p-value E-SNN | verdict E-SNN | accord |
|-----------|------|-------------|-------------|---------------|---------------|--------|
| H1 oubli | Welch unilatéral | 0,0000 | rejet H0 | 0,0000 | rejet H0 | OUI |
| H2 auto-équivalence | TOST (ε = 0,05) | 0,0000 | rejet H0 | 0,0000 | rejet H0 | OUI |
| H3 monotonicité | Jonckheere-Terpstra | 0,0248 | échec à rejeter | 0,0248 | échec à rejeter | OUI |
| H4 budget énergie | t un-échantillon (sup) | 0,0101 | rejet H0 | 0,0101 | rejet H0 | OUI |

- Nombre significatif MLX : **3 / 4**.
- Nombre significatif E-SNN : **3 / 4**.
- **Entièrement cohérent entre substrats** : **OUI**
  *(substitution synthétique — trivialement OUI par
  construction à prédicteur partagé)*.

### Guide de lecture de légende (substitution synthétique — pas de revendication empirique)

- H3 échoue à α = 0,0125 sur les deux substrats car le
  prédicteur mock émet une accuracy constante par profil
  (pas de dispersion par seed). La statistique de tendance
  Jonckheere-Terpstra ne peut pas franchir le seuil de
  Bonferroni sur un échantillon dégénéré. C'est une
  **propriété du prédicteur mock**, pas du framework.
- H2 passant comme `rejet H0` est une **passe de fumée
  d'auto-équivalence** héritée de l'Article 1 §7.6 ; le
  fait que P_max soit câblé en cycle 2 ne change pas le
  résultat à prédicteur partagé car le prédicteur ne
  distingue pas P_equ de P_max sous le mock.
- H1 et H4 rejetant H0 est la conception scriptée du
  prédicteur mock : le mock émet une accuracy plus grande
  sous P_equ que sous la baseline (donc
  `forgetting_P_equ < forgetting_baseline` tient) et émet
  un ratio d'énergie sous-seuil (donc le t un-échantillon
  rejette à la hausse).

## 7.3 Matrice d'accord (substitution synthétique — pas de revendication empirique)

**Table 7.3 — Accord de verdicts par hypothèse entre
substrats (substitution synthétique — pas de
revendication empirique).**

| hypothèse | verdicts égaux ? | rejet MLX | rejet E-SNN |
|-----------|------------------|-----------|-------------|
| H1 oubli | OUI | true | true |
| H2 auto-équivalence | OUI | true | true |
| H3 monotonicité | OUI | false | false |
| H4 budget énergie | OUI | true | true |

Les quatre hypothèses s'accordent entre substrats
(4 / 4 d'accord). *(substitution synthétique.)*  C'est le
résultat attendu quand le prédicteur sous-jacent est
identique entre les lignes de substrats — cela montre que
le chemin stats par substrat est correctement câblé et
émet des verdicts identiques sur des entrées identiques.
Une réplication à prédicteur divergent est la cible de
câblage réel cycle-3.

## 7.4 Matrice de Conformité DR-3 (référence croisée vers §4)

**Table 7.4 — 3 conditions × 2 substrats réels + 1
placeholder.**

| substrat | C1 typage signature | C2 tests propriétés axiomes | C3 invariants BLOCKING |
|----------|--------------------|-----------------------------|------------------------|
| `mlx_kiki_oniric` | PASS | PASS | PASS |
| `esnn_thalamocortical` | PASS *(substitution synthétique)* | PASS *(substitution synthétique — pas de matériel Loihi-2)* | PASS *(substitution synthétique — numpy LIF taux de spikes)* |
| `hypothetical_cycle3` | N/A | N/A | N/A |

Légende : matrice de Conformité DR-3 (source :
`docs/milestones/conformance-matrix.md`, commit `fd54df7`).
La ligne E-SNN porte le drapeau de substitution
synthétique selon §4.3. La ligne `hypothetical_cycle3`
est explicitement placeholder-seulement et ne doit pas
être lue comme passant / échouant. Cette matrice est la
preuve *architecturale* pour DR-3 ; la table H1-H4 du §7.2
est la preuve du *pipeline de réplication*. Les deux sont
nécessaires pour la revendication composite de
l'Article 2.

## 7.5 Ce que le §7 établit et ne établit pas

**Ce que le §7 établit :**

- Le pipeline de réplication inter-substrats s'exécute de
  bout en bout sur deux enregistrements de substrats
  structurellement distincts (§7.1 provenance + §7.2
  parité de p-values).
- Le chemin statistique par substrat émet des verdicts
  identiques sur des entrées identiques (§7.3 matrice
  d'accord).
- La matrice de Conformité DR-3 exhibe deux lignes de
  substrats PASSant (§7.4).
- Le contrat de reproductibilité R1 tient : chaque run_id
  au §7.1 circule aller-retour vers un artefact du dépôt.

**Ce que le §7 n'établit PAS :**

- Aucune **revendication de performance biologique ou
  neuromorphique** sur la dynamique E-SNN. Le substrat
  E-SNN est un squelette numpy LIF, pas Loihi-2.
- Aucune **revendication d'efficacité empirique de
  consolidation** sur des données linguistiques ou
  sensorielles réelles. Le prédicteur est un mock Python
  partagé ; le vrai câblage mega-v2 ou IRMf est repoussé.
- Aucune **réplication à prédicteur divergent**. L'accord
  inter-substrats est trivialement OUI par construction
  à prédicteur partagé.
- Aucun **ratio matériel d'énergie ou de latence**. Le
  `ratio d'énergie` de H4 est une valeur mock scriptée,
  pas une trace mesurée en temps d'horloge.

## 7.6 Résumé de gate (substitution synthétique — pas de revendication empirique)

Sur les 4 hypothèses pré-enregistrées (OSF, héritées de
l'Article 1), les verdicts de gate par substrat sont :

- **H1 oubli** : PASS *(substitution synthétique)*,
  rejetée sur les deux substrats.
- **H2 auto-équivalence** : PASS *(substitution
  synthétique, fumée d'auto-équivalence)*, rejetée sur les
  deux substrats.
- **H3 monotonicité** : ÉCHEC à α = 0,0125, PASS à
  α = 0,05 *(substitution synthétique, limitée par la
  dispersion du mock)*, échec à rejeter sur les deux
  substrats.
- **H4 budget énergie** : PASS *(substitution
  synthétique)*, rejetée sur les deux substrats.

**Verdict agrégé cycle-2 G9** : **CONDITIONAL-GO /
PARTIAL** selon
`docs/milestones/g9-cycle2-publication.md`. La fondation
d'ingénierie (conformité DR-3 E-SNN G7, P_max fonctionnel
G8, worker asynchrone C2.17) est complète ; le pipeline
de réplication inter-substrats est câblé ; la narration
Article 2 est brouillonnée ; mais la force empirique de la
revendication attend une réplication à prédicteur
divergent du cycle-3.

---

## Notes pour révision

- Ajouter Figure 7.1 : boîtes à moustaches côte à côte
  d'accuracy par profil sur MLX vs E-SNN (montrera des
  distributions identiques sous prédicteur partagé,
  confirmant la précaution synthétique visuellement).
- Ajouter Figure 7.2 : matrice de conformité en heat-map
  (cellules PASS / N/A).
- Référencer §6.4 prédicteur par substitution synthétique
  et §8.3 limitations une fois §8 livrée.
- Resserrer à ≤ 1500 mots à la passe de pré-soumission
  NeurIPS.
- Chaque légende de table doit conserver au moins un
  drapeau `(substitution synthétique)` ; ne pas supprimer
  lors des passes de resserrement.
