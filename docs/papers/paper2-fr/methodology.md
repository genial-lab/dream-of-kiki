<!--
SPDX-License-Identifier: CC-BY-4.0
Authorship byline : Saillant, Clément
License : Creative Commons Attribution 4.0 International (CC-BY-4.0)
-->

# §6 Méthodologie (Article 2, brouillon C2.15)

**Signataires** : *Saillant, Clément*
**Licence** : CC-BY-4.0

**Cible de longueur** : ~1 page markdown (≈ 900 mots)

---

## 6.1 Hypothèses pré-enregistrées (OSF, héritées de l'Article 1)

Les quatre hypothèses pré-enregistrées sont **héritées de
l'Article 1** — le cycle 2 n'enregistre *pas* de nouvelles
hypothèses, car l'Article 2 est un artefact de réplication,
pas une nouvelle revendication empirique. Le verrou OSF
(DOI en attente selon Article 1 §6.1) s'applique
verbatim.

- **H1 — Réduction de l'oubli.**
  `mean(forgetting_P_equ) < mean(forgetting_baseline)`.
  Test : test t de Welch, unilatéral.
- **H2 — Équivalence P_max.**
  `|mean(acc_P_max) - mean(acc_P_equ)| < 0.05`.
  Test : deux tests t unilatéraux (TOST), ε = 0,05.
- **H3 — Alignement monotone.**
  `mean(acc_P_min) < mean(acc_P_equ) < mean(acc_P_max)`.
  Test : Jonckheere-Terpstra.
- **H4 — Budget énergétique.**
  `mean(energy_dream / energy_awake) < 2.0`.
  Test : test t à un échantillon.

Note de portée cycle-2 : avec P_max entièrement câblé (G8,
`docs/milestones/g8-p-max-functional.md`), H2 et H3
admettent désormais une véritable structure à trois
groupes — mais l'évaluation numérique est toujours par
substitution synthétique car le prédicteur est partagé
entre les étiquettes de substrats (§6.4).

## 6.2 Pipeline statistique + α de Bonferroni

Les quatre tests s'exécutent sous un **seuil de
significativité corrigé par Bonferroni** : `α_par_hypothèse
= 0,05 / 4 = 0,0125`. L'implémentation est le module
`kiki_oniric.eval.statistics` du cycle 1 (inchangé en
cycle 2 — intentionnellement, pour que les p-values du
cycle 2 soient numériquement comparables à celles du
cycle 1 quand le même prédicteur est utilisé) :

- **`welch_one_sided`** (H1) : `scipy.stats.ttest_ind` avec
  `equal_var=False`, p-value divisée par 2 pour
  interprétation unilatérale.
- **`tost_equivalence`** (H2) : deux tests t unilatéraux
  manuels (borne basse `diff ≤ −ε`, borne haute
  `diff ≥ +ε`). Rejette H0 quand les deux passent à α
  (règle TOST max-p).
- **`jonckheere_trend`** (H3) : somme des counts par paires
  de Mann-Whitney U sur les groupes ordonnés, approximation
  z pour la p-value.
- **`one_sample_threshold`** (H4) : `scipy.stats.ttest_1samp`
  contre `popmean = threshold`, ajustement unilatéral pour
  échantillon sous seuil.

Les quatre retournent un `StatTestResult(test_name, p_value,
reject_h0, statistic)` uniforme pour la gestion aval.

## 6.3 Matrice d'ablation : 2 substrats × 3 profils × 3 seeds = 18 cellules

La dimension de la matrice de l'Article 2 est nouvelle :
l'Article 1 exécutait 3 profils × 3 seeds = 9 cellules.
L'Article 2 croise les substrats, donnant **2 × 3 × 3 = 18
cellules**. Chaque cellule produit :

- accuracy (pour H1, H2, H3)
- forgetting := 1 − accuracy (pour H1)
- ratio d'énergie := energy_dream / energy_awake (pour H4)

Les cellules sont générées par
`scripts/ablation_cycle2.py` ; les dumps Markdown + JSON
atterrissent dans `docs/milestones/cross-substrate-
results.{md,json}`.

Grille de seeds : `[42, 123, 7]` — identique au cycle 1,
donc la ligne MLX cycle-2 est bit-comparable à l'Article 1
§7.2 quand le prédicteur est inchangé.

## 6.4 Prédicteur par substitution synthétique (la précaution critique)

**(substitution synthétique — pas de revendication
empirique.)**  Cette sous-section est porteuse pour lire le
§7 correctement.

Les deux lignes de substrats dans la matrice 2 × 3 × 3
partagent le même prédicteur mock Python. La ligne du
substrat MLX et la ligne du substrat E-SNN passent leurs
représentations d'état (arrays MLX vs LIFState) à travers
les 4 factories d'op et émettent les canaux de sortie
attendus — mais quand `evaluate_retained` interroge pour
une prédiction par item, la même fonction mock retourne la
même étiquette quel que soit le substrat ayant produit
l'état.

Trois conséquences suivent :

1. Les **p-values par substrat sont bit-identiques** dans
   la limite d'un prédicteur partagé parfait, et très
   proches en pratique (voir table §7). Cet accord n'est
   **pas** une preuve de performance indépendante du
   substrat sur des données réelles ; c'est l'*absence*
   d'un signal spécifique au substrat dans le prédicteur.
2. Le **verdict d'accord inter-substrats** (4 / 4 d'accord)
   est trivialement OUI par construction.
3. La **revendication architecturale** — que le pipeline
   (runner → stats → dump) s'exécute de bout en bout sur
   deux enregistrements de substrats distincts sans
   duplication de code — est validée. C'est la moitié de
   la Conformité DR-3 que l'Article 2 gagne
   empiriquement.

Une réplication à prédicteur divergent (inférence
spécifique au substrat : forward pass MLX sur état MLX,
read-out LIFState sur LIFState) est la **cible cycle-3**.
L'Article 2 ne fait pas cette revendication.

## 6.5 Reproductibilité : contrat R1 + intégrité du benchmark

La matrice cycle-2 est reproductible par les deux mêmes
contrats que le cycle 1 :

- **R1 (run_id déterministe)** — chaque cellule résout vers
  un préfixe SHA-256 32-hex de `(c_version, profile, seed,
  commit_sha)`. Le wrapper de batch cycle-2 et le runner
  d'ablation émettent tous deux des run_ids enregistrés
  dans le dump JSON :
  - `cycle2_batch_id` : `3a94254190224ca82c70586e1f00d845`
  - `ablation_runner_run_id` :
    `45eccc12953e758440fca182244ddba2`
  - `harness_version` : `C-v0.6.0+PARTIAL`
- **R3 (intégrité du benchmark)** — le benchmark retenu
  stratifié mega-v2 500-items expédie un fichier
  `.sha256` ; le loader rejette tout fichier d'items dont
  le hash ne correspond pas à la référence gelée (en levant
  `RetainedIntegrityError`). Le cycle-2 utilise le
  fallback synthétique (flag
  `synthetic:c8a0712000b641...`) hérité du cycle 1, en
  attendant l'ablation mega-v2 réelle dans un cycle
  ultérieur.

Le tag DualVer `C-v0.6.0+PARTIAL` est attaché à chaque
artefact cycle-2. Les revendications empiriques sont
**valides seulement contre leur `c_version` déclarée** ;
bumper l'axe formel MAJEUR invaliderait les artefacts de
l'axe empirique et exigerait une ré-exécution.

## 6.6 Ce qui a changé vs méthodologie Article 1

L'Article 2 est délibérément minimal sur les ajouts de
méthodologie :

- **Mêmes hypothèses** (H1-H4, verrouillées OSF).
- **Même module statistique** (pas de nouveaux tests, pas de
  nouvelles corrections).
- **Même grille de seeds** (`[42, 123, 7]`).
- **Même α de Bonferroni** (0,0125).
- **Même classe de prédicteur** (mock partagé, substitution
  synthétique).

Ce qui est **nouveau** dans l'Article 2 :

- Dimension substrat dans la matrice (2 lignes).
- Verdict d'accord inter-substrats (agrégat sur H1-H4).
- Artefact de matrice de conformité câblé au runner
  (`scripts/conformance_matrix.py`).

Cette discipline — changer une dimension à la fois — est
comment un article de réplication préserve
l'interprétabilité de ses comparaisons. Le cycle-3 ajoutera
la dimension prédicteur ; le cycle-2 se restreint à la
dimension substrat sous un prédicteur fixe.

---

## Notes pour révision

- Lier le DOI OSF en ligne au §6.1 une fois verrouillé.
- Envisager une figure supplémentaire Méthodes : flux de
  données (benchmark → factory d'op de substrat → état →
  prédicteur → evaluate_retained → stats → dump).
- Resserrer à ≤ 800 mots à la passe de pré-soumission
  NeurIPS.
