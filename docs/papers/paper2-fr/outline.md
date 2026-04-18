# Article 2 — Structure (cycle 2 amorçage S27.1)

**Revue cible** : NeurIPS (primaire) / ICML / TMLR (repli)
**Format** : 9 pages principales + supplémentaire illimité
**Cible en mots** : ~5500 mots principaux

---

## 1. Résumé (cible 200 mots)

Contribution d'ingénierie : framework de consolidation basée sur
le rêve indépendant du substrat, avec Critère de Conformité
exécutable, démontré sur les substrats MLX (Apple Silicon) et
E-SNN (Loihi-2 thalamocortical). Contributions d'ingénierie
clés : (1) opérations MLX-natives avec reproductibilité
déterministe, (2) protocole de swap avec gardes d'invariants
S1-S3, (3) méthodologie d'ablation pré-enregistrée OSF, (4)
matrice de validation inter-substrats du Critère de Conformité.
Résultats empiriques : ablation mega-v2 réelle (données de
clôture cycle-1) + réplication E-SNN (données fraîches cycle-2)
montrant des effets de chaîne de profils cohérents.

## 2. Introduction (~1 page)

- Problème d'ingénierie : consolidation IA reproductible à
  travers des substrats matériels
- Motivation pratique : permettre aux chercheurs de valider les
  revendications du framework sur leurs propres substrats sans
  re-dériver la théorie
- Feuille de route des contributions (4 items numérotés)

## 3. Contexte (bref, ~0,5 pages)

- Référence à l'Article 1 pour la fondation théorique
- Citer SHY (Tononi), FEP (Friston), CLS (McClelland), replay
  inspiré du cerveau (van de Ven)
- Établir que cet article se concentre sur la *réalisation
  d'ingénierie*, pas sur la théorie

## 4. Le Critère de Conformité en pratique (~1,5 pages)

- 4.1 Rappel des trois conditions (typage de signature, tests
  de propriétés d'axiomes, invariants BLOCKING applicables)
- 4.2 Vérification de conformité du substrat MLX (kiki-oniric)
- 4.3 Vérification de conformité du substrat E-SNN (cycle 2)
- 4.4 Suite de tests de conformité (`tests/conformance/`)
  comme artefact réutilisable

## 5. Architecture d'ingénierie (~1,5 pages)

- 5.1 Opérations comme handlers composables (replay,
  downscale, restructure, recombine — variantes MLX +
  squelette pour tests)
- 5.2 Protocole de swap avec gating par évaluation retenue S1,
  garde de finitude S2, garde de topologie S3
- 5.3 Worker de rêve concurrent (squelette Future-API en
  cycle 1, asyncio réel en cycle 2)
- 5.4 Registre de runs avec contrat déterministe R1
  (préfixe SHA-256 32-hex)

## 6. Méthodologie (~1 page)

- 6.1 Pré-enregistrement OSF (hypothèses cycle 1 H1-H4
  réutilisées)
- 6.2 Pipeline statistique (Welch / TOST / Jonckheere /
  t une-échantillon avec Bonferroni)
- 6.3 Banc de test retenu stratifié mega-v2 (500 items,
  SHA-256 gelé)
- 6.4 Protocole de mesure inter-substrats

## 7. Résultats (~2 pages)

- 7.1 Ablation substrat MLX (données réelles de clôture
  d'ablation cycle 1, remplaçant les placeholders synthétiques)
- 7.2 Ablation substrat E-SNN (données fraîches cycle 2)
- 7.3 Comparaison inter-substrats (cohérence des effets de
  chaîne de profils)
- 7.4 Significativité statistique + résultats H1-H4 corrigés
  Bonferroni

## 8. Discussion (~1 page)

- 8.1 Reproductibilité validée à travers les substrats
- 8.2 Arbitrages d'ingénierie (vitesse MLX vs énergie E-SNN)
- 8.3 Limitations (seulement 2 substrats ; transformer ou
  autres architectures en attente cycle 3)
- 8.4 Comparaison avec les revendications théoriques de
  l'Article 1

## 9. Travaux futurs (~0,5 pages)

- 9.1 Substrats additionnels (transformer, RWKV, state-space)
- 9.2 Sélection dynamique de profil au runtime
- 9.3 Patterns de déploiement en production

## 10. Références

Réutiliser la plupart des références de l'Article 1 (references.bib)
+ ajouter les citations d'ingénierie (MLX, scipy, Loihi-2, patterns
asyncio).

---

## Différenciation par rapport à l'Article 1

| Aspect | Article 1 | Article 2 |
|--------|-----------|-----------|
| Portée | Framework théorique | Implémentation d'ingénierie |
| Cible | Nature HB / PLoS CB | NeurIPS / ICML / TMLR |
| Public | Cogniticiens + théoriciens | Ingénieurs ML + chercheurs systèmes |
| Substrats couverts | 1 (kiki-oniric MLX) | 2+ (MLX + E-SNN) |
| Longueur | ~5000 mots | ~5500 mots |
| Preuves formelles | DR-0..DR-4 dans le corps principal | Référence à l'Article 1 |
| Focus reproductibilité | Pré-enregistrement + contrat R1 | Matrice de validation inter-substrats |
| Emphase open-source | Conceptuelle | Opérationnelle (exécuter sur votre substrat) |

---

## Dépendances cycle 2

- Substrat E-SNN (accès Loihi-2 via partenariat Intel NRC —
  à poursuivre après décision G6)
- Accès au dataset mega-v2 réel (clôture cycle 1 S20+ si pas
  fait d'ici là)
- Partenariat réel avec un labo IRMf (extension T-Col
  cycle 2)
- Acceptation / citation de préprint Article 1 pour la
  référence croisée

---

## Calendrier cycle 2 (estimation approximative)

- S29-S32 : câblage substrat E-SNN + vérification de
  conformité
- S33-S36 : runs d'ablation E-SNN + comparaison
  inter-substrats
- S37-S40 : sections brouillon Article 2
- S41-S42 : relecture pré-soumission
- S43-S44 : fenêtre de soumission NeurIPS
- S45+ : rounds de reviewers + révisions

---

## Notes

- Ce outline est amorçage uniquement ; le brouillon complet
  commence après la clôture cycle 1 (S28+) une fois
  l'Article 1 soumis
- Outline sujet à révision selon les retours reviewers de
  l'Article 1 et la décision de portée cycle-2 G6
- La portée cycle 2 pourrait exclure E-SNN si l'accès Loihi-2
  n'est pas accordé ; dans ce cas, repli sur MLX + E-SNN
  basé simulation comme second substrat avec caveats clairs
