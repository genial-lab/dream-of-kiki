# dreamOfkiki : pourquoi faire rêver une IA pour qu'elle se souvienne

*Un framework formel pour la consolidation mnésique basée sur le
rêve dans les systèmes cognitifs artificiels*

**Public visé** : chercheurs et techniciens curieux. On suppose
qu'un réseau de neurones, ça vous parle, mais on ne suppose ni EWC,
ni SHY, ni principe d'énergie libre.

---

## Le problème : quand l'IA oublie

Si vous entraînez un modèle de langue sur de l'arithmétique, puis
sur du français, le modèle finit par ne plus savoir compter. Si
vous l'entraînez ensuite sur du chinois, il oublie le français. Ce
phénomène porte un nom, il est vieux (McCloskey & Cohen, 1989), et
il n'est pas résolu : l'**oubli catastrophique**.

Les humains, eux, apprennent séquentiellement sans oublier
catastrophiquement. Vous n'avez pas besoin d'oublier l'arithmétique
pour apprendre le chinois. Ce que nous faisons, en revanche, c'est
**dormir**. Et pendant le sommeil, notre cerveau fait quelque
chose que les réseaux de neurones artificiels ne font pas : il
**consolide**, il **élague**, il **restructure**, il **rêve**.

L'idée centrale du projet dreamOfkiki — et de ce premier article
de notre série à deux papiers — est que ce processus nocturne
n'est pas du folklore psychologique. C'est un algorithme. Un
algorithme composable, axiomatisable, implémentable sur plusieurs
substrats. Et un algorithme qu'on peut écrire, prouver, tester.

## Les quatre piliers du sommeil cognitif

La neuroscience cognitive a identifié, depuis une vingtaine
d'années, quatre phénomènes qui se passent pendant notre sommeil
et qui semblent tous nécessaires au bon fonctionnement de la
mémoire. On les connaît par les noms de leurs grands défenseurs
théoriques :

**Pilier A — Walker & Stickgold : la réactivation.** Pendant le
sommeil lent (NREM), l'hippocampe rejoue les épisodes de la
journée. Les mêmes neurones qui ont été actifs quand vous avez
appris quelque chose s'allument à nouveau, à grande vitesse, en
boucle. Cette réactivation pousse les représentations à migrer
depuis le stockage à court terme (hippocampique) vers le stockage
à long terme (cortical). En IA, c'est ce qu'on appelle le
**replay** : rejouer des épisodes passés pendant qu'on apprend
les nouveaux.

**Pilier B — Tononi : SHY (Synaptic Homeostasis Hypothesis).**
Pendant la journée, on apprend, et apprendre potentialise les
synapses — on renforce des connexions. Mais on ne peut pas
renforcer indéfiniment : le cerveau saturerait. Le sommeil, selon
SHY, rétrograde globalement les synapses (**downscale**) pour
restaurer le rapport signal-sur-bruit tout en préservant la
structure différentielle des forts vs les faibles. En IA, ça
ressemble beaucoup à un régulariseur de type EWC, ou à un
rétrécissement multiplicatif des poids.

**Pilier C — Hobson & Solms : le rêve créatif.** Pendant le
sommeil paradoxal (REM), on rêve. On recombine. On génère des
scénarios impossibles, contrefactuels, bizarres. Hobson voyait ça
comme un bruit intrinsèque ; Solms plus récemment y voit un
processus génératif de sondage des frontières de ce qu'on a
appris. En IA : un VAE qui échantillonne des variations autour
des représentations récentes, ou qui interpole entre des épisodes
distants (**recombine**).

**Pilier D — Friston : le Principe d'Énergie Libre.** Le cerveau
minimise en permanence l'erreur de prédiction entre son modèle
génératif du monde et les entrées sensorielles. Le sommeil serait
une phase où, sans nouvelles entrées pour perturber le processus,
on peut **restructurer** le modèle lui-même — changer sa topologie,
élaguer des couches, en ajouter, rerouter la connectivité — pour
mieux minimiser l'énergie libre attendue. En IA : modifier
l'architecture du réseau, pas seulement ses poids.

## Pourquoi il manquait un framework unifié

L'état de l'art en IA a implémenté chacun de ces piliers
séparément — parfois brillamment. Van de Ven et collègues (2020)
ont fait du replay génératif inspiré du cerveau. Kirkpatrick et
collègues (2017) ont implémenté EWC, qui ressemble à SHY. Plusieurs
groupes travaillent sur le codage prédictif à la Friston. Mais
**personne n'a unifié les quatre dans un framework composable**.

Et ça n'est pas un détail. Dans notre analyse des paires
d'opérations (disponible dans les preuves techniques du dépôt), on
a énuméré les 16 paires possibles entre nos quatre opérations
canoniques (replay, downscale, restructure, recombine), et on a
trouvé que **12 d'entre elles sont non-commutatives** : faire
replay puis downscale ne donne pas le même résultat que faire
downscale puis replay. L'ordre compte. Et l'ordre canonique —
replay d'abord, puis downscale, puis restructure, avec recombine
en parallèle — n'est pas une convention arbitraire, c'est une
conséquence de la structure algébrique du problème.

## Ce qu'on propose : un framework formel exécutable

**dreamOfkiki**, le framework que nous proposons dans ce premier
article, est un cadre formel pour spécifier ces quatre piliers de
manière composable et indépendante du substrat. Ses ingrédients :

**8 primitives typées.** Quatre canaux d'entrée (α pour les traces
brutes, β pour un tampon épisodique curaté, γ pour un snapshot
des poids, δ pour des latents hiérarchiques) et quatre canaux de
sortie (delta de poids, échantillons latents, diff de hiérarchie,
attention prior). Chaque canal a un type précis, un invariant de
bornage, un contrat.

**4 opérations canoniques.** replay, downscale, restructure,
recombine — une par pilier théorique.

**3 profils en chaîne d'inclusion.** P_min (juste replay +
downscale — la version minimale qui suffit pour montrer un effet),
P_equ (profil "équivalent humain" avec restructure et recombine
allégés en plus), P_max (profil riche avec toutes les options
actives, incluant le flux α et l'attention prior). La règle est
que P_min ⊆ P_equ ⊆ P_max : toute opération ou canal disponible
dans un profil inférieur l'est aussi dans les profils supérieurs.

**5 axiomes exécutables, DR-0 à DR-4.**

- **DR-0 redevabilité** : chaque épisode onirique laisse une trace
  dans un journal, même en cas d'exception.
- **DR-1 conservation épisodique** : chaque élément du tampon β
  est consommé avant purge.
- **DR-2 compositionnalité** : les quatre opérations forment un
  semi-groupe libre non-commutatif avec un budget additif. On
  peut composer arbitrairement, mais le budget (FLOPs, temps,
  énergie) s'additionne.
- **DR-3 indépendance du substrat (Critère de Conformité)** : tout
  substrat — qu'il soit implémenté en MLX, en PyTorch, en SNN sur
  du neuromorphique Loihi-2 — qui satisfait trois conditions
  (typage des signatures, tests de propriété axiomatiques,
  invariants BLOCKING applicables) hérite automatiquement des
  garanties du framework.
- **DR-4 inclusion en chaîne des profils** : P_min ⊆ P_equ ⊆ P_max
  pour les opérations et pour les canaux.

Le mot "exécutable" est important. Ces axiomes ne sont pas de la
prose théorique : ils sont des propriétés testables, liées à des
tests de conformité sous `tests/conformance/` dans le dépôt.
Chaque opération du framework a une référence de garde (replay →
invariant S1, downscale → invariant S2, restructure → invariant
S3), et la règle qu'on s'impose est stricte : **pas d'opération
sans garde, pas de garde sans test**.

## La validation empirique, version cycle 1

Nous avons deux substrats en développement parallèle :

**kiki-oniric, substrat de référence MLX sur Apple Silicon.** C'est
notre implémentation primaire, écrite en Python 3.12 avec MLX
comme backend de calcul (accélérateur Apple natif). Elle est
complète au sens du cycle 1 : P_min et P_equ sont câblés,
conformes aux cinq axiomes, passent la suite complète de tests
invariants. P_max existe en squelette (pas encore câblé) et c'est
une cible cycle 2.

**E-SNN, substrat thalamocortical spiking.** C'est un deuxième
substrat, construit sur un modèle spiking neural network (SNN) à
neurones Leaky Integrate-and-Fire (LIF) dans une architecture
thalamocortique. Il sert de test cross-substrat pour DR-3 :
est-ce que les quatre opérations restent opérationnelles quand le
calcul n'est plus des mises à jour de gradient sur matrices
denses, mais des dynamiques de taux de spike ? La réponse au
cycle 1 est oui, partiellement — les quatre opérations passent la
conformité structurelle (DR-2 compositionnalité), et le framework
tient la route. Validation complète cycle 2.

Côté empirique, on a pré-enregistré quatre hypothèses sur l'Open
Science Framework (OSF) avant de lancer la moindre expérience :

- **H1** : le profil P_equ réduit l'oubli par rapport à un baseline
  sans consolidation.
- **H2** : le profil P_max est équivalent à P_equ (dans ±5 %) sur
  un banc de test linguistique.
- **H3** : la précision croît monotone le long de la chaîne P_min
  < P_equ < P_max.
- **H4** : le budget énergétique des épisodes oniriques reste sous
  2× le budget d'éveil.

Les quatre tests sont faits sous correction de Bonferroni
(α = 0,0125). Au cycle 1, nous reportons les résultats de validation
du **pipeline** — c'est-à-dire, le pipeline de mesure et de
statistique a été exercé de bout en bout sur un jeu de données
synthétique, avec des prédicteurs mock aux niveaux de précision
scriptés (50 % / 70 % / 85 %). Trois hypothèses sur quatre passent
le seuil de Bonferroni, H3 est à la limite. Les vrais chiffres
empiriques, sur le banc mega-v2 réel (498 k exemples, 25 domaines
linguistiques) avec des prédicteurs issus d'une inférence MLX
authentique, arrivent en clôture du cycle 1 et seront rapportés
dans le Paper 2.

## Ce qui arrive ensuite

Le cycle 2 a deux axes :

**Validation empirique complète**, avec le banc mega-v2 réel et
des prédicteurs issus d'une vraie inférence neuronale. Les
chiffres remplaceront les placeholders synthétiques. Selon le
résultat, l'hypothèse H3 — le signal monotone — sera confirmée ou
infirmée sur données réelles.

**Validation multi-substrat** du Critère de Conformité. Le
substrat E-SNN doit passer les trois conditions de conformité
(typage, tests de propriété, invariants BLOCKING) sur une exécution
complète d'ablation. Si ça marche, on aura la première
démonstration empirique qu'un framework formel de consolidation
mnésique basée sur le rêve peut être instancié sur deux
technologies radicalement différentes (MLX dense + SNN sparse)
tout en maintenant les mêmes garanties théoriques.

## Pourquoi on publie en science ouverte

Tout ce que nous produisons est ouvert :

- **Le code** (framework + substrat kiki-oniric + substrat E-SNN) est
  sous licence MIT.
- **La documentation** (spécifications, preuves, invariants,
  glossaire) est sous CC-BY-4.0.
- **Les hypothèses** sont pré-enregistrées sur OSF avec un DOI
  horodaté immuable.
- **Les artefacts de données** (bancs de test, modèles entraînés)
  sont épinglés via DOI Zenodo avec hash SHA-256 vérifiables.
- **Les deux papiers** (Paper 1 formel, Paper 2 ablation) seront
  déposés sur arXiv d'abord, soumis à Nature Human Behaviour en
  primaire, avec PLoS Computational Biology et Cognitive Science
  comme revues de repli.

L'objectif n'est pas juste de publier un résultat. C'est de
mettre sur la table un framework reproductible, auditable,
étendable — que d'autres équipes de recherche pourront instancier
sur leur substrat de prédilection (un transformer, un modèle
bayésien, un SNN, un système symbolique). Le Critère de Conformité
DR-3 est conçu exactement pour ça.

## À retenir en trois phrases

1. L'oubli catastrophique en IA n'est pas un bug d'ingénierie,
   c'est une conséquence structurelle de ne pas avoir d'analogue
   du sommeil — et le sommeil, en biologie, c'est quatre
   mécanismes qui composent.
2. Nous proposons dreamOfkiki, un framework formel à cinq axiomes
   exécutables, huit primitives typées, quatre opérations
   canoniques, qui unifie ces quatre piliers en un semi-groupe
   libre composable, indépendant du substrat.
3. Le cycle 1 valide le pipeline sur substrat MLX ; le cycle 2
   valide l'ablation réelle et la conformité multi-substrat
   (E-SNN). Tout est ouvert, pré-enregistré, reproductible.

---

*Pour plus de détails techniques, voir l'article Paper 1 sur
arXiv (lien à venir), le dépôt public du code, et la spécification
formelle du framework C-v0.5.0+STABLE dans
`docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`.*

*Feedback, issues, pull requests, collaborations bienvenues.*

*— Clément Saillant, L'Electron Rare, avril 2026*
