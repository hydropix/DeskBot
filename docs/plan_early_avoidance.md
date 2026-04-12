# Plan de recherche — Early avoidance (loi de guidage inverse-distance)

**Pour** : session Claude séparée (prompt autonome)
**Auteur** : Bruno + Claude (session 10, 2026-04-12)
**Durée estimée** : 3-5 h de travail itératif

---

## Objectif

Remplacer le comportement bang-bang de Bug2 (ligne droite pleine vitesse
jusqu'à `SAFE_DIST = 0.36 m`, puis pivot + wall-follow) par un **biais de
lacet continu proportionnel à l'inverse de la distance frontale** appliqué
en GO_HEADING, dès qu'un obstacle apparaît dans la grille à longue
portée. L'idée centrale (cf. discussion préparatoire session 10) est :

$$
\omega_{\text{bias}} = k \cdot \text{side} \cdot \frac{1}{\max(D, D_{\text{sat}})}
\quad \text{si } D_{\min} < D < D_{\text{cut}}, \text{ sinon } 0
$$

À 2 m d'un obstacle, un biais de 2-4°/s est imperceptible pour le LQR
mais suffit à incurver la trajectoire en arc large (rayon ~5-10 m) qui
arrive à côté de l'obstacle *sans jamais toucher* le seuil SAFE_DIST.
En même temps, la rotation continue balaye le rayon FC latéralement à
travers les cellules inconnues entre FC et FL/FR, **remplissant la
grille proactivement** au lieu d'attendre un virage tardif. Double
bénéfice : navigation plus fluide (moins de CONTOUR) **et** mapping
plus dense (moins de gaps entre capteurs).

**Cible chiffrée** : battre la config actuelle Bug2+A* sur
`scripts/benchmark_random.py` (seed 42, 100 épisodes × 3 runs) avec :

- Succès global ≥ **89 %** (actuel : 89.0 %, cf. `docs/benchmark_astar_v1.txt`)
- 0 chute
- Pas de régression seed-à-seed > 1 épisode sur les chicanes
- IoU mapping ≥ baseline − 1 pt
- **Recall mapping (nouveau métric)** ≥ baseline +2 pts, mesure
  justifiant la valeur ajoutée côté cartographie

Si une seule condition régresse, early avoidance reste derrière un flag
et Bug2+A\* reste le défaut.

---

## Contexte à lire avant de coder

Dans l'ordre :

1. [CLAUDE.md](../CLAUDE.md) — règles de projet (sensor barrier, rigueur
   avant heuristique, pedagogical mode obligatoire).
2. [docs/plan_astar_local.md](plan_astar_local.md) — **modèle de
   référence** pour la forme de ce plan et la discipline d'exécution.
3. [docs/benchmark_astar_v1.txt](benchmark_astar_v1.txt) — baseline
   chiffrée à battre.
4. [docs/teb_evaluation.md](teb_evaluation.md) §6.2 — les 5 conditions
   strictes à respecter pour tout nouveau comportement de commande.
5. [deskbot/navigation.py](../deskbot/navigation.py) — spécifiquement :
   - `_state_go_heading()` → c'est là qu'on ajoute le biais
   - `_contour_side_from_virtual_scan()` (session 9) → c'est le
     sélecteur de `side` à réutiliser, déjà validé
   - `_enter_contour()` → **à ne pas toucher**, reste le filet de
     sécurité à `SAFE_DIST`
6. [deskbot/astar_local.py](../deskbot/astar_local.py) — A\* reste
   disponible comme alternative de sélection de `side` ; discuter en
   étape 4 si on le branche ou non.
7. [deskbot/mapping_eval.py](../deskbot/mapping_eval.py) +
   [scripts/eval_mapping.py](../scripts/eval_mapping.py) — pour le gate
   mapping étape 7.

---

## État de la base (ne pas le redécouvrir à tâtons)

- **Bug2 v2 + A\* (défaut session 9)** : 89 % sur seed 42, 100 ép × 3
  runs, 0 chute, σ 2.6 pts. Toute modification doit au minimum tenir
  ce chiffre.
- **Mapping IoU** = 0.380 (A\*) / 0.384 (Bug2) sur 15 eps seed 42, IoU
  essentiellement saturé par la couverture capteur, pas par le planner.
- **Modes d'échec résiduels (session 7 post-mortem + session 9)** :
  - ~11 % d'échecs restants sur les obstacles de type "wall with gap"
    quand le trou sort simultanément du champ FC et FL2/FR2.
  - Des chicanes serrées où Bug2 pivote tard et le CONTOUR dépasse la
    ligne avant de trouver le second obstacle.
  Les deux cas seraient théoriquement mitigés par un **virage léger
  anticipé** qui, cumulé sur ~2 m d'approche, produit ~15-20° de
  déviation — plus que suffisant pour éviter un obstacle décalé de
  30-40 cm.
- **SAFE_DIST et _enter_contour restent inchangés**. Early avoidance
  *précède* Bug2, elle ne le remplace pas. Si le biais ne suffit pas
  à éviter l'obstacle, `_min_front_dist < SAFE_DIST` déclenche CONTOUR
  comme avant — filet de sécurité garanti.

---

## Contraintes non négociables

1. **Sensor barrier préservée**. Early avoidance lit `Navigator.grid`
   (log-odds), `self._rf_compensated` (distances filtrées par
   GroundGeometry) et l'état FSM courant. Jamais `data.qpos` etc.
2. **Bug2 reste la couche stratégique intacte**. Early avoidance est
   *uniquement* un biais ω additif dans `_state_go_heading`. FSM,
   watchdogs, `_enter_contour`, `_state_contour`, `_check_stuck` :
   byte-identiques à la session 9. Si le biais est buggé, désactiver le
   flag doit restaurer exactement Bug2+A\*.
3. **Flag d'activation**. Option `--early-avoid {off,on}` sur
   `benchmark_random.py` et `eval_mapping.py`. Défaut `off`. Early
   avoidance ne devient défaut que si le gate §6 est strictement passé.
4. **Aucune rustine sur le filtrage pitch, les log-odds, le mapping,
   ou le LQR**. Si early avoidance se comporte mal, corriger *la
   formulation de la loi de guidage*, pas empiler des couches. Leçon
   explicite session 8.
5. **Zéro dépendance nouvelle**. numpy + math stdlib suffisent.
6. **Rigueur avant heuristique**. Chaque constante
   (`k`, `D_sat`, `D_cut`, `T_hold`, `T_clear`) a un docstring
   explicitant son unité, sa valeur initiale et la raison physique de
   cette valeur. Pas de tuning silencieux.
7. **Pas de re-entry à GO_HEADING pendant CONTOUR**. Le biais ω doit
   être strictement nul dans les états CONTOUR, REVERSE et IDLE. On ne
   touche qu'à GO_HEADING.

---

## Formulation mathématique (à figer avant coding)

### Variables

- $D$ : distance frontale à l'obstacle le plus proche confirmé par la
  grille, en mètres. Utilise `self._min_front_dist` déjà calculé.
- $\text{side} \in \{-1, +1\}$ : direction de contournement choisie
  par le sélecteur grille (voir §Step 2).
- $\omega_{\text{bias}}$ : yaw rate additionnel, rad/s, à ajouter au
  `yaw` produit par le contrôleur de cap P actuel.
- $\omega_{\text{go\_heading}}$ : le yaw nominal de GO_HEADING,
  `HEADING_P_GAIN * h_err`.

### Loi proposée (saturated inverse-distance)

$$
\omega_{\text{bias}}(D) =
\begin{cases}
0 & \text{si } D \ge D_{\text{cut}} \\
k \cdot \text{side} \cdot \dfrac{1}{\max(D, D_{\text{sat}})} & \text{si } D_{\text{trig}} < D < D_{\text{cut}} \\
0 & \text{si } D \le D_{\text{trig}}
\end{cases}
$$

avec :

- $D_{\text{cut}} = 2.0$ m — au-delà on considère que l'obstacle est
  hors d'horizon raisonnable (portée fiable des rangefinders).
- $D_{\text{sat}} = 0.5$ m — saturation basse : on ne laisse pas
  $\omega$ exploser quand $D$ devient petit. Le peak angulaire est
  $\omega_{\max} = k / D_{\text{sat}}$.
- $D_{\text{trig}} = 0.36$ m — en dessous, on *laisse la main* à
  `_enter_contour`. Le biais s'éteint brutalement, mais la continuité
  est assurée par le fait que CONTOUR prend le relais dans la même
  frame avec son propre `contour_side` (qui peut être égal ou différent
  de `side` early).
- $k$ : gain, rad·m/s, **valeur initiale $k = 0.08$**. Justification :
  à $D = 1$ m, $\omega = 0.08$ rad/s ≈ 4.6°/s, ordre de grandeur du
  virage CONTOUR_CRUISE mais distribué sur 2.5 s. Peut être tuné dans
  l'étape 6.

Tous les seuils doivent apparaître en tête du fichier avec des commentaires
explicatifs reprenant ces raisons, pas juste les valeurs.

### Sélection de `side` (cache avec hystéresis)

`side` est déterminé par un **appel unique** à
`_contour_side_from_virtual_scan(rf)` (ou `_contour_side_from_astar` si
`use_astar`) **dès que** $D < D_{\text{cut}}$. Ensuite :

- `side` est caché dans `self._early_side` et réutilisé tel quel.
- Le cache est **invalidé** si :
  (a) $D > D_{\text{cut}} + 0.2$ m pendant ≥ `T_clear = 1.0` s
      (obstacle dégagé), ou
  (b) on passe en CONTOUR/REVERSE/IDLE (FSM change d'état), ou
  (c) on fait un flip explicite via le watchdog de progression globale.

- Sinon, `side` reste constant pendant au moins `T_hold = 1.0` s, même
  si un nouveau scan donnerait un résultat différent. **Pas de flip
  de side avant T_hold**, pour éviter les oscillations de commande
  sur un obstacle légèrement asymétrique.

### Composition avec GO_HEADING

$$
\omega_{\text{cmd}} = \text{clip}\big(\omega_{\text{go\_heading}} + \omega_{\text{bias}},\; -\text{MAX\_NAV\_YAW},\; +\text{MAX\_NAV\_YAW}\big)
$$

Addition pure, clip final. **Pas de pondération relative entre les
deux termes** — le contrôleur de cap reste dominant pour les grands
écarts d'alignement ; le biais est un add-on qui modifie localement
la trajectoire sans casser la convergence vers le heading cible.

### Conditions d'activation

Le biais est *ajouté* seulement si :

1. FSM state == `GO_HEADING`
2. `D_trig < self._min_front_dist < D_cut`
3. `self._use_early_avoid == True`
4. `side` est défini (initialisé ou en cours de hold)

Sinon $\omega_{\text{bias}} = 0$.

---

## Plan incrémental — chaque étape a son test

### Étape 0 — Baseline gelée (15 min)

Re-courir `benchmark_random.py --episodes 100 --seed 42 --planner astar`
trois fois **avec et sans** `--early-avoid off` pour confirmer que le
flag neutre ne touche à rien. Gate : delta = 0. Si non : bug dans
l'intégration du flag, fixer avant de coder la loi.

### Étape 1 — Module `deskbot/early_avoidance.py` (30 min)

Fichier autonome contenant :

- `EarlyAvoidanceParams` (dataclass) avec les 5 constantes documentées
- `compute_bias_yaw(D, side, params) -> float` — loi pure, sans état
- Tests : `scripts/test_early_avoidance.py`
  - $D = 3$ m : bias = 0
  - $D = 2$ m (cutoff) : bias = 0
  - $D = 1$ m, side = +1 : bias = 0.08 rad/s (±1e-6)
  - $D = 0.5$ m, side = −1 : bias = −0.16 rad/s
  - $D = 0.3$ m (< D_trig) : bias = 0
  - $D = 0$ : bias = $k / D_{\text{sat}}$ (vérifier saturation)
  - monotonie : pour `side=+1`, `bias(0.5) ≥ bias(1.0) ≥ bias(2.0) = 0`

**Gate** : 7/7 tests passent avant toute intégration.

### Étape 2 — Gestion d'état `side` dans Navigator (45 min)

Ajouter à `Navigator.__init__` :

- `self._use_early_avoid: bool = False` (paramètre constructeur)
- `self._early_side: int | None = None`
- `self._early_side_timer: float = 0.0`   # temps depuis la sélection
- `self._early_clear_timer: float = 0.0`  # temps de vue dégagée

Nouvelle méthode `_update_early_side(rf, dt)` appelée dans `update()`
juste après `_min_front_dist` :

- Si FSM ≠ GO_HEADING → reset `_early_side = None`, timers à 0.
- Si `D_min > D_cut + 0.2` : incrémenter `_early_clear_timer`, si
  > T_clear reset `_early_side = None`.
- Sinon reset `_early_clear_timer`.
- Si `_early_side is None` et `D_trig < D_min < D_cut` : appeler le
  sélecteur (virtual scan, ou A\* si `_use_astar`) et fixer
  `_early_side`, reset `_early_side_timer`.
- Sinon incrémenter `_early_side_timer`.

**Test** : micro-script `scripts/test_early_side_state.py` qui simule
une séquence de `D_min` sur 10 secondes (approche obstacle, puis
dégagement) et vérifie que `_early_side` est stable pendant au moins
`T_hold`, bascule à None après `T_clear` de dégagement, et n'est jamais
défini en CONTOUR.

### Étape 3 — Intégration dans `_state_go_heading` (15 min)

Modifier le retour de `_state_go_heading` :

```python
yaw = HEADING_P_GAIN * h_err

if self._use_early_avoid and self._early_side is not None:
    from deskbot.early_avoidance import compute_bias_yaw, EARLY_PARAMS
    yaw += compute_bias_yaw(self._min_front_dist,
                            self._early_side, EARLY_PARAMS)

# reste inchangé : clip, vitesse, retour
```

Rien d'autre ne bouge. `_enter_contour` et tous les watchdogs restent
intacts. **Test** : smoke-run un épisode seed=42 avec `--early-avoid on`,
vérifier qu'il termine sans crash et que la trajectoire est "plus
courbe" qu'avant (visuellement via `mapviz`, ou métrique :
`avg|yaw_cmd|` > baseline en GO_HEADING).

### Étape 4 — Benchmark A/B contrôlé (1 h)

Ajouter `--early-avoid {off,on}` à `benchmark_random.py`. Faire tourner
3 runs × 100 ép × (off, on) sur seed 42. Aussi 3 runs × 100 ép sur
seed 43 pour contrôler le sur-apprentissage au seed 42.

**Gate** : `on` doit matcher `off` à ± 1 pt sur le succès global et
sur les chutes. **Si régression > 3 pts, arrêter** et rentrer en phase
diagnostic : identifier 2-3 seeds qui passaient en off et échouent en
on, visualiser la trajectoire via `mapviz`, corriger *la formulation*
(valeur de `k`, `D_sat`, ou logique de sélection `side`), **pas** empiler
des couches.

### Étape 5 — Test chicane dédié (30 min)

Créer `scripts/benchmark_chicanes.py` (copie minimale de
`benchmark_random.py` forçant `obs_type = "chicane"` et générant 30
seeds dédiés). Le but est d'isoler le risque identifié dans la
discussion préparatoire : early avoidance peut mal gérer un flip
rapide de `side` quand deux obstacles se succèdent.

**Gate** : succès chicanes `on` ≥ succès chicanes `off` − 1 seed.
**Si régression > 2 seeds sur 30**, diagnostic : l'hystéresis `T_hold`
est probablement la cause. **Corriger la formulation** (T_hold plus
long, ou détection de "nouveau obstacle" qui force un re-scan anticipé),
pas d'ajout de couche.

### Étape 6 — Benchmark final + tuning ciblé de `k` (1 h)

Si l'étape 4 donne `on == off`, tester 3 valeurs de `k` :
$k \in \{0.05, 0.08, 0.12\}$ sur seed 42, 100 ép, 3 runs chacune.
Garder la meilleure. **Ne PAS** sweep les autres constantes au premier
jet — un seul degré de liberté à la fois.

**Gate final** : meilleur `k` donne ≥ 89 % sur seed 42 ET seed 43, 0
chute, pas de régression > 1 seed sur chicanes vs baseline. Résultats
sauvés dans `docs/benchmark_early_avoid_v1.txt` avec la grille
complète.

### Étape 7 — Mesure mapping + nouveau métric "recall hors ligne
droite" (45 min)

Relancer `eval_mapping.py --episodes 15 --seed 42 --early-avoid {off,on}`
et comparer :

- IoU global (gate : ≥ baseline − 1 pt)
- **Recall** global (c'est là qu'on s'attend au gain)
- Nouveau métric (optionnel, à ajouter si utile) : "recall dans les
  cellules à ≥ 25° du heading courant" — c'est la partie de la grille
  que le balayage rotatif est censé remplir. Si on le voit +2-5 pts
  de recall dans cette bande, la théorie est confirmée
  empiriquement. Si on ne voit rien, la théorie est à revisiter avant
  de pousser early avoidance en défaut.

### Étape 8 — Mise à jour journal + mémoire (15 min)

Écrire la session 10 dans `JOURNAL.md` (mêmes règles narratives que
session 9 : ce qui a marché, ce qui a échoué, leçons, rôle de Claude).
Créer `project_early_avoidance.md` dans la mémoire avec les
paramètres validés et la décision finale (défaut ou opt-in).

---

## Ce qu'il ne faut PAS faire

- **Pas de retour à un APF classique** (potentiel répulsif + attractif).
  La loi de guidage 1/D *n'est pas* un APF — c'est une commande directe
  de yaw rate. Les problèmes d'APF (minima locaux, oscillations entre
  sources multiples) ne s'appliquent pas. Toute confusion dans le code
  ou les commentaires entre les deux est à rejeter.
- **Pas de pondération adaptative** de `k` selon la FSM, la grille, la
  vitesse ou le temps au premier jet. Constante pure, valeur
  justifiée, tuning si et seulement si le gate du §6 échoue.
- **Pas de flip de `side` plus rapide que `T_hold`**. L'hystéresis
  temporelle est structurellement nécessaire pour empêcher les
  oscillations sur obstacles quasi-symétriques. Si on est tenté de la
  retirer "parce que l'obstacle est clairement à gauche maintenant",
  c'est un signe qu'il faut corriger le sélecteur `side`, pas le
  timing.
- **Pas de modification de `_state_contour`**. Zéro. Si early
  avoidance échoue, le CONTOUR classique prend le relais intact.
- **Pas de biais actif en CONTOUR/REVERSE/IDLE**. `_update_early_side`
  doit reset le cache dès qu'on sort de GO_HEADING.
- **Pas d'ajout d'une "couche de sécurité" par-dessus early avoidance**
  (reactive front filter, emergency brake, etc.). `SAFE_DIST` est le
  filet de sécurité, unique, déjà en place.
- **Pas de tuning via un minimiseur ou une grid-search automatique**
  avant d'avoir compris manuellement pourquoi une valeur donnée marche.

---

## Livrables attendus

1. `deskbot/early_avoidance.py` — loi pure + dataclass de paramètres,
   testable indépendamment.
2. `scripts/test_early_avoidance.py` — 7 tests unitaires sur la loi
   pure.
3. `scripts/test_early_side_state.py` — test de la machine à état du
   cache `side` / timers.
4. Modifications chirurgicales de `deskbot/navigation.py` (constructeur,
   `reset`, `update`, `_state_go_heading`). Diff < 80 lignes.
5. Option `--early-avoid {off,on}` dans `benchmark_random.py` et
   `eval_mapping.py`.
6. `scripts/benchmark_chicanes.py` — subset dédié aux scènes chicane.
7. `docs/benchmark_early_avoid_v1.txt` — résultats bruts (seed 42,
   seed 43, chicanes, mapping IoU + recall).
8. Mise à jour `JOURNAL.md` (session 10) + mémoire
   `project_early_avoidance.md`.

---

## Critères d'arrêt de la session

**Arrête-toi et demande à Bruno si** :

- La régression globale dépasse 3 pts sur le benchmark seed 42 ET la
  cause n'est pas évidente en 30 min de diagnostic.
- Tu es tenté d'ajouter une rustine (safety layer, filtre pitch
  supplémentaire, clamp réactif, `side` qui flippe sans hystéresis) pour
  "faire marcher" early avoidance.
- Tu envisages de toucher `_state_contour`, `_enter_contour`, les
  watchdogs, le filtrage pitch, ou le LQR.
- Le gate mapping (étape 7) régresse d'au moins 2 pts d'IoU — c'est le
  signe que la rotation continue casse quelque chose qu'on ne voit pas
  (probablement la DR).
- Le benchmark prend plus de 20 minutes pour 100 épisodes en parallèle
  — bug de perf probable.
- Tu vois des chutes (> 0 falls) dans n'importe quel run. Le LQR doit
  rester stable sous toute trajectoire produite par early avoidance ;
  une chute est le signe que la composition `yaw_go_heading + ω_bias`
  sature mal ou que le clip est trop lâche.

**Continue sans demander si** :

- Tu tunes les constantes documentées (`k`, `D_sat`, `D_cut`,
  `T_hold`, `T_clear`) une à la fois, avec justification écrite du
  changement.
- Tu ajoutes des tests unitaires ou des asserts au module pur.
- Tu ajoutes des logs diagnostiques ou des snapshots mapviz.
- Tu améliores la formulation mathématique (ex. passer de saturé à
  bell-shaped $\omega = k \cdot D / (D^2 + D_{\text{sat}}^2)$) à
  condition de documenter le changement et de le passer par le gate du
  §4 avant de l'adopter.

---

## Annexe — notes sur les risques identifiés

### Risque A : chicane oscillante

Deux obstacles successifs à moins de 1.5 m l'un de l'autre. Le biais
sélectionné pour le premier peut être incohérent avec ce qui serait
optimal pour le second. Mitigation de premier jet : `T_hold = 1 s`, re-
scan obligatoire après dégagement (`T_clear`). Si ça ne suffit pas
(cf. étape 5), considérer une formulation où `side` est recalculé
aussi quand `_min_front_dist` remonte de >50 cm entre deux frames
(signe d'un obstacle passé, nouveau scan justifié). **Ne pas**
implémenter cette subtilité au premier jet.

### Risque B : obstacle parfaitement frontal

Un mur perpendiculaire exactement au cap laisse le virtual scan sans
asymétrie claire. Le sélecteur a déjà un tie-break (biais capteur FL
vs FR, ou last resort arbitrary). Early avoidance hérite
automatiquement de ce comportement — pas de travail additionnel.
Vérifier néanmoins sur un seed dédié dans le test chicane.

### Risque C : composition avec A\*

A\* à `_enter_contour` reste en place. Early avoidance agit *avant*
et peut faire éviter l'obstacle sans jamais toucher SAFE_DIST — donc
A\* n'est jamais consulté. Question ouverte : est-ce qu'on veut que le
*sélecteur* de `side` en early avoidance appelle A\* pour avoir une
meilleure vision du détour, ou rester sur le virtual scan qui est plus
rapide ? **Décision initiale** : rester sur virtual scan, A\* est
réservé au fallback CONTOUR. Re-évaluer en étape 6 si le gate est
limite.

### Risque D : biais qui "mange" le heading

Si `k` est trop grand, le biais domine le contrôleur de cap et le
robot tourne en rond autour de l'obstacle au lieu d'atteindre la
cible. Mitigation : le clip final à `MAX_NAV_YAW` borne physiquement
le yaw à 1.5 rad/s, donc même avec `k` saturé on ne dépasse jamais la
capacité physique du robot. Et le terme `HEADING_P_GAIN * h_err` reste
dans la somme, donc quand le robot est perpendiculaire à la cible, il
a une tendance forte à se réaligner qui contrecarre le biais. C'est
structurellement stable mais à vérifier empiriquement.

### Risque E : dégradation de la DR par rotation continue

Le modèle encodeur-pitch corrigé session 8 suppose que les rotations
sont relativement lentes. Un early avoidance à 0.08 rad/s continu
pendant 5 s, c'est 0.4 rad (23°) intégrés. Si la correction
encodeur-pitch introduit un biais avec la courbure, la DR peut
dériver. **Instrumentation** : logger `|pos_x_dr - qpos[0]|` sur
quelques épisodes et comparer off vs on. Si la dérive augmente de >
5 cm sur 12 m, c'est un signal à traiter comme un problème de
formulation, pas à ignorer.
