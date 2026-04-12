# Plan de recherche — A\* local planner sur la grille d'occupation

**Pour** : session Claude séparée (prompt autonome)
**Auteur** : Bruno + Claude (session 11, 2026-04-12)
**Durée estimée** : 2-4 h de travail itératif

---

## Objectif

Évaluer si un **A\* local sur la grille d'occupation** améliore le taux de succès de navigation de Bug2, en exploitant le mapping récemment corrigé (IoU 0.24 → 0.39). Le planificateur A\* remplace ou augmente le **virtual scan** actuel de Bug2 à l'entrée de `CONTOUR`, sans toucher au reste de la FSM.

**Cible chiffrée** : battre Bug2 v2 sur `scripts/benchmark_random.py` (seed 42, 100 épisodes) avec ≥ 85 % de succès global ET ≥ 82 % sur les murs, 0 chute. Si on n'atteint pas ça, on revient à Bug2 sans toucher au planificateur.

---

## Contexte à lire avant de coder

Dans l'ordre :

1. [CLAUDE.md](../CLAUDE.md) — règles du projet (sensor barrier, rigueur avant heuristique, pedagogical mode obligatoire).
2. [docs/teb_evaluation.md](teb_evaluation.md) — **lire en entier**. Explique pourquoi TEB a été écarté et pourquoi A\* local est le bon candidat. Le §6.2 donne les 5 conditions à respecter pour tout nouveau planificateur. Le §9 donne les critères d'acceptation numérique.
3. [deskbot/navigation.py](../deskbot/navigation.py) — Bug2 v2, la FSM actuelle. Regarder spécifiquement :
   - `_enter_contour()` (virtual scan 13 directions) → **c'est ce que A\* va remplacer**
   - `OccupancyGrid` (log-odds, Bresenham, shift) → **c'est l'input de A\***
   - `_can_resume_heading()` → exit criterion, à laisser tel quel
4. [deskbot/mapping_eval.py](../deskbot/mapping_eval.py) + [scripts/eval_mapping.py](../scripts/eval_mapping.py) — infrastructure de mesure du mapping, à réutiliser pour vérifier qu'A\* ne dégrade rien en amont.
5. [scripts/benchmark_random.py](../scripts/benchmark_random.py) — benchmark navigation Bug2, **à réutiliser tel quel** pour l'évaluation finale.

---

## État de la base (ne pas le découvrir à tâtons)

- **Bug2 v2** : 82-83 % sur 30-50 épisodes randomisés, seed 42. Validé sur > 100 épisodes historiques. Ne jamais le remplacer en totalité.
- **Mapping** : log-odds 60×60 @ 8 cm, corrigé par gyro-DR (voir [deskbot/control.py:44](../deskbot/control.py#L44) et [deskbot/navigation.py:513](../deskbot/navigation.py#L513)). IoU strict = 0.39, IoU tolérant-1 = 0.84. **Les obstacles sont localisés à 8 cm près** — c'est la précondition qui rend A\* viable.
- **Modes d'échec restants de Bug2** (à partir du post-mortem session 7) : ~22 % d'échec sur obstacles de type "wall with gap" quand le trou sort de la couverture FL2/FR2 (> 55° off-center). La cause est que le virtual scan actuel utilise seulement 13 directions et peut rater un passage étroit qui nécessite un détour en S. **C'est exactement le cas où A\* devrait briller.**

---

## Contraintes non négociables

1. **Sensor barrier préservée** : A\* lit `Navigator.grid.grid` (log-odds) et les `SensorReadings`. Jamais `data.qpos`, `data.xpos`, etc.
2. **Bug2 reste intact** : A\* est appelé *à l'intérieur* de `_enter_contour` comme alternative au virtual scan, pas comme remplacement de la FSM. Si A\* échoue ou timeout, fallback silencieux sur le virtual scan existant.
3. **Flag d'activation** : ajouter une option `--planner bug2|astar` au `benchmark_random.py` pour faire de l'A/B seed-à-seed. Bug2 reste la valeur par défaut tant qu'A\* n'a pas passé le gate numérique.
4. **Aucune rustine sur le filtrage pitch, les log-odds, ou le mapping** : si A\* se comporte mal, corriger *la formulation d'A\**, pas empiler des couches. Leçon explicite de la session 8.
5. **Zéro dépendance nouvelle** : heapq (stdlib) est suffisant pour A\*. Pas de networkx, pas de scipy nouveau.
6. **Rigueur avant heuristique** : documenter la fonction de coût, l'heuristique admissible, la règle de voisinage, et justifier chaque constante avant de tuner.

---

## Plan incrémental — chaque étape a son test

### Étape 1 — Structure et coût (30 min)

Créer `deskbot/astar_local.py` avec :
- `AStarPlanner(grid, start_cell, goal_cell) → list[cell] | None`
- Voisinage 8-connecté, coût = distance euclidienne entre centres de cellules
- Heuristique = distance octile (admissible pour 8-conn uniforme)
- **Inflation d'obstacles** : toute cellule dans un rayon de `INFLATE_CELLS = 2` d'une cellule occupée est interdite (robot radius ≈ 13 cm + 5 cm marge, à 8 cm/cellule → 2 cellules). Calculer par distance-transform sur le mask des cellules occupées, ou par dilatation binaire.
- Garde-fous : `max_iterations = 2000`, timeout sur nombre de cellules visitées

**Test unitaire** : grille 60×60 avec un mur à ci=30, deux points (5,5) et (55,55) de part et d'autre. A\* doit retourner un chemin qui contourne le mur. Imprimer le nombre de cellules visitées et la longueur du chemin.

### Étape 2 — Transformation start/goal monde → cellule (15 min)

- `start = robot current cell` (world_to_cell)
- `goal` : point sur la ligne cible à distance `LOCAL_HORIZON = 1.5 m` devant le robot (ou au bord de la grille si plus proche)
- Si le goal est dans une cellule occupée ou inflatée, prendre la cellule libre la plus proche dans un voisinage 5×5.

**Test unitaire** : placer le robot à (0,0), cap 0°, mur à x=1, goal théorique (1.5, 0) bloqué → doit renvoyer un goal valide contourné.

### Étape 3 — Intégration dans `_enter_contour` (30 min)

- Nouveau chemin appelé uniquement si `self._use_astar = True` (flag constructeur)
- Si A\* renvoie un chemin, extraire la **tangente initiale** (direction du 1er segment) → c'est ce qui remplace le `best_angle_deg` du virtual scan
- Convertir la tangente en `contour_side` (+1 ou −1) comme le fait déjà Bug2
- Si A\* timeout ou renvoie `None`, appeler le virtual scan existant (fallback)

**Test unitaire** : sur une scène synthétique avec un mur à gauche, A\* doit choisir `contour_side = +1` (contourner par la droite). Vérifier.

### Étape 4 — Benchmark A/B sans re-planification (1 h)

- Ajouter `--planner {bug2,astar}` à `benchmark_random.py`
- Passer `use_astar=(args.planner=="astar")` au Navigator
- Faire tourner 50 épisodes seed 42 dans chaque mode → comparer
- **Si A\* régresse** (< 82 %), ne PAS patcher : identifier un seed d'échec, debug avec `scripts/test_mapviz.py` + snapshots, corriger *la formulation* d'A\* (inflation, heuristique, voisinage, goal selection)

**Gate** : A\* doit matcher Bug2 à ce stade (± 2 pts). Sinon, revenir aux étapes précédentes.

### Étape 5 — Re-planification périodique (30 min)

Jusqu'ici A\* ne planifie qu'à l'*entrée* de CONTOUR. Si la grille évolue pendant le contour, le chemin peut devenir obsolète. Re-planifier toutes les `REPLAN_PERIOD = 1.0 s` pendant CONTOUR, et utiliser le nouveau chemin pour biaiser le wall-follow (heading pull gagne une composante "suivre le path").

**Test** : sur un seed où Bug2 échoue en contour (ex. seed 47 ou seed 51 du baseline de session 11), vérifier qu'A\* replan guide le robot à travers le passage.

### Étape 6 — Benchmark final (30 min)

- `python scripts/benchmark_random.py --episodes 100 --seed 42 --planner astar` vs `--planner bug2`
- Comparer seed-à-seed, pas seulement la moyenne
- Sauvegarder les résultats dans `docs/benchmark_astar_v1.txt`

**Gate numérique** (cf. teb_evaluation.md §9) :
- Succès global ≥ 82 % (Bug2 baseline)
- Succès murs ≥ 78 %
- Succès hors-murs ≥ 89 %
- 0 chute
- Pas de seed qui régresse de > 1 point d'IoU mapping (Astar ne doit pas dégrader la grille indirectement)

Si **toutes** ces conditions sont réunies, A\* devient le planificateur par défaut dans CONTOUR. Sinon il reste accessible via `--planner astar` pour investigation future, Bug2 reste défaut.

### Étape 7 — Mesure d'impact mapping (15 min)

Relancer `scripts/eval_mapping.py --episodes 15 --seed 42 --tag astar_v1` avec A\* activé et vérifier que l'IoU mapping ne régresse pas (sanity check — un planificateur qui fait faire des trajectoires différentes génère une grille différente, et on veut confirmer qu'elle reste au moins aussi bonne).

---

## Définitions mathématiques à documenter dans le code

Chaque fonction doit avoir un docstring qui expose :

- **Repères** : tous les (x, y) en frame DR, unités en mètres. Toutes les (ci, cj) en indices de cellule.
- **Transformation cellule ↔ monde** : formule exacte (reprendre celle de `OccupancyGrid.world_to_cell` pour cohérence).
- **Fonction de coût g(n)** : somme des distances euclidiennes des segments parcourus. Justification : mesure physique de la longueur du chemin parcouru.
- **Heuristique h(n)** : distance octile `octile(Δc, Δr) = max(Δc,Δr) + (√2 − 1) × min(Δc,Δr)`. Admissible et consistante pour 8-connectivité sur grille uniforme. Jamais d'heuristique non admissible — donne des chemins sous-optimaux silencieusement.
- **Inflation** : rayon en cellules = `ceil((robot_half_width + safety_margin) / GRID_RES)`. Nommer `INFLATE_CELLS`, commentaire avec le calcul.
- **Condition de terminaison** : goal atteint OU cellules_visitées > max_iter OU pas de chemin.
- **Complexité** : O(N log N) où N = cellules visitées, bornée par `max_iterations`.

---

## Ce qu'il ne faut PAS faire

- **Pas d'heuristique dynamique** (pondérée par occupancy locale, etc.) au premier jet. Commencer par l'heuristique admissible pure, valider, puis expérimenter.
- **Pas de lissage de chemin** type Douglas-Peucker ou spline au premier jet. Un chemin A\* brut (escalier 8-conn) est suffisant pour biaiser la tangente initiale du contour.
- **Pas de coût non-uniforme sur la grille** (type "pénaliser les cellules proches d'obstacles") au premier jet. L'inflation binaire fait déjà ce travail proprement, et les coûts continus rendent le debug beaucoup plus dur.
- **Pas de tentative de remplacer le wall-follow P-controller** par du pur path-tracking. La FSM reste Bug2. A\* donne seulement la *direction initiale* et un biais de re-planification.
- **Pas de retour à TEB "parce que A\* est trop simple"**. Le problème précédent était l'inadéquation structurelle, pas la simplicité.

---

## Livrables attendus

1. `deskbot/astar_local.py` — module A\* autonome, testable indépendamment
2. Modifications chirurgicales de `deskbot/navigation.py` (constructeur + `_enter_contour`)
3. Option `--planner` dans `scripts/benchmark_random.py`
4. Script `scripts/test_astar_local.py` — tests unitaires scène synthétique, imprime chemin + nombre d'itérations, lisible par Claude
5. `docs/benchmark_astar_v1.txt` — résultats bruts du benchmark final
6. Mise à jour de `JOURNAL.md` — session N+1, rigueur habituelle (ce qui a marché, ce qui a échoué, leçons)
7. Mise à jour de mémoire : si A\* passe le gate, créer un `project_astar_planner.md` décrivant l'architecture finale et les paramètres validés

---

## Critères d'arrêt de la session

**Arrête toi et demande à Bruno si** :
- A\* régresse de plus de 5 points sur le benchmark et la cause n'est pas évidente en 30 min
- Tu es tenté d'ajouter une rustine (safety layer, filtre pitch, heuristique bizarre) pour "faire marcher" A\*
- Tu envisages de toucher la FSM Bug2 (états, transitions, watchdogs)
- Tu envisages de toucher le filtrage pitch dans `GroundGeometry` ou `compensate_rangefinders`
- Le benchmark prend plus de 20 minutes à exécuter (probablement un bug ou une régression de perf)

**Continue sans demander si** :
- Tu tunes les constantes documentées (`INFLATE_CELLS`, `LOCAL_HORIZON`, `max_iterations`, `REPLAN_PERIOD`)
- Tu améliores la formulation mathématique de l'heuristique ou du coût en restant admissible
- Tu ajoutes des tests unitaires
- Tu améliores les logs et les snapshots diagnostiques
