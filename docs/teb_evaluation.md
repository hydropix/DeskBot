# Évaluation TEB (Timed Elastic Band) pour DeskBot

**Auteur** : Bruno + Claude (session 11, 2026-04-12)
**Statut** : Document de conception — **aucun code modifié** tant que le verdict n'est pas validé
**Référence à battre** : Bug2 v2 à **82 %** de succès sur le benchmark aléatoire (seed 42, 50 épisodes, couloir 3 m, 2-5 obstacles)

---

## 1. Contexte et objectif

TEB a déjà été tenté lors de la Session 8 (2026-04-12) et s'est soldé par un échec catastrophique : le robot fonçait dans les murs. Le post-mortem (`project_navigation_ai.md`, `feedback_nav_redesign.md`) identifie trois causes :

1. **Pas de benchmark randomisé avant commit.** L'implémentation avait été validée uniquement sur une scène synthétique à un seul mur.
2. **Rustines conflictuelles.** Une couche de sécurité réactive et un second filtre de pitch avaient été empilés par-dessus `compensate_rangefinders`, créant deux seuils de filtrage en concurrence.
3. **Aucun modèle mathématique documenté.** Les coûts, les contraintes, la topologie du graphe et les hyperparamètres étaient des boîtes noires copiées depuis `teb_local_planner` (ROS), sans justification.

Ce document pose le **socle mathématique** qui manquait, puis évalue si TEB est **physiquement pertinent** pour un auto-équilibreur à 2 roues avant toute ligne de code.

---

## 2. Formulation mathématique rigoureuse de TEB

### 2.1 Variables de décision

TEB optimise une *bande élastique temporisée* : une séquence alternée de **poses** et d'**intervalles de temps**.

$$
\mathcal{B} = \{ \mathbf{s}_1, \Delta T_1, \mathbf{s}_2, \Delta T_2, \ldots, \Delta T_{n-1}, \mathbf{s}_n \}
$$

avec :

- $\mathbf{s}_i = (x_i, y_i, \theta_i)^\top \in SE(2)$ — la $i$-ème pose planifiée dans le repère monde, en 2D plan + orientation yaw
- $\Delta T_i > 0$ — durée (secondes) pour passer de $\mathbf{s}_i$ à $\mathbf{s}_{i+1}$
- $n$ — nombre de poses (typiquement 10 à 20). La dimension du problème est $3n + (n-1) = 4n - 1$.

**Hypothèse implicite n°1** : la dynamique du robot est entièrement décrite par $(x, y, \theta)$ dans le plan. **C'est déjà problématique pour un auto-équilibreur** — on y reviendra en §4.

Les $\mathbf{s}_1$ et $\mathbf{s}_n$ sont **fixés** (contraintes d'égalité dures) à la pose courante et à la pose cible. Toutes les autres sont libres.

### 2.2 Fonction objectif : moindres carrés non linéaires

TEB minimise une somme pondérée de **termes de pénalité** (une par contrainte physique ou objectif) exprimés comme des moindres carrés :

$$
\mathbf{B}^* = \arg\min_{\mathcal{B}} \sum_{k} \sigma_k \cdot e_k(\mathcal{B})^2
$$

où $\sigma_k$ est un poids scalaire et $e_k$ un résidu. L'optimisation se fait par **Levenberg–Marquardt** sur le graphe creux (car chaque terme n'implique que 2-3 poses consécutives), typiquement avec la librairie **g2o** (Kümmerle et al., 2011).

### 2.3 Termes de coût (formulation standard Rösmann 2012-2017)

Tous les termes utilisent une pénalité *barrière quadratique* :

$$
e(x; x_{\min}, x_{\max}, \epsilon) =
\begin{cases}
(x - x_{\min} + \epsilon)^2 / \epsilon & \text{si } x < x_{\min} \\
(x - x_{\max} + \epsilon)^2 / \epsilon & \text{si } x > x_{\max} \\
0 & \text{sinon}
\end{cases}
$$

Ceci produit une violation proche-zéro tant qu'on reste dans la plage admissible, et explose quadratiquement dès qu'on en sort.

#### (a) **Temps minimal** — objectif principal

$$
e_{\text{time}} = \left( \sum_{i=1}^{n-1} \Delta T_i \right)^2
$$

C'est ce qui pousse TEB à *minimiser le temps de trajet* — l'objectif global de la bande.

#### (b) **Limites cinématiques** — vitesse linéaire et angulaire

Entre deux poses consécutives, on estime la vitesse linéaire effective :

$$
v_i = \frac{\|\mathbf{p}_{i+1} - \mathbf{p}_i\|}{\Delta T_i} \cdot \mathrm{sgn}\big((\mathbf{p}_{i+1} - \mathbf{p}_i) \cdot \hat{\mathbf{u}}_i\big)
$$

où $\hat{\mathbf{u}}_i = (\cos\theta_i, \sin\theta_i)$ est l'orientation au début de l'intervalle. Le `sgn` capture la marche arrière (produit scalaire négatif).

La vitesse angulaire discrète :

$$
\omega_i = \frac{\theta_{i+1} - \theta_i}{\Delta T_i}
$$

(avec wrap autour de $[-\pi, \pi]$). Les pénalités :

$$
e_v = \text{penalty}(v_i; -v_{\max}^{\text{back}}, +v_{\max}), \quad
e_\omega = \text{penalty}(\omega_i; -\omega_{\max}, +\omega_{\max})
$$

#### (c) **Limites dynamiques** — accélération linéaire et angulaire

$$
a_i = \frac{2(v_{i+1} - v_i)}{\Delta T_i + \Delta T_{i+1}}, \quad
\alpha_i = \frac{2(\omega_{i+1} - \omega_i)}{\Delta T_i + \Delta T_{i+1}}
$$

La différence finie centrée donne une estimation moyenne sur les deux intervalles.

#### (d) **Contrainte non holonomique** — différentielle pure

Pour un robot différentiel, le déplacement entre deux poses doit suivre un **arc de cercle** compatible avec l'orientation. Rösmann impose :

$$
e_{\text{nh}} = \left\| (\mathbf{p}_{i+1} - \mathbf{p}_i) \times
\big((\hat{\mathbf{u}}_i + \hat{\mathbf{u}}_{i+1})/\|\cdot\|\big) \right\|^2
$$

Ce terme est nul ssi le segment connectant les deux positions est colinéaire avec la bissectrice des deux orientations — condition géométrique nécessaire pour un arc cinématiquement faisable en différentiel.

#### (e) **Obstacles** — distance signée

Pour chaque obstacle $O_j$ (typiquement un point, une ligne ou un polygone issu de la grille d'occupation) et chaque pose $\mathbf{s}_i$ "proche" :

$$
d_{ij} = \text{dist}(\mathbf{p}_i, O_j), \quad
e_{\text{obs}} = \text{penalty}(d_{ij}; d_{\min}, +\infty)
$$

où $d_{\min}$ est la distance de sécurité (rayon du robot + marge). Les obstacles sont attachés aux *poses* (pas aux segments) pour garder la matrice jacobienne creuse.

**Point critique** : la liste d'obstacles est construite à partir de la grille d'occupation (ou d'un coût 2D), dont la qualité dépend des capteurs. **Si la grille est corrompue par les pitch oscillations, TEB optimise contre une réalité fantasmée.**

#### (f) **Via-points** (optionnel) — attracteurs doux

Pour forcer la bande à passer près d'un point désiré sans contrainte dure.

### 2.4 Topologie : nombre de poses variable

TEB **ajoute et retire dynamiquement des poses** pour maintenir une distance spatio-temporelle cible entre deux poses consécutives. C'est ce qui lui donne son nom *élastique* : la bande s'étire ou se resserre selon la géométrie locale.

- Si $\|\mathbf{p}_{i+1} - \mathbf{p}_i\| > d_{\text{ref}}$ → insérer une pose à mi-chemin
- Si $\|\mathbf{p}_{i+1} - \mathbf{p}_i\| < d_{\text{ref}} / 2$ → supprimer la pose $i+1$

Cela rend le problème **à taille variable** — chaque cycle d'optimisation peut changer le nombre de variables. C'est une source majeure de complexité d'implémentation.

### 2.5 Optimisation multi-topologie (TEB homotopique)

La version complète (Rösmann 2017) maintient *plusieurs* bandes en parallèle, chacune dans une **classe d'homotopie** distincte (ex. : contourner un obstacle par la gauche vs. par la droite), et choisit en fin de cycle celle qui a le plus petit coût. C'est le mécanisme qui permet à TEB de gérer les minima locaux dans lesquels DWA se fait piéger.

**Coût computationnel** : ce mécanisme multiplie la charge par 3-5x. Sur un ESP32-S3 cible, c'est rédhibitoire sans simplification drastique.

---

## 3. Hypothèses implicites de TEB (à confronter à DeskBot)

| # | Hypothèse TEB | Exigence physique | Valide pour DeskBot ? |
|---|---------------|-------------------|-----------------------|
| H1 | État robot = $(x, y, \theta) \in SE(2)$ | Robot stable en 2D plan | **Non** — 6 états (pitch, pitch_rate inclus) |
| H2 | Dynamique contrôlable découplée en $(v, \omega)$ | Actionneurs indépendants | Partiellement — $v$ est elle-même un setpoint d'une boucle pitch sous-actionnée |
| H3 | Obstacles connus et stables | Grille d'occupation fiable | **Fragile** — dépend du pitch filtering |
| H4 | Capteurs omni-directionnels | LIDAR ou caméra 360° | **Non** — 7 rangefinders, aveugle arrière, gaps 15-25° |
| H5 | Horizon de planification ≫ dimension robot | Permet optimisation multi-pose | **Tendu** — grille 4.8 m / longueur robot 0.2 m → 24 unités seulement |
| H6 | Fréquence optimisation 5-20 Hz suffisante | CPU disponible | **Oui** en simulation, **douteux** sur ESP32-S3 |
| H7 | Obstacles statiques ou lents | Pas de cible ni d'agent mobile | **Oui** dans notre cas |

**Trois hypothèses majoritairement fausses sur DeskBot : H1, H3, H4.**

---

## 4. Le problème fondamental : DeskBot n'est pas un robot planaire

### 4.1 Couplage pitch ↔ vitesse

La dynamique réelle de DeskBot est décrite par 6 états :
$\mathbf{x} = (\theta_{\text{pitch}}, \dot\theta_{\text{pitch}}, v, x, \psi, \dot\psi)$

Pour générer $v > 0$ (avancer), le LQR commande un **lean en avant** $\theta_{\text{pitch}} > 0$, ce qui :

1. **Bascule les rangefinders frontaux vers le sol.** Les capteurs FC/FL/FR sont montés à 6 cm, horizontaux. À $\theta_{\text{pitch}} = 10°$, ils touchent le sol à ~35 cm. `GroundGeometry` filtre ces coups de sol, mais au prix d'une perte d'information : le secteur frontal devient aveugle au-delà d'un certain pitch.
2. **Contamine la vitesse estimée** via le couplage encodeur–pitch (corrigé session 8 avec `+pitch_rate * R`), donc indirectement la dead reckoning → toute la grille d'occupation.

**Conséquence pour TEB** : une bande TEB qui demande $v_i$ élevé impose un $\theta_{\text{pitch}}$ non négligeable, ce qui dégrade la qualité des observations sur lesquelles TEB se fonde pour optimiser le coût d'obstacle. **C'est un couplage observabilité ↔ commande que TEB ne modélise pas.**

### 4.2 Contrainte non holonomique (e) vs auto-équilibreur

Le terme (e) de §2.3 impose un chemin en arc de cercle entre deux poses. Or DeskBot :
- Ne peut pas tourner sur place instantanément sans risquer de basculer (limite yaw_budget à 25 % du couple).
- Ne peut pas s'arrêter net pour pivoter : l'inertie en pitch impose une rampe de décélération.

Ces contraintes ne sont **pas exprimables** dans le terme (e) de TEB : elles dépendent de l'état pitch, pas de la géométrie du chemin. Il faudrait étendre TEB à **7 états** (ajouter pitch, pitch_rate, v) au lieu de 3 — ce qui transforme le problème en MPC non linéaire, plus en TEB.

### 4.3 Horizon de planification très court

La grille fait 4.8 m de côté, mais les rangefinders portent à ~1.5-2 m. En pratique, la connaissance fiable du monde au-delà de ~1.8 m devant le robot est nulle.

Avec $v_{\max} = 0.5$ m/s et un horizon de 1.5 m, TEB a **3 secondes** devant lui. Avec une résolution temporelle $\Delta T_{\text{ref}} = 0.3$ s, cela fait **10 poses**. C'est le plancher en dessous duquel TEB n'apporte plus grand-chose par rapport à DWA (qui regarde déjà ~1 s devant). **Le ratio horizon/vitesse est trop faible pour capter le bénéfice principal de TEB** (planification multi-pas dans des topologies complexes).

---

## 5. Diagnostic rétrospectif de l'échec de la session 8

Avec le socle mathématique du §2, on peut maintenant analyser ce qui avait cassé :

1. **Terme obstacles sous-pondéré ou distance de sécurité trop faible.** Le robot fonçait dans les murs → le terme (e) n'était pas assez dominant vs. le terme (a) "temps minimal". Classique : TEB mal pondéré privilégie la vitesse au détriment de la sécurité. **Correction** : $\sigma_{\text{obs}} \gg \sigma_{\text{time}}$ et $d_{\min}$ généreux (30 cm minimum, cf. SAFE_DIST actuel).

2. **Grille d'occupation corrompue par le pitch.** Le double filtrage pitch a détruit la cohérence de la grille. **Correction** : filtrage pitch **une seule fois**, dans `compensate_rangefinders` (désormais `GroundGeometry`). TEB ne doit **jamais** rajouter sa propre logique pitch.

3. **Pas de benchmark = pas de signal d'alerte.** Impossible de savoir si les changements amélioraient ou dégradaient les choses. **Correction** : `scripts/benchmark_random.py` existe, 100 épisodes minimum, comparaison A/B contre Bug2.

4. **TEB remplaçait Bug2 au lieu de le compléter.** La FSM Bug2 encode 15 heuristiques validées statistiquement. Les jeter toutes d'un coup efface le terrain d'atterrissage. **Correction** : architecture hybride (voir §7).

---

## 6. Verdict de pertinence

### 6.1 TEB pur en remplacement de Bug2 : **NON pertinent**

Raisons :
- H1, H3, H4 fausses → le modèle de TEB est structurellement inadapté.
- Horizon court → peu de valeur ajoutée vs. DWA, et aucune vs. Bug2 validé.
- Coût de développement élevé (g2o ou réimplémentation maison), risque de régression énorme.
- Coût CPU cible (ESP32-S3) : nonlinear graph optimization à 10 Hz n'est pas tenable.

### 6.2 TEB comme **sous-planificateur local** pendant CONTOUR : **potentiellement pertinent**

L'idée : Bug2 garde la couche stratégique (GO_HEADING ↔ CONTOUR), et TEB remplace **uniquement** le wall-following P-controller actuel par une mini-optimisation de bande sur 3-5 poses (horizon ~1 m).

Avantages :
- L'échec local d'une optimisation TEB ne détruit pas toute la navigation : Bug2 détecte le stuck (timer 3 s) et fait un REVERSE + flip de côté.
- Le domaine de pertinence de TEB (contournement multi-pas d'un obstacle) correspond exactement aux 22 % de murs où Bug2 échoue.
- On garde la FSM validée et le filet anti-régression.
- Coût CPU réduit : 3-5 poses, 3-4 itérations LM, pas de multi-topologie.

**Mais seulement sous 5 conditions strictes** :

1. **Formulation documentée.** Chaque terme de coût doit figurer dans le code avec sa formule et son unité.
2. **Bug2 reste fonctionnel en parallèle.** Flag `--planner bug2|hybrid` pour A/B.
3. **Seuil d'acceptation fixé à l'avance.** Hybride doit scorer ≥ 82 % sur le même seed (42) et ≥ 78 % en wall-only pour remplacer Bug2 par défaut. Aucun flou.
4. **Tests unitaires de chaque terme de coût** avant intégration : scène synthétique avec un seul obstacle, vérifier que le gradient pointe dans la bonne direction et que le minimum correspond à l'intuition géométrique.
5. **Un seul filtre pitch, zéro rustine réactive** par-dessus TEB. Si TEB échoue, on corrige la **formulation**, pas on empile une couche.

### 6.3 Alternative à considérer avant d'implémenter TEB hybride

Avant de se lancer dans TEB, il vaut la peine de regarder **où exactement** Bug2 échoue. Le benchmark montre 82 % global, 78 % en murs. **Un diagnostic ciblé** sur les 22 % échoués aurait peut-être un meilleur rendement :

- Sont-ils tous du même type (murs avec trou hors couverture FL2/FR2) ?
- Est-ce un problème de choix de côté initial ou de recherche de sortie ?
- Un balayage virtuel plus large sur la grille résoudrait-il le cas sans planificateur local ?

**Recommandation forte** : faire tourner `benchmark_random.py --episodes 200 --verbose --seed 42` et catégoriser manuellement les 20-40 échecs avant de décider si TEB est la bonne réponse.

---

## 7. Architecture hybride proposée (si Go)

```
┌─────────────────────────────────────────────────────────────┐
│  STRATEGIC (Bug2 v2, inchangé)                              │
│    FSM : IDLE → GO_HEADING → CONTOUR → REVERSE              │
│    Anti-régression watchdogs (par-contour + global)         │
│    Stuck detection, reverse + flip side                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
        GO_HEADING     │    CONTOUR
             │         │         │
             ▼         │         ▼
┌──────────────────┐   │   ┌────────────────────────────────┐
│  Heading P-ctrl  │   │   │  TEB local (3-5 poses, 1m)     │
│  (inchangé)      │   │   │  coût: temps + obs + nh + v    │
│                  │   │   │  fallback: wall-follow P-ctrl  │
└──────────────────┘   │   └────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  PERCEPTION (inchangé)                                      │
│    GroundGeometry (filtrage pitch RIGOUREUX, une seule fois)│
│    OccupancyGrid 60x60 @ 8cm (existant)                     │
│    7 rangefinders                                           │
└─────────────────────────────────────────────────────────────┘
```

**Contrats d'interface** :

- TEB reçoit : pose courante $(x, y, \theta)$, pose cible locale (proche point sur le mur suivi), grille d'occupation, côté contour ±1.
- TEB renvoie : $(v_{\text{cmd}}, \omega_{\text{cmd}})$ pour le prochain pas de temps, ou `None` si l'optimisation a divergé / timeout.
- Si `None`, Bug2 retombe sur le wall-follow P-controller actuel. Aucune tentative de rustine.

---

## 8. Plan d'implémentation incrémental (chaque étape avec son test)

**Étape 0 — Baseline** : `benchmark_random.py --seed 42 --episodes 100` avec Bug2 seul. Capturer le score exact (devrait être ~82 %). Sauver dans `docs/benchmark_baseline.txt`.

**Étape 1 — Structure de données TEB** : classe `TebBand` avec $(\mathbf{s}_i, \Delta T_i)$, méthodes `add_pose`, `remove_pose`, `compute_v_omega`. **Test unitaire** : créer une bande droite de 5 poses, vérifier que $v_i$ calculés sont constants et égaux à $L/(n-1)/\Delta T$.

**Étape 2 — Terme de coût temps** : $e_{\text{time}}(\mathcal{B})$. **Test unitaire** : gradient analytique vs numérique (écart < 1e-6).

**Étape 3 — Termes vitesse / accel** : $e_v, e_\omega, e_a, e_\alpha$ avec leurs gradients. **Test unitaire** : bande avec $\Delta T_i$ trop petit → le résidu doit être strictement positif et croissant avec la violation.

**Étape 4 — Terme non holonomique** : $e_{\text{nh}}$. **Test unitaire** : deux poses colinéaires → résidu zéro ; décalage latéral → résidu proportionnel.

**Étape 5 — Terme obstacles** : distance signée à une liste de points d'obstacle. **Test unitaire** : un obstacle à 20 cm avec $d_{\min} = 30$ cm → gradient pointe opposé à l'obstacle.

**Étape 6 — Optimiseur** : Levenberg–Marquardt maison (ou `scipy.optimize.least_squares` avec jacobienne creuse). **Test** : scène synthétique à un mur, bande initialisée droit → après optimisation elle doit contourner le mur. Visualiser la bande avant/après.

**Étape 7 — Topologie variable** : insertion/suppression de poses. **Test** : bande initialisée à 3 poses sur 2 m → après un cycle, devrait avoir ~7 poses.

**Étape 8 — Intégration Bug2** : nouvelle sous-routine `_state_contour_teb`, appelée uniquement si un flag `--planner hybrid`. `_state_contour` existant reste accessible en fallback. **Test** : benchmark 50 épisodes en mode `hybrid` vs. `bug2`, comparer.

**Étape 9 — Benchmark complet** : 200 épisodes, seed 42 + 43 + 44 pour marger la variance. **Critère d'acceptation** : hybride ≥ 82 % global ET ≥ 78 % wall-only ET 0 chute supplémentaire. Sinon on revert.

**Étape 10 — Diagnostic si échec** : ne pas rustiner. Identifier le terme de coût ou le poids qui déforme la bande dans les épisodes cassés, corriger la **formulation**, re-benchmark.

---

## 9. Critères d'acceptation numériques

Pour qu'une implémentation TEB hybride remplace le wall-follow P-controller par défaut :

| Métrique | Bug2 actuel (seed 42, 50 ep) | Seuil hybride |
|----------|------------------------------|---------------|
| Succès global | 82 % | ≥ 82 % |
| Succès murs | 78 % | ≥ 78 % |
| Succès hors-murs | 89 % | ≥ 89 % |
| Chutes | 0 | 0 |
| Vitesse moyenne | (à mesurer) | ≥ Bug2 |
| Durée moyenne des succès | (à mesurer) | ≤ Bug2 |

**Règle non négociable** : si une seule métrique régresse, TEB reste optionnel derrière un flag. Bug2 reste la valeur par défaut.

---

## 10. Risques et points de vigilance

1. **Dégradation silencieuse** sur des cas qui marchaient avec Bug2 mais échouent avec TEB. → Toujours comparer seed-à-seed, pas uniquement le score global.
2. **Divergence de l'optimiseur** en cas de grille bruitée. → Timeout dur (ex. 3 itérations max par cycle) + fallback Bug2 silencieux.
3. **Gradient d'obstacle non Lipschitz** près d'un coin. → Lisser le gradient de distance par un kernel gaussien de rayon 1 cellule, ou utiliser la transformée de distance de la grille (déjà calculable en $O(N)$).
4. **Drift du pitch filter** si TEB impose des demi-tours agressifs. → Saturer $v_{\max}$ dans TEB à $0.7 \times v_{\text{balance\_limit}}$.
5. **Over-fitting au seed 42**. → Benchmark final sur 3 seeds indépendants avant de déclarer victoire.

---

## 11. Références

- **Rösmann, C., Feiten, W., Wösch, T., Hoffmann, F., Bertram, T.** (2012). *Trajectory modification considering dynamic constraints of autonomous robots*. ROBOTIK 2012. → Papier fondateur, variables et fonction de coût.
- **Rösmann, C., Hoffmann, F., Bertram, T.** (2015). *Timed-elastic-bands for time-optimal point-to-point nonlinear model predictive control*. ECC 2015. → Lien avec le NMPC, conditions d'optimalité.
- **Rösmann, C., Hoffmann, F., Bertram, T.** (2017). *Integrated online trajectory planning and optimization in distinctive topologies*. Robotics and Autonomous Systems. → Exploration multi-topologie homotopique.
- **Kümmerle, R., Grisetti, G., Strasdat, H., Konolige, K., Burgard, W.** (2011). *g2o: A general framework for graph optimization*. ICRA 2011. → Solveur graphe creux utilisé en pratique.
- Code de référence open-source : `teb_local_planner` (ROS), pour inspiration d'implémentation uniquement, **pas** pour copier-coller.

---

## 12. Décision à prendre

**Trois options** :

- **A — Pas de TEB.** Accepter 82 % comme plafond et diagnostiquer finement les 22 % d'échecs de murs pour voir si une amélioration plus simple existe (meilleur balayage virtuel, plus de capteurs, ajustement de seuils).
- **B — TEB hybride (contour uniquement).** Suivre le plan §8 à la lettre, gate §9, retour à Bug2 si non concluant.
- **C — TEB pur en remplacement de Bug2.** **Déconseillé** — reproduit exactement les conditions de l'échec session 8.

**Recommandation Claude** : **A d'abord** (diagnostic 1-2 h), puis **B seulement si le diagnostic ne révèle pas de cause plus simple**. Jamais C.

Bruno décide.
