"""
Navigation par champ de potentiel (Potential Field) — Approche "fluide".

Le robot est repoussé par les obstacles via des forces virtuelles 
proportionnelles à l'inverse de la distance des lasers.
Comportement type "eau qui contourne" plutôt que FSM discret.
"""
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class FieldParams:
    """Paramètres du champ de potentiel."""
    # Force de répulsion: F = k_repulse / (d - d_min) pour d < d_threshold
    k_repulse: float = 0.8        # Gain de répulsion (rad/s par mètre)
    d_threshold: float = 1.5      # Distance d'activation (m)
    d_min: float = 0.25           # Distance minimale (évite singularité)
    
    # Force d'attraction vers le heading cible
    k_attract: float = 2.0        # Gain d'attraction (rad/s par rad)
    
    # Perpendiculaire: lorsque l'obstacle est exactement frontal,
    # on applique un "nudge" latéral aléatoire pour choisir un côté
    nudge_gain: float = 0.3       # Amplitude du nudge (rad/s)
    nudge_decay: float = 0.95     # Décroissance du nudge (mémorise le choix)


# Angles des lasers dans le repère robot (radians)
RF_ANGLES = {
    "rf_FC":  0.0,
    "rf_FL":  math.radians(30),
    "rf_FR":  math.radians(-30),
    "rf_FL2": math.radians(55),
    "rf_FR2": math.radians(-55),
    "rf_SL":  math.radians(90),
    "rf_SR":  math.radians(-90),
}

# Poids relatifs des capteurs (frontaux plus importants pour l'évitement)
RF_WEIGHTS = {
    "rf_FC":  2.0,   # Front central: priorité max
    "rf_FL":  1.5,
    "rf_FR":  1.5,
    "rf_FL2": 1.0,
    "rf_FR2": 1.0,
    "rf_SL":  0.5,   # Latéraux: moins prioritaires
    "rf_SR":  0.5,
}


class FieldNavigator:
    """
    Navigateur à champ de potentiel - comportement fluide type "eau".
    
    Pas de FSM, pas d'états. Juste une somme de forces:
    - Attraction vers le heading cible
    - Répulsion des obstacles détectés par lasers
    - Nudge latéral pour casser les symétries (choix de côté)
    """
    
    def __init__(self, dt: float, params: FieldParams = None):
        self.dt = dt
        self.params = params or FieldParams()
        
        # État persistent
        self._target_heading = 0.0
        self._active = False
        self._nudge_state = 0.0  # Mémorise le dernier nudge (décroissant)
        
        # Pour debug/visualisation
        self._last_repulsion = np.zeros(2)
        self._last_attraction = np.zeros(2)
        
        # Pour compatibilité GUI (simule NavState)
        from deskbot.navigation import NavState
        self.state = NavState()
        self.state.behavior = "field_nav"
        
    def set_heading(self, heading_deg: float):
        """Définit le heading cible (degrés, 0=+X)."""
        self._target_heading = math.radians(heading_deg)
        self._active = True
        self._nudge_state = 0.0
        self.state.active = True
        self.state.target_heading = self._target_heading
        
    def stop(self):
        self._active = False
        self.state.active = False
        
    def reset(self):
        self._active = False
        self._nudge_state = 0.0
        self._target_heading = 0.0
        self.state.active = False
        self.state.target_heading = 0.0
        
    def _compute_repulsion(self, rf: dict) -> np.ndarray:
        """
        Calcule la force de répulsion totale des obstacles.
        Retourne un vecteur [fx, fy] dans le repère robot.
        """
        force = np.zeros(2)
        
        for name, dist in rf.items():
            if dist <= 0 or dist > self.params.d_threshold:
                continue  # Pas d'obstacle ou trop loin
                
            angle = RF_ANGLES.get(name, 0.0)
            weight = RF_WEIGHTS.get(name, 1.0)
            
            # Distance effective (évite singularité à d=0)
            d_eff = max(dist, self.params.d_min)
            
            # Force répulsive: inverse de la distance
            # Plus proche = force plus forte
            f_mag = self.params.k_repulse * weight / (d_eff - self.params.d_min + 0.1)
            
            # Direction: opposée au capteur (s'éloigner de l'obstacle)
            fx = -f_mag * math.cos(angle)
            fy = -f_mag * math.sin(angle)
            
            force += np.array([fx, fy])
            
        return force
        
    def _compute_attraction(self, current_heading: float) -> np.ndarray:
        """
        Force d'attraction vers le heading cible.
        """
        heading_error = self._wrap_angle(self._target_heading - current_heading)
        
        # Attraction proportionnelle à l'erreur
        f_mag = self.params.k_attract * heading_error
        
        # Direction: suivant le heading cible
        return np.array([
            f_mag * math.cos(self._target_heading),
            f_mag * math.sin(self._target_heading)
        ])
        
    def _handle_perpendicular(self, rf: dict, current_heading: float) -> float:
        """
        Gère le cas perpendiculaire (obstacle exactement frontal).
        Ajoute un "nudge" latéral aléatoire qui persiste pour éviter
        l'hésitation entre gauche et droite.
        """
        fc_dist = rf.get("rf_FC", -1)
        
        # Si obstacle frontal proche et pas de latéraux détectés
        if 0 < fc_dist < 0.8:
            fl_dist = rf.get("rf_FL", 999)
            fr_dist = rf.get("rf_FR", 999)
            
            # Symétrie: choisir un côté si pas évident
            if abs(fl_dist - fr_dist) < 0.3:
                # Pas de préférence claire -> activer le nudge
                if abs(self._nudge_state) < 0.1:
                    # Choisir aléatoirement un côté
                    self._nudge_state = self.params.nudge_gain if (fc_dist * 100) % 2 > 1 else -self.params.nudge_gain
                    
        # Décroissance du nudge
        self._nudge_state *= self.params.nudge_decay
        
        return self._nudge_state
        
    @staticmethod
    def _wrap_angle(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))
        
    def update(self, rf: dict, current_heading: float, forward_vel: float) -> tuple[float, float]:
        """
        Met à jour la navigation et retourne (vel_cmd, yaw_cmd).
        
        Args:
            rf: Dict des rangefinders {name: distance}
            current_heading: Heading actuel du robot (radians)
            forward_vel: Vitesse actuelle (m/s) - pour l'échelle de temps
            
        Returns:
            (vel_cmd, yaw_cmd): Commandes de vitesse et yaw
        """
        if not self._active:
            self.state.active = False
            return 0.0, 0.0
        
        # Update state for GUI
        self.state.active = True
        self.state.target_heading = self._target_heading
            
        # 1. Forces de répulsion des obstacles
        repulsion = self._compute_repulsion(rf)
        self._last_repulsion = repulsion.copy()
        
        # 2. Force d'attraction vers le heading
        attraction = self._compute_attraction(current_heading)
        self._last_attraction = attraction.copy()
        
        # 3. Nudge pour casser les symétries (perpendiculaire)
        nudge = self._handle_perpendicular(rf, current_heading)
        
        # 4. Somme des forces
        # On veut le yaw_rate, donc on projette sur l'axe perpendiculaire
        # au mouvement (rotation du vecteur force en commande angulaire)
        
        total_force = repulsion + attraction
        
        # Convertir la force en commande de yaw
        # La composante Y de la force (latérale) crée une rotation
        # La composante X modifie la vitesse (ralentit si obstacle frontal)
        
        # Yaw = composante latérale de la force + nudge
        yaw_cmd = total_force[1] + nudge
        
        # Vitesse: réduite si obstacle frontal
        fc_dist = rf.get("rf_FC", 999)
        vel_scale = 1.0
        if 0 < fc_dist < 1.0:
            vel_scale = min(1.0, fc_dist / 0.6)  # Ralentit sous 60cm
            
        vel_cmd = 0.5 * vel_scale  # Vitesse max 0.5 m/s
        
        # Limites
        yaw_cmd = float(np.clip(yaw_cmd, -2.0, 2.0))
        vel_cmd = float(np.clip(vel_cmd, 0.0, 0.5))
        
        return vel_cmd, yaw_cmd
        
    def get_debug_info(self) -> dict:
        """Retourne les infos de debug (forces)."""
        return {
            "repulsion": self._last_repulsion.copy(),
            "attraction": self._last_attraction.copy(),
            "nudge": self._nudge_state,
        }
