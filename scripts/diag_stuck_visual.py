"""
Diagnostic visuel du comportement de navigation.
Génère des snapshots de la grille d'occupation pendant la simulation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deskbot.sim import run_headless_episode
from deskbot.navigation import Navigator, GRID_SIZE, GRID_RES, LOG_ODD_OCCUPIED_THRESHOLD


def save_grid_snapshot(navigator, episode, timestep, output_dir="snapshots"):
    """Sauvegarde une image de la grille d'occupation."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Grille centrée sur le robot
    grid = navigator.grid.grid.copy()
    
    # Afficher les obstacles (rouge), libre (vert), inconnu (gris)
    extent = [-GRID_SIZE*GRID_RES/2, GRID_SIZE*GRID_RES/2,
              -GRID_SIZE*GRID_RES/2, GRID_SIZE*GRID_RES/2]
    
    # Color map: bleu foncé = occupé, blanc = inconnu, vert clair = libre
    im = ax.imshow(grid.T, origin='lower', extent=extent, 
                   cmap='RdYlGn', vmin=-2, vmax=2, interpolation='nearest')
    
    # Position du robot (centre)
    ax.plot(0, 0, 'bo', markersize=15, label='Robot')
    
    # Direction du heading
    heading = navigator._heading
    ax.arrow(0, 0, 0.3*math.cos(heading), 0.3*math.sin(heading),
             head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    # Target heading
    target = navigator._target_heading
    ax.arrow(0, 0, 0.5*math.cos(target), 0.5*math.sin(target),
             head_width=0.05, head_length=0.05, fc='cyan', ec='cyan', 
             linestyle='--', alpha=0.7, label='Target')
    
    # Capteurs
    rf_angles = {
        "rf_FC": 0.0, "rf_FL": math.radians(30), "rf_FR": math.radians(-30),
        "rf_FL2": math.radians(55), "rf_FR2": math.radians(-55),
        "rf_SL": math.radians(90), "rf_SR": math.radians(-90),
    }
    
    for name, dist in navigator._rf_compensated.items():
        angle = rf_angles.get(name, 0) + heading
        if dist > 0:
            x = dist * math.cos(angle)
            y = dist * math.sin(angle)
            ax.plot([0, x], [0, y], 'g-', alpha=0.5, linewidth=1)
            ax.plot(x, y, 'g.', markersize=8)
        else:
            # Pas de détection - rayon max
            max_range = 2.0
            x = max_range * math.cos(angle)
            y = max_range * math.sin(angle)
            ax.plot([0, x], [0, y], 'r--', alpha=0.3, linewidth=0.5)
    
    # Cercles de distance
    for r in [0.36, 0.65, 1.0]:
        circle = plt.Circle((0, 0), r, fill=False, linestyle='--', alpha=0.3)
        ax.add_patch(circle)
        ax.text(r, 0, f'{r}m', fontsize=8, alpha=0.5)
    
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.4, 2.4)
    ax.set_aspect('equal')
    ax.set_title(f"Episode {episode} - t={timestep:.1f}s\n"
                 f"FSM={navigator._fsm.value}, "
                 f"Front={navigator._min_front_dist:.2f}m")
    ax.legend(loc='upper left')
    
    filename = f"{output_dir}/diag_ep{episode}_t{timestep:05.1f}.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def run_with_snapshots(seed=8147, max_time=30.0, snapshot_interval=2.0):
    """Exécute un épisode avec des snapshots réguliers."""
    import random
    from deskbot.robot import SCENES
    
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC VISUEL - Seed {seed}")
    print(f"{'='*60}\n")
    
    # Générer la scène aléatoire
    random.seed(seed)
    np.random.seed(seed)
    
    num_obstacles = random.randint(2, 5)
    obstacles = []
    
    corridor_half_width = 1.5
    
    for i in range(num_obstacles):
        x = 2.5 + i * (9.0 / max(num_obstacles, 1)) + random.uniform(-0.5, 0.5)
        x = max(1.5, min(11.0, x))
        
        obs_type = random.choice(["box", "cylinder", "wall"])
        
        if obs_type == "wall":
            gap_y = random.uniform(-0.5, 0.5)
            obstacles.append(("wall", x, gap_y))
        else:
            y = random.uniform(-corridor_half_width + 0.3, corridor_half_width - 0.3)
            if random.random() < 0.3:
                y = corridor_half_width - 0.2 if random.random() < 0.5 else -corridor_half_width + 0.2
            obstacles.append((obs_type, x, y))
    
    print(f"Obstacles: {obstacles}")
    
    # Créer le modèle MuJoCo temporaire
    from deskbot.sim import create_random_scene_xml
    xml = create_random_scene_xml(obstacles, corridor_half_width)
    
    with open("_diag_scene.xml", "w") as f:
        f.write(xml)
    
    # Charger le modèle
    model = mujoco.MjModel.from_xml_path("_diag_scene.xml")
    data = mujoco.MjData(model)
    
    # Initialiser le navigateur
    navigator = Navigator(dt=0.002, mj_model=model, use_astar=False, use_early_avoid=False)
    navigator.set_heading(0.0)
    
    # Exécution avec snapshots
    from deskbot.sensors import SensorModel
    from deskbot.control import StateEstimator, LQRController
    
    sensors = SensorModel(model, dt=0.002)
    estimator = StateEstimator()
    controller = LQRController(model)
    
    mujoco.mj_resetData(model, data)
    sensors.reset()
    estimator.reset()
    controller.reset()
    navigator.reset()
    navigator.set_heading(0.0)
    
    # Position initiale
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = 0.11
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    
    next_snapshot = 0.0
    step = 0
    
    while data.time < max_time:
        # Physics step
        mujoco.mj_step(model, data)
        step += 1
        
        if step % 10 == 0:  # 500 Hz -> 50 Hz control
            readings = sensors.read(data)
            estimator.update(readings, 0.002 * 10)
            
            if not estimator.fallen:
                vel_cmd, yaw_cmd = navigator.update(estimator, readings, 0.002 * 10)
                if vel_cmd is not None:
                    torque_L, torque_R = controller.update(estimator, vel_cmd, yaw_cmd)
                    data.ctrl[0] = torque_L
                    data.ctrl[1] = torque_R
            else:
                print(f"Fallen at t={data.time:.1f}s")
                break
        
        # Snapshot
        if data.time >= next_snapshot:
            save_grid_snapshot(navigator, seed, data.time)
            next_snapshot += snapshot_interval
        
        # Check success
        if data.qpos[0] >= 12.0:
            print(f"SUCCESS at t={data.time:.1f}s")
            save_grid_snapshot(navigator, seed, data.time)
            break
    else:
        print(f"TIMEOUT at X={data.qpos[0]:.1f}m")
        save_grid_snapshot(navigator, seed, data.time)
    
    # Cleanup
    os.remove("_diag_scene.xml")
    
    return data.qpos[0] >= 12.0


if __name__ == "__main__":
    # Tester les seeds qui échouent
    failing_seeds = [8147, 8161, 8164, 8127]
    
    for seed in failing_seeds[:2]:  # Limiter à 2 pour ne pas trop ralentir
        success = run_with_snapshots(seed=seed, max_time=30.0)
        print(f"\nSeed {seed}: {'SUCCESS' if success else 'FAIL'}\n")
