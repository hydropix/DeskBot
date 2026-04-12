"""
Test comparatif: Navigation fluide (Potential Field) vs Bug2.
Visualise le comportement "eau qui contourne" demandé.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import mujoco

# Test de navigation fluide - comparatif
from deskbot.navigation import Navigator
from deskbot.field_nav import FieldNavigator, FieldParams
from deskbot.sensors import SensorModel
from deskbot.control import StateEstimator, LQRController


def create_simple_scene_with_wall():
    """Crée une scène simple: mur perpendiculaire devant."""
    # Chemin absolu vers le robot
    import deskbot
    robot_path = os.path.join(os.path.dirname(deskbot.__file__), 'models', 'deskbot.xml')
    
    return f"""<mujoco model="test_scene">
  <include file="{robot_path}"/>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" rgba="0.9 0.9 0.9 1"/>
    <light directional="true" diffuse="0.8 0.8 0.8" pos="0 0 5"/>
    
    <!-- Mur perpendiculaire à X=5 -->
    <geom name="wall" type="box" size="0.1 2.0 1.0" pos="5.0 0 1.0" rgba="0.3 0.3 0.3 1"/>
    
    <!-- Ouverture à droite -->
    <geom name="wall_top" type="box" size="0.1 0.8 1.0" pos="5.0 1.2 1.0" rgba="0.3 0.3 0.3 1"/>
    <geom name="wall_bottom" type="box" size="0.1 0.8 1.0" pos="5.0 -1.2 1.0" rgba="0.3 0.3 0.3 1"/>
  </worldbody>
</mujoco>"""


def create_chicane_scene():
    """Scène en chicane: deux obstacles successifs."""
    import deskbot
    robot_path = os.path.join(os.path.dirname(deskbot.__file__), 'models', 'deskbot.xml')
    
    return f"""<mujoco model="test_scene">
  <include file="{robot_path}"/>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" rgba="0.9 0.9 0.9 1"/>
    <light directional="true" diffuse="0.8 0.8 0.8" pos="0 0 5"/>
    
    <!-- Chicane: obstacle à gauche puis à droite -->
    <geom name="obs1" type="box" size="0.3 0.5 0.5" pos="3.5 0.8 0.5" rgba="0.8 0.2 0.2 1"/>
    <geom name="obs2" type="box" size="0.3 0.5 0.5" pos="5.5 -0.8 0.5" rgba="0.2 0.8 0.2 1"/>
    <geom name="obs3" type="box" size="0.3 0.5 0.5" pos="7.5 0.8 0.5" rgba="0.2 0.2 0.8 1"/>
  </worldbody>
</mujoco>"""


def run_with_field_nav(scene_xml, max_time=30.0):
    """Exécute avec la navigation fluide."""
    with open("_test_scene.xml", "w") as f:
        f.write(scene_xml)
    
    model = mujoco.MjModel.from_xml_path("_test_scene.xml")
    data = mujoco.MjData(model)
    
    # Init
    sensors = SensorModel(model, dt=0.002)
    estimator = StateEstimator(dt=0.002)
    controller = LQRController(model)
    navigator = FieldNavigator(dt=0.02, params=FieldParams(
        k_repulse=1.0,      # Plus agressif
        d_threshold=2.0,    # Détecte plus loin
        k_attract=1.5,
    ))
    navigator.set_heading(0.0)
    
    mujoco.mj_resetData(model, data)
    sensors.reset()
    estimator.reset()
    controller.reset()
    
    # Pose initiale
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = 0.11
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    
    # Dead reckoning heading
    heading = 0.0
    
    positions = []
    
    while data.time < max_time:
        mujoco.mj_step(model, data)
        
        if int(data.time * 500) % 10 == 0:  # 50 Hz
            readings = sensors.read(data)
            estimator.update(readings)
            
            # Update heading from gyro
            heading += estimator.yaw_rate_gyro * 0.02
            heading = math.atan2(math.sin(heading), math.cos(heading))
            
            if not estimator.fallen:
                vel_cmd, yaw_cmd = navigator.update(
                    readings.rangefinders,
                    heading,
                    estimator.forward_vel
                )
                torque_L, torque_R = controller.compute(estimator, vel_cmd, yaw_cmd, 0.02)
                data.ctrl[0] = torque_L
                data.ctrl[1] = torque_R
                
                if int(data.time * 10) % 10 == 0:  # Log every second
                    positions.append((data.time, data.qpos[0], data.qpos[1], vel_cmd, yaw_cmd))
            else:
                print(f"  [FIELD] Fallen at t={data.time:.1f}s")
                break
        
        if data.qpos[0] >= 10.0:
            print(f"  [FIELD] Success at t={data.time:.1f}s")
            break
    else:
        print(f"  [FIELD] Timeout at X={data.qpos[0]:.1f}m")
    
    os.remove("_test_scene.xml")
    return positions, data.qpos[0] >= 10.0


def run_with_bug2(scene_xml, max_time=30.0):
    """Exécute avec Bug2 standard."""
    with open("_test_scene.xml", "w") as f:
        f.write(scene_xml)
    
    model = mujoco.MjModel.from_xml_path("_test_scene.xml")
    data = mujoco.MjData(model)
    
    # Init
    sensors = SensorModel(model, dt=0.002)
    estimator = StateEstimator(dt=0.002)
    controller = LQRController(model)
    navigator = Navigator(dt=0.002, mj_model=model, use_astar=False, use_early_avoid=False)
    navigator.set_heading(0.0)
    
    mujoco.mj_resetData(model, data)
    sensors.reset()
    estimator.reset()
    controller.reset()
    navigator.reset()
    navigator.set_heading(0.0)
    
    # Pose initiale
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = 0.11
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    
    positions = []
    
    while data.time < max_time:
        mujoco.mj_step(model, data)
        
        if int(data.time * 500) % 10 == 0:  # 50 Hz
            readings = sensors.read(data)
            estimator.update(readings)
            
            if not estimator.fallen:
                vel_cmd, yaw_cmd = navigator.update(estimator, readings, 0.02)
                if vel_cmd is not None:
                    torque_L, torque_R = controller.compute(estimator, vel_cmd, yaw_cmd, 0.02)
                    data.ctrl[0] = torque_L
                    data.ctrl[1] = torque_R
                
                if int(data.time * 10) % 10 == 0:  # Log every second
                    fsm = navigator._fsm.value if hasattr(navigator, '_fsm') else "?"
                    positions.append((data.time, data.qpos[0], data.qpos[1], fsm))
            else:
                print(f"  [BUG2] Fallen at t={data.time:.1f}s")
                break
        
        if data.qpos[0] >= 10.0:
            print(f"  [BUG2] Success at t={data.time:.1f}s")
            break
    else:
        print(f"  [BUG2] Timeout at X={data.qpos[0]:.1f}m")
    
    os.remove("_test_scene.xml")
    return positions, data.qpos[0] >= 10.0


def main():
    print("=" * 70)
    print("COMPARAISON: Navigation Fluide (Potential Field) vs Bug2")
    print("=" * 70)
    
    # Test 1: Mur perpendiculaire avec ouverture
    print("\n[Test 1] Mur perpendiculaire avec ouverture à droite")
    print("-" * 50)
    
    scene1 = create_simple_scene_with_wall()
    
    print("  FieldNav (approche fluide):")
    pos_field, success_field = run_with_field_nav(scene1)
    
    print("  Bug2 (approche FSM):")
    pos_bug2, success_bug2 = run_with_bug2(scene1)
    
    print(f"\n  Résultat: Field={success_field}, Bug2={success_bug2}")
    
    # Test 2: Chicane
    print("\n[Test 2] Chicane (3 obstacles alternés)")
    print("-" * 50)
    
    scene2 = create_chicane_scene()
    
    print("  FieldNav:")
    pos_field2, success_field2 = run_with_field_nav(scene2)
    
    print("  Bug2:")
    pos_bug2_2, success_bug2_2 = run_with_bug2(scene2)
    
    print(f"\n  Résultat: Field={success_field2}, Bug2={success_bug2_2}")
    
    # Résumé
    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"""
Approche Fluide (FieldNav):
  - Avantage: Comportement continu, "coule" autour des obstacles
  - Inconvénient: Peut rester coincé dans des minima locaux
  - Résultats: Test1={'✓' if success_field else '✗'}, Test2={'✓' if success_field2 else '✗'}

Approche Bug2:
  - Avantage: Garanties de complétude (si espace fini)
  - Inconvénient: À-coups (marche/arrêt), oscillations
  - Résultats: Test1={'✓' if success_bug2 else '✗'}, Test2={'✓' if success_bug2_2 else '✗'}
""")
    
    print("\nLa navigation fluide est disponible via:")
    print("  from deskbot.field_nav import FieldNavigator")
    print("  nav = FieldNavigator(dt=0.02)")
    print("  nav.set_heading(0)")
    print("  vel, yaw = nav.update(rangefinders, current_heading, forward_vel)")


if __name__ == "__main__":
    main()
