"""
Interactive simulation loop with real-time 3D viewer.

Sensor pipeline: MuJoCo physics -> SensorModel (adds noise) -> StateEstimator -> Controller
The controller never sees simulator internals.
"""
import time
import math
import numpy as np
import mujoco
import mujoco.viewer

from deskbot.robot import SCENES, DEFAULT_SCENE
from deskbot.sensors import SensorModel, RF_NAMES, RF_MAX_RANGE
from deskbot.control import BalanceController, LQRController, StateEstimator
from deskbot.navigation import Navigator
from deskbot.localization import create_apartment_localizer
from deskbot.gui import ControlPanel

MOVE_SPEED = 2.0      # m/s
TURN_SPEED = 3.0      # rad/s
PUSH_FORCE = 3.0      # N
PUSH_DURATION = 0.08  # seconds

# GLFW key codes
KEY_SPACE = 32
KEY_S = 83
KEY_R = 82
KEY_N = 78  # 'N' for Navigate stop


class Commands:
    def __init__(self):
        self.target_velocity = 0.0
        self.target_yaw_rate = 0.0
        self.push_timer = 0.0
        self.push_dir = np.zeros(3)
        self.reset_requested = False
        self._pitch_display = 0.0  # for GUI status readout
        self._rangefinder_display = {}  # for GUI: name -> distance
        # Navigation
        self.navigator = None  # set in run()
        self.nav_heading_request = None  # heading in degrees, set by GUI
        self.nav_stop_request = False


def _distance_color(dist: float) -> np.ndarray:
    """Return RGBA color based on rangefinder distance. Green=far, yellow=mid, red=close."""
    if dist < 0:
        return np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32)  # no hit: dim gray
    if dist < 0.20:
        return np.array([1.0, 0.0, 0.0, 0.9], dtype=np.float32)  # red: danger
    if dist < 0.50:
        t = (dist - 0.20) / 0.30  # 0..1
        return np.array([1.0, t, 0.0, 0.8], dtype=np.float32)    # red→yellow
    if dist < 1.00:
        t = (dist - 0.50) / 0.50  # 0..1
        return np.array([1.0 - t, 1.0, 0.0, 0.7], dtype=np.float32)  # yellow→green
    return np.array([0.0, 1.0, 0.0, 0.5], dtype=np.float32)      # green: safe


def _draw_rangefinders(viewer, model, data, readings):
    """Draw laser beams and hit points using viewer.user_scn."""
    scn = viewer.user_scn
    if scn is None:
        return
    scn.ngeom = 0

    _zeros3 = np.zeros(3)
    _eye9 = np.eye(3).flatten()

    for name in RF_NAMES:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            continue

        # Get world-space origin and ray direction (site +Z axis)
        origin = data.site_xpos[site_id].copy()
        site_mat = data.site_xmat[site_id].reshape(3, 3)
        direction = site_mat[:, 2]  # z-axis column

        dist = readings.rangefinders.get(name, -1.0)
        color = _distance_color(dist)

        # Compute end point
        if dist < 0:
            end = origin + direction * RF_MAX_RANGE
        else:
            end = origin + direction * dist

        # Draw beam line (thin capsule)
        if scn.ngeom < scn.maxgeom:
            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                _zeros3, _zeros3, _eye9, color,
            )
            mujoco.mjv_connector(
                geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                0.002,  # 2mm radius beam
                origin, end,
            )
            scn.ngeom += 1

        # Draw hit point sphere (only if there was a hit)
        if dist >= 0 and scn.ngeom < scn.maxgeom:
            hit_color = np.array([1.0, 0.2, 0.2, 0.9], dtype=np.float32)
            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.008, 0.008, 0.008]),  # 8mm radius sphere
                end.copy(),
                _eye9, hit_color,
            )
            scn.ngeom += 1


def run(scene_name: str = DEFAULT_SCENE, scene_xml: str | None = None,
        use_pid: bool = False, random_gen: dict | None = None):
    """
    Main simulation loop.

    Args:
        scene_name: Named scene to load from SCENES dict.
        scene_xml: Raw XML string (overrides scene_name).
        use_pid: Use legacy PID controller instead of LQR.
        random_gen: If provided, regenerate obstacles on each reset.
            Dict with keys: corridor (float), n_obs (int or None), seed_counter (list[int]).
    """
    if scene_xml is not None:
        model = mujoco.MjModel.from_xml_string(scene_xml)
    else:
        scene_path = SCENES.get(scene_name)
        if scene_path is None:
            available = ", ".join(SCENES.keys())
            raise ValueError(f"Unknown scene '{scene_name}'. Available: {available}")
        model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    sensor_model = SensorModel(model, dt)
    estimator = StateEstimator(dt)
    if use_pid:
        controller = BalanceController()
        print("  Controller: PID (legacy)")
    else:
        controller = LQRController(mj_model=model)
        print("  Controller: LQR (optimal)")
    navigator = Navigator(dt, mj_model=model)

    # Localization (WiFi + heading correction)
    if scene_name == "apartment":
        wifi_localizer, heading_corrector = create_apartment_localizer()
        print("  Localization: WiFi RSSI + wall-geometry heading correction")
    else:
        wifi_localizer, heading_corrector = None, None

    commands = Commands()
    commands.navigator = navigator
    chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

    mujoco.mj_forward(model, data)
    init_qpos = data.qpos.copy()
    init_qvel = data.qvel.copy()

    # Mutable flag: when random_gen is used, reset triggers a full reload
    reload_requested = [False]

    def handle_key(keycode):
        if keycode == KEY_S:
            commands.target_velocity = 0.0
            commands.target_yaw_rate = 0.0
        elif keycode == KEY_SPACE:
            angle = np.random.uniform(0, 2 * math.pi)
            commands.push_dir = np.array([math.cos(angle), math.sin(angle), 0.0])
            commands.push_timer = PUSH_DURATION
        elif keycode == KEY_R:
            if random_gen is not None:
                reload_requested[0] = True  # will close viewer and relaunch
            else:
                commands.reset_requested = True
        elif keycode == KEY_N:
            commands.nav_stop_request = True

    # Launch GUI control panel
    panel = ControlPanel(commands, MOVE_SPEED, TURN_SPEED)
    panel.start()

    print()
    print("  DeskBot Simulator")
    print("  Control panel opened in separate window.")
    if random_gen:
        print("  Keyboard: S=stop, SPACE=push, R=NEW TERRAIN, N=stop nav, ESC=quit")
    else:
        print("  Keyboard: S=stop, SPACE=push, R=reset, N=stop nav, ESC=quit")
    print()

    with mujoco.viewer.launch_passive(model, data, key_callback=handle_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = data.xpos[chassis_id]
        viewer.cam.distance = 1.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25

        while viewer.is_running():
            frame_start = time.perf_counter()

            # ── Random terrain reload ──
            if reload_requested[0]:
                reload_requested[0] = False
                viewer.close()
                break  # exit viewer loop, will relaunch below

            # ── Handle reset ──
            if commands.reset_requested:
                mujoco.mj_resetData(model, data)
                data.qpos[:] = init_qpos
                data.qvel[:] = init_qvel
                mujoco.mj_forward(model, data)
                commands.target_velocity = 0.0
                commands.target_yaw_rate = 0.0
                commands.push_timer = 0.0
                commands.push_dir[:] = 0
                sensor_model.reset()
                estimator.reset()
                controller.reset()
                navigator.reset()
                if wifi_localizer:
                    wifi_localizer.reset()
                if heading_corrector:
                    heading_corrector.reset()
                commands.reset_requested = False

            # ── Simulate one display frame (~8 physics steps at 500Hz) ──
            target_sim_time = data.time + 1.0 / 60.0

            # ── Handle navigation requests from GUI ──
            if commands.nav_heading_request is not None:
                navigator.set_heading(commands.nav_heading_request)
                commands.nav_heading_request = None
            if commands.nav_stop_request:
                navigator.stop()
                commands.nav_stop_request = False

            while data.time < target_sim_time:
                readings = sensor_model.read(data)
                estimator.update(readings)

                # Localization: WiFi scan + heading correction
                if wifi_localizer is not None:
                    scan = wifi_localizer.scan(
                        navigator._pos_x, navigator._pos_y, dt)
                    if scan is not None:
                        wx, wy, conf = wifi_localizer.estimate_position(scan)
                        # WiFi position available for future use (SLAM fusion)

                if heading_corrector is not None:
                    rf_comp = navigator.compensate_rangefinders(
                        readings.rangefinders, estimator.pitch)
                    correction, valid = heading_corrector.update(
                        navigator._heading, rf_comp)
                    # Apply small heading corrections to navigator's DR
                    if valid:
                        navigator._heading += correction * dt

                # Navigation AI: overrides joystick when active
                nav_vel, nav_yaw = navigator.update(estimator, readings, dt)
                if nav_vel is not None:
                    cmd_vel = nav_vel
                    cmd_yaw = nav_yaw
                else:
                    cmd_vel = commands.target_velocity
                    cmd_yaw = commands.target_yaw_rate

                if estimator.fallen:
                    data.ctrl[:] = 0
                else:
                    left, right = controller.compute(
                        estimator,
                        cmd_vel,
                        cmd_yaw,
                        dt,
                    )
                    data.ctrl[0] = left
                    data.ctrl[1] = right

                if commands.push_timer > 0:
                    data.xfrc_applied[chassis_id, :3] = commands.push_dir * PUSH_FORCE
                    commands.push_timer -= dt
                    if commands.push_timer <= 0:
                        data.xfrc_applied[chassis_id, :3] = 0
                        commands.push_timer = 0

                mujoco.mj_step(model, data)

            # Feed estimator state to GUI for display
            commands._pitch_display = estimator.pitch
            commands._rangefinder_display = readings.rangefinders.copy()

            # ── Draw rangefinder beams ──
            _draw_rangefinders(viewer, model, data, readings)

            # ── Camera follows robot ──
            viewer.cam.lookat[:] = data.xpos[chassis_id]

            viewer.sync()

            elapsed = time.perf_counter() - frame_start
            sleep_time = 1.0 / 60.0 - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ── If reload was requested (random terrain), generate new scene and relaunch ──
    if random_gen is not None and reload_requested[0] is False:
        # Viewer was closed via reload_requested → generate new terrain
        return "reload"
    return "quit"


def run_random_loop(corridor: float = 1.5, n_obs: int | None = None,
                    base_seed: int = 0, use_pid: bool = False):
    """
    Run the simulator in a loop, generating a new random terrain on each reset.
    """
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "scripts"))
    from benchmark_random import random_obstacles, generate_scene_xml

    seed = base_seed
    while True:
        rng = np.random.default_rng(seed)
        obs_count = n_obs if n_obs else int(rng.integers(2, 6))
        obstacles = random_obstacles(rng, obs_count, corridor)

        obs_desc = ", ".join(f"{o['type']}@X={o['x']:.1f}" for o in obstacles)
        print(f"\n  Random terrain #{seed}: {len(obstacles)} obstacles [{obs_desc}]")

        xml = generate_scene_xml(obstacles, corridor)
        random_gen = {"corridor": corridor, "n_obs": n_obs}
        result = run(scene_xml=xml, use_pid=use_pid, random_gen=random_gen)

        if result != "reload":
            break  # user closed the window with ESC
        seed += 1
