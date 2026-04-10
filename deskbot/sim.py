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

from deskbot.robot import SCENES, DEFAULT_SCENE, MODELS_DIR
from deskbot.sensors import SensorModel
from deskbot.control import BalanceController, StateEstimator

# GLFW key codes
KEY_UP = 265
KEY_DOWN = 264
KEY_LEFT = 263
KEY_RIGHT = 262
KEY_SPACE = 32
KEY_S = 83
KEY_R = 82

MOVE_SPEED = 0.5      # m/s when holding forward/back
TURN_SPEED = 1.5      # rad/s when holding left/right
IDLE_TIMEOUT = 0.50   # seconds without key → start decaying (covers OS repeat gap)
DECAY_RATE = 0.93     # per-frame multiplier when decaying (~1s to zero at 60fps)
PUSH_FORCE = 3.0      # N
PUSH_DURATION = 0.08  # seconds


class Commands:
    def __init__(self):
        self.target_velocity = 0.0
        self.target_yaw_rate = 0.0
        self.push_timer = 0.0
        self.push_dir = np.zeros(3)
        self.reset_requested = False
        # Timestamps for auto-decay
        self._last_vel_key = 0.0
        self._last_yaw_key = 0.0


def run(scene_name: str = DEFAULT_SCENE):
    scene_path = SCENES.get(scene_name)
    if scene_path is None:
        available = ", ".join(SCENES.keys())
        raise ValueError(f"Unknown scene '{scene_name}'. Available: {available}")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    sensor_model = SensorModel(model, dt)
    estimator = StateEstimator(dt)
    controller = BalanceController()

    commands = Commands()
    chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

    mujoco.mj_forward(model, data)
    init_qpos = data.qpos.copy()
    init_qvel = data.qvel.copy()

    def handle_key(keycode):
        now = time.perf_counter()
        if keycode == KEY_UP:
            commands.target_velocity = MOVE_SPEED
            commands._last_vel_key = now
        elif keycode == KEY_DOWN:
            commands.target_velocity = -MOVE_SPEED
            commands._last_vel_key = now
        elif keycode == KEY_LEFT:
            commands.target_yaw_rate = TURN_SPEED
            commands._last_yaw_key = now
        elif keycode == KEY_RIGHT:
            commands.target_yaw_rate = -TURN_SPEED
            commands._last_yaw_key = now
        elif keycode == KEY_S:
            commands.target_velocity = 0.0
            commands.target_yaw_rate = 0.0
        elif keycode == KEY_SPACE:
            angle = np.random.uniform(0, 2 * math.pi)
            commands.push_dir = np.array([math.cos(angle), math.sin(angle), 0.0])
            commands.push_timer = PUSH_DURATION
        elif keycode == KEY_R:
            commands.reset_requested = True

    _print_controls()

    with mujoco.viewer.launch_passive(model, data, key_callback=handle_key) as viewer:
        while viewer.is_running():
            frame_start = time.perf_counter()

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
                commands.reset_requested = False

            # ── Auto-decay: ramp to zero when keys released ──
            now = frame_start
            if now - commands._last_vel_key > IDLE_TIMEOUT:
                commands.target_velocity *= DECAY_RATE
                if abs(commands.target_velocity) < 0.01:
                    commands.target_velocity = 0.0
            if now - commands._last_yaw_key > IDLE_TIMEOUT:
                commands.target_yaw_rate *= DECAY_RATE
                if abs(commands.target_yaw_rate) < 0.01:
                    commands.target_yaw_rate = 0.0

            # ── Simulate one display frame (~8 physics steps at 500Hz) ──
            target_sim_time = data.time + 1.0 / 60.0

            while data.time < target_sim_time:
                readings = sensor_model.read(data)
                estimator.update(readings)

                if estimator.fallen:
                    data.ctrl[:] = 0
                else:
                    left, right = controller.compute(
                        estimator,
                        commands.target_velocity,
                        commands.target_yaw_rate,
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

            viewer.sync()

            elapsed = time.perf_counter() - frame_start
            sleep_time = 1.0 / 60.0 - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


def _print_controls():
    print()
    print("  +------------------------------------------+")
    print("  |         DeskBot Simulator                |")
    print("  |   Self-Balancing Two-Wheeled Robot       |")
    print("  +------------------------------------------+")
    print("  |  Hold UP/DOWN    Move forward/backward   |")
    print("  |  Hold LEFT/RIGHT Turn left/right         |")
    print("  |  Release keys    Auto-stop               |")
    print("  |  S               Immediate stop          |")
    print("  |  SPACE           Random push             |")
    print("  |  R               Reset position          |")
    print("  |  ESC             Quit                    |")
    print("  |                                          |")
    print("  |  Mouse: rotate/zoom camera               |")
    print("  +------------------------------------------+")
    print()
