"""
Diagnostic script for early avoidance side selection.

Runs one benchmark episode with early-avoid on, logs every tick where
the side cache transitions or where the bias is actively applied,
and prints a compact trace so we can see whether the virtual scan
is picking a sensible side at long range.

ASCII-only output for cp1252 consoles.
"""
import sys
import math
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import mujoco

from benchmark_random import random_obstacles, generate_scene_xml
from deskbot.sensors import SensorModel
from deskbot.control import BalanceController, StateEstimator
from deskbot.navigation import Navigator
from deskbot.early_avoidance import compute_bias_yaw, EARLY_PARAMS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--early-avoid", choices=["off", "on"], default="on")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    n_obs = int(rng.integers(2, 6))
    obstacles = random_obstacles(rng, n_obs, 1.5)
    xml = generate_scene_xml(obstacles, 1.5)

    print(f"seed={args.seed} n_obs={n_obs} early_avoid={args.early_avoid}")
    for i, o in enumerate(obstacles):
        print(f"  obs{i}: {o}")

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    sensor_model = SensorModel(model, dt)
    estimator = StateEstimator(dt)
    controller = BalanceController()
    nav = Navigator(dt, mj_model=model, use_astar=True,
                    use_early_avoid=(args.early_avoid == "on"))

    mujoco.mj_forward(model, data)
    for _ in range(int(1.0 / dt)):
        readings = sensor_model.read(data)
        estimator.update(readings)
        if not estimator.fallen:
            left, right = controller.compute(estimator, 0.0, 0.0, dt)
            data.ctrl[0] = left
            data.ctrl[1] = right
        mujoco.mj_step(model, data)

    nav.set_heading(0.0)
    print()
    print("tick  t    x     y     h      d_front fsm           side bias    "
          "rf_FL rf_C  rf_FR")
    print("-" * 98)

    sim_t = 0.0
    prev_side = None
    prev_fsm = None
    log_every = 0.2
    next_log = 0.0
    while sim_t < args.duration:
        readings = sensor_model.read(data)
        estimator.update(readings)
        if estimator.fallen:
            print("FELL")
            break
        vel_cmd, yaw_cmd = nav.update(estimator, readings, dt)
        if vel_cmd is None:
            break
        left, right = controller.compute(estimator, vel_cmd, yaw_cmd, dt)
        data.ctrl[0] = left
        data.ctrl[1] = right
        mujoco.mj_step(model, data)
        sim_t += dt

        side = nav._early_cache.side
        fsm = nav._fsm.value
        bias = 0.0
        if nav._use_early_avoid and side is not None:
            bias = compute_bias_yaw(nav._min_front_dist, side, EARLY_PARAMS)

        event = (side != prev_side) or (fsm != prev_fsm)
        if event or sim_t >= next_log:
            rf_FL = nav._rf_compensated.get("rf_FL", -1.0)
            rf_FC = nav._rf_compensated.get("rf_C", -1.0)
            rf_FR = nav._rf_compensated.get("rf_FR", -1.0)
            side_s = "--" if side is None else f"{side:+d}"
            print(f"{int(sim_t/dt):4d} {sim_t:5.2f} "
                  f"{nav._pos_x:5.2f} {nav._pos_y:+5.2f} "
                  f"{math.degrees(nav._heading):+5.0f}  "
                  f"{nav._min_front_dist:6.2f} "
                  f"{fsm:12s} {side_s:>4s} {bias:+5.2f}  "
                  f"{rf_FL:5.2f} {rf_FC:5.2f} {rf_FR:5.2f}")
            next_log = sim_t + log_every
        prev_side = side
        prev_fsm = fsm

        if data.qpos[0] >= 12.0:
            break

    print()
    print(f"final x={float(data.qpos[0]):.2f} y={float(data.qpos[1]):+.2f}")


if __name__ == "__main__":
    main()
