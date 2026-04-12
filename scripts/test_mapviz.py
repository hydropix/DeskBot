"""
Headless end-to-end test for the occupancy map visualization.

Loads a real scene, runs the navigator for a few seconds, and dumps a
PNG snapshot of the live occupancy map. The test is designed so Claude
can Read the resulting PNG without needing the tkinter viewer.

Usage:
    python scripts/test_mapviz.py                 # nav_test scene, 5 s
    python scripts/test_mapviz.py --scene random  # random corridor
    python scripts/test_mapviz.py --duration 10   # longer run
"""
import argparse
import sys
import os
import time
import math
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deskbot.robot import MODELS_DIR, SCENES
from deskbot.sensors import SensorModel
from deskbot.control import LQRController, StateEstimator
from deskbot.navigation import Navigator
from deskbot.mapviz import (
    MapFrame, render_rgb, save_png, extract_gt_obstacles,
    default_snapshot_dir,
)


def run(scene_name: str, duration: float, heading_deg: float,
        snapshot_every: float, out_tag: str):
    if scene_name == "random":
        # Import on demand to avoid hard dependency in the headless path
        from benchmark_random import random_obstacles, generate_scene_xml
        rng = np.random.default_rng(42)
        obstacles = random_obstacles(rng, 3, 1.5)
        xml = generate_scene_xml(obstacles, 1.5)
        model = mujoco.MjModel.from_xml_string(xml)
    else:
        scene_path = SCENES.get(scene_name)
        if scene_path is None:
            raise ValueError(f"Unknown scene: {scene_name}")
        model = mujoco.MjModel.from_xml_path(str(scene_path))

    data = mujoco.MjData(model)
    dt = model.opt.timestep

    sensor_model = SensorModel(model, dt)
    estimator = StateEstimator(dt)
    controller = LQRController(mj_model=model)
    navigator = Navigator(dt, mj_model=model)
    gt_obstacles = extract_gt_obstacles(model)

    print(f"  Scene: {scene_name}, dt={dt}, duration={duration}s")
    print(f"  Ground-truth obstacles: {len(gt_obstacles)}")
    for o in gt_obstacles:
        print(f"    {o}")

    mujoco.mj_forward(model, data)

    # Warmup so the balancer stabilizes
    for _ in range(int(1.0 / dt)):
        readings = sensor_model.read(data)
        estimator.update(readings)
        if not estimator.fallen:
            left, right = controller.compute(estimator, 0.0, 0.0, dt)
            data.ctrl[0] = left
            data.ctrl[1] = right
        mujoco.mj_step(model, data)

    if estimator.fallen:
        print("  Fell during warmup!")
        return

    navigator.set_heading(heading_deg)

    out_dir = default_snapshot_dir()
    sim_time = 0.0
    last_snapshot = -snapshot_every
    snapshot_idx = 0

    while sim_time < duration:
        readings = sensor_model.read(data)
        estimator.update(readings)
        if estimator.fallen:
            print(f"  Fell at t={sim_time:.1f}s")
            break

        nav_vel, nav_yaw = navigator.update(estimator, readings, dt)
        if nav_vel is None:
            nav_vel = 0.0
            nav_yaw = 0.0
        left, right = controller.compute(estimator, nav_vel, nav_yaw, dt)
        data.ctrl[0] = left
        data.ctrl[1] = right
        mujoco.mj_step(model, data)
        sim_time += dt

        if sim_time - last_snapshot >= snapshot_every:
            rf_comp = navigator.compensate_rangefinders(
                readings.rangefinders, estimator.pitch
            )
            frame = MapFrame(
                grid=navigator.grid.grid.copy(),
                grid_cx=navigator.grid.cx,
                grid_cy=navigator.grid.cy,
                robot_x=navigator._pos_x,
                robot_y=navigator._pos_y,
                heading=navigator._heading,
                target_heading=navigator._target_heading,
                nav_active=navigator._active,
                fsm_state=navigator._fsm.value,
                rf_compensated=rf_comp,
                gt_obstacles=gt_obstacles,
            )
            img = render_rgb(frame)
            path = os.path.join(
                out_dir,
                f"mapviz_{out_tag}_{snapshot_idx:03d}_t{sim_time:04.1f}s.png",
            )
            save_png(path, img)
            print(f"  t={sim_time:5.2f}s  fsm={navigator._fsm.value:12s}  "
                  f"pos=({navigator._pos_x:+.2f},{navigator._pos_y:+.2f})  "
                  f"front={navigator._min_front_dist:.2f}m  -> {os.path.basename(path)}")
            last_snapshot = sim_time
            snapshot_idx += 1

    print(f"\n  Final REAL pos: ({data.qpos[0]:+.2f}, {data.qpos[1]:+.2f})")
    print(f"  Final DR pos:   ({navigator._pos_x:+.2f}, {navigator._pos_y:+.2f})")
    print(f"  Snapshots in: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Mapviz end-to-end test")
    parser.add_argument("--scene", default="nav_test",
                        help="Scene name or 'random' (default: nav_test)")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--heading", type=float, default=0.0,
                        help="Target heading in degrees")
    parser.add_argument("--every", type=float, default=1.0,
                        help="Snapshot interval in seconds")
    parser.add_argument("--tag", default="run")
    args = parser.parse_args()
    run(args.scene, args.duration, args.heading, args.every, args.tag)


if __name__ == "__main__":
    main()
