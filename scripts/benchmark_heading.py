"""
Headless benchmark for DeskBot heading-based navigation v2.

Tests the robot's ability to maintain a heading while avoiding obstacles
on the nav_test scene. Runs at max CPU speed without viewer.

Usage:
    python scripts/benchmark_heading.py                  # default test
    python scripts/benchmark_heading.py --verbose        # per-step debug
    python scripts/benchmark_heading.py --duration 20    # longer episodes
"""
import sys
import time
import math
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deskbot.robot import SCENES
from deskbot.sensors import SensorModel
from deskbot.control import BalanceController, StateEstimator
from deskbot.navigation import Navigator


@dataclass
class EpisodeResult:
    name: str
    reached_x: float       # final X position (progress along heading)
    final_x: float
    final_y: float          # lateral deviation
    fell: bool
    stuck_count: int
    contour_count: int      # number of times contour mode was entered
    avg_speed: float
    duration: float
    heading_error_avg: float  # average heading error in degrees


def run_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    init_qpos: np.ndarray,
    init_qvel: np.ndarray,
    target_heading_deg: float = 0.0,
    target_x: float = 12.0,
    max_duration: float = 30.0,
    verbose: bool = False,
) -> EpisodeResult:
    """Run one navigation episode. Robot starts at origin, heading 0."""
    dt = model.opt.timestep

    # Reset
    mujoco.mj_resetData(model, data)
    data.qpos[:] = init_qpos
    data.qvel[:] = init_qvel
    mujoco.mj_forward(model, data)

    sensor_model = SensorModel(model, dt)
    estimator = StateEstimator(dt)
    controller = BalanceController()
    navigator = Navigator(dt, mj_model=model)

    # Warmup: 1s of balancing
    warmup_steps = int(1.0 / dt)
    for _ in range(warmup_steps):
        readings = sensor_model.read(data)
        estimator.update(readings)
        if not estimator.fallen:
            left, right = controller.compute(estimator, 0.0, 0.0, dt)
            data.ctrl[0] = left
            data.ctrl[1] = right
        mujoco.mj_step(model, data)

    if estimator.fallen:
        return EpisodeResult("warmup_fall", 0, 0, 0, True, 0, 0, 0, 0, 0)

    # Set heading
    navigator.set_heading(target_heading_deg)

    # Metrics
    contour_count = 0
    stuck_count = 0
    was_contour = False
    speed_samples = []
    heading_errors = []

    sim_time = 0.0
    debug_interval = 1.0
    next_debug = debug_interval

    while sim_time < max_duration:
        readings = sensor_model.read(data)
        estimator.update(readings)

        if estimator.fallen:
            break

        nav_vel, nav_yaw = navigator.update(estimator, readings, dt)

        if nav_vel is None:
            break

        left, right = controller.compute(estimator, nav_vel, nav_yaw, dt)
        data.ctrl[0] = left
        data.ctrl[1] = right

        mujoco.mj_step(model, data)
        sim_time += dt

        # Track metrics
        speed_samples.append(abs(estimator.forward_vel))

        h_err = navigator._target_heading - navigator._heading
        h_err = math.atan2(math.sin(h_err), math.cos(h_err))
        heading_errors.append(abs(math.degrees(h_err)))

        is_contour = navigator._fsm.value == "contour"
        if is_contour and not was_contour:
            contour_count += 1
        was_contour = is_contour

        if navigator._fsm.value == "reverse":
            stuck_count += 1

        # Check if we passed the target X
        if navigator._pos_x >= target_x:
            break

        # Debug output
        if verbose and sim_time >= next_debug:
            real_x, real_y = data.qpos[0], data.qpos[1]
            rf_str = " ".join(
                f"{k[-2:]}={v:.2f}" for k, v in readings.rangefinders.items() if v > 0
            )
            print(f"  t={sim_time:5.1f} fsm={navigator._fsm.value:12s} "
                  f"DR=({navigator._pos_x:+.2f},{navigator._pos_y:+.2f}) "
                  f"REAL=({real_x:+.2f},{real_y:+.2f}) "
                  f"h={math.degrees(navigator._heading):+.0f} "
                  f"front={navigator._min_front_dist:.2f} "
                  f"v={nav_vel:+.2f} yaw={nav_yaw:+.2f} "
                  f"rf=[{rf_str}]")
            next_debug += debug_interval

    avg_speed = float(np.mean(speed_samples)) if speed_samples else 0.0
    avg_h_err = float(np.mean(heading_errors)) if heading_errors else 0.0

    return EpisodeResult(
        name="heading_test",
        reached_x=navigator._pos_x,
        final_x=float(data.qpos[0]),
        final_y=float(data.qpos[1]),
        fell=estimator.fallen,
        stuck_count=stuck_count // 500,  # deduplicate (500Hz)
        contour_count=contour_count,
        avg_speed=avg_speed,
        duration=sim_time,
        heading_error_avg=avg_h_err,
    )


def main():
    parser = argparse.ArgumentParser(description="DeskBot Heading Navigation Benchmark")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--heading", type=float, default=0.0,
                        help="Target heading in degrees (default: 0 = +X)")
    args = parser.parse_args()

    scene_path = SCENES["nav_test"]
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    init_qpos = data.qpos.copy()
    init_qvel = data.qvel.copy()

    print("=" * 60)
    print("  HEADING NAVIGATION BENCHMARK — nav_test scene")
    print(f"  Target heading: {args.heading} deg, duration: {args.duration}s")
    print("=" * 60)
    print()

    results = []
    for ep in range(args.episodes):
        result = run_episode(
            model, data, init_qpos, init_qvel,
            target_heading_deg=args.heading,
            max_duration=args.duration,
            verbose=args.verbose,
        )
        results.append(result)

        status = "FELL" if result.fell else f"X={result.reached_x:.2f}"
        print(f"  ep={ep:2d} [{status}] "
              f"real=({result.final_x:.2f},{result.final_y:.2f}) "
              f"t={result.duration:.1f}s "
              f"spd={result.avg_speed:.2f} "
              f"contours={result.contour_count} "
              f"stuck={result.stuck_count} "
              f"h_err={result.heading_error_avg:.1f}")

    print()
    print("=" * 60)
    successes = [r for r in results if not r.fell and r.reached_x > 10.0]
    fell = sum(1 for r in results if r.fell)
    avg_x = np.mean([r.reached_x for r in results])
    print(f"  Passed X=10: {len(successes)}/{len(results)}")
    print(f"  Falls: {fell}/{len(results)}")
    print(f"  Avg X reached: {avg_x:.2f}m")
    if successes:
        avg_time = np.mean([r.duration for r in successes])
        avg_y_dev = np.mean([abs(r.final_y) for r in successes])
        print(f"  Avg time (successes): {avg_time:.1f}s")
        print(f"  Avg lateral deviation: {avg_y_dev:.2f}m")
    print("=" * 60)


if __name__ == "__main__":
    main()
