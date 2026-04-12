"""
Headless navigation benchmark for DeskBot.

Runs the full sensor→estimator→navigator→controller pipeline without any
viewer or GUI, at maximum CPU speed. A 10-second episode typically completes
in ~50ms, enabling rapid parameter iteration.

Usage:
    python scripts/benchmark_nav.py                    # run default test suite
    python scripts/benchmark_nav.py --sweep            # parameter sweep
    python scripts/benchmark_nav.py --episodes 50      # more episodes per route
    python scripts/benchmark_nav.py --duration 15      # 15s per episode
    python scripts/benchmark_nav.py --verbose          # print per-episode details
"""
import sys
import time
import math
import argparse
import itertools
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import mujoco

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deskbot.robot import SCENES
from deskbot.sensors import SensorModel
from deskbot.control import BalanceController, StateEstimator
from deskbot.navigation import (
    Navigator, APARTMENT_WAYPOINTS, APARTMENT_ROUTES,
)
import deskbot.navigation as nav_module


# ── Test routes ──────────────────────────────────────────────────
# "route" = multi-waypoint route name from APARTMENT_ROUTES (optional)
# "goal" = single waypoint name (used if no route)
TEST_ROUTES = [
    # Simple: straight line in salon
    {"name": "salon_straight", "start": (0.5, -0.25), "heading": 0.0,
     "goal": "salon_center", "difficulty": "easy"},

    # Medium: salon → corridor via door
    {"name": "salon_to_corridor", "start": (0.5, -0.25), "heading": 0.0,
     "goal": "couloir_west", "difficulty": "medium"},

    # Medium: corridor traverse
    {"name": "corridor_traverse", "start": (1.5, 1.60), "heading": 0.0,
     "goal": "couloir_mid", "difficulty": "medium"},

    # Hard: salon → cuisine (multi-waypoint route)
    {"name": "salon_to_cuisine", "start": (0.5, -0.25), "heading": 0.0,
     "route": "salon_to_cuisine", "difficulty": "hard"},

    # Hard: corridor → chambre (multi-waypoint route, start near door)
    {"name": "corridor_to_chambre", "start": (1.90, 1.60), "heading": 0.0,
     "route": ["chambre_door", "chambre_entry", "chambre_center"],
     "difficulty": "hard"},
]


@dataclass
class EpisodeResult:
    route_name: str
    reached: bool
    time_to_goal: float       # seconds (sim time), or max_duration if failed
    final_dist: float         # meters from goal at end
    collisions: int           # number of emergency brakes triggered
    fell: bool                # did the robot fall over?
    stuck_recoveries: int     # number of stuck recovery maneuvers
    avg_speed: float          # average forward speed (m/s)
    path_efficiency: float    # straight_line_dist / actual_distance_traveled


def run_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    init_qpos: np.ndarray,
    init_qvel: np.ndarray,
    route: dict,
    max_duration: float = 10.0,
    nav_params: dict | None = None,
) -> EpisodeResult:
    """Run a single headless navigation episode. Returns metrics."""
    dt = model.opt.timestep

    # Reset simulation state
    mujoco.mj_resetData(model, data)
    data.qpos[:] = init_qpos
    data.qvel[:] = init_qvel

    # Position robot at route start (free joint: qpos[0:3]=pos, qpos[3:7]=quat)
    data.qpos[0] = route["start"][0]
    data.qpos[1] = route["start"][1]
    # qpos[2] (Z height) stays at default
    heading = route["heading"]
    data.qpos[3] = math.cos(heading / 2)  # qw
    data.qpos[4] = 0.0                     # qx
    data.qpos[5] = 0.0                     # qy
    data.qpos[6] = math.sin(heading / 2)  # qz

    mujoco.mj_forward(model, data)

    # Create fresh instances
    sensor_model = SensorModel(model, dt)
    estimator = StateEstimator(dt)
    controller = BalanceController()
    navigator = Navigator(dt, mj_model=model)

    # Apply parameter overrides if sweeping
    if nav_params:
        for key, val in nav_params.items():
            setattr(nav_module, key, val)

    # Warmup: run 1s of physics with zero commands to let the balance
    # controller stabilize after teleportation
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
        return EpisodeResult(
            route_name=route["name"], reached=False, time_to_goal=0.0,
            final_dist=99.0, collisions=0, fell=True,
            stuck_recoveries=0, avg_speed=0.0, path_efficiency=0.0,
        )

    # Set goal or route (after warmup so balance is stable)
    if "route" in route:
        r = route["route"]
        if isinstance(r, str):
            navigator.set_route_absolute(APARTMENT_ROUTES[r])
        else:
            navigator.set_route_absolute(r)
        goal_xy = APARTMENT_WAYPOINTS[r[-1]] if isinstance(r, list) else APARTMENT_WAYPOINTS[APARTMENT_ROUTES[r][-1]]
    else:
        goal_xy = APARTMENT_WAYPOINTS[route["goal"]]
        navigator.set_goal_absolute(goal_xy[0], goal_xy[1])

    # Override dead reckoning start position/heading
    navigator._pos_x = route["start"][0]
    navigator._pos_y = route["start"][1]
    navigator._heading = route["heading"]
    navigator._last_progress_pos = np.array(route["start"])

    # Metrics tracking
    collisions = 0
    stuck_recoveries = 0
    total_distance = 0.0
    prev_pos = np.array(route["start"])
    speed_samples = []
    start_dist = np.linalg.norm(np.array(goal_xy) - np.array(route["start"]))
    debug = nav_params and nav_params.get("_debug", False)
    debug_interval = 1.0
    next_debug = debug_interval

    # Main loop — no viewer, no frame cap, pure physics
    sim_time = 0.0
    reached = False
    fell = False

    while sim_time < max_duration:
        readings = sensor_model.read(data)
        estimator.update(readings)

        if estimator.fallen:
            fell = True
            break

        nav_vel, nav_yaw = navigator.update(estimator, readings, dt)

        if nav_vel is None:
            # Navigation finished (reached goal or deactivated)
            if navigator.state.reached:
                reached = True
            break

        left, right = controller.compute(estimator, nav_vel, nav_yaw, dt)
        data.ctrl[0] = left
        data.ctrl[1] = right

        mujoco.mj_step(model, data)
        sim_time += dt

        # Track metrics
        cur_pos = np.array([navigator._pos_x, navigator._pos_y])
        step_dist = np.linalg.norm(cur_pos - prev_pos)
        total_distance += step_dist
        prev_pos = cur_pos.copy()
        speed_samples.append(abs(estimator.forward_vel))

        if navigator.state.behavior == "EMERGENCY":
            collisions += 1
        if navigator._fsm.value == "reverse":
            stuck_recoveries += 1

        # Debug output
        if debug and sim_time >= next_debug:
            rf_str = " ".join(f"{k[-2:]}={v:.2f}" for k, v in readings.rangefinders.items() if v > 0)
            real_x, real_y = data.qpos[0], data.qpos[1]
            pitch_deg = math.degrees(estimator.pitch)
            print(f"    t={sim_time:5.1f} beh={navigator.state.behavior:10s} "
                  f"DR=({navigator._pos_x:.2f},{navigator._pos_y:.2f}) "
                  f"REAL=({real_x:.2f},{real_y:.2f}) "
                  f"h={math.degrees(navigator._heading):+.0f}° "
                  f"pitch={pitch_deg:+.1f}° "
                  f"cmd_v={nav_vel:+.2f} fwd_v={estimator.forward_vel:+.2f} "
                  f"torq=({data.ctrl[0]:+.2f},{data.ctrl[1]:+.2f}) "
                  f"rf=[{rf_str}]")
            next_debug += debug_interval

    # Final metrics
    final_pos = np.array([navigator._pos_x, navigator._pos_y])
    final_dist = float(np.linalg.norm(np.array(goal_xy) - final_pos))
    avg_speed = float(np.mean(speed_samples)) if speed_samples else 0.0
    path_eff = start_dist / max(total_distance, 0.01) if start_dist > 0.01 else 1.0

    return EpisodeResult(
        route_name=route["name"],
        reached=reached,
        time_to_goal=sim_time,
        final_dist=final_dist,
        collisions=collisions // 50,  # deduplicate (500Hz → ~50 steps per collision event)
        fell=fell,
        stuck_recoveries=stuck_recoveries // 500,  # deduplicate
        avg_speed=avg_speed,
        path_efficiency=min(path_eff, 1.0),
    )


def run_benchmark(
    episodes_per_route: int = 10,
    max_duration: float = 10.0,
    nav_params: dict | None = None,
    verbose: bool = False,
    label: str = "",
) -> dict:
    """Run full benchmark suite. Returns aggregate metrics."""
    scene_path = SCENES["apartment"]
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)
    init_qpos = data.qpos.copy()
    init_qvel = data.qvel.copy()

    all_results = []
    wall_start = time.perf_counter()

    for route in TEST_ROUTES:
        route_results = []
        for ep in range(episodes_per_route):
            result = run_episode(
                model, data, init_qpos, init_qvel,
                route, max_duration, nav_params,
            )
            route_results.append(result)
            all_results.append(result)

            if verbose:
                status = "OK" if result.reached else ("FELL" if result.fell else "FAIL")
                print(f"  [{status}] {route['name']:25s} ep={ep:2d} "
                      f"t={result.time_to_goal:5.1f}s "
                      f"d={result.final_dist:.2f}m "
                      f"spd={result.avg_speed:.2f} "
                      f"eff={result.path_efficiency:.0%} "
                      f"col={result.collisions} "
                      f"rec={result.stuck_recoveries}")

        # Per-route summary
        successes = sum(1 for r in route_results if r.reached)
        avg_time = np.mean([r.time_to_goal for r in route_results if r.reached]) if successes else float('inf')
        if not verbose:
            print(f"  {route['name']:25s} "
                  f"success={successes}/{episodes_per_route} "
                  f"avg_t={avg_time:5.1f}s "
                  f"[{route['difficulty']}]")

    wall_elapsed = time.perf_counter() - wall_start

    # Aggregate
    total = len(all_results)
    success_rate = sum(1 for r in all_results if r.reached) / total
    fall_rate = sum(1 for r in all_results if r.fell) / total
    avg_collisions = np.mean([r.collisions for r in all_results])
    avg_efficiency = np.mean([r.path_efficiency for r in all_results if r.reached]) if success_rate > 0 else 0
    avg_speed = np.mean([r.avg_speed for r in all_results])

    summary = {
        "label": label,
        "success_rate": success_rate,
        "fall_rate": fall_rate,
        "avg_collisions": avg_collisions,
        "avg_efficiency": avg_efficiency,
        "avg_speed": avg_speed,
        "total_episodes": total,
        "wall_time": wall_elapsed,
        "episodes_per_sec": total / wall_elapsed,
    }

    print(f"\n{'='*60}")
    if label:
        print(f"  Config: {label}")
    print(f"  Success rate:    {success_rate:.0%} ({int(success_rate*total)}/{total})")
    print(f"  Fall rate:       {fall_rate:.0%}")
    print(f"  Avg collisions:  {avg_collisions:.1f}")
    print(f"  Path efficiency: {avg_efficiency:.0%}")
    print(f"  Avg speed:       {avg_speed:.2f} m/s")
    print(f"  Wall time:       {wall_elapsed:.1f}s ({total/wall_elapsed:.0f} episodes/s)")
    print(f"{'='*60}\n")

    return summary


def parameter_sweep():
    """Sweep key navigation parameters and find the best configuration."""
    print("=" * 60)
    print("  PARAMETER SWEEP — Navigation AI")
    print("=" * 60)

    # Define parameter grid
    param_grid = {
        "GOAL_ATTRACT_GAIN":     [0.8, 1.2, 1.6],
        "OBSTACLE_REPULSE_GAIN": [0.5, 0.8, 1.2],
        "COMMAND_SMOOTHING":     [0.60, 0.75, 0.85],
        "CAUTION_DIST":          [0.35, 0.50, 0.65],
    }

    # Save defaults
    defaults = {k: getattr(nav_module, k) for k in param_grid}

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"\n  Testing {len(combinations)} configurations "
          f"({len(combinations) * len(TEST_ROUTES) * 5} total episodes)\n")

    results = []
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        label = " | ".join(f"{k.split('_')[-1]}={v}" for k, v in params.items())
        print(f"\n[{i+1}/{len(combinations)}] {label}")

        summary = run_benchmark(
            episodes_per_route=5,
            max_duration=10.0,
            nav_params=params,
            verbose=False,
            label=label,
        )
        summary["params"] = params
        results.append(summary)

        # Restore defaults after each run
        for k, v in defaults.items():
            setattr(nav_module, k, v)

    # Rank by composite score: success_rate * 2 + efficiency - fall_rate * 3
    for r in results:
        r["score"] = (r["success_rate"] * 2.0
                      + r["avg_efficiency"]
                      - r["fall_rate"] * 3.0
                      - r["avg_collisions"] * 0.1)

    results.sort(key=lambda r: r["score"], reverse=True)

    print("\n" + "=" * 60)
    print("  TOP 5 CONFIGURATIONS")
    print("=" * 60)
    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1} (score={r['score']:.3f})")
        print(f"     success={r['success_rate']:.0%} "
              f"eff={r['avg_efficiency']:.0%} "
              f"falls={r['fall_rate']:.0%} "
              f"cols={r['avg_collisions']:.1f}")
        for k, v in r["params"].items():
            print(f"     {k} = {v}")

    print(f"\n  Worst: score={results[-1]['score']:.3f} "
          f"success={results[-1]['success_rate']:.0%}")
    print()

    return results


def main():
    parser = argparse.ArgumentParser(description="DeskBot Navigation Benchmark")
    parser.add_argument("--sweep", action="store_true",
                        help="Run parameter sweep to find optimal config")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Episodes per route (default: 10)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Max episode duration in seconds (default: 10)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-episode results")
    parser.add_argument("--debug", type=str, default=None,
                        help="Debug a specific route by name (e.g. salon_straight)")
    args = parser.parse_args()

    if args.debug:
        print(f"  DEBUG MODE: route={args.debug}\n")
        route = next((r for r in TEST_ROUTES if r["name"] == args.debug), None)
        if route is None:
            print(f"  Unknown route. Available: {[r['name'] for r in TEST_ROUTES]}")
            return
        scene_path = SCENES["apartment"]
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        result = run_episode(
            model, data, data.qpos.copy(), data.qvel.copy(),
            route, args.duration, nav_params={"_debug": True},
        )
        status = "OK" if result.reached else ("FELL" if result.fell else "FAIL")
        print(f"\n  Result: [{status}] t={result.time_to_goal:.1f}s "
              f"d={result.final_dist:.2f}m spd={result.avg_speed:.2f}")
        return

    if args.sweep:
        parameter_sweep()
    else:
        print("=" * 60)
        print("  NAVIGATION BENCHMARK — Headless")
        print("=" * 60)
        print()
        run_benchmark(
            episodes_per_route=args.episodes,
            max_duration=args.duration,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
