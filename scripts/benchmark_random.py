"""
Randomized obstacle benchmark for DeskBot navigation v2.

Generates random obstacle configurations in a corridor and tests the
robot's ability to navigate through them while maintaining heading.

Obstacle types:
  - Box: random size 30-80cm, random Y offset
  - Cylinder: random radius 15-35cm, random Y offset
  - Wall with gap: spans most of corridor width, gap on random side
  - Chicane: two offset obstacles forcing an S-curve

Usage:
    python scripts/benchmark_random.py                    # 50 random episodes
    python scripts/benchmark_random.py --episodes 200     # more episodes
    python scripts/benchmark_random.py --verbose          # per-step debug
    python scripts/benchmark_random.py --seed 42          # reproducible
    python scripts/benchmark_random.py --corridor 2.0     # narrower corridor
"""
import os
import sys
import time
import math
import argparse
import textwrap
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deskbot.robot import MODELS_DIR
from deskbot.sensors import SensorModel
from deskbot.control import BalanceController, StateEstimator
from deskbot.navigation import Navigator


# ── Scene generation ─────────────────────────────────────────────

def generate_scene_xml(
    obstacles: list[dict],
    corridor_half_width: float = 1.5,
    corridor_length: float = 15.0,
) -> str:
    """Generate a MuJoCo XML scene with the given obstacles."""
    deskbot_path = str(MODELS_DIR / "deskbot.xml").replace("\\", "/")

    obstacle_geoms = []
    for i, obs in enumerate(obstacles):
        if obs["type"] == "box":
            geom = (f'    <geom name="obs_{i}" type="box" '
                    f'size="{obs["sx"]:.3f} {obs["sy"]:.3f} 0.15" '
                    f'pos="{obs["x"]:.3f} {obs["y"]:.3f} 0.15" '
                    f'rgba="0.7 0.25 0.2 1"/>')
        elif obs["type"] == "cylinder":
            geom = (f'    <geom name="obs_{i}" type="cylinder" '
                    f'size="{obs["r"]:.3f} 0.15" '
                    f'pos="{obs["x"]:.3f} {obs["y"]:.3f} 0.15" '
                    f'rgba="0.7 0.25 0.2 1"/>')
        elif obs["type"] == "wall":
            geom = (f'    <geom name="obs_{i}" type="box" '
                    f'size="0.05 {obs["sy"]:.3f} 0.15" '
                    f'pos="{obs["x"]:.3f} {obs["y"]:.3f} 0.15" '
                    f'rgba="0.55 0.55 0.6 1"/>')
        else:
            continue
        obstacle_geoms.append(geom)

    geom_block = "\n".join(obstacle_geoms)
    chw = corridor_half_width

    xml = textwrap.dedent(f"""\
    <mujoco model="deskbot_random_test">
      <include file="{deskbot_path}"/>
      <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast">
        <flag multiccd="enable"/>
      </option>
      <visual>
        <headlight diffuse="0.5 0.5 0.5" ambient="0.3 0.3 0.3"/>
      </visual>
      <statistic center="5 0 0.15" extent="8"/>
      <worldbody>
        <light pos="5 -2 4" dir="0 0.3 -1" diffuse="0.7 0.7 0.7" castshadow="true"/>
        <geom name="floor" type="plane" size="20 10 0.05"
              rgba="0.42 0.42 0.45 1" contype="1" conaffinity="1"/>
        <geom name="wall_N" type="box" size="{corridor_length:.1f} 0.05 0.15"
              pos="{corridor_length/2:.1f} {chw:.2f} 0.15" rgba="0.55 0.55 0.6 1"/>
        <geom name="wall_S" type="box" size="{corridor_length:.1f} 0.05 0.15"
              pos="{corridor_length/2:.1f} {-chw:.2f} 0.15" rgba="0.55 0.55 0.6 1"/>
    {geom_block}
      </worldbody>
    </mujoco>
    """)
    return xml


def random_obstacles(
    rng: np.random.Generator,
    n_obstacles: int,
    corridor_half_width: float = 1.5,
    corridor_length: float = 15.0,
    robot_width: float = 0.26,
) -> list[dict]:
    """Generate random obstacle configurations ensuring passability."""
    obstacles = []
    # Place obstacles along X axis with minimum spacing
    min_spacing = 2.0
    max_x = corridor_length - 2.0
    x_positions = []
    x = 2.5 + rng.uniform(0, 1.0)
    for _ in range(n_obstacles):
        if x > max_x:
            break
        x_positions.append(x)
        x += min_spacing + rng.uniform(0, 2.0)

    chw = corridor_half_width
    usable = chw - 0.10  # margin from walls

    for x in x_positions:
        obs_type = rng.choice(["box", "cylinder", "wall", "chicane"],
                              p=[0.35, 0.25, 0.25, 0.15])

        if obs_type == "box":
            sx = rng.uniform(0.15, 0.40)  # half-size X
            sy = rng.uniform(0.15, 0.40)  # half-size Y
            # Random Y within corridor, ensuring gap on at least one side
            max_y_offset = usable - sy - robot_width * 0.6
            y = rng.uniform(-max_y_offset, max_y_offset)
            obstacles.append({"type": "box", "x": x, "y": y, "sx": sx, "sy": sy})

        elif obs_type == "cylinder":
            r = rng.uniform(0.10, 0.30)
            max_y_offset = usable - r - robot_width * 0.6
            y = rng.uniform(-max_y_offset, max_y_offset)
            obstacles.append({"type": "cylinder", "x": x, "y": y, "r": r})

        elif obs_type == "wall":
            # Wall spanning most of corridor, gap on one side
            gap_width = rng.uniform(0.55, 0.80)
            gap_side = rng.choice([-1, 1])  # -1 = south, +1 = north
            wall_extent = chw - gap_width / 2
            wall_center_y = -gap_side * (gap_width / 2 + wall_extent) / 2
            # Compute wall half-size and center to leave gap
            if gap_side > 0:
                # Gap on north side: wall from south wall to (chw - gap_width)
                wy_min = -usable
                wy_max = chw - gap_width
                sy = (wy_max - wy_min) / 2
                y = (wy_max + wy_min) / 2
            else:
                # Gap on south side: wall from (-(chw - gap_width)) to north wall
                wy_min = -(chw - gap_width)
                wy_max = usable
                sy = (wy_max - wy_min) / 2
                y = (wy_max + wy_min) / 2
            obstacles.append({"type": "wall", "x": x, "y": y, "sy": sy})

        elif obs_type == "chicane":
            # Two offset obstacles creating an S-curve
            sy1 = rng.uniform(0.15, 0.30)
            sy2 = rng.uniform(0.15, 0.30)
            side = rng.choice([-1, 1])
            y1 = side * rng.uniform(0.2, usable - sy1)
            y2 = -side * rng.uniform(0.2, usable - sy2)
            obstacles.append({"type": "box", "x": x, "y": y1, "sx": 0.20, "sy": sy1})
            obstacles.append({"type": "box", "x": x + 1.5, "y": y2, "sx": 0.20, "sy": sy2})

    return obstacles


# ── Episode runner ───────────────────────────────────────────────

@dataclass
class EpisodeResult:
    seed: int
    n_obstacles: int
    reached_x: float
    final_x: float
    final_y: float
    fell: bool
    stuck_count: int
    contour_count: int
    avg_speed: float
    duration: float
    success: bool


def run_episode(
    scene_xml: str,
    target_heading: float = 0.0,
    target_x: float = 12.0,
    max_duration: float = 60.0,
    seed: int = 0,
    n_obstacles: int = 0,
    verbose: bool = False,
    planner: str = "bug2",
    early_avoid: bool = False,
    early_k: float | None = None,
) -> EpisodeResult:
    """Run one navigation episode with a generated scene."""
    model = mujoco.MjModel.from_xml_string(scene_xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    mujoco.mj_forward(model, data)
    init_qpos = data.qpos.copy()

    sensor_model = SensorModel(model, dt)
    estimator = StateEstimator(dt)
    controller = BalanceController()
    navigator = Navigator(
        dt, mj_model=model, use_astar=(planner == "astar"),
        use_early_avoid=early_avoid,
        early_k=early_k,
    )

    # Warmup
    for _ in range(int(1.0 / dt)):
        readings = sensor_model.read(data)
        estimator.update(readings)
        if not estimator.fallen:
            left, right = controller.compute(estimator, 0.0, 0.0, dt)
            data.ctrl[0] = left
            data.ctrl[1] = right
        mujoco.mj_step(model, data)

    if estimator.fallen:
        return EpisodeResult(seed, n_obstacles, 0, 0, 0, True, 0, 0, 0, 0, False)

    navigator.set_heading(target_heading)

    contour_count = 0
    stuck_count = 0
    was_contour = False
    speed_samples = []
    sim_time = 0.0
    debug_interval = 2.0
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

        speed_samples.append(abs(estimator.forward_vel))
        is_contour = navigator._fsm.value == "contour"
        if is_contour and not was_contour:
            contour_count += 1
        was_contour = is_contour
        if navigator._fsm.value == "reverse":
            stuck_count += 1

        if data.qpos[0] >= target_x:
            break

        if verbose and sim_time >= next_debug:
            real_x, real_y = data.qpos[0], data.qpos[1]
            print(f"  t={sim_time:5.1f} fsm={navigator._fsm.value:12s} "
                  f"REAL=({real_x:+.2f},{real_y:+.2f}) "
                  f"h={math.degrees(navigator._heading):+.0f} "
                  f"front={navigator._min_front_dist:.2f}")
            next_debug += debug_interval

    real_x = float(data.qpos[0])
    real_y = float(data.qpos[1])
    avg_speed = float(np.mean(speed_samples)) if speed_samples else 0.0
    success = real_x >= target_x * 0.9 and not estimator.fallen

    return EpisodeResult(
        seed=seed,
        n_obstacles=n_obstacles,
        reached_x=navigator._pos_x,
        final_x=real_x,
        final_y=real_y,
        fell=estimator.fallen,
        stuck_count=stuck_count // 500,
        contour_count=contour_count,
        avg_speed=avg_speed,
        duration=sim_time,
        success=success,
    )


# ── Parallel worker ──────────────────────────────────────────────

def _worker(job: dict) -> tuple[int, EpisodeResult]:
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    ep = job["ep"]
    ep_seed = job["ep_seed"]
    chw = job["chw"]
    min_obs = job["min_obs"]
    max_obs = job["max_obs"]
    duration = job["duration"]
    planner = job.get("planner", "bug2")
    early_avoid = job.get("early_avoid", False)
    early_k = job.get("early_k", None)

    rng = np.random.default_rng(ep_seed)
    n_obs = int(rng.integers(min_obs, max_obs + 1))
    obstacles = random_obstacles(rng, n_obs, chw)
    xml = generate_scene_xml(obstacles, chw)

    result = run_episode(
        xml,
        target_heading=0.0,
        max_duration=duration,
        seed=ep_seed,
        n_obstacles=len(obstacles),
        verbose=False,
        planner=planner,
        early_avoid=early_avoid,
        early_k=early_k,
    )
    return ep, result


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Randomized Navigation Benchmark")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--corridor", type=float, default=1.5,
                        help="Corridor half-width in meters (default: 1.5)")
    parser.add_argument("--min-obs", type=int, default=2)
    parser.add_argument("--max-obs", type=int, default=5)
    parser.add_argument(
        "--planner", choices=["bug2", "astar"], default="bug2",
        help="Contour-entry planner. bug2 = legacy virtual scan (default); "
             "astar = A* on the inflated occupancy grid with silent "
             "fallback to bug2 on timeout/failure.",
    )
    parser.add_argument(
        "--early-avoid", choices=["off", "on"], default="off",
        help="Early avoidance (session 10). off = legacy behaviour "
             "(default); on = add continuous inverse-distance yaw bias "
             "in GO_HEADING before SAFE_DIST trigger.",
    )
    parser.add_argument(
        "--early-k", type=float, default=None,
        help="Override the early avoidance gain k (default from "
             "EarlyAvoidanceParams). Only used when --early-avoid on. "
             "Step 6 sweep values: 0.05, 0.08, 0.12.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel workers (default: cpu_count-1, use 1 for serial debug)",
    )
    args = parser.parse_args()
    if args.verbose:
        args.jobs = 1  # verbose output is unreadable when interleaved

    base_seed = args.seed if args.seed is not None else int(time.time()) % 100000
    chw = args.corridor

    print("=" * 65)
    print("  RANDOMIZED NAVIGATION BENCHMARK")
    print(f"  {args.episodes} episodes, corridor={chw*2:.1f}m wide, "
          f"{args.min_obs}-{args.max_obs} obstacles, seed={base_seed}")
    print(f"  Planner: {args.planner}")
    print(f"  Early avoid: {args.early_avoid}")
    print(f"  Parallel workers: {args.jobs}")
    print("=" * 65)
    print()

    def _print_result(ep, result, done, total):
        status = "OK" if result.success else ("FELL" if result.fell else "FAIL")
        marker = "  " if result.success else ">>"
        print(f"{marker} [{done:3d}/{total}] ep={ep:3d} [{status:4s}] "
              f"X={result.final_x:5.1f} Y={result.final_y:+5.1f} "
              f"t={result.duration:5.1f}s "
              f"obs={result.n_obstacles} "
              f"cont={result.contour_count} "
              f"stuck={result.stuck_count} "
              f"spd={result.avg_speed:.2f} "
              f"seed={result.seed}")

    results = []
    failures = []
    wall_start = time.perf_counter()

    jobs = [
        {
            "ep": ep,
            "ep_seed": base_seed + ep,
            "chw": chw,
            "min_obs": args.min_obs,
            "max_obs": args.max_obs,
            "duration": args.duration,
            "planner": args.planner,
            "early_avoid": (args.early_avoid == "on"),
            "early_k": args.early_k,
        }
        for ep in range(args.episodes)
    ]

    if args.jobs <= 1:
        # Serial path (debug / verbose)
        for job in jobs:
            ep = job["ep"]
            ep_seed = job["ep_seed"]
            rng = np.random.default_rng(ep_seed)
            n_obs = rng.integers(args.min_obs, args.max_obs + 1)
            obstacles = random_obstacles(rng, n_obs, chw)
            xml = generate_scene_xml(obstacles, chw)

            if args.verbose:
                obs_desc = ", ".join(
                    f"{o['type']}@X={o['x']:.1f}" for o in obstacles
                )
                print(f"\n  ep={ep} seed={ep_seed} obs=[{obs_desc}]")

            result = run_episode(
                xml,
                target_heading=0.0,
                max_duration=args.duration,
                seed=ep_seed,
                n_obstacles=len(obstacles),
                verbose=args.verbose,
                planner=args.planner,
                early_avoid=(args.early_avoid == "on"),
                early_k=args.early_k,
            )
            results.append(result)
            if not args.verbose:
                _print_result(ep, result, len(results), args.episodes)
            if not result.success:
                failures.append(result)
    else:
        # Parallel path
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            future_to_ep = {pool.submit(_worker, job): job["ep"] for job in jobs}
            for fut in as_completed(future_to_ep):
                ep, result = fut.result()
                results.append(result)
                _print_result(ep, result, len(results), args.episodes)
                if not result.success:
                    failures.append(result)
        results.sort(key=lambda r: r.seed)
        failures.sort(key=lambda r: r.seed)

    wall_elapsed = time.perf_counter() - wall_start

    # Summary
    total = len(results)
    successes = sum(1 for r in results if r.success)
    falls = sum(1 for r in results if r.fell)
    stuck_episodes = sum(1 for r in results if r.stuck_count > 0)
    avg_speed = np.mean([r.avg_speed for r in results])
    avg_contours = np.mean([r.contour_count for r in results])
    avg_time_success = np.mean([r.duration for r in results if r.success]) if successes else 0

    print()
    print("=" * 65)
    print(f"  SUCCESS RATE:    {successes}/{total} ({100*successes/total:.0f}%)")
    print(f"  Falls:           {falls}")
    print(f"  Stuck episodes:  {stuck_episodes}")
    print(f"  Avg contours:    {avg_contours:.1f}")
    print(f"  Avg speed:       {avg_speed:.2f} m/s")
    if successes:
        print(f"  Avg time (pass): {avg_time_success:.1f}s")
        avg_dev = np.mean([abs(r.final_y) for r in results if r.success])
        print(f"  Avg |Y| dev:     {avg_dev:.2f}m")
    print(f"  Wall time:       {wall_elapsed:.1f}s "
          f"({total/wall_elapsed:.1f} ep/s)")
    print("=" * 65)

    if failures:
        print(f"\n  FAILURE SEEDS (reproduce with --seed N --episodes 1 --verbose):")
        for f in failures[:10]:
            status = "FELL" if f.fell else "FAIL"
            print(f"    seed={f.seed} [{status}] X={f.final_x:.1f} "
                  f"obs={f.n_obstacles} stuck={f.stuck_count}")
    print()


if __name__ == "__main__":
    main()
