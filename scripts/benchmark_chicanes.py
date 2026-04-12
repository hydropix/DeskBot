"""
Chicane-focused benchmark for DeskBot navigation v2.

Generates scenes where every episode features exactly one chicane
(two offset boxes 1.5 m apart forcing an S-curve). Used in session 10
step 5 to isolate the chicane risk of early avoidance: a side cached
on the first box may be wrong for the second.

Usage:
    python scripts/benchmark_chicanes.py                       # 30 eps
    python scripts/benchmark_chicanes.py --early-avoid on
    python scripts/benchmark_chicanes.py --planner astar

ASCII-only output for cp1252 consoles.
"""
import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_random import generate_scene_xml, run_episode


def chicane_scene(rng: np.random.Generator, corridor_half: float):
    """Produce exactly one chicane obstacle at a randomized X + offsets."""
    chw = corridor_half
    usable = chw - 0.10
    x = 3.0 + rng.uniform(0.0, 4.0)
    sy1 = rng.uniform(0.15, 0.30)
    sy2 = rng.uniform(0.15, 0.30)
    side = int(rng.choice([-1, 1]))
    y1 = side * rng.uniform(0.2, max(0.21, usable - sy1))
    y2 = -side * rng.uniform(0.2, max(0.21, usable - sy2))
    return [
        {"type": "box", "x": x,       "y": y1, "sx": 0.20, "sy": sy1},
        {"type": "box", "x": x + 1.5, "y": y2, "sx": 0.20, "sy": sy2},
    ]


def _worker(job):
    ep = job["ep"]
    ep_seed = job["ep_seed"]
    chw = job["chw"]
    duration = job["duration"]
    planner = job["planner"]
    early_avoid = job["early_avoid"]

    rng = np.random.default_rng(ep_seed)
    obstacles = chicane_scene(rng, chw)
    xml = generate_scene_xml(obstacles, chw)

    r = run_episode(
        xml,
        target_heading=0.0,
        max_duration=duration,
        seed=ep_seed,
        n_obstacles=len(obstacles),
        verbose=False,
        planner=planner,
        early_avoid=early_avoid,
    )
    return ep, r


def main():
    p = argparse.ArgumentParser(description="Chicane-only navigation benchmark")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--corridor", type=float, default=1.5)
    p.add_argument("--planner", choices=["bug2", "astar"], default="astar")
    p.add_argument("--early-avoid", choices=["off", "on"], default="off")
    p.add_argument("--jobs", type=int,
                   default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()

    print("=" * 65)
    print("  CHICANE-ONLY BENCHMARK")
    print(f"  {args.episodes} eps, corridor={args.corridor*2:.1f}m, seed={args.seed}")
    print(f"  Planner: {args.planner}   Early avoid: {args.early_avoid}")
    print(f"  Parallel workers: {args.jobs}")
    print("=" * 65)

    jobs = [
        {
            "ep": ep,
            "ep_seed": args.seed + ep,
            "chw": args.corridor,
            "duration": args.duration,
            "planner": args.planner,
            "early_avoid": (args.early_avoid == "on"),
        }
        for ep in range(args.episodes)
    ]

    wall = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(_worker, j): j["ep"] for j in jobs}
        for fut in as_completed(futs):
            ep, r = fut.result()
            results.append(r)
            status = "OK" if r.success else ("FELL" if r.fell else "FAIL")
            marker = "  " if r.success else ">>"
            print(f"{marker} [{len(results):3d}/{args.episodes}] ep={ep:3d} "
                  f"[{status:4s}] X={r.final_x:5.1f} Y={r.final_y:+5.1f} "
                  f"t={r.duration:5.1f}s cont={r.contour_count} "
                  f"stuck={r.stuck_count} seed={r.seed}")
    results.sort(key=lambda r: r.seed)
    wall = time.perf_counter() - wall

    succ = sum(1 for r in results if r.success)
    falls = sum(1 for r in results if r.fell)
    print()
    print("=" * 65)
    print(f"  SUCCESS:  {succ}/{args.episodes} ({100*succ/args.episodes:.0f} %)")
    print(f"  Falls:    {falls}")
    print(f"  Avg cont: {np.mean([r.contour_count for r in results]):.1f}")
    print(f"  Wall:     {wall:.1f}s ({args.episodes/wall:.1f} ep/s)")
    print("=" * 65)

    failures = [r for r in results if not r.success]
    if failures:
        print()
        print("  FAILING SEEDS:")
        for f in failures:
            kind = "FELL" if f.fell else "FAIL"
            print(f"    seed={f.seed} [{kind}] X={f.final_x:.1f} "
                  f"cont={f.contour_count} stuck={f.stuck_count}")


if __name__ == "__main__":
    main()
