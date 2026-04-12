"""Entry point: python -m deskbot [--scene ...] [--random [SEED]] [--pid] [--planner ...]"""
import argparse
import sys
import time

from deskbot.robot import SCENES, DEFAULT_SCENE
from deskbot.sim import run, run_random_loop

parser = argparse.ArgumentParser(description="DeskBot Simulator")
parser.add_argument(
    "--scene", choices=list(SCENES.keys()), default=DEFAULT_SCENE,
    help="Scene to load (default: %(default)s)",
)
parser.add_argument(
    "--random", nargs="?", const=-1, type=int, default=None, metavar="SEED",
    help="Generate random obstacle corridors. R=new terrain. Optional seed.",
)
parser.add_argument(
    "--corridor", type=float, default=1.5,
    help="Corridor half-width in meters for --random (default: 1.5)",
)
parser.add_argument(
    "--obstacles", type=int, default=None, metavar="N",
    help="Number of obstacles for --random (default: random 2-5)",
)
parser.add_argument(
    "--pid", action="store_true",
    help="Use legacy PID controller instead of LQR",
)
parser.add_argument(
    "--planner", choices=["bug2", "astar", "field"], default="astar",
    help="Navigation planner: bug2, astar (Bug2+A*), field (Potential Field/fluide) (default: astar)",
)
args = parser.parse_args()

if args.random is not None:
    seed = args.random if args.random >= 0 else int(time.time()) % 100000
    run_random_loop(
        corridor=args.corridor,
        n_obs=args.obstacles,
        base_seed=seed,
        use_pid=args.pid,
        planner=args.planner,
    )
else:
    run(scene_name=args.scene, use_pid=args.pid, planner=args.planner)
