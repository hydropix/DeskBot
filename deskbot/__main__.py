"""Entry point: python -m deskbot [--scene flat|skatepark]"""
import argparse

from deskbot.robot import SCENES, DEFAULT_SCENE
from deskbot.sim import run

parser = argparse.ArgumentParser(description="DeskBot Simulator")
parser.add_argument(
    "--scene", choices=list(SCENES.keys()), default=DEFAULT_SCENE,
    help="Scene to load (default: %(default)s)",
)
args = parser.parse_args()
run(scene_name=args.scene)
