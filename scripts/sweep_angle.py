"""
Sweep the angle of FL2/FR2 sensors to find the optimal placement.

Modifies the MJCF model in-memory to test different angles, then runs
the randomized benchmark for each configuration.

Usage:
    python scripts/sweep_angle.py
    python scripts/sweep_angle.py --episodes 60
"""
import sys
import math
import argparse
import numpy as np
import mujoco
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deskbot.robot import MODELS_DIR
from deskbot.sensors import SensorModel
from deskbot.control import BalanceController, StateEstimator
from deskbot.navigation import Navigator, RF_BODY_ANGLES, RF_PITCH_FACTOR
import deskbot.navigation as nav_mod
from scripts.benchmark_random import random_obstacles, generate_scene_xml, run_episode


def patch_scene_xml_angle(base_xml: str, angle_deg: float) -> str:
    """Replace FL2/FR2 zaxis directions in the XML for a given angle."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    # Include -5deg tilt: multiply sin by cos(-5°), add sin(-5°) to z
    tilt = math.radians(-5)
    cos_t = math.cos(tilt)  # ~0.9962
    sin_t = math.sin(tilt)  # ~-0.0872

    zx = cos_a * cos_t
    zy_l = sin_a * cos_t
    zz = sin_t

    # Replace FL2 zaxis
    import re
    xml = re.sub(
        r'(name="rf_FL2"[^/]*zaxis=")[^"]*(")',
        f'\\g<1>{zx:.4f} {zy_l:.4f} {zz:.4f}\\2',
        base_xml
    )
    # Replace FR2 zaxis
    xml = re.sub(
        r'(name="rf_FR2"[^/]*zaxis=")[^"]*(")',
        f'\\g<1>{zx:.4f} {-zy_l:.4f} {zz:.4f}\\2',
        xml
    )
    return xml


def run_sweep_at_angle(angle_deg: float, n_episodes: int, base_seed: int):
    """Run randomized benchmark at a specific FL2/FR2 angle."""

    # Patch the navigation module's angle tables
    angle_rad = math.radians(angle_deg)
    RF_BODY_ANGLES["rf_FL2"] = angle_rad
    RF_BODY_ANGLES["rf_FR2"] = -angle_rad
    RF_PITCH_FACTOR["rf_FL2"] = math.cos(angle_rad)
    RF_PITCH_FACTOR["rf_FR2"] = math.cos(angle_rad)

    # Read base deskbot.xml and patch it
    deskbot_xml = (MODELS_DIR / "deskbot.xml").read_text()
    patched_deskbot = patch_scene_xml_angle(deskbot_xml, angle_deg)

    # Write patched version temporarily
    patched_path = MODELS_DIR / "_deskbot_sweep.xml"
    patched_path.write_text(patched_deskbot)

    results_wall = []
    results_no_wall = []

    try:
        for ep in range(n_episodes):
            rng = np.random.default_rng(base_seed + ep)
            n_obs = rng.integers(2, 6)
            obstacles = random_obstacles(rng, n_obs, 1.5)
            has_wall = any(o['type'] == 'wall' for o in obstacles)

            # Generate scene XML using the patched deskbot
            xml = generate_scene_xml(obstacles, 1.5)
            # Replace deskbot.xml include with patched version
            xml = xml.replace(
                str(MODELS_DIR / "deskbot.xml").replace("\\", "/"),
                str(patched_path).replace("\\", "/")
            )

            r = run_episode(xml, 0.0, 12.0, 60.0, base_seed + ep, len(obstacles))

            if has_wall:
                results_wall.append(r)
            else:
                results_no_wall.append(r)
    finally:
        # Clean up
        if patched_path.exists():
            patched_path.unlink()

    s_all = sum(1 for r in results_wall + results_no_wall if r.success)
    n_all = len(results_wall) + len(results_no_wall)
    s_wall = sum(1 for r in results_wall if r.success)
    s_no = sum(1 for r in results_no_wall if r.success)

    return {
        "angle": angle_deg,
        "total": f"{s_all}/{n_all} ({100*s_all/n_all:.0f}%)",
        "wall": f"{s_wall}/{len(results_wall)} ({100*s_wall/max(len(results_wall),1):.0f}%)",
        "no_wall": f"{s_no}/{len(results_no_wall)} ({100*s_no/max(len(results_no_wall),1):.0f}%)",
        "success_rate": s_all / n_all,
        "wall_rate": s_wall / max(len(results_wall), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep FL2/FR2 angle")
    parser.add_argument("--episodes", type=int, default=80,
                        help="Episodes per angle (default: 80)")
    parser.add_argument("--seed", type=int, default=8000)
    args = parser.parse_args()

    angles = [45, 50, 55, 60, 65, 70, 75]

    print("=" * 65)
    print(f"  FL2/FR2 ANGLE SWEEP — {args.episodes} episodes per angle")
    print("=" * 65)
    print()

    results = []
    for angle in angles:
        print(f"  Testing {angle}° ...", end="", flush=True)
        r = run_sweep_at_angle(angle, args.episodes, args.seed)
        results.append(r)
        print(f"  overall={r['total']}  wall={r['wall']}  no_wall={r['no_wall']}")

    # Restore default angles
    RF_BODY_ANGLES["rf_FL2"] = math.radians(60)
    RF_BODY_ANGLES["rf_FR2"] = math.radians(-60)

    print()
    print("=" * 65)
    print(f"  {'Angle':>5s}  {'Overall':>12s}  {'Wall':>12s}  {'No wall':>12s}")
    print("-" * 65)
    for r in results:
        marker = " **" if r["success_rate"] == max(x["success_rate"] for x in results) else ""
        print(f"  {r['angle']:>4.0f}°  {r['total']:>12s}  {r['wall']:>12s}  {r['no_wall']:>12s}{marker}")
    print("=" * 65)

    best = max(results, key=lambda x: x["success_rate"])
    print(f"\n  Best angle: {best['angle']}° (overall {best['total']}, wall {best['wall']})")
    print()


if __name__ == "__main__":
    main()
