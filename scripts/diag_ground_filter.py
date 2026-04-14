"""
Ground-filter false-positive diagnostic.

For every rangefinder reading, compares two independent views:

  1. Ground truth (raycast + geom_id)
     We re-raycast directly with mj_ray and inspect the geom actually hit.
     If the hit belongs to the floor plane → the beam saw ground.
     Anything else → the beam saw a real obstacle.

  2. GroundGeometry.classify() output
     The live filter that navigation.compensate_rangefinders relies on,
     fed with the noisy sensor reading and the estimator's pitch.

Each sample is bucketed into one of four cells of a confusion matrix:

    GT vs Filter        flat / no_reading       obstacle / hole
    floor               TN                      FP  (ground leak)
    real obstacle       FN                      TP

The script runs several randomized benchmark_random-style episodes with
navigation enabled so the robot actually pitches, turns, and collides.
Every false positive is logged with:
    sensor, pitch_true, pitch_est, Δpitch, pitch_rate, measured, expected,
    tolerance, deviation, ray-to-horizon angle.

At the end, aggregate counts per sensor + histograms of the FP conditions
are printed. Goal: find the pitch/rate regimes where the current filter
leaks, so the filter can be tightened without sacrificing real detections.
"""
import sys
import math
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deskbot.robot import MODELS_DIR
from deskbot.sensors import SensorModel, RF_NAMES
from deskbot.control import BalanceController, StateEstimator
from deskbot.navigation import Navigator, POST_COLLISION_LOCKOUT_S
from deskbot.perception import GroundGeometry

# Reuse the random-scene generator from the main benchmark.
from benchmark_random import generate_scene_xml, random_obstacles


# ── Sample record ─────────────────────────────────────────────────

@dataclass
class FpSample:
    ep: int
    t: float
    sensor: str
    pitch_true: float
    pitch_est: float
    pitch_err: float
    pitch_rate: float
    measured: float
    expected: float
    tol: float
    deviation: float
    ray_angle_deg: float   # ray direction vs horizontal plane (deg,
                           # positive = pointing down)
    kind: str              # filter classification ("obstacle"/"hole"/...)


@dataclass
class EpisodeStats:
    samples: int = 0
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    fp_list: list[FpSample] = field(default_factory=list)
    per_sensor: dict[str, Counter] = field(
        default_factory=lambda: {n: Counter() for n in RF_NAMES}
    )


def chassis_pitch_true(model, data) -> float:
    """Extract true chassis pitch (radians) from the compiled state.

    Positive = leaning forward, matching the estimator/perception
    convention. Computed from the chassis body orientation matrix using
    the yaw-robust formula pitch = atan2(sin(θ), cos(θ)).
    """
    chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    R = data.xmat[chassis_id].reshape(3, 3)
    # After Rz(yaw) * Ry(pitch): R[2,0] = -sin(pitch), R[2,2] = cos(pitch)
    return math.atan2(-R[2, 0], R[2, 2])


def ray_angle_to_horizontal(model, data, site_id) -> float:
    """Current direction of a sensor ray vs the world horizontal plane.

    Returns degrees, positive = pointing downward. This is the quantity
    that controls where the beam hits the floor; near 0° means the ray
    is almost parallel to the ground (ill-conditioned regime).
    """
    z_axis_world = data.site_xmat[site_id].reshape(3, 3)[:, 2]
    return math.degrees(math.asin(-z_axis_world[2]))


def run_episode(ep_idx: int, seed: int, max_duration: float,
                chw: float, min_obs: int, max_obs: int) -> EpisodeStats:
    rng = np.random.default_rng(seed)
    n_obs = int(rng.integers(min_obs, max_obs + 1))
    obstacles = random_obstacles(rng, n_obs, chw)
    xml = generate_scene_xml(obstacles, chw)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    mujoco.mj_forward(model, data)

    sensor_model = SensorModel(model, dt)
    estimator = StateEstimator(dt)
    controller = BalanceController()
    navigator = Navigator(dt, mj_model=model)
    ground = GroundGeometry(model)

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

    # Site ids for re-raycasting on ground truth.
    site_ids = {}
    for name in RF_NAMES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            site_ids[name] = sid

    geomid_buf = np.zeros(1, dtype=np.int32)

    # Warm-up: let the balance loop settle before any stats.
    for _ in range(int(1.0 / dt)):
        readings = sensor_model.read(data)
        estimator.update(readings)
        if estimator.fallen:
            break
        left, right = controller.compute(estimator, 0.0, 0.0, dt)
        data.ctrl[0] = left
        data.ctrl[1] = right
        mujoco.mj_step(model, data)

    if estimator.fallen:
        return EpisodeStats()

    navigator.set_heading(0.0)

    stats = EpisodeStats()
    t = 0.0
    prev_pitch_true = chassis_pitch_true(model, data)
    lockout_t = 0.0

    while t < max_duration:
        readings = sensor_model.read(data)
        estimator.update(readings)

        if estimator.fallen:
            break

        # ── Ground truth via a clean (noise-free) raycast ──
        pitch_true = chassis_pitch_true(model, data)
        pitch_rate = (pitch_true - prev_pitch_true) / dt
        prev_pitch_true = pitch_true
        pitch_est = float(estimator.pitch)

        # Mirror Navigator's post-collision lockout so diag counts
        # reflect the filter as it is actually applied online.
        if lockout_t > 0.0:
            lockout_t = max(0.0, lockout_t - dt)
        if readings.collision_detected:
            lockout_t = POST_COLLISION_LOCKOUT_S
        lockout_active = lockout_t > 0.0

        for name, site_id in site_ids.items():
            origin = data.site_xpos[site_id].copy()
            direction = data.site_xmat[site_id].reshape(3, 3)[:, 2].copy()
            gt_dist = mujoco.mj_ray(
                model, data, origin, direction,
                None, 1, chassis_id, geomid_buf,
            )
            gt_geom = int(geomid_buf[0])

            measured = readings.rangefinders[name]
            gr = ground.classify(name, measured, pitch_est)

            # Reproduce the online lockout: during the post-collision
            # window, ground-facing beams are force-suppressed before
            # they ever reach the classifier.
            if lockout_active and ground.is_ground_facing(name):
                gr = type(gr)(
                    name, measured, gr.expected,
                    float("nan"), float("nan"), "no_reading",
                )

            # Decide truth label:
            #   floor   → "ground"
            #   other   → "obstacle"
            #   no hit  → "free"
            if gt_dist < 0:
                truth = "free"
            elif gt_geom == floor_id:
                truth = "ground"
            else:
                truth = "obstacle"

            # Filter label: map to what compensate_rangefinders actually
            # pushes into the grid. "flat", "hole" and "no_reading" all
            # become -1 (suppressed = free); "obstacle" and
            # "no_ground_expected" become a kept measurement (obstacle).
            filt = gr.kind
            if filt in ("obstacle", "no_ground_expected"):
                filt_decision = "obstacle"
            else:
                filt_decision = "flat"

            stats.samples += 1
            stats.per_sensor[name][f"{truth}|{filt_decision}"] += 1

            if truth == "ground" and filt_decision == "obstacle":
                stats.fp += 1
                stats.fp_list.append(FpSample(
                    ep=ep_idx, t=t, sensor=name,
                    pitch_true=pitch_true, pitch_est=pitch_est,
                    pitch_err=pitch_est - pitch_true,
                    pitch_rate=pitch_rate,
                    measured=measured, expected=gr.expected,
                    tol=gr.tolerance, deviation=gr.deviation,
                    ray_angle_deg=ray_angle_to_horizontal(model, data, site_id),
                    kind=filt,
                ))
            elif truth == "ground" and filt_decision == "flat":
                stats.tn += 1
            elif truth == "obstacle" and filt_decision == "obstacle":
                stats.tp += 1
            elif truth == "obstacle" and filt_decision == "flat":
                stats.fn += 1
            # "free" (no-hit ground truth) is not counted — we cannot
            # say whether the filter was right or wrong when the ray
            # saw nothing at all.

        nav_vel, nav_yaw = navigator.update(estimator, readings, dt)
        if nav_vel is None:
            break
        left, right = controller.compute(estimator, nav_vel, nav_yaw, dt)
        data.ctrl[0] = left
        data.ctrl[1] = right
        mujoco.mj_step(model, data)
        t += dt

        if data.qpos[0] >= 12.0:
            break

    return stats


# ── Reporting ─────────────────────────────────────────────────────

def print_confusion(all_stats: list[EpisodeStats]):
    tp = sum(s.tp for s in all_stats)
    tn = sum(s.tn for s in all_stats)
    fp = sum(s.fp for s in all_stats)
    fn = sum(s.fn for s in all_stats)
    total = tp + tn + fp + fn
    print()
    print("=" * 60)
    print(f"CONFUSION MATRIX  (total classified samples: {total})")
    print("=" * 60)
    print(f"                    filter=flat       filter=obstacle")
    print(f"  truth=ground      TN={tn:7d}      FP={fp:7d}  <<< leak")
    print(f"  truth=obstacle    FN={fn:7d}      TP={tp:7d}")
    if fp + tn > 0:
        leak = 100 * fp / (fp + tn)
        print(f"\nGround leak rate : {leak:6.3f}%  ({fp} of {fp+tn} ground hits)")
    if tp + fn > 0:
        recall = 100 * tp / (tp + fn)
        print(f"Obstacle recall  : {recall:6.3f}%")
    print()


def print_per_sensor(all_stats: list[EpisodeStats]):
    agg: dict[str, Counter] = {n: Counter() for n in RF_NAMES}
    for s in all_stats:
        for name, c in s.per_sensor.items():
            agg[name].update(c)

    print("=" * 60)
    print("PER-SENSOR BREAKDOWN")
    print("=" * 60)
    print(f"{'sensor':8s} | {'ground/OK':>12s} | {'ground/LEAK':>12s} | "
          f"{'obs/OK':>10s} | {'obs/MISS':>10s}")
    print("-" * 60)
    for name in RF_NAMES:
        c = agg[name]
        tn = c["ground|flat"]
        fp = c["ground|obstacle"]
        tp = c["obstacle|obstacle"]
        fn = c["obstacle|flat"]
        flag = " <<" if fp > 0 else ""
        print(f"{name:8s} | {tn:12d} | {fp:12d}{flag:3s} | {tp:10d} | {fn:10d}")
    print()


def print_fp_details(all_stats: list[EpisodeStats], top_n: int = 25):
    all_fps: list[FpSample] = []
    for s in all_stats:
        all_fps.extend(s.fp_list)
    if not all_fps:
        print("No ground-leak false positives observed.")
        return

    print("=" * 60)
    print(f"FALSE-POSITIVE CONDITIONS  (n={len(all_fps)})")
    print("=" * 60)

    # Histograms
    def buckets(values, edges, label):
        print(f"\n{label}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            n = sum(1 for v in values if lo <= v < hi)
            bar = "#" * min(60, n)
            print(f"  [{lo:+6.2f} .. {hi:+6.2f})  {n:5d}  {bar}")

    pitches_true = [fp.pitch_true for fp in all_fps]
    pitches_err = [fp.pitch_err for fp in all_fps]
    pitch_rates = [fp.pitch_rate for fp in all_fps]
    ray_angs = [fp.ray_angle_deg for fp in all_fps]
    dev_over_tol = [
        (fp.expected - fp.measured) / fp.tol
        if (fp.tol > 0 and math.isfinite(fp.tol)) else math.nan
        for fp in all_fps
    ]

    buckets(pitches_true,
            [-0.25, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.25],
            "pitch_true (rad, +=forward)")
    buckets(pitches_err,
            [-0.10, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.10],
            "pitch_est − pitch_true (rad)")
    buckets(pitch_rates,
            [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0],
            "pitch_rate (rad/s)")
    buckets(ray_angs,
            [-30, -15, -5, 0, 5, 15, 30, 60, 90],
            "ray-to-horizontal (deg, += points down)")
    good_dovt = [x for x in dev_over_tol if math.isfinite(x)]
    if good_dovt:
        buckets(good_dovt,
                [-10, -5, -3, -2, -1, 0, 1, 2, 5, 10],
                "(expected−measured)/tol  (how far over the filter tolerance)")

    # Worst offenders
    print("\nTop offenders (largest |pitch_err|):")
    for fp in sorted(all_fps, key=lambda f: abs(f.pitch_err), reverse=True)[:top_n]:
        print(f"  ep={fp.ep:2d} t={fp.t:5.2f} {fp.sensor:5s} "
              f"pitch_true={math.degrees(fp.pitch_true):+6.1f}° "
              f"pitch_est={math.degrees(fp.pitch_est):+6.1f}° "
              f"Δ={math.degrees(fp.pitch_err):+5.1f}° "
              f"rate={fp.pitch_rate:+5.2f} "
              f"ray={fp.ray_angle_deg:+5.1f}° "
              f"meas={fp.measured:5.2f} exp={fp.expected:5.2f} "
              f"dev={fp.deviation:+.3f} tol={fp.tol:.3f} "
              f"kind={fp.kind}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=25.0)
    parser.add_argument("--corridor", type=float, default=1.5)
    parser.add_argument("--min-obs", type=int, default=2)
    parser.add_argument("--max-obs", type=int, default=5)
    args = parser.parse_args()

    all_stats: list[EpisodeStats] = []
    for ep in range(args.episodes):
        print(f"[ep {ep+1:2d}/{args.episodes}] seed={args.seed+ep} ... ", end="",
              flush=True)
        stats = run_episode(
            ep, args.seed + ep, args.duration,
            args.corridor, args.min_obs, args.max_obs,
        )
        print(f"samples={stats.samples:6d} fp={stats.fp:4d} "
              f"tp={stats.tp:5d} fn={stats.fn:4d}")
        all_stats.append(stats)

    print_confusion(all_stats)
    print_per_sensor(all_stats)
    print_fp_details(all_stats)


if __name__ == "__main__":
    main()
