"""
Diagnostic: pure mapping quality with ZERO dead-reckoning drift.

This script runs the same episodes as the baseline but *overrides* the
navigator's dead-reckoned pose with the MuJoCo ground-truth pose every
timestep. This simulates perfect localization: all DR drift is eliminated.

The resulting mapping IoU is the *ceiling* the mapping algorithm alone
can reach. If this number is high (>0.8), fixing DR drift would recover
most of the gap. If it is low (<0.6), the mapping algorithm itself is
the bottleneck and DR fixes alone won't save us.
"""
import sys, math
from pathlib import Path
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from deskbot.sensors import SensorModel
from deskbot.control import LQRController, StateEstimator
from deskbot.navigation import Navigator
from deskbot.mapviz import extract_gt_obstacles
from deskbot.mapping_eval import rasterize_gt, evaluate_grid
from benchmark_random import random_obstacles, generate_scene_xml


def quat_to_yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def run(seed: int, cheat_pose: bool) -> tuple:
    rng = np.random.default_rng(seed)
    n_obs = int(rng.integers(2, 6))
    obstacles = random_obstacles(rng, n_obs, 1.5)
    xml = generate_scene_xml(obstacles, 1.5)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    sm = SensorModel(model, dt)
    est = StateEstimator(dt)
    ctl = LQRController(mj_model=model)
    nav = Navigator(dt, mj_model=model)
    gt_obs = extract_gt_obstacles(model)
    chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

    mujoco.mj_forward(model, data)
    # Warmup
    for _ in range(500):
        r = sm.read(data); est.update(r)
        if not est.fallen:
            l, rt = ctl.compute(est, 0, 0, dt)
            data.ctrl[0], data.ctrl[1] = l, rt
        mujoco.mj_step(model, data)

    if est.fallen:
        return None

    nav.set_heading(0.0)
    t = 0.0
    while t < 15.0:
        r = sm.read(data); est.update(r)
        if est.fallen: break

        # CHEAT: overwrite DR with ground truth AFTER navigator's internal
        # integration but BEFORE grid update happens. We override here
        # just before the nav.update() call uses these values for the
        # grid shift and ray projection.
        if cheat_pose:
            nav._pos_x = float(data.qpos[0])
            nav._pos_y = float(data.qpos[1])
            nav._heading = quat_to_yaw(data.xquat[chassis_id])

        nv, ny = nav.update(est, r, dt)
        if nv is None: nv = 0.0; ny = 0.0
        l, rt = ctl.compute(est, nv, ny, dt)
        data.ctrl[0], data.ctrl[1] = l, rt
        mujoco.mj_step(model, data)
        t += dt

        if data.qpos[0] >= 12.0:
            break

    gt = rasterize_gt(gt_obs, nav.grid.cx, nav.grid.cy)
    m = evaluate_grid(nav.grid.grid, gt)
    return m


def main():
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]
    print("=" * 75)
    print("  PURE MAPPING (cheat pose = ground truth)  vs  BASELINE (DR)")
    print("=" * 75)
    baseline_ious = []
    cheat_ious = []
    for s in seeds:
        m_base = run(s, cheat_pose=False)
        m_cheat = run(s, cheat_pose=True)
        if m_base is None or m_cheat is None:
            print(f"  seed={s} FELL"); continue
        baseline_ious.append(m_base.iou)
        cheat_ious.append(m_cheat.iou)
        print(f"  seed={s:3d}  baseline IoU={m_base.iou:.3f}  "
              f"P={m_base.precision:.3f} R={m_base.recall:.3f}  |  "
              f"cheat IoU={m_cheat.iou:.3f} P={m_cheat.precision:.3f} "
              f"R={m_cheat.recall:.3f}  "
              f"delta={m_cheat.iou - m_base.iou:+.3f}")

    def mean(xs): return sum(xs) / len(xs) if xs else float('nan')
    print("=" * 75)
    print(f"  Mean baseline IoU: {mean(baseline_ious):.3f}")
    print(f"  Mean cheat    IoU: {mean(cheat_ious):.3f}")
    print(f"  Mean gain        : {mean(cheat_ious) - mean(baseline_ious):+.3f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
