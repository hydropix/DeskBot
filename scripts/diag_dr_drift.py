"""
Diagnostic: is the bad mapping caused by DR drift, or by the mapping algorithm itself?

Runs a single random episode and reports IoU computed in TWO frames:
  - real world frame (naive rasterization): measures mapping + localization
  - DR frame (drift-corrected rasterization): measures pure mapping

Delta = DR - real reveals how much of the error is due to DR drift.
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
from deskbot.mapping_eval import (
    rasterize_gt, rasterize_gt_in_dr_frame, evaluate_grid,
)
from benchmark_random import random_obstacles, generate_scene_xml


def quat_to_yaw(q):
    # q = (w, x, y, z)
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def run(seed):
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
    for _ in range(500):
        r = sm.read(data); est.update(r)
        if not est.fallen:
            l, rt = ctl.compute(est, 0, 0, dt)
            data.ctrl[0], data.ctrl[1] = l, rt
        mujoco.mj_step(model, data)

    nav.set_heading(0.0)
    t = 0
    while t < 15.0:
        r = sm.read(data); est.update(r)
        if est.fallen: break
        nv, ny = nav.update(est, r, dt)
        if nv is None: nv = 0.0; ny = 0.0
        l, rt = ctl.compute(est, nv, ny, dt)
        data.ctrl[0], data.ctrl[1] = l, rt
        mujoco.mj_step(model, data)
        t += dt

    # At end, grab poses
    real_x = float(data.qpos[0])
    real_y = float(data.qpos[1])
    real_yaw = quat_to_yaw(data.xquat[chassis_id])
    dr_x = float(nav._pos_x)
    dr_y = float(nav._pos_y)
    dr_yaw = float(nav._heading)

    drift_x = dr_x - real_x
    drift_y = dr_y - real_y
    drift_yaw = dr_yaw - real_yaw

    print(f"  seed={seed}  t={t:.1f}s")
    print(f"  real pose: ({real_x:+.3f}, {real_y:+.3f}, yaw={math.degrees(real_yaw):+.2f}°)")
    print(f"  dr   pose: ({dr_x:+.3f}, {dr_y:+.3f}, yaw={math.degrees(dr_yaw):+.2f}°)")
    print(f"  DRIFT:     ({drift_x:+.3f}, {drift_y:+.3f}, yaw={math.degrees(drift_yaw):+.2f}°)")

    # Metric in real frame (naive)
    gt_real = rasterize_gt(gt_obs, nav.grid.cx, nav.grid.cy)
    m_real = evaluate_grid(nav.grid.grid, gt_real)

    # Metric in DR frame (drift-corrected)
    gt_dr = rasterize_gt_in_dr_frame(
        gt_obs, nav.grid.cx, nav.grid.cy,
        real_pose=(real_x, real_y, real_yaw),
        dr_pose=(dr_x, dr_y, dr_yaw),
    )
    m_dr = evaluate_grid(nav.grid.grid, gt_dr)

    def fmt(m, label):
        print(f"    {label}: IoU={m.iou:.3f}  P={m.precision:.3f}  R={m.recall:.3f}  "
              f"TP={m.tp} FP={m.fp} FN={m.fn}")

    fmt(m_real, "real frame")
    fmt(m_dr,   "DR frame  ")
    if m_dr.iou == m_dr.iou and m_real.iou == m_real.iou:
        print(f"    delta IoU (DR - real) = {m_dr.iou - m_real.iou:+.3f}")


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [47, 51, 48, 42, 54]
    for s in seeds:
        print()
        run(s)
