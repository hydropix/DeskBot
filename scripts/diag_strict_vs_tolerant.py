"""
Diagnostic: strict cell-center metric vs 1-cell-dilated tolerant metric.

Both computed with cheat-pose (ground-truth localization) to isolate
the pure mapping error. The gap between strict and tolerant tells us
how much of the error is just cell-level quantization vs genuine
phantoms scattered across free space.
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


def run(seed: int):
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
    t = 0.0
    while t < 12.0:
        r = sm.read(data); est.update(r)
        if est.fallen: return None
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
    m_strict = evaluate_grid(nav.grid.grid, gt, tolerance_cells=0)
    m_tol1   = evaluate_grid(nav.grid.grid, gt, tolerance_cells=1)
    return m_strict, m_tol1


def main():
    seeds = list(range(42, 57))
    print("=" * 95)
    print("  CHEAT POSE: strict vs 1-cell tolerant metric (fair at 8 cm grid)")
    print("=" * 95)
    strict_ious, tol_ious = [], []
    strict_ps, tol_ps = [], []
    strict_rs, tol_rs = [], []
    for s in seeds:
        out = run(s)
        if out is None:
            print(f"  seed={s:3d} FELL"); continue
        ms, m1 = out
        strict_ious.append(ms.iou); tol_ious.append(m1.iou)
        strict_ps.append(ms.precision); tol_ps.append(m1.precision)
        strict_rs.append(ms.recall); tol_rs.append(m1.recall)
        print(f"  seed={s:3d}  strict IoU={ms.iou:.3f} P={ms.precision:.3f} R={ms.recall:.3f} "
              f"FP={ms.fp:4d} FN={ms.fn:3d} |  "
              f"tol IoU={m1.iou:.3f} P={m1.precision:.3f} R={m1.recall:.3f} "
              f"FP={m1.fp:4d} FN={m1.fn:3d}")

    def mean(xs): return sum(xs)/len(xs) if xs else float('nan')
    print("=" * 95)
    print(f"  Mean IoU:       strict={mean(strict_ious):.3f}  tol-1={mean(tol_ious):.3f}")
    print(f"  Mean Precision: strict={mean(strict_ps):.3f}  tol-1={mean(tol_ps):.3f}")
    print(f"  Mean Recall:    strict={mean(strict_rs):.3f}  tol-1={mean(tol_rs):.3f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
