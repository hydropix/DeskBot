"""
Compare yaw integration sources: encoder (current) vs gyro vs complementary.

Hypothesis: encoder-derived yaw_rate accumulates error because of wheel
slip during contour maneuvers. Gyro-derived yaw_rate has bias drift but
no slip. A complementary filter should beat both.
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
from benchmark_random import random_obstacles, generate_scene_xml


def quat_to_yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def run(seed: int, duration: float = 15.0):
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
    chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

    mujoco.mj_forward(model, data)
    for _ in range(500):
        r = sm.read(data); est.update(r)
        if not est.fallen:
            l, rt = ctl.compute(est, 0, 0, dt)
            data.ctrl[0], data.ctrl[1] = l, rt
        mujoco.mj_step(model, data)

    # Init tracked headings to real heading to eliminate warmup artifacts
    real_yaw_init = quat_to_yaw(data.xquat[chassis_id])
    heading_enc = real_yaw_init
    heading_gyro = real_yaw_init
    heading_comp = real_yaw_init
    ALPHA = 0.05  # complementary: 5% encoder, 95% gyro short-term

    nav._heading = real_yaw_init  # navigator uses encoder internally
    nav.set_heading(math.degrees(real_yaw_init))

    t = 0.0
    snapshots = []
    next_t = 2.0
    while t < duration:
        r = sm.read(data); est.update(r)
        if est.fallen: return None

        # Collect sources
        gyro_z = r.gyro[2]       # z-axis rate, rad/s
        enc_yaw_rate = est.yaw_rate  # (vr - vl) / L

        heading_enc  = heading_enc  + enc_yaw_rate * dt
        heading_gyro = heading_gyro + gyro_z * dt
        heading_comp = heading_comp + (ALPHA * enc_yaw_rate + (1 - ALPHA) * gyro_z) * dt

        nv, ny = nav.update(est, r, dt)
        if nv is None: nv = 0.0; ny = 0.0
        l, rt = ctl.compute(est, nv, ny, dt)
        data.ctrl[0], data.ctrl[1] = l, rt
        mujoco.mj_step(model, data)
        t += dt

        if t >= next_t:
            real_yaw = quat_to_yaw(data.xquat[chassis_id])
            snapshots.append((t, real_yaw, heading_enc, heading_gyro, heading_comp))
            next_t += 2.0

        if data.qpos[0] >= 12.0:
            break

    return snapshots


def main():
    seeds = [42, 45, 47, 50, 51, 54, 55]
    err_enc_all = []; err_gyro_all = []; err_comp_all = []
    for s in seeds:
        print(f"\n  seed={s}")
        print(f"    {'t':>5}  {'real':>8}  {'enc':>8}  {'gyro':>8}  {'comp':>8}"
              f"   err_enc   err_gyro   err_comp")
        snaps = run(s)
        if snaps is None:
            print("    FELL"); continue
        for t, r, e, g, c in snaps:
            def fmt(a): return f"{math.degrees(a):+7.2f}°"
            err_e = math.degrees(e - r)
            err_g = math.degrees(g - r)
            err_c = math.degrees(c - r)
            print(f"    {t:5.1f}  {fmt(r)}  {fmt(e)}  {fmt(g)}  {fmt(c)}"
                  f"  {err_e:+8.2f}° {err_g:+8.2f}° {err_c:+8.2f}°")
            err_enc_all.append(abs(err_e))
            err_gyro_all.append(abs(err_g))
            err_comp_all.append(abs(err_c))

    def mean(xs): return sum(xs)/len(xs) if xs else float('nan')
    print()
    print("=" * 75)
    print(f"  Mean |yaw error| over all snapshots:")
    print(f"    encoder (current): {mean(err_enc_all):6.2f}°")
    print(f"    gyro only        : {mean(err_gyro_all):6.2f}°")
    print(f"    complementary 5/95: {mean(err_comp_all):6.2f}°")
    print("=" * 75)


if __name__ == "__main__":
    main()
