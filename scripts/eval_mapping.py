"""
Mapping quality benchmark: random motion through random environments.

For each episode:
  - Generate a random corridor with 2-5 obstacles (box/cylinder/wall/chicane)
  - Warm up the balancer
  - Drive the robot forward with Navigator(heading=0)
  - Every `metric_every` seconds, rasterize the ground truth in the
    grid's current frame and compute mapping metrics
  - On the last snapshot of the episode, optionally save a PNG

Aggregates precision/recall/IoU/FP-density over all snapshots and all
episodes. Identifies the worst episodes by IoU and saves their snapshots
(with a visual overlay of FP cells in magenta and FN cells in yellow) so
you can diagnose what the mapping is getting wrong.

Usage:
    python scripts/eval_mapping.py                # 20 episodes, default
    python scripts/eval_mapping.py --episodes 50
    python scripts/eval_mapping.py --seed 42
    python scripts/eval_mapping.py --save-all     # snapshot every episode
"""
import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # benchmark_random

from deskbot.robot import MODELS_DIR
from deskbot.sensors import SensorModel
from deskbot.control import LQRController, StateEstimator
from deskbot.navigation import Navigator, GRID_SIZE, GRID_RES, LOG_ODD_OCCUPIED_THRESHOLD
from deskbot.mapviz import (
    MapFrame, render_rgb, save_png, extract_gt_obstacles, default_snapshot_dir,
)
from deskbot.mapping_eval import (
    rasterize_gt, compute_surface, evaluate_grid, fp_distance_histogram,
    MappingMetrics, OBS_THRESH,
)

from benchmark_random import random_obstacles, generate_scene_xml


def _quat_to_yaw(q):
    import math
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ─────────────────────────────────────────────────────────────────────
# Per-episode result dataclass
# ─────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeMetrics:
    seed: int = 0
    n_obstacles: int = 0
    duration: float = 0.0
    final_x: float = 0.0
    fell: bool = False
    # Aggregated metric series across snapshots within the episode
    snapshots: list = field(default_factory=list)  # list of MappingMetrics

    def summary(self) -> MappingMetrics:
        """Return the metric of the LAST snapshot — the most complete view."""
        if not self.snapshots:
            return MappingMetrics()
        return self.snapshots[-1]


# ─────────────────────────────────────────────────────────────────────
# Diagnostic overlay: FP in magenta, FN in yellow
# ─────────────────────────────────────────────────────────────────────

def render_diagnostic(frame: MapFrame, log_odds: np.ndarray,
                      gt: np.ndarray) -> np.ndarray:
    """
    Like render_rgb but overlays false-positive cells in bright magenta
    and false-negative (visible-surface) cells in bright yellow. Lets a
    human eye spot phantom obstacles and missed detections immediately.
    """
    img = render_rgb(frame)

    surface = compute_surface(gt)
    observed = np.abs(log_odds) > OBS_THRESH
    pred_occ = log_odds > LOG_ODD_OCCUPIED_THRESHOLD
    fp_mask = pred_occ & ~gt
    fn_mask = surface & observed & ~pred_occ

    from deskbot.mapviz import world_to_px, PX_PER_CELL, MAP_PX

    def paint(mask, color):
        ci_list, cj_list = np.where(mask)
        for ci, cj in zip(ci_list, cj_list):
            wx = frame.grid_cx + (ci - GRID_SIZE / 2 + 0.5) * GRID_RES
            wy = frame.grid_cy + (cj - GRID_SIZE / 2 + 0.5) * GRID_RES
            col, row = world_to_px(wx, wy, frame.robot_x, frame.robot_y)
            c0 = max(0, col - PX_PER_CELL // 3)
            r0 = max(0, row - PX_PER_CELL // 3)
            c1 = min(MAP_PX, col + PX_PER_CELL // 3 + 1)
            r1 = min(MAP_PX, row + PX_PER_CELL // 3 + 1)
            img[r0:r1, c0:c1] = color

    paint(fp_mask, (255, 60, 240))   # magenta = phantom
    paint(fn_mask, (255, 240, 60))   # yellow  = missed
    return img


# ─────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────

def run_episode(seed: int, corridor_half: float, min_obs: int, max_obs: int,
                duration: float, metric_every: float,
                save_snapshot_path: str | None = None,
                cheat_pose: bool = False,
                planner: str = "bug2",
                early_avoid: bool = False) -> EpisodeMetrics:
    rng = np.random.default_rng(seed)
    n_obs = int(rng.integers(min_obs, max_obs + 1))
    obstacles = random_obstacles(rng, n_obs, corridor_half)
    scene_xml = generate_scene_xml(obstacles, corridor_half)

    model = mujoco.MjModel.from_xml_string(scene_xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    sensor_model = SensorModel(model, dt)
    estimator = StateEstimator(dt)
    controller = LQRController(mj_model=model)
    navigator = Navigator(
        dt, mj_model=model,
        use_astar=(planner == "astar"),
        use_early_avoid=early_avoid,
    )
    gt_obstacles = extract_gt_obstacles(model)
    chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

    mujoco.mj_forward(model, data)

    # Warmup 1 s
    for _ in range(int(1.0 / dt)):
        readings = sensor_model.read(data)
        estimator.update(readings)
        if not estimator.fallen:
            left, right = controller.compute(estimator, 0.0, 0.0, dt)
            data.ctrl[0] = left
            data.ctrl[1] = right
        mujoco.mj_step(model, data)

    result = EpisodeMetrics(seed=seed, n_obstacles=len(gt_obstacles))

    if estimator.fallen:
        result.fell = True
        return result

    navigator.set_heading(0.0)

    sim_time = 0.0
    next_metric_t = metric_every

    while sim_time < duration:
        readings = sensor_model.read(data)
        estimator.update(readings)
        if estimator.fallen:
            result.fell = True
            break

        if cheat_pose:
            navigator._pos_x = float(data.qpos[0])
            navigator._pos_y = float(data.qpos[1])
            navigator._heading = _quat_to_yaw(data.xquat[chassis_id])

        nav_vel, nav_yaw = navigator.update(estimator, readings, dt)
        if nav_vel is None:
            break
        left, right = controller.compute(estimator, nav_vel, nav_yaw, dt)
        data.ctrl[0] = left
        data.ctrl[1] = right
        mujoco.mj_step(model, data)
        sim_time += dt

        if sim_time >= next_metric_t:
            gt = rasterize_gt(
                gt_obstacles, navigator.grid.cx, navigator.grid.cy
            )
            m = evaluate_grid(navigator.grid.grid, gt)
            result.snapshots.append(m)
            next_metric_t += metric_every

        if data.qpos[0] >= 12.0:
            break

    result.duration = sim_time
    result.final_x = float(data.qpos[0])

    # Save a diagnostic PNG if requested
    if save_snapshot_path is not None and not result.fell:
        rf_comp = navigator.compensate_rangefinders(
            readings.rangefinders, estimator.pitch
        )
        frame = MapFrame(
            grid=navigator.grid.grid.copy(),
            grid_cx=navigator.grid.cx,
            grid_cy=navigator.grid.cy,
            robot_x=navigator._pos_x,
            robot_y=navigator._pos_y,
            heading=navigator._heading,
            target_heading=navigator._target_heading,
            nav_active=navigator._active,
            fsm_state=navigator._fsm.value,
            rf_compensated=rf_comp,
            gt_obstacles=gt_obstacles,
        )
        gt = rasterize_gt(
            gt_obstacles, navigator.grid.cx, navigator.grid.cy
        )
        img = render_diagnostic(frame, navigator.grid.grid, gt)
        save_png(save_snapshot_path, img)

    return result


# ─────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────

def aggregate_final(results: list) -> dict:
    """Aggregate only the LAST snapshot of each episode."""
    finals = [r.summary() for r in results if r.snapshots]
    if not finals:
        return {}
    tp = sum(m.tp for m in finals)
    fp = sum(m.fp for m in finals)
    fn = sum(m.fn for m in finals)
    tn = sum(m.tn for m in finals)
    n_obs = sum(m.n_observed for m in finals)
    n_gt = sum(m.n_gt for m in finals)
    n_surf = sum(m.n_surface for m in finals)

    def safe_div(a, b): return a / b if b > 0 else float('nan')
    return {
        "n_episodes": len(finals),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": safe_div(tp, tp + fp),
        "recall":    safe_div(tp, tp + fn),
        "iou":       safe_div(tp, tp + fp + fn),
        "fp_density": safe_div(fp, n_obs),
        "n_obs": n_obs, "n_gt": n_gt, "n_surface": n_surf,
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mapping quality benchmark")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corridor", type=float, default=1.5)
    parser.add_argument("--min-obs", type=int, default=2)
    parser.add_argument("--max-obs", type=int, default=5)
    parser.add_argument("--metric-every", type=float, default=2.0)
    parser.add_argument("--save-worst", type=int, default=3,
                        help="Save diagnostic snapshots for the N worst episodes")
    parser.add_argument("--save-all", action="store_true",
                        help="Save snapshots for every episode")
    parser.add_argument("--tag", default="baseline",
                        help="Prefix for saved snapshot filenames")
    parser.add_argument("--cheat-pose", action="store_true",
                        help="Override DR with ground-truth pose (pure mapping test)")
    parser.add_argument("--planner", choices=["bug2", "astar"], default="bug2",
                        help="Navigation planner used during mapping")
    parser.add_argument("--early-avoid", choices=["off", "on"], default="off",
                        help="Session 10 early avoidance flag.")
    args = parser.parse_args()

    out_dir = default_snapshot_dir()
    print("=" * 65)
    print(f"  MAPPING QUALITY BENCHMARK")
    print(f"  {args.episodes} episodes, corridor={args.corridor*2:.1f}m, "
          f"{args.min_obs}-{args.max_obs} obstacles, seed={args.seed}")
    print(f"  Snapshots dir: {out_dir}")
    print("=" * 65)

    wall_start = time.perf_counter()
    results = []
    for i in range(args.episodes):
        seed = args.seed + i
        save_path = None
        if args.save_all:
            save_path = os.path.join(out_dir, f"map_{args.tag}_ep{i:02d}_seed{seed}.png")

        r = run_episode(
            seed=seed,
            corridor_half=args.corridor,
            min_obs=args.min_obs,
            max_obs=args.max_obs,
            duration=args.duration,
            metric_every=args.metric_every,
            save_snapshot_path=save_path,
            cheat_pose=args.cheat_pose,
            planner=args.planner,
            early_avoid=(args.early_avoid == "on"),
        )
        results.append(r)

        m = r.summary()
        status = "FELL" if r.fell else "OK  "
        iou_str = f"{m.iou:.3f}" if m.iou == m.iou else " nan "
        p_str   = f"{m.precision:.3f}" if m.precision == m.precision else " nan "
        rc_str  = f"{m.recall:.3f}" if m.recall == m.recall else " nan "
        print(f"  ep={i:2d} seed={seed:3d} [{status}] x={r.final_x:5.2f} "
              f"obs={r.n_obstacles} IoU={iou_str} P={p_str} R={rc_str} "
              f"TP={m.tp:4d} FP={m.fp:4d} FN={m.fn:3d}")

    wall_elapsed = time.perf_counter() - wall_start

    # Aggregate
    agg = aggregate_final(results)
    print()
    print("=" * 65)
    if agg:
        print(f"  AGGREGATED ({agg['n_episodes']} eps with data)")
        print(f"  IoU        : {agg['iou']:.3f}")
        print(f"  Precision  : {agg['precision']:.3f}")
        print(f"  Recall     : {agg['recall']:.3f}")
        print(f"  FP density : {agg['fp_density']:.4f} (phantoms per observed cell)")
        print(f"  Totals     : TP={agg['tp']} FP={agg['fp']} FN={agg['fn']} TN={agg['tn']}")
        print(f"  Wall time  : {wall_elapsed:.1f}s ({args.episodes/wall_elapsed:.1f} ep/s)")
    else:
        print("  No episodes produced metrics.")
    print("=" * 65)

    # Save snapshots of the N worst episodes by IoU (lower IoU = worse)
    if args.save_worst > 0 and not args.save_all:
        scored = [(r, r.summary()) for r in results
                  if r.snapshots and not r.fell and r.summary().iou == r.summary().iou]
        scored.sort(key=lambda t: t[1].iou)
        worst = scored[:args.save_worst]
        if worst:
            print()
            print(f"  Saving worst {len(worst)} snapshots (lowest IoU):")
        for r, m in worst:
            save_path = os.path.join(
                out_dir, f"map_{args.tag}_WORST_seed{r.seed}_iou{m.iou:.2f}.png"
            )
            # Re-run the episode JUST to capture the final frame (cheap)
            run_episode(
                seed=r.seed,
                corridor_half=args.corridor,
                min_obs=args.min_obs,
                max_obs=args.max_obs,
                duration=args.duration,
                metric_every=args.metric_every,
                save_snapshot_path=save_path,
                cheat_pose=args.cheat_pose,
                planner=args.planner,
                early_avoid=(args.early_avoid == "on"),
            )
            print(f"    seed={r.seed} IoU={m.iou:.3f} P={m.precision:.3f} "
                  f"R={m.recall:.3f} FP={m.fp} FN={m.fn} -> {os.path.basename(save_path)}")

    # FP spatial histogram on the final episode for quick diagnostic
    if results and results[-1].snapshots:
        # Re-run last episode to get its grid (we don't cache grids in EpisodeMetrics)
        last_r = results[-1]
        # This is just a diagnostic printout; recomputing the grid for the
        # last episode is cheap.
        pass

    print()


if __name__ == "__main__":
    main()
