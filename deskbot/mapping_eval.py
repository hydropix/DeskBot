"""
Mapping quality evaluation — ground truth vs occupancy grid.

This module provides the *math* base for comparing the robot's occupancy
grid (Navigator.grid, a log-odds array) against a rasterized ground truth
built from the MJCF obstacle list. The metrics it produces are the input
signal for iterative mapping improvements.

Reference frame (critical): both grids live in the *same* world frame,
centered at (grid.cx, grid.cy) — the position to which the log-odds grid
has been shifted during robot motion. Cell (ci, cj) corresponds to world:

    wx = cx + (ci - GRID_SIZE/2 + 0.5) * GRID_RES
    wy = cy + (cj - GRID_SIZE/2 + 0.5) * GRID_RES

which matches exactly the convention used by OccupancyGrid.world_to_cell
in deskbot/navigation.py (the +0.5 accounts for cell centers).

Error taxonomy
--------------
A ray-tracing occupancy grid cannot observe a cell that no ray reaches:
cells deep inside a solid are never touched, so marking them "unknown"
is not an error. Our evaluation therefore distinguishes:

  surface(gt)    — GT cells that are adjacent to at least one non-GT
                   cell (4-connected). These are the cells a rangefinder
                   ray *can* hit. They are the only reasonable targets
                   for the True Positive / False Negative count.

  observed(grid) — cells whose |log_odds| > OBS_THRESH, i.e. updated at
                   least once by some ray.

  pred_occ       — cells whose log_odds > LOG_ODD_OCCUPIED_THRESHOLD.

With these, we define:

  TP = surface ∩ observed ∩ pred_occ       # surface correctly detected
  FN = surface ∩ observed ∩ ¬pred_occ      # surface seen but missed
  FP = pred_occ ∩ ¬gt                       # phantom obstacle anywhere
  TN = observed ∩ ¬gt ∩ ¬pred_occ           # correctly reported free

  IoU(surface)  = TP / (TP + FP + FN)
  Precision     = TP / (TP + FP)
  Recall        = TP / (TP + FN)
  FP_density    = FP / n_observed           # phantom cells per observation

The FP term intentionally counts any predicted-occupied cell outside the
raw GT set, not just those outside the surface — a phantom obstacle in
empty space is always a bug, regardless of how far from a real wall.
"""
import math
from dataclasses import dataclass

import numpy as np

from deskbot.navigation import (
    GRID_SIZE, GRID_RES,
    LOG_ODD_OCCUPIED_THRESHOLD,
)


# Cells whose |log_odds| exceeds this are considered "observed" (touched
# at least once by a ray). Chosen slightly above zero to exclude pristine
# cells but below a single free update (|-0.3|) so a single observation
# counts as observed.
OBS_THRESH = 0.1


# ─────────────────────────────────────────────────────────────────────
# Ground-truth rasterization
# ─────────────────────────────────────────────────────────────────────

def rasterize_gt(gt_obstacles: list, grid_cx: float, grid_cy: float) -> np.ndarray:
    """
    Rasterize a GT obstacle list into a boolean grid in the same frame as
    the log-odds occupancy grid.

    Cell centers are used for the inclusion test — a cell is marked
    occupied iff its center falls inside the obstacle shape. This is the
    coarsest-but-most-honest criterion at the grid's native resolution
    (8 cm cells). We do not inflate.

    Vectorized with numpy; all obstacles are tested against the full grid
    in one pass via broadcasting.
    """
    gt = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    # World center of each cell, precomputed once.
    idx = np.arange(GRID_SIZE, dtype=np.float32)
    offsets = (idx - GRID_SIZE / 2 + 0.5) * GRID_RES
    # WX varies along axis 0 (ci), WY varies along axis 1 (cj)
    WX = grid_cx + offsets[:, None]
    WY = grid_cy + offsets[None, :]

    for o in gt_obstacles:
        t = o.get("type")
        if t == "box":
            mask = (np.abs(WX - o["x"]) <= o["sx"]) & (np.abs(WY - o["y"]) <= o["sy"])
        elif t == "cylinder":
            dx = WX - o["x"]
            dy = WY - o["y"]
            mask = (dx * dx + dy * dy) <= (o["r"] * o["r"])
        elif t == "wall":
            sx = o.get("sx", 0.05)
            mask = (np.abs(WX - o["x"]) <= sx) & (np.abs(WY - o["y"]) <= o["sy"])
        else:
            continue
        gt |= mask

    return gt


def rasterize_gt_in_dr_frame(
    gt_obstacles: list,
    grid_cx: float, grid_cy: float,
    real_pose: tuple, dr_pose: tuple,
) -> np.ndarray:
    """
    Rasterize GT in the robot's DR frame, accounting for dead-reckoning drift.

    Setup: the occupancy grid lives in the robot's DR frame (its beliefs
    about the world). The GT obstacles are defined in the *real* world
    frame. If the DR pose has drifted from the real pose, then placing
    GT obstacles naively at their real-world coordinates inside the DR
    grid gives a comparison where a perfect mapper looks bad only
    because the robot's self-localization is off.

    To measure *pure mapping quality*, we apply the rigid transform that
    maps real poses to DR poses (the drift transform) to every GT
    obstacle before comparison. Equivalently, and more numerically
    stable at the grid level, we instead *inverse-transform each grid
    cell center from DR frame back into world frame*, then test the
    unchanged GT predicates against the transformed cell center. This
    is a change of variables in the rasterization integral; no obstacle
    shape rotation is needed.

    Args:
        real_pose: (rx, ry, rθ) — true robot pose from MuJoCo
        dr_pose:   (dx, dy, dθ) — robot's DR estimate

    Returns a bool GT grid aligned with the log-odds grid at (cx, cy).
    """
    rx, ry, rth = real_pose
    dx, dy, dth = dr_pose
    # Drift heading: how much DR believes it is rotated vs real.
    d_theta = dth - rth
    # We want the transform world -> DR:
    #   p_dr = R(d_theta) * (p_world - real_pose_xy) + dr_pose_xy
    # The inverse (DR -> world) is:
    #   p_world = R(-d_theta) * (p_dr - dr_pose_xy) + real_pose_xy
    cos_a = math.cos(-d_theta)
    sin_a = math.sin(-d_theta)

    idx = np.arange(GRID_SIZE, dtype=np.float32)
    offsets = (idx - GRID_SIZE / 2 + 0.5) * GRID_RES
    WX_dr = grid_cx + offsets[:, None]
    WY_dr = grid_cy + offsets[None, :]

    # Relative to DR robot
    RX = WX_dr - dx
    RY = WY_dr - dy
    # Rotate back by -d_theta, then translate to real robot origin
    WX_real = cos_a * RX - sin_a * RY + rx
    WY_real = sin_a * RX + cos_a * RY + ry

    gt = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    for o in gt_obstacles:
        t = o.get("type")
        if t == "box":
            mask = (np.abs(WX_real - o["x"]) <= o["sx"]) & (np.abs(WY_real - o["y"]) <= o["sy"])
        elif t == "cylinder":
            dxo = WX_real - o["x"]
            dyo = WY_real - o["y"]
            mask = (dxo * dxo + dyo * dyo) <= (o["r"] * o["r"])
        elif t == "wall":
            sx = o.get("sx", 0.05)
            mask = (np.abs(WX_real - o["x"]) <= sx) & (np.abs(WY_real - o["y"]) <= o["sy"])
        else:
            continue
        gt |= mask
    return gt


def compute_surface(gt: np.ndarray) -> np.ndarray:
    """
    Visible surface of GT obstacles: cells occupied AND adjacent (4-conn)
    to at least one non-GT cell. Cells deep inside a solid are excluded
    since no ray can ever reach them.

    Implemented with 1-pixel edge-padded neighbors so cells on the grid
    border are treated correctly (grid-edge = unknown = "free neighbor").
    """
    padded = np.pad(gt, 1, mode='constant', constant_values=False)
    up    = ~padded[:-2, 1:-1]
    down  = ~padded[2:,   1:-1]
    left  = ~padded[1:-1, :-2]
    right = ~padded[1:-1, 2:]
    return gt & (up | down | left | right)


# ─────────────────────────────────────────────────────────────────────
# Metric dataclass
# ─────────────────────────────────────────────────────────────────────

@dataclass
class MappingMetrics:
    """Counts and derived scores for one grid-vs-GT comparison."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    n_observed: int = 0
    n_surface: int = 0
    n_gt: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d > 0 else float('nan')

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d > 0 else float('nan')

    @property
    def iou(self) -> float:
        d = self.tp + self.fp + self.fn
        return self.tp / d if d > 0 else float('nan')

    @property
    def fp_density(self) -> float:
        return self.fp / self.n_observed if self.n_observed > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p != p or r != r or (p + r) == 0:  # nan check
            return float('nan')
        return 2 * p * r / (p + r)


# ─────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────

def dilate_bool(g: np.ndarray, n: int = 1) -> np.ndarray:
    """
    Binary dilation by `n` cells, 4-connected. n=0 is a no-op.

    Used to produce a "forgiving" GT that tolerates cell-level quantization
    errors: a hit that lands in a cell adjacent to the true wall center is
    still considered correct. Justified by the physical fact that an 8 cm
    cell cannot localize a 10 cm wall to better than 1 cell.
    """
    if n <= 0:
        return g.copy()
    out = g.copy()
    for _ in range(n):
        up    = np.zeros_like(out)
        down  = np.zeros_like(out)
        left  = np.zeros_like(out)
        right = np.zeros_like(out)
        up[1:, :]    = out[:-1, :]
        down[:-1, :] = out[1:, :]
        left[:, 1:]  = out[:, :-1]
        right[:, :-1] = out[:, 1:]
        out = out | up | down | left | right
    return out


def evaluate_grid(log_odds: np.ndarray, gt: np.ndarray,
                  tolerance_cells: int = 0) -> MappingMetrics:
    """
    Compute mapping metrics with an optional tolerance radius.

    Args:
        log_odds : float32 grid from Navigator.grid
        gt       : bool GT grid from rasterize_gt()
        tolerance_cells :
            0 → strict cell-center comparison (unforgiving on thin walls).
            1 → a predicted cell within 1 cell of any GT cell counts as
                TP (not as FP), and a surface cell with any prediction
                within 1 cell counts as detected. Physically motivated:
                at 8 cm cell size, a 10 cm-wide wall cannot be localized
                to better than 1 cell. Tolerance 1 is the "fair" metric
                at the current grid resolution.
            2 → overly permissive, useful only to see the ceiling.

    Surface is always computed on the *unmodified* GT so the recall
    denominator remains stable across tolerances.
    """
    observed = np.abs(log_odds) > OBS_THRESH
    pred_occ = log_odds > LOG_ODD_OCCUPIED_THRESHOLD
    surface = compute_surface(gt)

    if tolerance_cells <= 0:
        gt_expanded = gt
        pred_expanded = pred_occ
    else:
        gt_expanded = dilate_bool(gt, tolerance_cells)
        pred_expanded = dilate_bool(pred_occ, tolerance_cells)

    # TP: surface cell detected within tolerance (some prediction near it)
    tp = int(np.sum(surface & observed & pred_expanded))
    # FN: surface cell observed but no prediction within tolerance
    fn = int(np.sum(surface & observed & ~pred_expanded))
    # FP: predicted occupied outside the dilated GT zone (true phantom)
    fp = int(np.sum(pred_occ & ~gt_expanded))
    # TN: observed free, not near any GT and not predicted occupied
    tn = int(np.sum(observed & ~gt_expanded & ~pred_occ))

    return MappingMetrics(
        tp=tp, fp=fp, fn=fn, tn=tn,
        n_observed=int(np.sum(observed)),
        n_surface=int(np.sum(surface)),
        n_gt=int(np.sum(gt)),
    )


# ─────────────────────────────────────────────────────────────────────
# FP spatial analysis — where are the phantoms?
# ─────────────────────────────────────────────────────────────────────

def fp_distance_histogram(log_odds: np.ndarray, gt: np.ndarray,
                          max_dist_m: float = 0.5) -> dict:
    """
    For each false-positive cell, compute its distance to the nearest
    GT cell, and bucketize. Cells very close to GT (< 1 cell) are border
    jitter; cells far from GT are true phantoms.

    Returns a dict: distance_bucket_m -> count.
    """
    pred_occ = log_odds > LOG_ODD_OCCUPIED_THRESHOLD
    fp_mask = pred_occ & ~gt

    if not np.any(fp_mask):
        return {}

    # Euclidean distance transform via scipy, or a dumb BFS if scipy
    # missing. Scipy is already a requirement, so use it.
    from scipy.ndimage import distance_transform_edt
    # distance_transform_edt returns Euclidean distance to the nearest
    # *zero* cell; we want distance to the nearest True GT cell, so we
    # invert and pass ~gt.
    dist_cells = distance_transform_edt(~gt)
    dist_m = dist_cells * GRID_RES

    buckets = {}
    edges = [0.0, 0.08, 0.16, 0.24, 0.32, 0.40, max_dist_m, float('inf')]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = fp_mask & (dist_m >= lo) & (dist_m < hi)
        buckets[f"{lo:.2f}-{hi:.2f}"] = int(np.sum(mask))
    return buckets
