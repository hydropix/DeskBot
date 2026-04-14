"""
A* local planner on the egocentric occupancy grid.

Purpose
-------
At the entry of Bug2's CONTOUR state, we need to pick a direction that
leads around the blocking obstacle. The legacy virtual scan samples 13
discrete directions and picks the best clearance×alignment score — it
cannot reason about two-step detours (e.g. an S-curve through a narrow
wall gap). This module provides a rigorous alternative: classical A*
over the existing log-odds occupancy grid, returning a full cell path
whose initial tangent replaces the scan's chosen angle.

Mathematical specification
--------------------------
Frames:
    - (wx, wy) : world (dead-reckoned) coordinates in metres.
    - (ci, cj) : grid cell indices, same convention as
      deskbot.navigation.OccupancyGrid.world_to_cell.
      The grid is egocentric: ci maps +X (forward), cj maps +Y (left).

Occupancy mask:
    A cell is OCCUPIED iff  log_odds[ci, cj] > LOG_ODD_OCCUPIED_THRESHOLD.
    (Cells outside the grid are treated as unknown → passable.)

Obstacle inflation:
    INFLATE_CELLS = ceil((robot_half_width + safety_margin) / GRID_RES)
                  = ceil((0.13 m + 0.11 m) / 0.08 m)
                  = 3 cells  (24 cm Chebyshev buffer).
    Any free cell within Chebyshev distance ≤ INFLATE_CELLS of an
    occupied cell is marked impassable. Encodes the robot's physical
    footprint + safety buffer. The 11 cm margin (vs. the strict 3 cm
    needed to cover half_width) absorbs dead-reckoning drift, pitch-
    compensation residuals, and rangefinder quantisation — without it,
    A* plans paths that graze the inflation boundary and the real robot
    clips obstacles in practice.

Neighbourhood:
    8-connected. Step cost:
        c_step = 1            if |Δci|+|Δcj| = 1   (axial)
        c_step = √2           if |Δci|=|Δcj|=1     (diagonal)
    Diagonal moves are forbidden when either of the two axial
    "shoulder" cells is blocked (no corner cutting through obstacles).

Cost g(n):
    g(n) = Σ (c_step + φ(n')) along the expanded path, where n' is the
    cell being entered. Units: grid cells.

    φ(n') is the optional repulsive potential — zero by default, or
    α · exp(-d(n') / λ) when the planner is constructed with a positive
    `potential_gain`. d(n') is the Euclidean distance (in cells) from n'
    to the nearest cell of the inflated obstacle mask, obtained via
    `scipy.ndimage.distance_transform_edt`. This makes cells near the
    inflation boundary strictly more expensive to traverse, pushing A*
    paths toward the middle of free corridors without changing the
    algorithm itself.

    Admissibility of the octile heuristic is preserved: φ ≥ 0, so the
    actual cost from any node to the goal is ≥ the pure geometric cost
    that octile computes, hence octile still underestimates it.

Heuristic h(n):
    Octile distance, admissible and consistent for 8-connectivity on a
    uniform grid:
        Δc = |ci - gi| , Δr = |cj - gj|
        h = max(Δc, Δr) + (√2 - 1) · min(Δc, Δr)
    It is the exact cost of the cheapest 8-conn path in an empty grid,
    so it never overestimates the true cost in any grid.

Termination:
    Success              → goal cell popped from the open set.
    Impossibility        → open set empty (no path).
    Budget exceeded      → expanded > max_iterations (timeout).

Complexity:
    O(N log N) where N = cells expanded, bounded by max_iterations
    (default 2000, i.e. ~55 % of the 60×60 grid).

Dependencies:
    numpy (already used across the project) + heapq from stdlib. No
    new runtime dependency.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


# ─────────────────────────────────────────────────────────────────────
# Tunable constants — justified in the module docstring.
# ─────────────────────────────────────────────────────────────────────

#: Inflation radius, in cells. 3 @ 8 cm = 24 cm Chebyshev buffer, which
#: covers the 13 cm half-width of DeskBot and leaves ~11 cm of margin.
#: Bumped from 2 to 3 after empirically observing the robot clip obstacles
#: when the planned path hugged the inflation boundary — 3 cm of margin
#: was not enough to absorb DR drift + pitch residuals + encoder noise.
INFLATE_CELLS = 3

#: Hard bound on cells expanded by A*. Matches ~55 % of the 60x60 grid
#: — well above the distance any realistic contour detour takes.
MAX_ITERATIONS = 2000

#: Planning horizon in metres. Goal cell is picked this far ahead of
#: the robot along the target heading.
LOCAL_HORIZON = 1.5

#: Radius (in cells) of the fallback neighbourhood scan when the nominal
#: goal falls inside an inflated obstacle. 2 cells = 16 cm search box.
GOAL_SEARCH_RADIUS = 2

#: Default repulsive-potential gain α (dimensionless, same units as the
#: per-step A* cost — so α=0.6 means a cell glued to the inflation
#: boundary costs as much as 0.6 extra axial steps to enter). Zero
#: disables the potential field entirely, giving the legacy pure-geometric
#: cost. Value picked so that a cell at d=0 is ~half as expensive as a
#: diagonal step (√2 ≈ 1.41), i.e. strong bias but never dominating the
#: shortest-path term.
POTENTIAL_GAIN = 0.6

#: Default decay length λ in cells. exp(-d/λ) drops to ~5% at d = 3λ, so
#: λ=3 means the repulsion is still meaningful ~9 cells (~72 cm) away
#: from an obstacle and nearly zero beyond that. Matches roughly the
#: CAUTION_DIST envelope used by the reactive layer in navigation.py.
POTENTIAL_LAMBDA_CELLS = 3.0


def inflate_mask(occupied: np.ndarray, radius: int) -> np.ndarray:
    """
    Dilate `occupied` by `radius` cells under Chebyshev distance.

    Iterative 3×3 square dilation: each pass OR-merges every cell with
    its eight neighbours, so after k passes any cell within Chebyshev
    distance k of an originally occupied cell is set. Cost is
    O(k · rows · cols) — a few thousand ops on a 60×60 grid at k=2.

    Exposed as a module-level helper so `OccupancyGrid` can share the
    same dilation kernel as `AStarPlanner` without bouncing through a
    private static method.
    """
    mask = occupied.astype(bool, copy=True)
    if radius <= 0:
        return mask
    for _ in range(int(radius)):
        out = mask.copy()
        out[1:, :]   |= mask[:-1, :]
        out[:-1, :]  |= mask[1:, :]
        out[:, 1:]   |= mask[:, :-1]
        out[:, :-1]  |= mask[:, 1:]
        out[1:, 1:]    |= mask[:-1, :-1]
        out[:-1, :-1]  |= mask[1:, 1:]
        out[1:, :-1]   |= mask[:-1, 1:]
        out[:-1, 1:]   |= mask[1:, :-1]
        mask = out
    return mask


# ─────────────────────────────────────────────────────────────────────
# Cached neighbourhood: (Δci, Δcj, step_cost)
# ─────────────────────────────────────────────────────────────────────
_SQRT2 = math.sqrt(2.0)
_OCTILE_D2 = _SQRT2 - 1.0

_NEIGHBOURS: tuple[tuple[int, int, float], ...] = (
    (-1, -1, _SQRT2), (-1, 0, 1.0), (-1, 1, _SQRT2),
    ( 0, -1, 1.0),                  ( 0, 1, 1.0),
    ( 1, -1, _SQRT2), ( 1, 0, 1.0), ( 1, 1, _SQRT2),
)


def _build_cost_field(
    blocked: np.ndarray,
    gain: float,
    lambda_cells: float,
) -> np.ndarray | None:
    """
    Build the per-cell repulsive potential φ(cell) = α · exp(-d / λ).

    d(cell) is the Euclidean distance (in cells) from `cell` to the
    nearest True pixel of `blocked`, computed by
    `scipy.ndimage.distance_transform_edt` on the inverted mask.
    Returns None when the field is disabled (gain ≤ 0 or λ ≤ 0), or
    when there is no obstacle at all in the grid (distance transform
    is undefined — every cell would have infinite distance and zero
    cost anyway).
    """
    if gain <= 0.0 or lambda_cells <= 0.0:
        return None
    if not blocked.any():
        return None
    dist_cells = distance_transform_edt(~blocked)
    field = gain * np.exp(-dist_cells / float(lambda_cells))
    return field.astype(np.float32, copy=False)


@dataclass
class AStarResult:
    """Outcome of a single plan() call."""
    path: list[tuple[int, int]] | None
    expanded: int
    reason: str  # "ok", "start_blocked", "goal_blocked", "no_path", "budget"


class AStarPlanner:
    """
    8-connected A* over an inflated occupancy mask, optionally with a
    repulsive potential field baked into the step cost.

    Parameters
    ----------
    log_odds : np.ndarray (GRID_SIZE, GRID_SIZE), float32
        Raw log-odds grid from OccupancyGrid.grid. The constructor
        thresholds it and computes the inflated mask once; subsequent
        plan() calls reuse this pre-computed map.
    occ_threshold : float
        Log-odds threshold above which a cell is considered occupied.
        Should match OccupancyGrid.LOG_ODD_OCCUPIED_THRESHOLD.
    inflate_cells : int
        Chebyshev dilation radius applied to the occupied mask.
    max_iterations : int
        Hard cap on cells expanded by plan().
    potential_gain : float
        α in φ(cell) = α · exp(-d / λ). Zero disables the field and the
        planner behaves exactly like the legacy pure-geometric A*.
    potential_lambda_cells : float
        λ in cells. Ignored when `potential_gain` is zero. Must be > 0
        when the field is active.
    """

    def __init__(
        self,
        log_odds: np.ndarray,
        occ_threshold: float,
        inflate_cells: int = INFLATE_CELLS,
        max_iterations: int = MAX_ITERATIONS,
        potential_gain: float = 0.0,
        potential_lambda_cells: float = POTENTIAL_LAMBDA_CELLS,
    ):
        occupied = log_odds > occ_threshold
        self._blocked = inflate_mask(occupied, inflate_cells)
        self._max_iterations = int(max_iterations)
        self._rows, self._cols = self._blocked.shape
        self._cost_field = _build_cost_field(
            self._blocked, potential_gain, potential_lambda_cells,
        )

    # ── Queries ─────────────────────────────────────────────────────

    @property
    def blocked_mask(self) -> np.ndarray:
        """Read-only view of the inflated obstacle mask (for tests/viz)."""
        return self._blocked

    @property
    def cost_field(self) -> np.ndarray | None:
        """
        Read-only view of the repulsive potential φ(cell), or None if
        the field is disabled. Useful for visualization: heatmap this
        and you see the soft inflation gradient around every obstacle.
        """
        return self._cost_field

    def is_free(self, ci: int, cj: int) -> bool:
        if ci < 0 or ci >= self._rows or cj < 0 or cj >= self._cols:
            return False
        return not self._blocked[ci, cj]

    # ── Core search ─────────────────────────────────────────────────

    @staticmethod
    def _octile(ci: int, cj: int, gi: int, gj: int) -> float:
        dc = abs(ci - gi)
        dr = abs(cj - gj)
        lo, hi = (dc, dr) if dc < dr else (dr, dc)
        return hi + _OCTILE_D2 * lo

    def plan(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> AStarResult:
        """
        Compute a shortest 8-connected path from `start` to `goal`.

        Returns an AStarResult with the cell path (inclusive of both
        endpoints) or None plus a human-readable reason. Also exposes
        the number of cells expanded for diagnostics.
        """
        if not self.is_free(*start):
            return AStarResult(None, 0, "start_blocked")
        if not self.is_free(*goal):
            return AStarResult(None, 0, "goal_blocked")
        if start == goal:
            return AStarResult([start], 0, "ok")

        gi, gj = goal
        # Tie-breaker counter so heapq never compares tuples on the
        # second field when two f-scores coincide.
        counter = 0
        g_score: dict[tuple[int, int], float] = {start: 0.0}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        open_heap: list[tuple[float, int, tuple[int, int]]] = []
        heapq.heappush(
            open_heap,
            (self._octile(start[0], start[1], gi, gj), counter, start),
        )
        closed: set[tuple[int, int]] = set()
        expanded = 0

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                path = self._reconstruct(came_from, current)
                return AStarResult(path, expanded, "ok")
            closed.add(current)

            expanded += 1
            if expanded > self._max_iterations:
                return AStarResult(None, expanded, "budget")

            ci, cj = current
            g_cur = g_score[current]
            cost_field = self._cost_field
            for dci, dcj, step in _NEIGHBOURS:
                ni, nj = ci + dci, cj + dcj
                if not self.is_free(ni, nj):
                    continue
                # No corner cutting: a diagonal step must have both
                # axial shoulder cells clear. This preserves the octile
                # heuristic's admissibility near wall corners.
                if dci != 0 and dcj != 0:
                    if self._blocked[ci + dci, cj] or self._blocked[ci, cj + dcj]:
                        continue
                tentative = g_cur + step
                if cost_field is not None:
                    tentative += float(cost_field[ni, nj])
                nb = (ni, nj)
                if tentative < g_score.get(nb, math.inf):
                    g_score[nb] = tentative
                    came_from[nb] = current
                    f = tentative + self._octile(ni, nj, gi, gj)
                    counter += 1
                    heapq.heappush(open_heap, (f, counter, nb))

        return AStarResult(None, expanded, "no_path")

    @staticmethod
    def _reconstruct(
        came_from: dict[tuple[int, int], tuple[int, int]],
        current: tuple[int, int],
    ) -> list[tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


# ─────────────────────────────────────────────────────────────────────
# Goal selection helpers
# ─────────────────────────────────────────────────────────────────────

def nearest_free_cell(
    planner: AStarPlanner,
    ci: int,
    cj: int,
    radius: int = GOAL_SEARCH_RADIUS,
) -> tuple[int, int] | None:
    """
    Return the closest free cell to (ci, cj) within a ±radius square,
    using Chebyshev-distance rings expanding outward.

    This is the "nudge the goal out of the inflated obstacle" step from
    the plan: if the nominal horizon point falls inside an obstacle's
    safety buffer, we search a small neighbourhood for the nearest free
    cell and use it instead. Rings are iterated by Chebyshev distance
    (BFS-style on a square), so the first free cell we hit is a closest
    one (ties broken by row-major order).
    """
    if planner.is_free(ci, cj):
        return (ci, cj)
    for d in range(1, radius + 1):
        for di in range(-d, d + 1):
            for dj in range(-d, d + 1):
                # Only the ring at Chebyshev distance exactly d.
                if max(abs(di), abs(dj)) != d:
                    continue
                ni, nj = ci + di, cj + dj
                if planner.is_free(ni, nj):
                    return (ni, nj)
    return None


def path_initial_tangent(
    path: list[tuple[int, int]],
    look_ahead: int = 4,
) -> float | None:
    """
    Angle (in the world frame, radians) of the path's initial tangent.

    We average over the first `look_ahead` segments so a single diagonal
    quirk near the start doesn't dominate. Returns None if the path is
    too short.

    IMPORTANT: the grid convention is ci→X, cj→Y, so the world vector
    from start cell to target cell is (Δci, Δcj). math.atan2(Δcj, Δci)
    therefore yields the heading in the same convention Navigator uses.
    """
    if not path or len(path) < 2:
        return None
    start = path[0]
    end = path[min(look_ahead, len(path) - 1)]
    dci = end[0] - start[0]
    dcj = end[1] - start[1]
    if dci == 0 and dcj == 0:
        return None
    return math.atan2(float(dcj), float(dci))
