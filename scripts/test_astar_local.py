"""
Unit tests for deskbot/astar_local.py on synthetic occupancy grids.

Each test builds a log-odds grid by hand, runs the planner, and prints a
human-readable summary (path length, expanded cells, ASCII map of the
grid with the path overlaid). No framework dependency — just run it:

    python scripts/test_astar_local.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deskbot.astar_local import (
    AStarPlanner,
    nearest_free_cell,
    path_initial_tangent,
    INFLATE_CELLS,
)


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

GRID_SIZE = 60
OCC_THRESHOLD = 0.5
STRONG_OCC = 2.0  # well above threshold


def empty_grid() -> np.ndarray:
    return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)


def put_wall(grid: np.ndarray, ci: int, cj_min: int, cj_max: int):
    """Vertical wall at column ci spanning [cj_min, cj_max]."""
    grid[ci, cj_min:cj_max + 1] = STRONG_OCC


def put_box(grid: np.ndarray, ci_min: int, ci_max: int, cj_min: int, cj_max: int):
    grid[ci_min:ci_max + 1, cj_min:cj_max + 1] = STRONG_OCC


# ─────────────────────────────────────────────────────────────────────
# ASCII rendering
# ─────────────────────────────────────────────────────────────────────

def ascii_render(
    log_odds: np.ndarray,
    blocked: np.ndarray,
    path: list[tuple[int, int]] | None,
    start: tuple[int, int],
    goal: tuple[int, int],
    window: int = 22,
) -> str:
    """Print a window centered on start-goal midpoint."""
    ci_c = (start[0] + goal[0]) // 2
    cj_c = (start[1] + goal[1]) // 2
    i0 = max(0, ci_c - window)
    i1 = min(GRID_SIZE, ci_c + window)
    j0 = max(0, cj_c - window)
    j1 = min(GRID_SIZE, cj_c + window)

    path_set = set(path) if path else set()
    lines: list[str] = []
    # Print j decreasing downward so +Y is up visually.
    for j in range(j1 - 1, j0 - 1, -1):
        row_chars: list[str] = []
        for i in range(i0, i1):
            ch = "."
            if log_odds[i, j] > OCC_THRESHOLD:
                ch = "#"
            elif blocked[i, j]:
                ch = "o"
            if (i, j) in path_set:
                ch = "*"
            if (i, j) == start:
                ch = "S"
            if (i, j) == goal:
                ch = "G"
            row_chars.append(ch)
        lines.append("".join(row_chars))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────

def test_empty_grid_straight_line():
    print("\n-- test_empty_grid_straight_line --")
    grid = empty_grid()
    planner = AStarPlanner(grid, OCC_THRESHOLD)

    start = (10, 30)
    goal = (40, 30)
    r = planner.plan(start, goal)

    assert r.path is not None, "A* failed on empty grid"
    assert r.path[0] == start and r.path[-1] == goal
    # On an empty grid, cost = max(Δci, Δcj) = 30.
    # The reported path has 31 cells for a 30-cell displacement.
    print(f"  path length     = {len(r.path)} cells")
    print(f"  expanded        = {r.expanded}")
    print(f"  reason          = {r.reason}")
    assert len(r.path) == 31, f"expected 31 cells, got {len(r.path)}"
    tangent = path_initial_tangent(r.path)
    print(f"  initial tangent = {math.degrees(tangent):+.1f}°")
    assert abs(tangent) < 1e-6  # straight along +X
    print("  PASS")


def test_wall_with_gap():
    """
    Vertical wall at ci=30 spanning the whole grid except a gap at
    cj∈[28,32]. Start on the left, goal on the right. A* must route
    through the gap.
    """
    print("\n-- test_wall_with_gap --")
    grid = empty_grid()
    put_wall(grid, 30, 0, 27)
    put_wall(grid, 30, 33, GRID_SIZE - 1)

    planner = AStarPlanner(grid, OCC_THRESHOLD)
    start = (5, 5)
    goal = (55, 55)
    r = planner.plan(start, goal)

    assert r.path is not None, f"A* failed: {r.reason}"
    print(f"  path length = {len(r.path)} cells")
    print(f"  expanded    = {r.expanded}")
    print(f"  reason      = {r.reason}")
    # Sanity: the path must cross ci=30 via the gap, i.e. at a cj in
    # [28-INFLATE_CELLS+1 .. 32+INFLATE_CELLS-1]. With 2-cell inflation
    # the gap width 5 is just barely passable at the centre.
    crossings = [cj for (ci, cj) in r.path if ci == 30]
    print(f"  ci=30 crossings at cj = {crossings}")
    assert crossings, "path does not cross the wall column"
    for cj in crossings:
        assert 28 <= cj <= 32, f"crossing at cj={cj} is inside the wall"
    print("  PASS")


def test_inflation_blocks_narrow_passage():
    """
    Narrow gap of width 3 in the middle of a wall is blocked by the
    2-cell inflation. A* must detour via the large open strips left at
    each end of the wall (cj < 10 and cj > 49).
    """
    print("\n-- test_inflation_blocks_narrow_passage --")
    grid = empty_grid()
    # Wall segments with a 3-cell nominal gap at cj=[29,31] and wide
    # open regions at cj<10 and cj>49.
    put_wall(grid, 30, 10, 28)
    put_wall(grid, 30, 32, 49)

    planner = AStarPlanner(grid, OCC_THRESHOLD)
    start = (5, 30)
    goal = (55, 30)
    r = planner.plan(start, goal)

    assert r.path is not None, f"A* failed: {r.reason}"
    crossings = [cj for (ci, cj) in r.path if ci == 30]
    print(f"  path length = {len(r.path)} cells  expanded={r.expanded}")
    print(f"  ci=30 crossings at cj = {crossings}")
    # Inflation expands the wall endpoints by 2 cells, so the only
    # free corridors at ci=30 are cj <= 7 or cj >= 52.
    for cj in crossings:
        assert cj <= 7 or cj >= 52, (
            f"path squeezed through blocked gap at cj={cj}"
        )
    print("  PASS")


def test_goal_inside_obstacle_nudged():
    print("\n-- test_goal_inside_obstacle_nudged --")
    grid = empty_grid()
    put_box(grid, 38, 42, 28, 32)

    planner = AStarPlanner(grid, OCC_THRESHOLD)
    start = (5, 30)
    raw_goal = (40, 30)  # inside the box
    nudged = nearest_free_cell(planner, *raw_goal, radius=5)
    assert nudged is not None, "nearest_free_cell returned None"
    assert planner.is_free(*nudged)
    print(f"  raw goal     = {raw_goal} (blocked)")
    print(f"  nudged goal  = {nudged}")
    # Must be outside the inflated box (which extends 2 cells out).
    assert nudged[0] < 36 or nudged[0] > 44 or nudged[1] < 26 or nudged[1] > 34

    r = planner.plan(start, nudged)
    assert r.path is not None
    print(f"  path length  = {len(r.path)} cells  expanded={r.expanded}")
    print("  PASS")


def test_tangent_choice_around_left_wall():
    """
    Wall on the robot's LEFT in body frame (positive Y). Robot heading
    is +X. A* should route by going right (negative Y) → initial tangent
    yaw should be negative.
    """
    print("\n-- test_tangent_choice_around_left_wall --")
    grid = empty_grid()
    # Wall at ci=32, occupying cj=[10..59]. With 2-cell inflation the
    # wall grows to ci=[30..34], cj=[8..59]. The ONLY free crossing at
    # the wall column is cj <= 7, so A* is forced to detour south
    # (toward negative Y) → initial tangent must point to -Y.
    put_wall(grid, 32, 10, GRID_SIZE - 1)

    planner = AStarPlanner(grid, OCC_THRESHOLD)
    start = (15, 30)
    goal = (50, 30)
    r = planner.plan(start, goal)
    assert r.path is not None, f"A* failed: {r.reason}"

    tangent = path_initial_tangent(r.path, look_ahead=6)
    print(f"  path length     = {len(r.path)}  expanded={r.expanded}")
    print(f"  initial tangent = {math.degrees(tangent):+.1f}°  (expected < 0)")
    assert tangent < 0, "tangent should point to negative Y (avoid left-side wall)"

    print(ascii_render(grid, planner.blocked_mask, r.path, start, goal))
    print("  PASS")


def test_max_iterations_budget():
    print("\n-- test_max_iterations_budget --")
    grid = empty_grid()
    planner = AStarPlanner(grid, OCC_THRESHOLD, max_iterations=5)
    r = planner.plan((0, 0), (59, 59))
    print(f"  reason={r.reason}  expanded={r.expanded}")
    assert r.reason == "budget"
    assert r.path is None
    print("  PASS")


def test_no_path():
    print("\n-- test_no_path --")
    grid = empty_grid()
    # Full vertical wall at ci=50 spanning all cj: nothing can cross.
    put_wall(grid, 50, 0, GRID_SIZE - 1)
    planner = AStarPlanner(grid, OCC_THRESHOLD)
    r = planner.plan((5, 30), (55, 30))
    print(f"  reason={r.reason}  expanded={r.expanded}")
    # Either truly no_path or goal_blocked after inflation — both OK.
    assert r.path is None
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_empty_grid_straight_line,
    test_wall_with_gap,
    test_inflation_blocks_narrow_passage,
    test_goal_inside_obstacle_nudged,
    test_tangent_choice_around_left_wall,
    test_max_iterations_budget,
    test_no_path,
]


def main():
    print("=" * 60)
    print(f"  A* LOCAL PLANNER -- UNIT TESTS  (INFLATE_CELLS={INFLATE_CELLS})")
    print("=" * 60)
    failed = 0
    for fn in ALL_TESTS:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {fn.__name__}: {e}")
    print()
    print("=" * 60)
    if failed:
        print(f"  {failed}/{len(ALL_TESTS)} tests failed")
        sys.exit(1)
    print(f"  {len(ALL_TESTS)}/{len(ALL_TESTS)} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
