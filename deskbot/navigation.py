"""
Navigation v2 — Bug2 + local occupancy grid + VFH-lite steering.

Architecture (3 layers):
  STRATEGIC  — TangentBug-inspired FSM: keep heading or contour obstacle
  TACTICAL   — VFH-lite: choose steering direction from occupancy grid
  REACTIVE   — emergency braking, pitch-compensated rangefinder filtering

Uses ONLY sensor data (rangefinders, encoders, IMU) — never simulator
internals. Rangefinder interpretation is delegated to GroundGeometry,
which does exact vector-based ground projection with pitch-uncertainty
propagation; no scalar approximations remain in this module.

FSM has 2 main states plus 2 auxiliaries:
  GO_HEADING — drive toward the desired heading (infinite target)
  CONTOUR    — wall-follow an obstacle, return to heading when clear
  REVERSE    — back up after stuck detection
  IDLE       — no goal
"""
import math
from enum import Enum

import numpy as np
from dataclasses import dataclass

from deskbot.robot import WHEEL_RADIUS, WHEEL_SEPARATION
from deskbot.perception import GroundGeometry


# ─────────────────────────────────────────────────────────────────────
# Geometry constants (cached from GroundGeometry; nominal values below
# are used for module-level scans and fall-back reasoning only).
# ─────────────────────────────────────────────────────────────────────

# Yaw angles of each rangefinder in the body frame (for grid projection
# and scan direction sampling). These match the MJCF mounting but are
# only used as a coarse angular index — precise ray directions come from
# GroundGeometry at runtime.
RF_BODY_ANGLES = {
    "rf_FC":  0.0,
    "rf_FL":  math.radians(30),
    "rf_FR":  math.radians(-30),
    "rf_FL2": math.radians(55),
    "rf_FR2": math.radians(-55),
    "rf_SL":  math.radians(90),
    "rf_SR":  math.radians(-90),
}


# ─────────────────────────────────────────────────────────────────────
# Navigation physical parameters
# ─────────────────────────────────────────────────────────────────────

# Distance thresholds (meters).
#
# IMPORTANT — semantics: these are distances from the ROBOT PIVOT AXIS
# (wheel axle projected onto the floor), as returned by
# GroundGeometry.horizontal_distance. The front sensor pod sits about
# 5.5 cm forward of the axle and the side pod sits 2.5 cm off-axis, so
# a value here is ~5 cm larger than the raw sensor reading for front
# rays and ~2.5 cm larger for side rays. The thresholds below already
# include those offsets, so the effective physical trigger points are:
#   SAFE_DIST        → ~30 cm from the robot front face
#   CAUTION_DIST     → ~60 cm from the front face
#   WALL_FOLLOW_DIST → ~22 cm from the side face
#
# SAFE_DIST is the hard trigger for entering contour mode; CAUTION_DIST
# is where cruise speed starts to roll off linearly. WALL_FOLLOW_DIST
# is the P-controller setpoint for the side-laser wall tracker — tuned
# to keep wheel/body clearance while staying above the VL53L0X grazing
# threshold (~15 cm on matte surfaces).
SAFE_DIST = 0.36
CAUTION_DIST = 0.65
WALL_FOLLOW_DIST = 0.25

# Cruise envelope (m/s, rad/s). MAX_NAV_SPEED is held conservative so
# that the self-balancer never saturates its pitch actuator during
# aggressive steering; MIN_NAV_SPEED prevents the velocity PI from
# collapsing to zero in tight corners.
MAX_NAV_SPEED = 0.5
MIN_NAV_SPEED = 0.15
MAX_NAV_YAW   = 1.5

# ── Exit-from-contour criteria ──
# Heading must be clear for at least CONTOUR_CLEAR_DIST straight ahead,
# and for CONTOUR_CLEAR_SIDE_RATIO * CONTOUR_CLEAR_DIST at ±20° offsets,
# before we dare resume the M-line.
CONTOUR_CLEAR_DIST = 1.00
CONTOUR_CLEAR_SIDE_OFFSET = math.radians(20)
CONTOUR_CLEAR_SIDE_RATIO = 0.70

# Anti-loop: if cumulative yaw during contour exceeds this, flip side.
MAX_CONTOUR_ANGLE = math.radians(300)

# ── Hysteresis / watchdogs ──
# Minimum time in CONTOUR before the exit check is considered. Without
# hysteresis the FSM flaps between GO/CONTOUR every frame on noise.
CONTOUR_MIN_DWELL = 1.5          # seconds
# Local anti-regression: if, after this dwell, we have moved backward
# along the target heading by more than the threshold, give up contour.
CONTOUR_REGRESS_DWELL = 4.0
CONTOUR_REGRESS_DIST = 0.30
# Global anti-regression watchdog period and trigger.
GLOBAL_PROGRESS_PERIOD = 4.0
GLOBAL_PROGRESS_MIN = 0.0        # if progress < this over one period, flip side
# Stuck recovery.
STUCK_TIME = 3.0
STUCK_DIST = 0.06
REVERSE_DURATION = 0.8
REVERSE_SPEED = -0.25

# ── Control gains ──
# All gains are proportional; units documented next to each one. They
# came out of empirical tuning but are now named and grouped so that
# their meaning is explicit and future sweeps can target them.
HEADING_P_GAIN = 1.5             # rad/s per rad of heading error (GO_HEADING)
CONTOUR_HEADING_P_GAIN = 2.0     # rad/s per rad (wall lost → pull to heading)
WALL_FOLLOW_P_GAIN = 2.5         # rad/s per meter of wall distance error
CONTOUR_HEADING_PULL_GAIN = 0.4  # rad/s per rad (bias toward M-line during wall-follow)
FRONT_AVOID_GAIN = 2.0           # rad/s per unit of normalized front urgency

# Speed shaping. Below ALIGNED_THRESHOLD in cos(h_err) we turn in place;
# below PERPENDICULAR_THRESHOLD we crawl; above we cruise.
ALIGNED_COS_THRESHOLD = 0.3
PERPENDICULAR_COS_THRESHOLD = -0.3
CONTOUR_CRUISE_FACTOR = 0.6      # contour speed relative to MAX_NAV_SPEED

# ── Output smoothing ──
# First-order low-pass. Time constant ≈ dt / (1 - alpha).
SMOOTH_ALPHA = 0.85

# ─────────────────────────────────────────────────────────────────────
# Occupancy grid parameters
# ─────────────────────────────────────────────────────────────────────

GRID_SIZE = 60                  # cells per side
GRID_RES = 0.08                 # meters per cell
GRID_HALF = GRID_SIZE * GRID_RES / 2.0   # 2.4 m — half-extent

# Log-odds update model. Tuned so a single hit moves a "free" cell
# (log-odds ≈ -1) to "unknown" (~0) in about 1.5 readings, and an
# established "occupied" cell requires several consecutive frees to
# erase. Clamp range limits how confident the grid can become, so it
# keeps reacting to the world even after many agreeing observations.
LOG_ODD_FREE = -0.3
LOG_ODD_OCC  = +0.7
LOG_ODD_MAX  = +3.0
LOG_ODD_MIN  = -2.0
LOG_ODD_OCCUPIED_THRESHOLD = 0.5  # posterior p ≈ 0.62

# Grid is updated out to this distance along each ray even when the
# sensor sees nothing — the unknown beyond stays at log-odds zero.
GRID_RAY_MAX = 1.5


class FSMState(Enum):
    IDLE = "idle"
    GO_HEADING = "go_heading"
    CONTOUR = "contour"
    REVERSE = "reverse"


@dataclass
class NavState:
    """Observable navigation state for GUI display."""
    active: bool = False
    target_heading: float = 0.0
    pos_x: float = 0.0
    pos_y: float = 0.0
    heading: float = 0.0
    behavior: str = "idle"
    contour_side: str = ""       # "left" or "right"
    contour_deviation: float = 0.0
    min_front_dist: float = 999.0


class OccupancyGrid:
    """
    Small egocentric log-odds occupancy grid.

    The grid is centered on the robot and shifts as the robot moves.
    Each cell stores a log-odds value:
        log_odds = log(p_occ / (1 - p_occ))
    Updates are additive (Bayesian fusion in log space), which is both
    numerically stable and O(1) per cell per ray.

    Ray tracing uses a Bresenham line in integer cell coordinates — the
    standard algorithm used in every occupancy-grid implementation since
    Moravec & Elfes (1985). Each cell along the ray is visited exactly
    once; diagonal rays don't skip cells.
    """

    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        # World position of grid center (tracks robot).
        self.cx = 0.0
        self.cy = 0.0

    def reset(self):
        self.grid[:] = 0.0
        self.cx = 0.0
        self.cy = 0.0

    def shift(self, robot_x: float, robot_y: float):
        """Shift grid to keep robot centered. Cells that fall off are dropped."""
        dx_cells = int(round((robot_x - self.cx) / GRID_RES))
        dy_cells = int(round((robot_y - self.cy) / GRID_RES))

        if dx_cells == 0 and dy_cells == 0:
            return

        if abs(dx_cells) >= GRID_SIZE or abs(dy_cells) >= GRID_SIZE:
            self.grid[:] = 0.0
        else:
            self.grid = np.roll(self.grid, -dx_cells, axis=0)
            self.grid = np.roll(self.grid, -dy_cells, axis=1)
            if dx_cells > 0:
                self.grid[-dx_cells:, :] = 0.0
            elif dx_cells < 0:
                self.grid[:(-dx_cells), :] = 0.0
            if dy_cells > 0:
                self.grid[:, -dy_cells:] = 0.0
            elif dy_cells < 0:
                self.grid[:, :(-dy_cells)] = 0.0

        self.cx += dx_cells * GRID_RES
        self.cy += dy_cells * GRID_RES

    def world_to_cell(self, wx: float, wy: float) -> tuple[int, int]:
        """Convert world coordinates to grid cell indices."""
        ci = int((wx - self.cx) / GRID_RES + GRID_SIZE / 2)
        cj = int((wy - self.cy) / GRID_RES + GRID_SIZE / 2)
        return ci, cj

    def in_bounds(self, ci: int, cj: int) -> bool:
        return 0 <= ci < GRID_SIZE and 0 <= cj < GRID_SIZE

    def _bresenham(self, r0: int, c0: int, r1: int, c1: int):
        """
        Yield every integer cell on the line from (r0, c0) to (r1, c1),
        endpoints included. Standard 2D Bresenham — integer-only, visits
        each column/row exactly once.
        """
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc
        r, c = r0, c0
        while True:
            yield r, c
            if r == r1 and c == c1:
                return
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

    def update_ray(self, robot_x: float, robot_y: float,
                   hit_x: float, hit_y: float, hit: bool):
        """
        Update grid along a ray from the robot to the endpoint.

        Cells strictly between origin and endpoint are marked free;
        the endpoint is additionally marked occupied if `hit` is True.
        Updates are additive log-odds, clamped to [LOG_ODD_MIN, LOG_ODD_MAX].
        """
        ri, rj = self.world_to_cell(robot_x, robot_y)
        hi, hj = self.world_to_cell(hit_x, hit_y)

        cells = list(self._bresenham(ri, rj, hi, hj))
        # Free update for all cells EXCEPT the endpoint (so a hit doesn't
        # simultaneously get decremented and incremented).
        for ci, cj in cells[:-1]:
            if self.in_bounds(ci, cj):
                v = self.grid[ci, cj] + LOG_ODD_FREE
                if v < LOG_ODD_MIN:
                    v = LOG_ODD_MIN
                self.grid[ci, cj] = v

        # Endpoint: occupied if the ray hit something, else free.
        last_ci, last_cj = cells[-1]
        if self.in_bounds(last_ci, last_cj):
            if hit:
                v = self.grid[last_ci, last_cj] + LOG_ODD_OCC
                if v > LOG_ODD_MAX:
                    v = LOG_ODD_MAX
            else:
                v = self.grid[last_ci, last_cj] + LOG_ODD_FREE
                if v < LOG_ODD_MIN:
                    v = LOG_ODD_MIN
            self.grid[last_ci, last_cj] = v

    def is_clear_direction(self, robot_x: float, robot_y: float,
                           angle: float, distance: float) -> bool:
        """
        Return True if no occupied cell lies within `distance` along
        the ray at `angle` from the robot. Cells outside the grid are
        treated as unknown and therefore clear.
        """
        steps = int(distance / GRID_RES)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        for s in range(1, steps + 1):
            d = s * GRID_RES
            wx = robot_x + d * cos_a
            wy = robot_y + d * sin_a
            ci, cj = self.world_to_cell(wx, wy)
            if not self.in_bounds(ci, cj):
                return True
            if self.grid[ci, cj] > LOG_ODD_OCCUPIED_THRESHOLD:
                return False
        return True

    def clearance_in_direction(self, robot_x: float, robot_y: float,
                               angle: float, max_dist: float = 1.0) -> float:
        """Distance to the first occupied cell along the given ray."""
        steps = int(max_dist / GRID_RES)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        for s in range(1, steps + 1):
            d = s * GRID_RES
            wx = robot_x + d * cos_a
            wy = robot_y + d * sin_a
            ci, cj = self.world_to_cell(wx, wy)
            if not self.in_bounds(ci, cj):
                return max_dist
            if self.grid[ci, cj] > LOG_ODD_OCCUPIED_THRESHOLD:
                return d
        return max_dist


class Navigator:
    """
    Bug2-inspired navigator with local occupancy grid.

    Goal: maintain a heading toward an infinitely distant point,
    avoiding obstacles by contouring around them.
    """

    def __init__(self, dt: float, mj_model):
        """
        `mj_model` is REQUIRED. The navigator depends on GroundGeometry
        for exact pitch compensation; the legacy scalar fallback has been
        removed. Pass the compiled MuJoCo model used by the simulator.
        """
        if mj_model is None:
            raise ValueError(
                "Navigator requires an mj_model. Pass the compiled MuJoCo "
                "model so GroundGeometry can read sensor mountings exactly."
            )
        self.dt = dt
        self.state = NavState()
        self.grid = OccupancyGrid()
        self._ground = GroundGeometry(mj_model)

        # Dead reckoning
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._heading = 0.0

        # Target heading (the "M-line" direction)
        self._target_heading = 0.0
        self._active = False

        # FSM
        self._fsm = FSMState.IDLE

        # Smoothed output
        self._smooth_vel = 0.0
        self._smooth_yaw = 0.0

        # Contour state
        self._contour_side = 1       # +1 = wall on left (go right), -1 = wall on right
        self._contour_deviation = 0.0
        self._contour_timer = 0.0
        self._contour_start_x = 0.0

        # Reverse state
        self._reverse_timer = 0.0

        # Stuck detection
        self._stuck_timer = 0.0
        self._last_progress_pos = np.zeros(2)

        # Global heading progress watchdog
        self._progress_check_timer = 0.0
        self._progress_check_x = 0.0

        # Cached sensor data
        self._min_front_dist = 999.0
        self._rf_compensated: dict[str, float] = {}

    def reset(self):
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._heading = 0.0
        self._target_heading = 0.0
        self._active = False
        self._fsm = FSMState.IDLE
        self._smooth_vel = 0.0
        self._smooth_yaw = 0.0
        self._contour_side = 1
        self._contour_deviation = 0.0
        self._contour_timer = 0.0
        self._contour_start_x = 0.0
        self._reverse_timer = 0.0
        self._stuck_timer = 0.0
        self._last_progress_pos = np.zeros(2)
        self._progress_check_timer = 0.0
        self._progress_check_x = 0.0
        self._min_front_dist = 999.0
        self._rf_compensated = {}
        self.grid.reset()
        self.state = NavState()

    # ── Goal setting ──────────────────────────────────────────────

    def set_heading(self, heading_deg: float):
        """Set target heading in degrees. 0=+X, 90=+Y."""
        self._target_heading = math.radians(heading_deg)
        self._active = True
        self._fsm = FSMState.GO_HEADING
        self._contour_deviation = 0.0
        self._stuck_timer = 0.0
        self._last_progress_pos = np.array([self._pos_x, self._pos_y])
        self._smooth_vel = 0.0
        self._smooth_yaw = 0.0
        self._progress_check_x = self._heading_axis_projection()
        self._progress_check_timer = 0.0
        self.state.active = True
        self.state.target_heading = self._target_heading

    def stop(self):
        self._active = False
        self._fsm = FSMState.IDLE
        self.state.active = False
        self.state.behavior = "idle"

    # ── Rangefinder processing (delegates to GroundGeometry) ──────

    def compensate_rangefinders(self, rf: dict, pitch: float) -> dict:
        """
        Return horizontal distances to obstacles, with ground hits and
        holes filtered out.

        Classification is performed by GroundGeometry (exact vector-based
        projection + pitch uncertainty propagation). Any reading tagged
        `flat` or `hole` is suppressed (-1); `obstacle` readings are
        projected into the horizontal plane relative to the robot's
        pivot axis; `no_ground_expected` readings are passed through as
        obstacles since the ray never reaches the floor.
        """
        compensated: dict[str, float] = {}
        for name, dist in rf.items():
            if dist < 0.0:
                compensated[name] = -1.0
                continue
            gr = self._ground.classify(name, dist, pitch)
            if gr.kind in ("flat", "hole", "no_reading"):
                compensated[name] = -1.0
            else:
                compensated[name] = self._ground.horizontal_distance(
                    name, pitch, dist
                )
        return compensated

    # ── Grid update from sensors ──────────────────────────────────

    def _update_grid(self, rf: dict):
        """Project rangefinder readings into occupancy grid."""
        for name, dist in rf.items():
            angle = RF_BODY_ANGLES.get(name)
            if angle is None:
                continue
            world_angle = self._heading + angle
            cos_a = math.cos(world_angle)
            sin_a = math.sin(world_angle)

            if dist > 0:
                hit_x = self._pos_x + dist * cos_a
                hit_y = self._pos_y + dist * sin_a
                self.grid.update_ray(
                    self._pos_x, self._pos_y, hit_x, hit_y, hit=True
                )
            else:
                end_x = self._pos_x + GRID_RAY_MAX * cos_a
                end_y = self._pos_y + GRID_RAY_MAX * sin_a
                self.grid.update_ray(
                    self._pos_x, self._pos_y, end_x, end_y, hit=False
                )

    # ── Helpers ───────────────────────────────────────────────────

    def _heading_axis_projection(self) -> float:
        """Scalar progress along the target heading unit vector."""
        return (self._pos_x * math.cos(self._target_heading)
                + self._pos_y * math.sin(self._target_heading))

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    # ── Main update ───────────────────────────────────────────────

    def update(self, estimator, readings, dt: float):
        """Called every physics step. Returns (vel_cmd, yaw_cmd) or (None, None)."""

        # ── 1. Dead reckoning ──
        self._heading = self._wrap_angle(self._heading + estimator.yaw_rate * dt)
        self._pos_x += estimator.forward_vel * math.cos(self._heading) * dt
        self._pos_y += estimator.forward_vel * math.sin(self._heading) * dt

        self.state.pos_x = self._pos_x
        self.state.pos_y = self._pos_y
        self.state.heading = self._heading

        if not self._active or estimator.fallen:
            return None, None

        # ── 2. Compensate rangefinders ──
        rf = self.compensate_rangefinders(readings.rangefinders, estimator.pitch)
        self._rf_compensated = rf

        front_names = ("rf_FC", "rf_FL", "rf_FR")
        front_dists = [rf[n] for n in front_names if rf.get(n, -1) > 0]
        self._min_front_dist = min(front_dists) if front_dists else 999.0
        self.state.min_front_dist = self._min_front_dist

        # ── 3. Update occupancy grid ──
        self.grid.shift(self._pos_x, self._pos_y)
        self._update_grid(rf)

        # ── 4. FSM dispatch ──
        vel_cmd, yaw_cmd = 0.0, 0.0

        if self._fsm == FSMState.GO_HEADING:
            vel_cmd, yaw_cmd = self._state_go_heading(rf, dt)
        elif self._fsm == FSMState.CONTOUR:
            vel_cmd, yaw_cmd = self._state_contour(rf, dt)
        elif self._fsm == FSMState.REVERSE:
            vel_cmd, yaw_cmd = self._state_reverse(dt)

        # ── 5. Stuck detection ──
        self._check_stuck(dt)

        # ── 6. Global heading progress watchdog ──
        self._progress_check_timer += dt
        if self._progress_check_timer > GLOBAL_PROGRESS_PERIOD:
            heading_x = self._heading_axis_projection()
            progress = heading_x - self._progress_check_x
            if progress < GLOBAL_PROGRESS_MIN and self._fsm == FSMState.CONTOUR:
                self._fsm = FSMState.GO_HEADING
                self._contour_side *= -1
                self._contour_deviation = 0.0
            self._progress_check_x = heading_x
            self._progress_check_timer = 0.0

        # ── 7. Smooth output ──
        self._smooth_vel = SMOOTH_ALPHA * self._smooth_vel + (1 - SMOOTH_ALPHA) * vel_cmd
        self._smooth_yaw = SMOOTH_ALPHA * self._smooth_yaw + (1 - SMOOTH_ALPHA) * yaw_cmd

        self._smooth_vel = float(np.clip(self._smooth_vel, -MAX_NAV_SPEED, MAX_NAV_SPEED))
        self._smooth_yaw = float(np.clip(self._smooth_yaw, -MAX_NAV_YAW, MAX_NAV_YAW))

        self.state.behavior = self._fsm.value
        self.state.contour_side = "left" if self._contour_side > 0 else "right"
        self.state.contour_deviation = self._contour_deviation

        return self._smooth_vel, self._smooth_yaw

    # ── FSM: GO_HEADING ───────────────────────────────────────────

    def _state_go_heading(self, rf, dt):
        """Drive toward the target heading. Switch to CONTOUR if blocked."""

        h_err = self._wrap_angle(self._target_heading - self._heading)

        if self._min_front_dist < SAFE_DIST:
            self._enter_contour(rf)
            return 0.0, 0.0

        yaw = HEADING_P_GAIN * h_err

        alignment = math.cos(h_err)
        if alignment > ALIGNED_COS_THRESHOLD:
            vel = MAX_NAV_SPEED * min(alignment, 1.0)
            vel *= min(self._min_front_dist / CAUTION_DIST, 1.0)
            vel = max(vel, MIN_NAV_SPEED)
        elif alignment > PERPENDICULAR_COS_THRESHOLD:
            vel = MIN_NAV_SPEED * 0.5
        else:
            vel = 0.0

        return vel, yaw

    # ── FSM: CONTOUR ──────────────────────────────────────────────

    def _enter_contour(self, rf):
        """Start wall-following. Choose side via virtual grid scan."""
        self._fsm = FSMState.CONTOUR
        self._contour_deviation = 0.0
        self._contour_timer = 0.0
        self._contour_start_x = self._heading_axis_projection()

        # Virtual scan on the occupancy grid — pick the direction with
        # max clearance * alignment-to-heading weight.
        best_angle_deg = 0.0
        best_score = -1.0

        for deg_offset in range(-90, 91, 15):
            angle = self._heading + math.radians(deg_offset)
            clearance = self.grid.clearance_in_direction(
                self._pos_x, self._pos_y, angle, GRID_RAY_MAX
            )
            alignment = math.cos(math.radians(deg_offset))
            score = clearance * (1.0 + 0.5 * alignment)
            if score > best_score:
                best_score = score
                best_angle_deg = deg_offset

        fl = rf.get("rf_FL", -1.0)
        fr = rf.get("rf_FR", -1.0)
        fl_val = fl if fl > 0 else 0.0
        fr_val = fr if fr > 0 else 0.0
        sensor_bias = fl_val - fr_val  # positive = left is clearer

        if best_angle_deg > 10 or (best_angle_deg == 0 and sensor_bias > 0.1):
            self._contour_side = -1
        elif best_angle_deg < -10 or (best_angle_deg == 0 and sensor_bias < -0.1):
            self._contour_side = 1
        else:
            self._contour_side = -1 if sensor_bias >= 0 else 1

    def _state_contour(self, rf, dt):
        """
        Wall-follow along the obstacle, then return to heading.

        contour_side = +1: obstacle is on LEFT,  robot goes RIGHT of it
        contour_side = -1: obstacle is on RIGHT, robot goes LEFT  of it
        """

        if self._contour_side > 0:
            wall_dist = rf.get("rf_SL", -1.0)
        else:
            wall_dist = rf.get("rf_SR", -1.0)

        wall_visible = wall_dist > 0

        # ── Exit criterion ──
        self._contour_timer += dt
        if (self._contour_timer > CONTOUR_MIN_DWELL
                and self._can_resume_heading()):
            self._fsm = FSMState.GO_HEADING
            return 0.0, 0.0

        # ── Local anti-regression ──
        heading_progress = self._heading_axis_projection()
        regression = self._contour_start_x - heading_progress
        if (self._contour_timer > CONTOUR_REGRESS_DWELL
                and regression > CONTOUR_REGRESS_DIST):
            self._fsm = FSMState.GO_HEADING
            return 0.0, 0.0

        # ── Anti-loop ──
        if abs(self._contour_deviation) > MAX_CONTOUR_ANGLE:
            self._contour_side *= -1
            self._contour_deviation = 0.0

        # ── Sub-behavior 1: front blocked → turn in place ──
        if self._min_front_dist < SAFE_DIST:
            yaw = -self._contour_side * MAX_NAV_YAW
            clearance_ratio = max(self._min_front_dist / SAFE_DIST, 0.15)
            vel = MIN_NAV_SPEED * clearance_ratio
            self._contour_deviation += yaw * dt
            return vel, yaw

        # ── Sub-behavior 2: wall visible → P-controller + heading pull ──
        if wall_visible:
            vel = MAX_NAV_SPEED * CONTOUR_CRUISE_FACTOR
            wall_error = wall_dist - WALL_FOLLOW_DIST
            yaw = self._contour_side * WALL_FOLLOW_P_GAIN * wall_error

            h_err = self._wrap_angle(self._target_heading - self._heading)
            heading_yaw = CONTOUR_HEADING_PULL_GAIN * h_err
            pulls_away_from_wall = (heading_yaw * self._contour_side) < 0
            if pulls_away_from_wall:
                yaw += heading_yaw

            if self._min_front_dist < CAUTION_DIST:
                urgency = 1.0 - self._min_front_dist / CAUTION_DIST
                yaw += -self._contour_side * urgency * FRONT_AVOID_GAIN

            vel *= min(self._min_front_dist / CAUTION_DIST, 1.0)
            vel = max(vel, MIN_NAV_SPEED)
            self._contour_deviation += yaw * dt
            return vel, yaw

        # ── Sub-behavior 3: wall lost → steer back to heading ──
        h_err = self._wrap_angle(self._target_heading - self._heading)
        yaw = CONTOUR_HEADING_P_GAIN * h_err
        vel = MAX_NAV_SPEED * 0.5
        self._contour_deviation += yaw * dt
        return vel, yaw

    def _can_resume_heading(self) -> bool:
        """Check whether the target heading is clear enough to exit contour."""
        if not self.grid.is_clear_direction(
            self._pos_x, self._pos_y,
            self._target_heading, CONTOUR_CLEAR_DIST,
        ):
            return False

        side_dist = CONTOUR_CLEAR_DIST * CONTOUR_CLEAR_SIDE_RATIO
        for offset in (CONTOUR_CLEAR_SIDE_OFFSET, -CONTOUR_CLEAR_SIDE_OFFSET):
            if not self.grid.is_clear_direction(
                self._pos_x, self._pos_y,
                self._target_heading + offset, side_dist,
            ):
                return False
        return True

    # ── FSM: REVERSE ──────────────────────────────────────────────

    def _state_reverse(self, dt):
        """Back up to create space, then resume heading."""
        self._reverse_timer -= dt
        if self._reverse_timer <= 0:
            self._fsm = FSMState.GO_HEADING
            return 0.0, 0.0
        return REVERSE_SPEED, 0.0

    # ── Stuck detection ───────────────────────────────────────────

    def _check_stuck(self, dt):
        current_pos = np.array([self._pos_x, self._pos_y])
        progress = float(np.linalg.norm(current_pos - self._last_progress_pos))

        if progress > STUCK_DIST:
            self._stuck_timer = 0.0
            self._last_progress_pos = current_pos.copy()
        else:
            self._stuck_timer += dt

        if self._stuck_timer > STUCK_TIME:
            self._stuck_timer = 0.0
            self._last_progress_pos = current_pos.copy()
            self._fsm = FSMState.REVERSE
            self._reverse_timer = REVERSE_DURATION
            self._contour_side *= -1
            self._contour_deviation = 0.0
            self._smooth_vel = 0.0
            self._smooth_yaw = 0.0
