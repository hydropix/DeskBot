"""
Early avoidance guidance law (session 10).

Adds a continuous yaw-rate bias in GO_HEADING that is proportional to
the inverse of the frontal obstacle distance, so that the robot starts
curving softly several meters *before* SAFE_DIST is reached. At 2 m
the bias is a few degrees per second — imperceptible for the LQR but,
integrated over the approach, yields ~15-25 degrees of course change
which routinely steers the robot past the obstacle without ever
tripping the CONTOUR fallback.

Pure function + dataclass. No state, no I/O. The Navigator owns all
state (side cache, timers) and only calls `compute_bias_yaw` to get
the scalar contribution that is then added to the heading-P output
before the final yaw clip.

Not an Artificial Potential Field. The output is a direct yaw-rate
command, not a virtual force. APF-style failure modes (local minima,
oscillations between repelling sources) do not apply because there is
no attractive term and because heading convergence is handled by the
unchanged `HEADING_P_GAIN * h_err` term of `_state_go_heading`.
"""
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class EarlyAvoidanceParams:
    """
    Early avoidance guidance-law constants.

    k : gain of the inverse-distance law, units rad*m/s. Peak yaw rate
        is k / d_sat. Initial value 0.08 rad*m/s gives 0.16 rad/s
        (~9 deg/s) at saturation and 0.04 rad/s (~2.3 deg/s) at 2 m,
        two orders of magnitude below MAX_NAV_YAW so heading P stays
        dominant on large alignment errors.

    d_sat : low-distance saturation, meters. Below d_sat the law stops
        growing; avoids singularity at d=0 and caps the peak yaw rate.
        Value 0.5 m chosen so the saturation floor sits well above
        SAFE_DIST (0.36 m), guaranteeing the bias has already reached
        its plateau before the CONTOUR fallback kicks in.

    d_trig : lower activation threshold, meters. Below d_trig the bias
        is switched off to leave the frame cleanly to `_enter_contour`.
        Pinned at SAFE_DIST = 0.36 m on purpose so the handoff is
        atomic: one frame has the bias, the next is in CONTOUR mode.

    d_cut : upper activation threshold, meters. Above d_cut the bias
        is zero. 2.0 m matches the reliable range of the front
        rangefinders and the depth at which the occupancy grid is
        still well populated ahead of the robot.

    t_hold : hysteresis time on the `side` decision, seconds. After
        an initial side pick, the Navigator must keep the same side
        for at least this long even if a later virtual scan would
        disagree. Prevents bang-bang oscillations on nearly symmetric
        obstacles. Value 1.0 s covers roughly 2 m of travel at nominal
        cruise.

    t_clear : dwell time before the `side` cache is reset after the
        obstacle has cleared, seconds. 1.0 s ensures a brief gap in
        the grid (single frame of noise) does not reset the selection.
    """
    k: float = 0.08
    d_sat: float = 0.5
    d_trig: float = 0.36
    d_cut: float = 2.0
    t_hold: float = 1.0
    t_clear: float = 1.0


EARLY_PARAMS = EarlyAvoidanceParams()


def compute_bias_yaw(d: float, side: int,
                     params: EarlyAvoidanceParams = EARLY_PARAMS) -> float:
    """
    Return the yaw-rate bias (rad/s) produced by the early avoidance
    guidance law for a frontal obstacle distance `d` (meters) and a
    precomputed turn direction `side`.

    `side` convention HERE is the yaw-rate sign convention used by
    the Navigator's heading controller: +1 means "the bias should
    push yaw positive" (turn left in the world frame, +Y), -1 means
    "push yaw negative" (turn right). This is the OPPOSITE of
    `Navigator._contour_side` (+1 = "contour by going right" = yaw
    negative). The Navigator's `_update_early_side` is responsible
    for negating the virtual-scan result before caching it here; see
    that method for the rationale.

    Pure and stateless. `side` must be one of {-1, +1}; any other
    value yields 0.0.

    Piecewise definition (matches plan_early_avoidance.md):
        d >= d_cut              -> 0
        d_trig < d < d_cut      -> side * k / max(d, d_sat)
        d <= d_trig             -> 0   (handoff to _enter_contour)
    """
    if side not in (-1, 1):
        return 0.0
    if d >= params.d_cut:
        return 0.0
    if d <= params.d_trig:
        return 0.0
    effective = d if d > params.d_sat else params.d_sat
    return float(side) * params.k / effective


@dataclass
class EarlySideCache:
    """Mutable cache for the early-avoidance side decision.

    Attributes
    ----------
    side : +1, -1 or None
        Cached turn direction in yaw-rate convention (+1 = turn left,
        -1 = turn right). None means "no obstacle currently tracked".
    hold_timer : float
        Seconds elapsed since the current `side` was first selected.
        Used to enforce the T_hold hysteresis — side is never
        re-evaluated until hold_timer >= params.t_hold.
    clear_timer : float
        Seconds of "clear view" (d_front > d_cut + 0.2 m). Reset to 0
        the moment an obstacle reappears inside d_cut + 0.2. The cache
        is invalidated when clear_timer exceeds params.t_clear.
    """
    side: Optional[int] = None
    hold_timer: float = 0.0
    clear_timer: float = 0.0


CLEAR_MARGIN = 0.2  # meters; dwell must be beyond d_cut + this before reset


def update_side_cache(cache: EarlySideCache,
                      d_front: float,
                      fsm_is_go_heading: bool,
                      dt: float,
                      pick_side: Callable[[], Optional[int]],
                      params: EarlyAvoidanceParams = EARLY_PARAMS) -> None:
    """
    Mutate `cache` in place to reflect one Navigator tick of early
    avoidance side management.

    Rules (matches plan_early_avoidance.md Step 2):

    1. If FSM is not GO_HEADING, the cache is fully reset. The plan
       forbids any bias in CONTOUR/REVERSE/IDLE.

    2. If `d_front > d_cut + CLEAR_MARGIN`, accumulate `clear_timer`.
       When it exceeds `t_clear` the side decision is invalidated.
       This is asymmetric with the trigger (`< d_cut`) to add
       hysteresis across the cut-off distance, which would otherwise
       flap on a single rangefinder noise spike.

    3. Otherwise reset `clear_timer` to 0 (obstacle is back in range).

    4. If no side is cached and `d_trig < d_front < d_cut`, ask the
       caller's `pick_side()` for a new direction. The callback owns
       whichever selector is currently active (virtual scan or A*)
       and must return a turn direction in yaw-rate convention
       (+1 = turn left, -1 = turn right) or None to decline.

    5. Otherwise, if a side is already cached, advance `hold_timer`.
       The plan enforces `t_hold` as a no-flip interval; the current
       implementation simply re-uses the cached side until step 2
       invalidates it, so `hold_timer` is only observational. A later
       iteration may add a "force re-scan on new obstacle" rule; this
       helper exposes the timer to make that extension trivial.
    """
    if not fsm_is_go_heading:
        cache.side = None
        cache.hold_timer = 0.0
        cache.clear_timer = 0.0
        return

    if d_front > params.d_cut + CLEAR_MARGIN:
        cache.clear_timer += dt
        if cache.clear_timer >= params.t_clear:
            cache.side = None
            cache.hold_timer = 0.0
        return

    cache.clear_timer = 0.0

    if cache.side is None:
        if params.d_trig < d_front < params.d_cut:
            picked = pick_side()
            if picked in (-1, 1):
                cache.side = int(picked)
                cache.hold_timer = 0.0
        return

    cache.hold_timer += dt
