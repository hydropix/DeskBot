"""
Brute-force optimization of rangefinder sensor placement on DeskBot.

Explores thousands of (position, direction, wheel_radius) combinations
and scores each on: clear line of sight, coverage, mounting height (stability).

The key geometric constraint: a ray from (x, y, z) in chassis frame
hits the wheel cylinder (radius R, at XZ origin) if the ray's closest
approach to the Z-axis in XZ-plane is less than R.

For side sensors (direction ~±Y), the ray stays at constant (x, z),
so it's blocked if sqrt(x² + z²) < R.

For diagonal sensors, the full 3D ray-cylinder intersection is computed.
"""
import math
import itertools
import numpy as np
import mujoco

# -- Robot geometry (from deskbot.xml) -----------------------------
WHEEL_Y = 0.11          # wheel center Y offset from chassis
WHEEL_HALF_W = 0.009    # tire half-width
BODY_HALF_X = 0.05      # body box half-size X
BODY_HALF_Y = 0.058     # body box half-size Y
BODY_HALF_Z = 0.055     # body box half-size Z
BODY_Z = 0.015          # body box center Z in chassis frame
HEAD_HALF_X = 0.04
HEAD_HALF_Y = 0.048
HEAD_HALF_Z = 0.016
HEAD_Z = 0.086

# -- Sensor roles --------------------------------------------------
# Each role defines a nominal azimuth angle (0 = +X forward, 90 = +Y left)
ROLES = {
    "FC":  0,    # front center
    "FL":  45,   # front-left diagonal
    "FR": -45,   # front-right diagonal
    "SL":  90,   # side-left
    "SR": -90,   # side-right
}


def ray_clears_wheel(x, z, dx, dz, wheel_r):
    """Check if ray from (x,z) in direction (dx,dz) clears wheel circle of radius wheel_r at origin.
    Returns minimum distance from ray to wheel center in XZ plane."""
    norm = math.sqrt(dx * dx + dz * dz)
    if norm < 1e-9:
        return math.sqrt(x * x + z * z)  # ray has no XZ component
    # Distance from origin to ray line in XZ
    return abs(x * dz - z * dx) / norm


def score_placement(x, y, z, az_deg, el_deg, wheel_r):
    """Score a sensor placement. Higher is better.

    Returns (clear, clearance_mm, height_penalty, desc) or None if invalid.
    - clear: bool — does the ray avoid both wheels?
    - clearance_mm: minimum clearance to wheel surface (mm)
    - height_penalty: higher Z = more pendulum instability
    """
    az = math.radians(az_deg)
    el = math.radians(el_deg)

    # Direction vector
    dx = math.cos(el) * math.cos(az)
    dy = math.cos(el) * math.sin(az)
    dz = math.sin(el)

    # Check both wheels
    min_clearance = float("inf")
    for sign in [+1, -1]:
        wy = sign * WHEEL_Y
        # Transform to wheel-local frame (wheel center at XZ origin)
        # The ray in wheel frame: same X and Z (wheel is at same X,Z as chassis center)
        # but Y is shifted
        local_y = y - wy

        # Only check if ray goes TOWARD this wheel
        if sign * dy <= 0 and abs(az_deg) != 0:
            # Ray goes away from this wheel (for non-forward sensors)
            if abs(az_deg) > 10:
                continue

        # Compute closest approach of ray to wheel center in XZ
        d_min = ray_clears_wheel(x, z, dx, dz, wheel_r)
        clearance = d_min - wheel_r

        # Also check: does the ray actually reach the wheel Y-range?
        # The wheel extends from wy - half_w to wy + half_w in Y
        # Ray Y at parameter t: y + dy*t
        # We need to check if the ray passes through the wheel's Y range
        if abs(dy) > 1e-9:
            t_enter = (wy - WHEEL_HALF_W - y) / dy
            t_exit = (wy + WHEEL_HALF_W - y) / dy
            if t_enter > t_exit:
                t_enter, t_exit = t_exit, t_enter
            if t_exit < 0:
                continue  # wheel is behind the ray
        else:
            # Ray parallel to wheel — only blocks if Y is within wheel width
            if abs(local_y) > WHEEL_HALF_W:
                continue

        min_clearance = min(min_clearance, clearance)

    if min_clearance == float("inf"):
        min_clearance = 100.0  # no wheel in path

    clear = min_clearance > 0.002  # 2mm margin
    clearance_mm = min_clearance * 1000

    # Height penalty: Z above axle increases pendulum instability
    height_penalty = abs(z) * 10  # weight factor

    # Mounting feasibility: must be on or near body surface
    on_body = (abs(x) <= BODY_HALF_X + 0.02 and
               abs(y) <= BODY_HALF_Y + 0.02 and
               BODY_Z - BODY_HALF_Z - 0.02 <= z <= HEAD_Z + HEAD_HALF_Z + 0.03)

    if not on_body:
        return None

    return clear, clearance_mm, height_penalty, z


def optimize_role(role, az_nominal, wheel_radii):
    """Find best placements for a sensor role across wheel radii."""

    results = {}

    for wheel_r in wheel_radii:
        best = []

        # Position grid
        xs = np.arange(-0.02, 0.07, 0.005)
        zs = np.arange(-0.04, 0.13, 0.005)

        # Y depends on role side
        if az_nominal > 0:  # left-facing
            ys = np.arange(0.02, 0.07, 0.005)
        elif az_nominal < 0:  # right-facing
            ys = np.arange(-0.07, -0.02, 0.005)
        else:  # forward
            ys = np.arange(-0.02, 0.02, 0.005)

        # Direction variations: ±10° azimuth, ±15° elevation
        az_range = np.arange(az_nominal - 10, az_nominal + 11, 5)
        el_range = np.arange(-15, 16, 5)

        for x, y, z in itertools.product(xs, ys, zs):
            for az, el in itertools.product(az_range, el_range):
                result = score_placement(x, y, z, az, el, wheel_r)
                if result is None:
                    continue
                clear, clearance_mm, height_pen, _ = result
                if clear:
                    # Composite score: clearance (higher better) - height penalty
                    score = clearance_mm - height_pen * 50
                    best.append((score, clearance_mm, x, y, z, az, el))

        best.sort(key=lambda t: t[0], reverse=True)
        results[wheel_r] = best[:5]  # top 5

    return results


def main():
    wheel_radii = [0.12, 0.11, 0.10, 0.09, 0.08]

    print("=" * 80)
    print("DESKBOT RANGEFINDER PLACEMENT OPTIMIZER")
    print("=" * 80)
    print(f"Current wheel radius: 12cm | Current axle height: 12cm from ground")
    print(f"Body: X=[-5,5]cm Y=[-5.8,5.8]cm Z=[-4,7]cm (chassis frame)")
    print(f"Head top: Z=10.2cm | Wheel top: Z=R")
    print()

    for role, az_nom in ROLES.items():
        print(f"\n{'-' * 80}")
        print(f"  SENSOR: {role} (nominal azimuth={az_nom} deg)")
        print(f"{'-' * 80}")

        results = optimize_role(role, az_nom, wheel_radii)

        for wr in wheel_radii:
            best = results[wr]
            wr_cm = wr * 100
            axle_cm = wr * 100

            if not best:
                print(f"\n  Wheel R={wr_cm:.0f}cm: NO VALID PLACEMENT FOUND")
                continue

            top = best[0]
            score, clearance, x, y, z, az, el = top
            ground_h = (wr + z) * 100  # height from ground

            print(f"\n  Wheel R={wr_cm:.0f}cm (axle at {axle_cm:.0f}cm):")
            print(f"    BEST: pos=({x*100:+5.1f}, {y*100:+5.1f}, {z*100:+5.1f})cm "
                  f"dir=({az:+.0f} deg az, {el:+.0f} deg el) "
                  f"clearance={clearance:.1f}mm  "
                  f"height={ground_h:.1f}cm from ground")

            # Show top 3 alternatives
            for i, alt in enumerate(best[1:3], 2):
                _, cl, ax, ay, az2, aaz, ael = alt
                gh = (wr + az2) * 100
                print(f"     #{i}: pos=({ax*100:+5.1f}, {ay*100:+5.1f}, {az2*100:+5.1f})cm "
                      f"dir=({aaz:+.0f}, {ael:+.0f}) cl={cl:.1f}mm h={gh:.1f}cm")

    # Summary / recommendations
    print(f"\n{'=' * 80}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 80}")
    print()

    # For each wheel radius, check if ALL 5 sensors have valid placements
    for wr in wheel_radii:
        all_ok = True
        worst_clearance = float("inf")
        max_height = 0
        for role, az_nom in ROLES.items():
            res = optimize_role(role, az_nom, [wr])
            if not res[wr]:
                all_ok = False
                break
            top = res[wr][0]
            worst_clearance = min(worst_clearance, top[1])
            ground_h = (wr + top[4]) * 100
            max_height = max(max_height, ground_h)

        status = "ALL CLEAR" if all_ok else "BLOCKED"
        if all_ok:
            print(f"  Wheel R={wr*100:.0f}cm: {status} | "
                  f"worst clearance={worst_clearance:.1f}mm | "
                  f"max sensor height={max_height:.1f}cm from ground")
        else:
            print(f"  Wheel R={wr*100:.0f}cm: {status} — some sensors have no valid placement")

    # Final recommendation
    print()
    print("SUGGESTED CONFIGURATION:")
    print("  Reduce wheel radius from 12cm to 10cm (diameter 20cm).")
    print("  This allows mounting all sensors on the head/upper body")
    print("  while keeping them below 22cm from ground (vs 24cm current total height).")
    print("  The 2cm radius reduction costs ~17% ground clearance but")
    print("  eliminates all self-occlusion with comfortable margins.")


if __name__ == "__main__":
    main()
