"""
Unit tests for the early avoidance guidance law.

Runs 7 pure-function tests on deskbot.early_avoidance.compute_bias_yaw
matching the Step 1 gate of docs/plan_early_avoidance.md. No MuJoCo,
no Navigator, no RNG. ASCII-only output for cp1252 consoles.
"""
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deskbot.early_avoidance import (
    EarlyAvoidanceParams,
    EARLY_PARAMS,
    compute_bias_yaw,
)


TOL = 1e-9


def _approx(got, want, tol=TOL):
    return abs(got - want) <= tol


def test_far_above_cutoff():
    # D = 3 m > d_cut (2 m) -> bias must be 0
    b = compute_bias_yaw(3.0, +1, EARLY_PARAMS)
    assert _approx(b, 0.0), f"D=3 side=+1: expected 0, got {b}"


def test_exactly_at_cutoff():
    # D = 2 m == d_cut -> bias must be 0 (strict <)
    b = compute_bias_yaw(EARLY_PARAMS.d_cut, +1, EARLY_PARAMS)
    assert _approx(b, 0.0), f"D=d_cut side=+1: expected 0, got {b}"


def test_nominal_midrange():
    # D = 1 m, side = +1 -> k / 1 = 0.08 rad/s
    b = compute_bias_yaw(1.0, +1, EARLY_PARAMS)
    want = EARLY_PARAMS.k / 1.0
    assert _approx(b, want, 1e-6), f"D=1 side=+1: expected {want}, got {b}"


def test_saturated_negative_side():
    # D = 0.5 m == d_sat, side = -1 -> -k / 0.5 = -0.16 rad/s
    b = compute_bias_yaw(0.5, -1, EARLY_PARAMS)
    want = -EARLY_PARAMS.k / EARLY_PARAMS.d_sat
    assert _approx(b, want, 1e-6), f"D=0.5 side=-1: expected {want}, got {b}"


def test_below_trig_is_zero():
    # D = 0.3 m < d_trig (0.36) -> bias must be 0 (handoff to CONTOUR)
    b = compute_bias_yaw(0.3, +1, EARLY_PARAMS)
    assert _approx(b, 0.0), f"D=0.3 side=+1: expected 0, got {b}"


def test_saturation_floor():
    # D = 0 is below d_trig so must be 0 regardless of saturation.
    # Verify the saturation behaviour just above d_trig instead:
    # at D = d_trig + epsilon the effective distance is d_sat (since
    # d_trig < d_sat), so bias should equal k / d_sat.
    eps = 1e-6
    b = compute_bias_yaw(EARLY_PARAMS.d_trig + eps, +1, EARLY_PARAMS)
    want = EARLY_PARAMS.k / EARLY_PARAMS.d_sat
    assert _approx(b, want, 1e-6), f"d_trig+eps side=+1: expected {want}, got {b}"


def test_monotonic_in_active_range():
    # For side=+1, bias must be weakly decreasing as D grows
    # over [d_sat+eps, d_cut). Below d_sat the bias is constant at
    # peak (saturated), above d_cut it is 0.
    vals = []
    for d in (0.6, 0.8, 1.0, 1.5, 1.9):
        vals.append((d, compute_bias_yaw(d, +1, EARLY_PARAMS)))
    for (d1, v1), (d2, v2) in zip(vals, vals[1:]):
        assert v1 >= v2, f"monotonicity broken: bias({d1})={v1} < bias({d2})={v2}"
    # And bias at 2.0 must be 0
    assert _approx(compute_bias_yaw(2.0, +1, EARLY_PARAMS), 0.0)


def main():
    tests = [
        ("far above cutoff         ", test_far_above_cutoff),
        ("exactly at cutoff         ", test_exactly_at_cutoff),
        ("nominal midrange D=1 side+", test_nominal_midrange),
        ("saturated D=0.5 side-    ", test_saturated_negative_side),
        ("below d_trig (handoff)    ", test_below_trig_is_zero),
        ("saturation floor          ", test_saturation_floor),
        ("monotonic in active range ", test_monotonic_in_active_range),
    ]
    print("=" * 60)
    print("  early_avoidance.py -- unit tests")
    print("=" * 60)
    ok = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            ok += 1
        except AssertionError as e:
            print(f"  [FAIL] {name}  -- {e}")
    print("-" * 60)
    print(f"  {ok}/{len(tests)} tests passed")
    print("=" * 60)
    return 0 if ok == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
