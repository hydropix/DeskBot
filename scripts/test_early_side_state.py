"""
Unit tests for the early avoidance side cache state machine.

Exercises `update_side_cache` in isolation with a fake `pick_side`
callback and a scripted sequence of `d_front` values. No MuJoCo,
no Navigator. ASCII-only output for cp1252 consoles.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deskbot.early_avoidance import (
    EarlyAvoidanceParams,
    EARLY_PARAMS,
    EarlySideCache,
    CLEAR_MARGIN,
    update_side_cache,
)


DT = 0.1  # 10 Hz simulation tick; shorter than t_hold/t_clear (1 s each)


def make_picker(value):
    """Build a pick_side callback that always returns `value`."""
    calls = {"n": 0}

    def _pick():
        calls["n"] += 1
        return value

    _pick.calls = calls
    return _pick


def test_not_go_heading_resets_cache():
    cache = EarlySideCache(side=+1, hold_timer=0.5, clear_timer=0.3)
    picker = make_picker(+1)
    update_side_cache(cache, d_front=0.8,
                      fsm_is_go_heading=False, dt=DT,
                      pick_side=picker)
    assert cache.side is None, f"expected None, got {cache.side}"
    assert cache.hold_timer == 0.0
    assert cache.clear_timer == 0.0
    assert picker.calls["n"] == 0, "picker must not be called outside GO_HEADING"


def test_side_selected_when_obstacle_in_range():
    cache = EarlySideCache()
    picker = make_picker(-1)
    update_side_cache(cache, d_front=1.2,
                      fsm_is_go_heading=True, dt=DT,
                      pick_side=picker)
    assert cache.side == -1, f"expected side=-1, got {cache.side}"
    assert picker.calls["n"] == 1


def test_side_not_selected_above_cutoff():
    cache = EarlySideCache()
    picker = make_picker(+1)
    update_side_cache(cache, d_front=3.0,
                      fsm_is_go_heading=True, dt=DT,
                      pick_side=picker)
    assert cache.side is None
    assert picker.calls["n"] == 0


def test_side_not_selected_below_trig():
    cache = EarlySideCache()
    picker = make_picker(+1)
    update_side_cache(cache, d_front=0.30,  # < d_trig (0.36)
                      fsm_is_go_heading=True, dt=DT,
                      pick_side=picker)
    assert cache.side is None
    assert picker.calls["n"] == 0


def test_side_holds_for_at_least_t_hold():
    """After initial selection, side must stay stable while obstacle
    is in range, regardless of what the picker says later."""
    cache = EarlySideCache()
    initial_picker = make_picker(+1)
    update_side_cache(cache, 1.5, True, DT, initial_picker)
    assert cache.side == +1

    # Flip the picker to say -1, but the cache is NOT supposed to ask
    # it again until the clear-timer invalidates the side.
    contradictory = make_picker(-1)
    t = 0.0
    while t < 2.0 * EARLY_PARAMS.t_hold:
        update_side_cache(cache, 1.5, True, DT, contradictory)
        t += DT
    assert cache.side == +1, (
        f"side flipped to {cache.side} mid-hold, expected stable +1"
    )
    assert contradictory.calls["n"] == 0, (
        "contradictory picker was called; hysteresis broken"
    )
    # hold_timer should have accumulated close to 2 * t_hold
    assert cache.hold_timer > EARLY_PARAMS.t_hold


def test_cache_reset_after_t_clear_obstacle_gone():
    """Simulate: approach obstacle, lock side, obstacle clears, wait
    more than t_clear, cache should reset and next obstacle is fresh."""
    cache = EarlySideCache()
    update_side_cache(cache, 1.0, True, DT, make_picker(+1))
    assert cache.side == +1

    # Obstacle gone: d_front well above d_cut + CLEAR_MARGIN.
    d_clear = EARLY_PARAMS.d_cut + CLEAR_MARGIN + 0.5
    t = 0.0
    while t < EARLY_PARAMS.t_clear + 0.5:
        update_side_cache(cache, d_clear, True, DT, make_picker(+1))
        t += DT
    assert cache.side is None, (
        f"cache not invalidated after {t}s clear, still {cache.side}"
    )

    # Next obstacle: picker says -1, cache picks it.
    update_side_cache(cache, 1.2, True, DT, make_picker(-1))
    assert cache.side == -1


def test_clear_timer_resets_on_obstacle_re_entry():
    """A brief glimpse over d_cut + margin followed by immediate re-
    appearance within d_cut must not invalidate the side."""
    cache = EarlySideCache()
    update_side_cache(cache, 1.2, True, DT, make_picker(+1))
    assert cache.side == +1

    # Intermittent clear/present sequence, none longer than t_clear.
    d_clear = EARLY_PARAMS.d_cut + CLEAR_MARGIN + 0.3
    for _ in range(3):
        # 0.5 s of clear, then 0.5 s of present
        t = 0.0
        while t < 0.5:
            update_side_cache(cache, d_clear, True, DT, make_picker(-1))
            t += DT
        t = 0.0
        while t < 0.5:
            update_side_cache(cache, 1.0, True, DT, make_picker(-1))
            t += DT
    assert cache.side == +1, (
        f"cache flipped to {cache.side} under intermittent noise"
    )


def test_below_trig_does_not_reset_cache():
    """When d_front drops below d_trig (entering CONTOUR territory),
    we must keep the side cached so a later return to the active
    range reuses the same direction. Reset is the FSM's job."""
    cache = EarlySideCache()
    update_side_cache(cache, 1.0, True, DT, make_picker(+1))
    assert cache.side == +1

    # d drops below d_trig for a few ticks
    for _ in range(10):
        update_side_cache(cache, 0.30, True, DT, make_picker(-1))
    assert cache.side == +1


def main():
    tests = [
        ("not GO_HEADING -> full reset     ", test_not_go_heading_resets_cache),
        ("initial side selection (in range)", test_side_selected_when_obstacle_in_range),
        ("no selection above d_cut         ", test_side_not_selected_above_cutoff),
        ("no selection below d_trig        ", test_side_not_selected_below_trig),
        ("side hold >= t_hold              ", test_side_holds_for_at_least_t_hold),
        ("reset after t_clear gone         ", test_cache_reset_after_t_clear_obstacle_gone),
        ("intermittent clear/present       ", test_clear_timer_resets_on_obstacle_re_entry),
        ("d_front < d_trig keeps cache     ", test_below_trig_does_not_reset_cache),
    ]
    print("=" * 60)
    print("  early_avoidance.py -- side cache state tests")
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
