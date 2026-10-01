"""Tests for av_core.py (pure Python, no web framework needed)."""
import av_core as c


def test_no_collisions_in_any_scenario():
    for i in range(5):
        assert c.simulate(i, "ours", 11 + i, 25)["metrics"]["collisions"] == 0


def test_deterministic_for_same_seed():
    a = c.simulate(3, "ours", 7, 15)["metrics"]
    b = c.simulate(3, "ours", 7, 15)["metrics"]
    for k in ("collisions", "min_gap_m", "distance_m", "rms_lateral_accel_mps2"):
        assert a[k] == b[k]


def test_planner_makes_more_progress_than_brake_only():
    ours = sum(c.simulate(i, "ours", 11 + i, 25)["metrics"]["distance_m"] for i in range(5))
    base = sum(c.simulate(i, "brake", 11 + i, 25)["metrics"]["distance_m"] for i in range(5))
    assert ours > base


def test_assess_empty_road_is_low_risk():
    a = c.assess([], ego_speed=9)
    assert a["overall_risk"] == "LOW" and a["decision"] == "CRUISE"


def test_assess_cow_in_lane_is_high_risk_and_probabilities_valid():
    a = c.assess([{"type": "cow", "x": 0.0, "y": 14, "vx": 0, "vy": 0}], ego_speed=9)
    assert a["overall_risk"] == "HIGH"
    assert all(0 <= o["collision_probability"] <= 1 for o in a["objects"])
    assert all(0 <= p["collision_risk"] <= 1 for p in a["candidate_paths"])


def test_driver_state_thresholds():
    assert c.driver_state(.1) == "OK" and c.driver_state(.4) == "WARN"
    assert c.driver_state(.6) == "CONSERVATIVE" and c.driver_state(.9) == "TAKEOVER"


def test_severe_impairment_ends_in_safe_stop():
    m = c.simulate(0, "ours", 5, 25, sev=1.0)["metrics"]
    assert m["driver_state"] == "MRM" and m["final_speed_kmh"] == 0.0 and m["collisions"] == 0


def test_sober_driver_stays_ok():
    assert c.simulate(1, "ours", 5, 10, sev=0.0)["metrics"]["driver_state"] == "OK"
