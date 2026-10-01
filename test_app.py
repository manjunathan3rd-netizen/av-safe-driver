"""Pytest suite for the FastAPI service (requires fastapi, httpx, pytest)."""
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    r = client.get("/")
    assert r.status_code == 200 and r.json()["problem_id"] == "SIH26037"


def test_scenarios_lists_five():
    r = client.get("/api/v1/scenarios")
    assert r.status_code == 200 and len(r.json()) == 5


def test_risk_assess_cow_in_lane():
    payload = {"ego_speed_mps": 9, "objects": [{"type": "cow", "x": 0.0, "y": 14}]}
    r = client.post("/api/v1/risk/assess", json=payload)
    assert r.status_code == 200
    d = r.json()
    assert d["overall_risk"] == "HIGH" and d["objects"][0]["type"] == "cow"


def test_risk_assess_rejects_bad_input():
    r = client.post("/api/v1/risk/assess", json={"objects": [{"type": "dragon", "x": 0, "y": 5}]})
    assert r.status_code == 422


def test_driver_assess_impaired():
    r = client.post("/api/v1/driver/assess", json={"perclos": .5, "reaction_time_s": 1.2,
                                                    "lane_weave_m": .6, "breath_alcohol_mgl": .5})
    d = r.json()
    assert r.status_code == 200 and d["state"] == "TAKEOVER" and d["ignition_locked"] is True


def test_simulate_returns_metrics():
    r = client.post("/api/v1/simulate", json={"scenario_index": 4, "seconds": 10})
    assert r.status_code == 200 and "collisions" in r.json()["metrics"]


def test_benchmark_has_ten_rows():
    r = client.get("/api/v1/benchmark")
    assert r.status_code == 200 and len(r.json()["rows"]) == 10
