"""
FastAPI microservice: Adaptive Path Planning and Collision Avoidance for Autonomous
Vehicles on Unstructured Indian Roads (SIH26037).

Exposes the algorithms in av_core.py (tracking, Monte-Carlo collision risk, risk-aware
lateral-lattice planner, driver-impairment monitor, 5-scenario benchmark) over REST.
"""
import datetime
from typing import List, Literal

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import av_core

app = FastAPI(
    title="SIH26037 - Adaptive Path Planning & Collision Avoidance (Indian Roads)",
    description="Risk-aware planner, Monte-Carlo collision risk and driver-impairment monitor (2-D simulation).",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

ObjType = Literal["auto", "car", "bike", "truck", "ped", "cow", "cart", "rock"]


class DetectedObject(BaseModel):
    type: ObjType = Field(..., description="Detected class")
    x: float = Field(..., ge=-20, le=20, description="Lateral position (m), + = right of road centre")
    y: float = Field(..., ge=-5, le=100, description="Distance ahead of ego front bumper (m)")
    vx: float = Field(0.0, ge=-15, le=15, description="Lateral velocity (m/s)")
    vy: float = Field(0.0, ge=-15, le=40, description="Forward velocity in the world frame (m/s)")
    conf: float = Field(0.9, ge=0, le=1)


class RiskRequest(BaseModel):
    ego_speed_mps: float = Field(9.0, ge=0, le=40)
    ego_lateral_m: float = Field(0.0, ge=-8, le=8)
    road_half_width_m: float = Field(3.5, ge=2.0, le=8.0)
    impairment: float = Field(0.0, ge=0, le=1, description="Driver impairment 0..1 (simulated)")
    seed: int = 1
    objects: List[DetectedObject] = Field(default_factory=list, max_length=30)


class DriverRequest(BaseModel):
    perclos: float = Field(0.08, ge=0, le=1, description="Fraction of time eyes closed")
    reaction_time_s: float = Field(0.3, ge=0.1, le=3.0)
    lane_weave_m: float = Field(0.05, ge=0, le=2.0)
    breath_alcohol_mgl: float = Field(0.0, ge=0, le=1.5)


class SimRequest(BaseModel):
    scenario_index: int = Field(0, ge=0, le=4)
    planner: Literal["ours", "brake"] = "ours"
    seconds: float = Field(25.0, ge=5, le=60)
    seed: int = 5
    impairment: float = Field(0.0, ge=0, le=1)
    trace: bool = False


@app.get("/", tags=["Health & Metadata"])
def root():
    return {"problem_id": "SIH26037",
            "title": "Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads",
            "organization": "MathWorks", "theme": "Smart Vehicles", "status": "OPERATIONAL",
            "version": "1.0.0", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}


@app.get("/api/v1/scenarios", tags=["Simulation"])
def scenarios():
    return [{"index": i, "name": s["n"], "cruise_speed_mps": s["cv"], "road_half_width_m": s["hw"]}
            for i, s in enumerate(av_core.SC)]


@app.post("/api/v1/risk/assess", tags=["Inference"])
def risk_assess(req: RiskRequest):
    """Collision-risk assessment + safe-path decision for one set of detections."""
    return av_core.assess([o.model_dump() for o in req.objects], req.ego_speed_mps, req.ego_lateral_m,
                          req.road_half_width_m, req.impairment, req.seed)


@app.post("/api/v1/driver/assess", tags=["Driver Monitoring"])
def driver_assess(req: DriverRequest):
    dr = av_core.driver_risk(req.perclos, req.reaction_time_s, req.lane_weave_m, req.breath_alcohol_mgl)
    state = av_core.driver_state(dr)
    action = {"OK": "NORMAL_OPERATION", "WARN": "WARN_DRIVER", "CONSERVATIVE": "CAP_SPEED_AND_WIDEN_MARGINS",
              "TAKEOVER": "AUTONOMY_TAKES_CONTROL"}[state]
    return {"driver_risk": round(dr, 3), "state": state, "recommended_action": action,
            "ignition_locked": req.breath_alcohol_mgl > 0.3}


@app.post("/api/v1/simulate", tags=["Simulation"])
def simulate(req: SimRequest):
    return av_core.simulate(req.scenario_index, req.planner, req.seed, req.seconds, req.impairment, req.trace)


@app.get("/api/v1/benchmark", tags=["Simulation"])
def benchmark():
    """5 Indian-road scenarios x {risk-aware planner, brake-only baseline}, same seeds."""
    return {"seconds_per_run": 25, "rows": av_core.benchmark(25.0)}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
