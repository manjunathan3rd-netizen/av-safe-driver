# SIH26037 - Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads

Project codebase for the Smart India Hackathon 2026 problem statement `SIH26037` (MathWorks, Smart Vehicles).

## What is in this project

| File | Purpose |
| :--- | :--- |
| `index.html` | Dashboard (single file, no dependencies): moving dash-cam view with detections, risk Low/Medium/High with reason, safe-path planner view, driver-impairment monitor, 5-scenario benchmark, clip recording, optional webcam / uploaded video background |
| `av_core.py` | The algorithms (pure Python): tracker, multi-modal prediction, Monte-Carlo collision risk, risk-aware lateral-lattice planner, decision FSM, driver-impairment monitor, benchmark |
| `app.py` | FastAPI service exposing `av_core.py` over REST |
| `test_core.py` | Tests for the algorithms (no web framework needed) |
| `test_app.py` | Tests for the REST API |
| `solution.md` | Technical write-up: architecture, algorithms, scenarios, results, limitations |
| `Dockerfile`, `docker-compose.yml`, `requirements.txt` | Containerisation |

## Run

### Dashboard only (zero dependencies)
```bash
python -m http.server 8080
```
Open http://localhost:8080 . Webcam and "Record clip" need HTTPS or localhost.

### Backend API
```bash
pip install -r requirements.txt
python app.py
```
- API: http://127.0.0.1:8000
- Interactive docs: http://127.0.0.1:8000/docs

### Tests
```bash
pytest -v
```

### Docker
```bash
docker-compose up --build
```
Dashboard on :8080, API on :8000.

## API

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/` | GET | Service metadata |
| `/api/v1/scenarios` | GET | The 5 Indian-road scenarios |
| `/api/v1/risk/assess` | POST | Detections in, risk level + collision probabilities + safe-path decision out |
| `/api/v1/driver/assess` | POST | Driver signals in, impairment risk + state + ignition-lock flag out |
| `/api/v1/simulate` | POST | Run one scenario with either planner; returns metrics (and optional trace) |
| `/api/v1/benchmark` | GET | 5 scenarios x {ours, brake-only baseline} |

Example:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/risk/assess \
  -H "Content-Type: application/json" \
  -d '{"ego_speed_mps": 9, "objects": [{"type": "cow", "x": 0.0, "y": 14}]}'
```

## Important notes (read before presenting)

- This is a **2-D kinematic simulation**. It is not a MATLAB/Simulink, RoadRunner or CARLA model, and detections are simulated (no camera/LiDAR model is run).
- The JavaScript in `index.html` and the Python in `av_core.py` implement the same algorithm independently. Their random streams differ, so numbers will be close but not identical.
- The dashboard does not call the API; the API is a separate, testable implementation of the same logic.
- `test_core.py` was run and passes. `test_app.py` and `app.py` were only syntax-checked when this package was produced, so run `pytest -v` once after installing the requirements.
