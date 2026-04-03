# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastAPI microservice that predicts cognitive performance levels (low/medium/high) for patients with cognitive deterioration using an SVM classifier exported to ONNX. It receives session data from a Unity VR application, calculates cognitive metrics, returns a classification with difficulty adjustment recommendation, and persists results to PostgreSQL (TimescaleDB).

## Commands

```bash
# Install dependencies (use the .venv virtual environment)
.venv/Scripts/pip.exe install -r requirements.txt

# Train the model (generates model.onnx + scaler.joblib)
.venv/Scripts/python.exe train.py

# Run the server (requires model.onnx, scaler.joblib, and a running PostgreSQL)
.venv/Scripts/python.exe main.py

# Test the endpoint
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"patient_id": 1, "total_objects": 10, "correct_objects": 7, "total_events": 5, "correct_events": 4, "comprehension_score": 2, "response_times": [2.1, 3.5, 1.8], "total_questions": 10, "incorrect_answers": 2, "interaction_events": 8, "expected_interactions": 10}'
```

## Architecture

Multi-file architecture:

- **`metrics.py`** — Pydantic models (`SessionInput`, `SessionMetrics`, `PredictionResponse`), metric calculation functions, recommendation logic, and feature vector assembly
- **`train.py`** — Standalone training script: reads `dataset/synthetic_vr_dataset.csv`, trains StandardScaler + SVC, exports model to ONNX via skl2onnx, saves scaler with joblib
- **`model_handler.py`** — Loads `model.onnx` (onnxruntime) and `scaler.joblib` (joblib) at startup; exposes `predict()` function
- **`database.py`** — Async PostgreSQL connection pool (asyncpg), DDL for `schema_telemetria.metricas_sesion`, insert/query functions
- **`main.py`** — FastAPI app with lifespan (model + DB init), CORS middleware, `POST /predict` endpoint

## Key Conventions

- Classification labels: `{0: "low", 1: "medium", 2: "high"}` — explicit mapping, NOT LabelEncoder
- Feature vector order: `[ORS, ERS, SCS, RTA, ATS, ER, SPS]` — must match across train.py, metrics.py, and model_handler.py
- The scaler and ONNX model must always be saved/loaded as a pair
- ONNX inference requires float32 input (scaler outputs float64, cast is required)
- DB credentials loaded from `.env` via python-dotenv
- Recommendation thresholds: SPS < 0.4 → decrease; 0.4 ≤ SPS ≤ 0.7 → maintain; SPS > 0.7 → increase
