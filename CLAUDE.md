# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastAPI microservice that predicts cognitive performance levels (low/medium/high) for patients with cognitive deterioration using an SVM classifier exported to ONNX, and produces a personalized difficulty adjustment recommendation using patient longitudinal history. It receives session data from a Unity VR application, calculates cognitive metrics, classifies performance, personalizes the difficulty recommendation based on patient history (baseline, trend, delta), and persists results to PostgreSQL (TimescaleDB).

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

## Architecture — DDD + CQRS

The project follows a layered Domain-Driven Design + CQRS structure:

```
domain/                                    # Pure business types, no framework deps
  session_metrics.py                       # RawSessionData, SessionMetrics, CognitiveLevel
  patient_context.py                       # HistoricalSession, PatientContext, TrendType
  difficulty_recommendation.py             # DifficultyRecommendation enum

app/                                       # Orchestration (CQRS)
  commands/predict_session.py              # PredictSessionCommand + Handler (full flow)
  queries/get_patient_history.py           # GetPatientHistoryQuery + Handler

infrastructure/                            # Technical adapters
  config.py                                # Env settings, paths, HISTORY_WINDOW
  ml/onnx_classifier.py                    # SVM ONNX + scaler wrapper
  ml/personalization_engine.py             # Clinical rule-based personalization
  persistence/postgres_pool.py             # asyncpg pool + DDL
  persistence/session_repository.py        # insert_session, get_patient_history

interfaces/                                # External entry points
  http/api.py                              # FastAPI app + lifespan + /predict
  http/schemas.py                          # Pydantic input/output DTOs

main.py                                    # Shim that boots interfaces.http.api
train.py                                   # Standalone training script (unchanged)
```

### Dependency rule
`interfaces → app → domain`, `infrastructure → domain`. Domain never imports from other layers. The `interfaces/http/api.py` lifespan is the composition root that wires concrete infrastructure into app handlers.

## Endpoint flow (`POST /predict`)

1. Pydantic validates the incoming `SessionInput`
2. `GetPatientHistoryHandler` fetches up to 10 recent sessions of the patient
3. `SessionMetrics.from_raw()` computes the 7 cognitive metrics
4. `PatientContext.from_history()` derives baseline (weighted MA), trend (linear slope), delta
5. `OnnxClassifier.classify()` runs the SVM on the feature vector
6. `PersonalizationEngine.recommend()` produces the personalized recommendation using classification + context
7. `SessionRepository.insert_session()` persists raw data, metrics, prediction, recommendation, and context used
8. Endpoint returns `PredictionResponse` with metrics, prediction, recommendation, and context

## Key Conventions

- Classification labels: `{0: "low", 1: "medium", 2: "high"}` — explicit mapping, NOT LabelEncoder
- Feature vector order: `[ORS, ERS, SCS, RTA, ATS, ER, SPS]` — defined in `SessionMetrics.as_feature_vector()`
- The scaler and ONNX model must always be saved/loaded as a pair
- ONNX inference requires float32 input (scaler outputs float64, cast is required)
- DB credentials loaded from `.env` via python-dotenv
- Base SPS thresholds: `< 0.4` decrease, `0.4 - 0.7` maintain, `> 0.7` increase
- Personalization constants live in `infrastructure/ml/personalization_engine.py` (TREND_THRESHOLD in `domain/patient_context.py`)
- History window: 10 sessions (`HISTORY_WINDOW` in `infrastructure/config.py`)
- Cold start (`session_count == 0`): falls back to base SPS thresholds without personalization
