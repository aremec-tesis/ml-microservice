# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastAPI microservice that recommends a personalized difficulty adjustment (`decrease` / `maintain` / `increase`) for patients with cognitive deterioration using a **stateful** SVM classifier exported to ONNX. The model consumes the 7 cognitive metrics of the current VR session **and** 9 aggregated features derived from the patient's longitudinal history (10-session window). It also derives a deterministic `cognitive_level` (low/medium/high) from the SPS as informational data for the therapist. All inferences (features, prediction, probabilities) are persisted to PostgreSQL (TimescaleDB) for clinical traceability.

## Commands

```bash
# Install dependencies (use the .venv virtual environment)
.venv/Scripts/pip.exe install -r requirements.txt

# Generate the synthetic longitudinal dataset (writes misc/dataset/synthetic_vr_dataset.csv)
.venv/Scripts/python.exe misc/generate_dataset.py

# Train the model (generates misc/model.onnx + misc/scaler.joblib)
.venv/Scripts/python.exe misc/train.py

# Run the server (requires model.onnx, scaler.joblib, and a running PostgreSQL)
.venv/Scripts/python.exe main.py

# Test the endpoint
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"patient_id": 1, "correct_key_objects": 4, "correct_secondary_objects": 5, "incorrect_objects": 1, "total_key_objects": 5, "total_secondary_objects": 8, "total_events": 5, "correct_events": 4, "comprehension_score": 2, "response_times": [2.1, 3.5, 1.8, 2.0, 2.5, 1.9, 2.3, 2.1, 2.0, 1.7], "total_questions": 10, "incorrect_answers": 2, "interaction_events": 8, "expected_interactions": 10}'
```

## Architecture — DDD + CQRS

The project follows a layered Domain-Driven Design + CQRS structure:

```
domain/                                    # Pure business types, no framework deps
  session_metrics.py                       # RawSessionData, SessionMetrics, CognitiveLevel, cognitive_level_from_sps
  patient_context.py                       # HistoricalSession, PatientContext, feature_vector
  difficulty_recommendation.py             # DifficultyRecommendation enum

app/                                       # Orchestration (CQRS)
  commands/predict_session.py              # PredictSessionCommand + Handler (full flow)
  queries/get_patient_history.py           # GetPatientHistoryQuery + Handler

infrastructure/                            # Technical adapters
  config.py                                # Env settings, paths, HISTORY_WINDOW
  ml/onnx_classifier.py                    # Stateful SVM ONNX + scaler wrapper, returns label + probabilities
  persistence/postgres_pool.py             # asyncpg pool + DDL
  persistence/session_repository.py        # insert_session, get_patient_history

interfaces/                                # External entry points
  http/api.py                              # FastAPI app + lifespan + /predict
  http/schemas.py                          # Pydantic input/output DTOs

main.py                                    # Shim that boots interfaces.http.api

misc/
  generate_dataset.py                      # Synthetic longitudinal dataset generator
  train.py                                 # Training script (16 features, target=recommendation)
  dataset/synthetic_vr_dataset.csv         # ~6k sessions, 500 patients
  model.onnx + scaler.joblib               # Trained artifacts
```

### Dependency rule
`interfaces → app → domain`, `infrastructure → domain`. Domain never imports from other layers. The `interfaces/http/api.py` lifespan is the composition root that wires concrete infrastructure into app handlers.

## Endpoint flow (`POST /predict`)

1. Pydantic validates the incoming `SessionInput`
2. `GetPatientHistoryHandler` fetches up to 10 recent sessions (sps, ors, ers, er, rta) of the patient
3. `SessionMetrics.from_raw()` computes the 7 cognitive metrics (ORS uses the new key/secondary weighted formula)
4. `PatientContext.from_history()` derives the 9 aggregated history features (baseline, slope, delta, means, std, count) — neutralized in cold start
5. `feature_vector(metrics, context)` builds the 16-feature vector
6. `OnnxClassifier.classify()` runs the stateful SVM and returns `recommendation` + per-class probabilities
7. `cognitive_level_from_sps()` derives an informational level (low/medium/high) from the current SPS
8. `SessionRepository.insert_session()` persists raw data, metrics, history features, recommendation, probabilities and cognitive_level
9. Endpoint returns `PredictionResponse` with metrics, cognitive_level, recommendation, probabilities, context

## Key Conventions

- Recommendation labels: `{0: "decrease_difficulty", 1: "maintain_difficulty", 2: "increase_difficulty"}` — explicit mapping, NOT LabelEncoder
- 16-feature vector order (defined in `domain/patient_context.feature_vector`):
  `[ORS, ERS, SCS, RTA, ATS, ER, SPS, baseline_sps, slope_sps, delta_sps, mean_ors, mean_ers, mean_er, mean_rta, std_sps, session_count]`
- ORS formula (new): `((correct_key*2) + (correct_secondary*1) - (incorrect*1)) / ((total_key*2) + (total_secondary*1))`. Can be negative; clamp not applied.
- The scaler and ONNX model must always be saved/loaded as a pair
- ONNX inference requires float32 input (scaler outputs float64, cast is required)
- ONNX model exposes 2 outputs: predicted label and per-class probabilities (`zipmap=False` at export)
- DB credentials loaded from `.env` via python-dotenv
- `cognitive_level` is purely derived from SPS thresholds (`<0.4` low, `0.4-0.7` medium, `>0.7` high) — informational only, not consumed by the ML
- History window: 10 sessions (`HISTORY_WINDOW` in `infrastructure/config.py`)
- Cold start (`session_count == 0`): history features neutralized to current-session values, ML still produces recommendation
- Clinical traceability: every prediction persists the 16 input features and the 3 class probabilities so any decision can be explained
