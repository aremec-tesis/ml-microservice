# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stateless FastAPI microservice that recommends a personalized difficulty adjustment (`decrease` / `maintain` / `increase`) for patients with cognitive deterioration using a **stateful** SVM classifier exported to ONNX. The model consumes the 6 cognitive metrics of the current VR session **and** 9 aggregated features derived from the patient's longitudinal history. The microservice does **not** access any database — both the 6 current-session metrics and the 9 aggregated history features are supplied by the upstream **Central API**. It also derives a deterministic `cognitive_level` (low/medium/high) from the SPS as informational data for the therapist.

## Commands

```bash
# Install dependencies (use the .venv virtual environment)
.venv/Scripts/pip.exe install -r requirements.txt

# Generate the synthetic longitudinal dataset (writes misc/dataset/synthetic_vr_dataset.csv)
.venv/Scripts/python.exe misc/generate_dataset.py

# Train the model (generates misc/model.onnx + misc/scaler.joblib)
.venv/Scripts/python.exe misc/train.py

# Run the server (only requires model.onnx + scaler.joblib — no database)
.venv/Scripts/python.exe main.py

# Test the endpoint
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "patient_id": "5a1f3c8a-...-...",
  "session_metrics": {"ors": 0.78, "ers": 0.80, "scs": 1.0, "rta": 2.10, "er": 0.20, "sps": 0.72},
  "patient_history": {"baseline_sps": 0.65, "slope_sps": 0.02, "delta_sps": 0.07, "mean_ors": 0.70, "mean_ers": 0.75, "mean_er": 0.20, "mean_rta": 2.3, "std_sps": 0.05, "session_count": 8}
}'
```

## Architecture — DDD + CQRS

The project follows a layered Domain-Driven Design + CQRS structure:

```
domain/                                    # Pure business types, no framework deps
  session_metrics.py                       # SessionMetrics, CognitiveLevel, cognitive_level_from_sps
  patient_context.py                       # PatientContext, feature_vector
  difficulty_recommendation.py             # DifficultyRecommendation enum

app/                                       # Orchestration (CQRS)
  commands/predict_session.py              # PredictSessionCommand + Handler

infrastructure/                            # Technical adapters
  config.py                                # ONNX/scaler paths
  ml/onnx_classifier.py                    # Stateful SVM ONNX + scaler wrapper, returns label + probabilities

interfaces/                                # External entry points
  http/api.py                              # FastAPI app + lifespan + /predict
  http/schemas.py                          # Pydantic input/output DTOs

main.py                                    # Shim that boots interfaces.http.api

misc/
  generate_dataset.py                      # Synthetic longitudinal dataset generator
  train.py                                 # Training script (15 features, target=recommendation)
  dataset/synthetic_vr_dataset.csv         # ~6k sessions, 500 patients
  model.onnx + scaler.joblib               # Trained artifacts
```

### Dependency rule
`interfaces → app → domain`, `infrastructure → domain`. Domain never imports from other layers. The `interfaces/http/api.py` lifespan is the composition root that wires the ONNX classifier into the command handler.

## Endpoint flow (`POST /predict`)

1. Pydantic validates the incoming `SessionInput` (patient_id + session_metrics + patient_history)
2. `SessionInput.session_metrics.to_domain()` builds the `SessionMetrics` value object
3. `SessionInput.patient_history.to_domain()` builds the `PatientContext` value object
4. `feature_vector(metrics, context)` builds the 15-feature vector in canonical order
5. `OnnxClassifier.classify()` runs the stateful SVM and returns `recommendation` + per-class probabilities
6. `cognitive_level_from_sps()` derives an informational level (low/medium/high) from the current SPS
7. Endpoint returns `PredictionResponse` with patient_id, metrics, cognitive_level, recommendation, probabilities, context

## Key Conventions

- The microservice is **stateless**: no database reads, no database writes. Persistence and clinical traceability are the Central API's responsibility.
- The Central API is responsible for computing the 6 cognitive metrics (ORS, ERS, SCS, RTA, ER, SPS) from raw Unity telemetry and for aggregating the 9 history features (baseline_sps, slope_sps, delta_sps, mean_ors, mean_ers, mean_er, mean_rta, std_sps, session_count) from the patient's previous sessions.
- Recommendation labels: `{0: "decrease_difficulty", 1: "maintain_difficulty", 2: "increase_difficulty"}` — explicit mapping, NOT LabelEncoder
- 15-feature vector order (defined in `domain/patient_context.feature_vector`):
  `[ORS, ERS, SCS, RTA, ER, SPS, baseline_sps, slope_sps, delta_sps, mean_ors, mean_ers, mean_er, mean_rta, std_sps, session_count]`
- The scaler and ONNX model must always be saved/loaded as a pair
- ONNX inference requires float32 input (scaler outputs float64, cast is required)
- ONNX model exposes 2 outputs: predicted label and per-class probabilities (`zipmap=False` at export)
- `cognitive_level` is purely derived from SPS thresholds (`<0.4` low, `0.4-0.7` medium, `>0.7` high) — informational only, not consumed by the ML
- Cold start (`session_count == 0`): the Central API neutralizes history features to current-session values (e.g., `baseline_sps = sps`, `slope_sps = 0`, `delta_sps = 0`, `mean_* = current_*`). `PatientContext.cold_start` is derived locally as `session_count == 0`.
