# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastAPI microservice that predicts cognitive performance levels (bajo/medio/alto) for Alzheimer's patients using an SVM classifier. It receives behavioral metrics from a Unity VR application and returns a classification. All code lives in a single file (`main.py`).

## Commands

```bash
# Install dependencies (use the .venv virtual environment)
.venv/Scripts/pip.exe install -r requirements.txt

# Run the server (auto-trains model on first run if .pkl files are missing)
.venv/Scripts/python.exe main.py

# Test the endpoint
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"tiempo_reaccion": 1200, "aciertos": 8, "errores": 2, "tiempo_total": 300}'

# Retrain the model manually
.venv/Scripts/python.exe -c "from main import entrenar_modelo; entrenar_modelo()"
```

## Architecture

Single-file architecture (`main.py`) with these sections:

1. **Pydantic models** — `MetricasPaciente` (input validation) and `PrediccionResponse` (output)
2. **`entrenar_modelo()`** — Generates synthetic training data (60 samples per class), fits StandardScaler + SVC(kernel='rbf'), saves both to `.pkl` files
3. **`lifespan`** — FastAPI lifespan handler that loads existing `.pkl` files or triggers training if they don't exist
4. **`POST /predict`** — Receives metrics, scales with the loaded scaler, predicts with the loaded SVM, maps numeric prediction to label via `ETIQUETAS` dict

The model and scaler are held as module-level globals, loaded once at startup. The feature order is fixed: `[tiempo_reaccion, aciertos, errores, tiempo_total]`.

## Key Conventions

- All code, comments, variable names, and API responses are in Spanish
- Classification labels: `{0: "bajo", 1: "medio", 2: "alto"}`
- The scaler and model must always be saved/loaded as a pair — retraining one without the other will break predictions
