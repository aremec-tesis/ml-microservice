"""Loads the ONNX model and scaler into memory; exposes a predict function."""

from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.onnx"
SCALER_PATH = BASE_DIR / "scaler.joblib"

LABELS = {0: "low", 1: "medium", 2: "high"}

ort_session: ort.InferenceSession | None = None
scaler = None


def load_model() -> None:
    global ort_session, scaler

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found at {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")

    scaler = joblib.load(SCALER_PATH)
    ort_session = ort.InferenceSession(str(MODEL_PATH))


def predict(features: list[float]) -> str:
    if ort_session is None or scaler is None: 
        raise RuntimeError("Model not loaded. Call load_model() first.")

    X = np.array([features], dtype=np.float64)
    X_scaled = scaler.transform(X).astype(np.float32)

    input_name = ort_session.get_inputs()[0].name
    result = ort_session.run(None, {input_name: X_scaled})
    label_index = int(result[0][0])

    return LABELS[label_index]
