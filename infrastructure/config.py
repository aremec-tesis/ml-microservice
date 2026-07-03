"""Runtime configuration: paths to the ONNX model and scaler artifacts."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "misc" / "model.onnx"
SCALER_PATH = BASE_DIR / "misc" / "scaler.joblib"

# URL of the Central API — set this env var before running the server.
# Example: CENTRAL_API_ORIGIN=https://api.aremec.com
CENTRAL_API_ORIGIN = os.getenv("CENTRAL_API_ORIGIN", "http://localhost:3000")