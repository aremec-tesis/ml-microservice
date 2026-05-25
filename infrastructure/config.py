"""Runtime configuration: paths to the ONNX model and scaler artifacts."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "misc" / "model.onnx"
SCALER_PATH = BASE_DIR / "misc" / "scaler.joblib"
