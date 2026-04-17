"""ONNX + scaler wrapper that classifies a feature vector into a CognitiveLevel."""

from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort

from domain.session_metrics import CognitiveLevel

_LABEL_BY_INDEX = {
    0: CognitiveLevel.LOW,
    1: CognitiveLevel.MEDIUM,
    2: CognitiveLevel.HIGH,
}


class OnnxClassifier:
    def __init__(self, model_path: Path, scaler_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")

        self._session = ort.InferenceSession(str(model_path))
        self._scaler = joblib.load(scaler_path)
        self._input_name = self._session.get_inputs()[0].name

    def classify(self, features: list[float]) -> CognitiveLevel:
        X = np.array([features], dtype=np.float64)
        X_scaled = self._scaler.transform(X).astype(np.float32)
        result = self._session.run(None, {self._input_name: X_scaled})
        label_index = int(result[0][0])
        return _LABEL_BY_INDEX[label_index]
