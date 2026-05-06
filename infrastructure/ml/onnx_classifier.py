"""ONNX + scaler wrapper that classifies a 15-feature vector into a DifficultyRecommendation.

Returns both the predicted label and the per-class probabilities so the inference is
traceable end-to-end (the probabilities are persisted alongside the prediction).
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort

from domain.difficulty_recommendation import DifficultyRecommendation

_LABEL_BY_INDEX = {
    0: DifficultyRecommendation.DECREASE,
    1: DifficultyRecommendation.MAINTAIN,
    2: DifficultyRecommendation.INCREASE,
}


@dataclass(frozen=True)
class ClassificationResult:
    recommendation: DifficultyRecommendation
    prob_decrease: float
    prob_maintain: float
    prob_increase: float


class OnnxClassifier:
    def __init__(self, model_path: Path, scaler_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")

        self._session = ort.InferenceSession(str(model_path))
        self._scaler = joblib.load(scaler_path)
        self._input_name = self._session.get_inputs()[0].name

    def classify(self, features: list[float]) -> ClassificationResult:
        X = np.array([features], dtype=np.float64)
        X_scaled = self._scaler.transform(X).astype(np.float32)
        outputs = self._session.run(None, {self._input_name: X_scaled})
        label_index = int(outputs[0][0])
        probs = outputs[1][0]
        return ClassificationResult(
            recommendation=_LABEL_BY_INDEX[label_index],
            prob_decrease=float(probs[0]),
            prob_maintain=float(probs[1]),
            prob_increase=float(probs[2]),
        )
