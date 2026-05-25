"""Predict session command: stateless prediction flow.

Flow:
    1. Receive pre-computed session metrics and patient context (from the Central API)
    2. Build the 15-feature vector and run the stateful ML
    3. Derive cognitive_level deterministically from SPS for therapist reporting
    4. Return the rich result to the interface layer
"""

from dataclasses import dataclass

from domain.patient_context import PatientContext, feature_vector
from domain.session_metrics import (
    CognitiveLevel,
    SessionMetrics,
    cognitive_level_from_sps,
)
from infrastructure.ml.onnx_classifier import ClassificationResult, OnnxClassifier


@dataclass(frozen=True)
class PredictSessionCommand:
    metrics: SessionMetrics
    context: PatientContext


@dataclass(frozen=True)
class PredictSessionResult:
    metrics: SessionMetrics
    context: PatientContext
    cognitive_level: CognitiveLevel
    classification: ClassificationResult


class PredictSessionHandler:
    def __init__(self, classifier: OnnxClassifier) -> None:
        self._classifier = classifier

    def handle(self, command: PredictSessionCommand) -> PredictSessionResult:
        cognitive_level = cognitive_level_from_sps(command.metrics.sps)
        classification = self._classifier.classify(
            feature_vector(command.metrics, command.context)
        )

        return PredictSessionResult(
            metrics=command.metrics,
            context=command.context,
            cognitive_level=cognitive_level,
            classification=classification,
        )
