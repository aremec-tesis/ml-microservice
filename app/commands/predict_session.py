"""Predict session command: orchestrates the full stateful prediction flow.

Flow:
    1. Fetch patient history (query)
    2. Compute session metrics from raw input
    3. Derive patient longitudinal context (9 aggregated features)
    4. Build the 16-feature vector and run the stateful ML
    5. Derive cognitive_level deterministically from SPS for therapist reporting
    6. Persist session with full traceability (features + probabilities)
    7. Return rich result to the interface layer
"""

from dataclasses import dataclass

from app.queries.get_patient_history import (
    GetPatientHistoryHandler,
    GetPatientHistoryQuery,
)
from domain.patient_context import PatientContext, feature_vector
from domain.session_metrics import (
    CognitiveLevel,
    RawSessionData,
    SessionMetrics,
    cognitive_level_from_sps,
)
from infrastructure.config import HISTORY_WINDOW
from infrastructure.ml.onnx_classifier import ClassificationResult, OnnxClassifier
from infrastructure.persistence.session_repository import SessionRepository


@dataclass(frozen=True)
class PredictSessionCommand:
    raw: RawSessionData


@dataclass(frozen=True)
class PredictSessionResult:
    metrics: SessionMetrics
    context: PatientContext
    cognitive_level: CognitiveLevel
    classification: ClassificationResult


class PredictSessionHandler:
    def __init__(
        self,
        classifier: OnnxClassifier,
        repository: SessionRepository,
        history_handler: GetPatientHistoryHandler,
    ) -> None:
        self._classifier = classifier
        self._repository = repository
        self._history_handler = history_handler

    async def handle(self, command: PredictSessionCommand) -> PredictSessionResult:
        raw = command.raw

        history = await self._history_handler.handle(
            GetPatientHistoryQuery(patient_id=raw.patient_id, limit=HISTORY_WINDOW)
        )

        metrics = SessionMetrics.from_raw(raw)
        context = PatientContext.from_history(history, current=metrics)
        cognitive_level = cognitive_level_from_sps(metrics.sps)

        classification = self._classifier.classify(feature_vector(metrics, context))

        await self._repository.insert_session(
            raw=raw,
            metrics=metrics,
            context=context,
            cognitive_level=cognitive_level,
            classification=classification,
        )

        return PredictSessionResult(
            metrics=metrics,
            context=context,
            cognitive_level=cognitive_level,
            classification=classification,
        )
