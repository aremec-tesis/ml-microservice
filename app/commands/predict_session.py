"""Predict session command: orchestrates the full hybrid stateful prediction flow.

Flow:
    1. Fetch patient history (query)
    2. Compute session metrics from raw input
    3. Derive patient longitudinal context
    4. Classify with SVM (ONNX)
    5. Personalize difficulty recommendation using context
    6. Persist session with computed context
    7. Return rich result to the interface layer
"""

from dataclasses import dataclass

from app.queries.get_patient_history import (
    GetPatientHistoryHandler,
    GetPatientHistoryQuery,
)
from domain.difficulty_recommendation import DifficultyRecommendation
from domain.patient_context import PatientContext
from domain.session_metrics import CognitiveLevel, RawSessionData, SessionMetrics
from infrastructure.config import HISTORY_WINDOW
from infrastructure.ml.onnx_classifier import OnnxClassifier
from infrastructure.ml.personalization_engine import PersonalizationEngine
from infrastructure.persistence.session_repository import SessionRepository


@dataclass(frozen=True)
class PredictSessionCommand:
    raw: RawSessionData


@dataclass(frozen=True)
class PredictSessionResult:
    metrics: SessionMetrics
    prediction: CognitiveLevel
    recommendation: DifficultyRecommendation
    context: PatientContext


class PredictSessionHandler:
    def __init__(
        self,
        classifier: OnnxClassifier,
        personalization: PersonalizationEngine,
        repository: SessionRepository,
        history_handler: GetPatientHistoryHandler,
    ) -> None:
        self._classifier = classifier
        self._personalization = personalization
        self._repository = repository
        self._history_handler = history_handler

    async def handle(self, command: PredictSessionCommand) -> PredictSessionResult:
        raw = command.raw

        history = await self._history_handler.handle(
            GetPatientHistoryQuery(patient_id=raw.patient_id, limit=HISTORY_WINDOW)
        )

        metrics = SessionMetrics.from_raw(raw)
        context = PatientContext.from_history(history, current_sps=metrics.sps)

        prediction = self._classifier.classify(metrics.as_feature_vector())
        recommendation = self._personalization.recommend(
            current_sps=metrics.sps,
            context=context,
        )

        await self._repository.insert_session(
            raw=raw,
            metrics=metrics,
            prediction=prediction,
            recommendation=recommendation,
            context=context,
        )

        return PredictSessionResult(
            metrics=metrics,
            prediction=prediction,
            recommendation=recommendation,
            context=context,
        )
