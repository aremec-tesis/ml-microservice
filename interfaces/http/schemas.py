"""Pydantic DTOs for the HTTP layer: input validation and response shaping."""

from typing import Literal

from pydantic import BaseModel, Field

from domain.session_metrics import RawSessionData


class SessionInput(BaseModel):
    patient_id: int
    total_objects: int = Field(ge=0)
    correct_objects: int = Field(ge=0)
    total_events: int = Field(ge=0)
    correct_events: int = Field(ge=0)
    comprehension_score: int = Field(ge=0, le=2)
    response_times: list[float]
    total_questions: int = Field(ge=0)
    incorrect_answers: int = Field(ge=0)
    interaction_events: int = Field(ge=0)
    expected_interactions: int = Field(ge=0)

    def to_domain(self) -> RawSessionData:
        return RawSessionData(
            patient_id=self.patient_id,
            total_objects=self.total_objects,
            correct_objects=self.correct_objects,
            total_events=self.total_events,
            correct_events=self.correct_events,
            comprehension_score=self.comprehension_score,
            response_times=tuple(self.response_times),
            total_questions=self.total_questions,
            incorrect_answers=self.incorrect_answers,
            interaction_events=self.interaction_events,
            expected_interactions=self.expected_interactions,
        )


class SessionMetricsOut(BaseModel):
    ors: float
    ers: float
    scs: float
    rta: float
    ats: float
    er: float
    sps: float


class PatientContextOut(BaseModel):
    baseline_sps: float
    trend: Literal["improving", "stable", "declining", "cold_start"]
    delta_sps: float
    session_count: int
    cold_start: bool


class PredictionResponse(BaseModel):
    metrics: SessionMetricsOut
    prediction: Literal["low", "medium", "high"]
    recommendation: Literal[
        "increase_difficulty", "maintain_difficulty", "decrease_difficulty"
    ]
    context: PatientContextOut
