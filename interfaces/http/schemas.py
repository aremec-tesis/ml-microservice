"""Pydantic DTOs for the HTTP layer: input validation and response shaping."""

from typing import Literal

from pydantic import BaseModel, Field

from domain.session_metrics import RawSessionData


class SessionInput(BaseModel):
    # Identifiers
    patient_id: str
    user_id: str

    # VR environment context
    level: str
    variation: str
    difficulty: str
    duration_min: int = Field(ge=0)

    # Raw metrics from Unity
    correct_key_objects: int = Field(ge=0)
    correct_secondary_objects: int = Field(ge=0)
    incorrect_objects: int = Field(ge=0)
    total_key_objects: int = Field(ge=0)
    total_secondary_objects: int = Field(ge=0)
    total_events: int = Field(ge=0)
    correct_events: int = Field(ge=0)
    comprehension_score: int = Field(ge=0, le=2)
    response_times: list[float]
    total_questions: int = Field(ge=0)
    incorrect_answers: int = Field(ge=0)

    def to_domain(self) -> RawSessionData:
        return RawSessionData(
            patient_id=self.patient_id,
            user_id=self.user_id,
            level=self.level,
            variation=self.variation,
            difficulty=self.difficulty,
            duration_min=self.duration_min,
            correct_key_objects=self.correct_key_objects,
            correct_secondary_objects=self.correct_secondary_objects,
            incorrect_objects=self.incorrect_objects,
            total_key_objects=self.total_key_objects,
            total_secondary_objects=self.total_secondary_objects,
            total_events=self.total_events,
            correct_events=self.correct_events,
            comprehension_score=self.comprehension_score,
            response_times=tuple(self.response_times),
            total_questions=self.total_questions,
            incorrect_answers=self.incorrect_answers,
        )


class SessionMetricsOut(BaseModel):
    ors: float
    ers: float
    scs: float
    rta: float
    er: float
    sps: float


class PatientContextOut(BaseModel):
    baseline_sps: float
    slope_sps: float
    delta_sps: float
    mean_ors: float
    mean_ers: float
    mean_er: float
    mean_rta: float
    std_sps: float
    session_count: int
    cold_start: bool


class ProbabilitiesOut(BaseModel):
    decrease_difficulty: float
    maintain_difficulty: float
    increase_difficulty: float


class PredictionResponse(BaseModel):
    metrics: SessionMetricsOut
    cognitive_level: Literal["low", "medium", "high"]
    recommendation: Literal[
        "decrease_difficulty", "maintain_difficulty", "increase_difficulty"
    ]
    probabilities: ProbabilitiesOut
    context: PatientContextOut
