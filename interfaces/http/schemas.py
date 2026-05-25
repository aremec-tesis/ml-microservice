"""Pydantic DTOs for the HTTP layer: input validation and response shaping."""

from typing import Literal

from pydantic import BaseModel, Field

from domain.patient_context import PatientContext
from domain.session_metrics import SessionMetrics


class SessionMetricsIn(BaseModel):
    ors: float
    ers: float
    scs: float = Field(ge=0.0, le=1.0)
    rta: float = Field(ge=0.0)
    er: float = Field(ge=0.0, le=1.0)
    sps: float

    def to_domain(self) -> SessionMetrics:
        return SessionMetrics(**self.model_dump())


class PatientHistoryIn(BaseModel):
    baseline_sps: float
    slope_sps: float
    delta_sps: float
    mean_ors: float
    mean_ers: float
    mean_er: float
    mean_rta: float
    std_sps: float = Field(ge=0.0)
    session_count: int = Field(ge=0)

    def to_domain(self) -> PatientContext:
        return PatientContext(**self.model_dump())


class SessionInput(BaseModel):
    patient_id: str
    session_metrics: SessionMetricsIn
    patient_history: PatientHistoryIn


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
    patient_id: str
    metrics: SessionMetricsOut
    cognitive_level: Literal["low", "medium", "high"]
    recommendation: Literal[
        "decrease_difficulty", "maintain_difficulty", "increase_difficulty"
    ]
    probabilities: ProbabilitiesOut
    context: PatientContextOut
