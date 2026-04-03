from typing import Literal

from pydantic import BaseModel, Field


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


class SessionMetrics(BaseModel):
    ors: float
    ers: float
    scs: float
    rta: float
    ats: float
    er: float
    sps: float


class PredictionResponse(BaseModel):
    metrics: SessionMetrics
    prediction: Literal["low", "medium", "high"]
    recommendation: Literal[
        "increase_difficulty", "maintain_difficulty", "decrease_difficulty"
    ]


def calculate_metrics(data: SessionInput) -> SessionMetrics:
    ors = data.correct_objects / data.total_objects if data.total_objects else 0.0
    ers = data.correct_events / data.total_events if data.total_events else 0.0
    scs = data.comprehension_score / 2.0
    rta = (
        sum(data.response_times) / len(data.response_times)
        if data.response_times
        else 0.0
    )
    ats = (
        data.interaction_events / data.expected_interactions
        if data.expected_interactions
        else 0.0
    )
    er = data.incorrect_answers / data.total_questions if data.total_questions else 0.0
    sps = 0.3 * ors + 0.3 * ers + 0.2 * scs + 0.2 * (1 - er)

    return SessionMetrics(ors=ors, ers=ers, scs=scs, rta=rta, ats=ats, er=er, sps=sps)


def get_recommendation(sps: float) -> str:
    if sps < 0.4:
        return "decrease_difficulty"
    elif sps > 0.7:
        return "increase_difficulty"
    return "maintain_difficulty"


def metrics_to_feature_vector(m: SessionMetrics) -> list[float]:
    return [m.ors, m.ers, m.scs, m.rta, m.ats, m.er, m.sps]
