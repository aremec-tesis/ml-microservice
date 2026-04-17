"""Domain value objects for raw session data, computed cognitive metrics, and classification labels."""

from dataclasses import dataclass
from enum import Enum


class CognitiveLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class RawSessionData:
    patient_id: int
    total_objects: int
    correct_objects: int
    total_events: int
    correct_events: int
    comprehension_score: int
    response_times: tuple[float, ...]
    total_questions: int
    incorrect_answers: int
    interaction_events: int
    expected_interactions: int


@dataclass(frozen=True)
class SessionMetrics:
    ors: float
    ers: float
    scs: float
    rta: float
    ats: float
    er: float
    sps: float

    @classmethod
    def from_raw(cls, raw: RawSessionData) -> "SessionMetrics":
        ors = raw.correct_objects / raw.total_objects if raw.total_objects else 0.0
        ers = raw.correct_events / raw.total_events if raw.total_events else 0.0
        scs = raw.comprehension_score / 2.0
        rta = (
            sum(raw.response_times) / len(raw.response_times)
            if raw.response_times
            else 0.0
        )
        ats = (
            raw.interaction_events / raw.expected_interactions
            if raw.expected_interactions
            else 0.0
        )
        er = raw.incorrect_answers / raw.total_questions if raw.total_questions else 0.0
        sps = 0.3 * ors + 0.3 * ers + 0.2 * scs + 0.2 * (1 - er)

        return cls(ors=ors, ers=ers, scs=scs, rta=rta, ats=ats, er=er, sps=sps)

    def as_feature_vector(self) -> list[float]:
        return [self.ors, self.ers, self.scs, self.rta, self.ats, self.er, self.sps]
