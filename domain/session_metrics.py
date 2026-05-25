"""Domain value objects for cognitive metrics and classification labels.

The microservice receives the 6 cognitive metrics pre-computed by the Central API,
so the raw-to-metrics transformation no longer lives here.
"""

from dataclasses import dataclass
from enum import Enum


class CognitiveLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


LOW_SPS_THRESHOLD = 0.4
HIGH_SPS_THRESHOLD = 0.7


def cognitive_level_from_sps(sps: float) -> CognitiveLevel:
    if sps < LOW_SPS_THRESHOLD:
        return CognitiveLevel.LOW
    if sps > HIGH_SPS_THRESHOLD:
        return CognitiveLevel.HIGH
    return CognitiveLevel.MEDIUM


@dataclass(frozen=True)
class SessionMetrics:
    ors: float
    ers: float
    scs: float
    rta: float
    er: float
    sps: float
