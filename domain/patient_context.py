"""Patient longitudinal context derived from session history (baseline, trend, delta)."""

from dataclasses import dataclass
from enum import Enum


class TrendType(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    COLD_START = "cold_start"


TREND_THRESHOLD = 0.02


@dataclass(frozen=True)
class HistoricalSession:
    sps: float


@dataclass(frozen=True)
class PatientContext:
    baseline_sps: float
    trend: TrendType
    delta_sps: float
    session_count: int
    cold_start: bool

    @classmethod
    def from_history(
        cls,
        history: list[HistoricalSession],
        current_sps: float,
    ) -> "PatientContext":
        if not history:
            return cls(
                baseline_sps=current_sps,
                trend=TrendType.COLD_START,
                delta_sps=0.0,
                session_count=0,
                cold_start=True,
            )

        baseline = _weighted_baseline([s.sps for s in history])
        trend = _linear_trend([s.sps for s in history])
        delta = current_sps - baseline

        return cls(
            baseline_sps=baseline,
            trend=trend,
            delta_sps=delta,
            session_count=len(history),
            cold_start=False,
        )


def _weighted_baseline(sps_values: list[float]) -> float:
    # history is most-recent-first; index 0 weights 1.0, older sessions decay
    weights = [1.0 / (i + 1) for i in range(len(sps_values))]
    total_weight = sum(weights)
    weighted_sum = sum(w * v for w, v in zip(weights, sps_values))
    return weighted_sum / total_weight


def _linear_trend(sps_values: list[float]) -> TrendType:
    if len(sps_values) < 2:
        return TrendType.STABLE

    # reverse so x=0 is oldest, x=n-1 is newest → positive slope means improvement
    ordered = list(reversed(sps_values))
    n = len(ordered)
    mean_x = (n - 1) / 2
    mean_y = sum(ordered) / n
    numerator = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(ordered))
    denominator = sum((i - mean_x) ** 2 for i in range(n))

    if denominator == 0:
        return TrendType.STABLE

    slope = numerator / denominator
    if slope > TREND_THRESHOLD:
        return TrendType.IMPROVING
    if slope < -TREND_THRESHOLD:
        return TrendType.DECLINING
    return TrendType.STABLE
