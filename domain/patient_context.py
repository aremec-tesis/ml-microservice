"""Patient longitudinal context: aggregated history features the stateful ML consumes."""

from dataclasses import dataclass

from domain.session_metrics import SessionMetrics


@dataclass(frozen=True)
class HistoricalSession:
    sps: float
    ors: float
    ers: float
    er: float
    rta: float


@dataclass(frozen=True)
class PatientContext:
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

    @classmethod
    def from_history(
        cls,
        history: list[HistoricalSession],
        current: SessionMetrics,
    ) -> "PatientContext":
        if not history:
            return cls(
                baseline_sps=current.sps,
                slope_sps=0.0,
                delta_sps=0.0,
                mean_ors=current.ors,
                mean_ers=current.ers,
                mean_er=current.er,
                mean_rta=current.rta,
                std_sps=0.0,
                session_count=0,
                cold_start=True,
            )

        sps_vals = [h.sps for h in history]
        baseline = _weighted_baseline(sps_vals)
        slope = _linear_slope(sps_vals)
        std_sps = _std(sps_vals)
        delta = current.sps - baseline

        return cls(
            baseline_sps=baseline,
            slope_sps=slope,
            delta_sps=delta,
            mean_ors=_mean([h.ors for h in history]),
            mean_ers=_mean([h.ers for h in history]),
            mean_er=_mean([h.er for h in history]),
            mean_rta=_mean([h.rta for h in history]),
            std_sps=std_sps,
            session_count=len(history),
            cold_start=False,
        )


def feature_vector(metrics: SessionMetrics, context: PatientContext) -> list[float]:
    """The 16 features the stateful ML consumes, in canonical order."""
    return [
        metrics.ors,
        metrics.ers,
        metrics.scs,
        metrics.rta,
        metrics.ats,
        metrics.er,
        metrics.sps,
        context.baseline_sps,
        context.slope_sps,
        context.delta_sps,
        context.mean_ors,
        context.mean_ers,
        context.mean_er,
        context.mean_rta,
        context.std_sps,
        float(context.session_count),
    ]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _weighted_baseline(sps_values: list[float]) -> float:
    # history is most-recent-first; index 0 weights 1.0, older sessions decay
    weights = [1.0 / (i + 1) for i in range(len(sps_values))]
    total_weight = sum(weights)
    weighted_sum = sum(w * v for w, v in zip(weights, sps_values))
    return weighted_sum / total_weight


def _linear_slope(sps_values: list[float]) -> float:
    if len(sps_values) < 2:
        return 0.0
    # reverse so x=0 is oldest, x=n-1 is newest → positive slope means improvement
    ordered = list(reversed(sps_values))
    n = len(ordered)
    mean_x = (n - 1) / 2
    mean_y = sum(ordered) / n
    numerator = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(ordered))
    denominator = sum((i - mean_x) ** 2 for i in range(n))
    return numerator / denominator if denominator else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return var ** 0.5
