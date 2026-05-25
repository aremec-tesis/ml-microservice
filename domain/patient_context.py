"""Patient longitudinal context: aggregated history features the stateful ML consumes.

The Central API is responsible for aggregating the patient's historical sessions and
sending the 9 derived features ready to use. This module only models the value object
and assembles the 15-feature vector.
"""

from dataclasses import dataclass

from domain.session_metrics import SessionMetrics


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

    @property
    def cold_start(self) -> bool:
        return self.session_count == 0


def feature_vector(metrics: SessionMetrics, context: PatientContext) -> list[float]:
    """The 15 features the stateful ML consumes, in canonical order."""
    return [
        metrics.ors,
        metrics.ers,
        metrics.scs,
        metrics.rta,
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
