"""Clinical personalization engine: combines SVM classification with patient longitudinal context.

Applies interpretable rules to produce a personalized difficulty recommendation. Designed to
respect patient trajectory so the system does not push difficulty up on a declining patient
nor keep difficulty flat on a patient showing sustained improvement.
"""

from domain.difficulty_recommendation import DifficultyRecommendation
from domain.patient_context import PatientContext, TrendType

LOW_SPS_THRESHOLD = 0.4
HIGH_SPS_THRESHOLD = 0.7
DELTA_SIGNIFICANT = -0.15
SPS_IMPROVEMENT_FLOOR = 0.6


class PersonalizationEngine:
    def recommend(
        self,
        current_sps: float,
        context: PatientContext,
    ) -> DifficultyRecommendation:
        base = self._base_recommendation(current_sps)

        if context.cold_start:
            return base

        if context.delta_sps < DELTA_SIGNIFICANT:
            return DifficultyRecommendation.DECREASE

        if (
            context.trend == TrendType.DECLINING
            and base == DifficultyRecommendation.INCREASE
        ):
            return DifficultyRecommendation.MAINTAIN

        if (
            context.trend == TrendType.IMPROVING
            and base == DifficultyRecommendation.MAINTAIN
            and current_sps > SPS_IMPROVEMENT_FLOOR
        ):
            return DifficultyRecommendation.INCREASE

        return base

    @staticmethod
    def _base_recommendation(sps: float) -> DifficultyRecommendation:
        if sps < LOW_SPS_THRESHOLD:
            return DifficultyRecommendation.DECREASE
        if sps > HIGH_SPS_THRESHOLD:
            return DifficultyRecommendation.INCREASE
        return DifficultyRecommendation.MAINTAIN
