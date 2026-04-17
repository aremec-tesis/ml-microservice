"""Session repository: reads patient history and persists completed sessions."""

from datetime import datetime, timezone

import asyncpg

from domain.difficulty_recommendation import DifficultyRecommendation
from domain.patient_context import HistoricalSession, PatientContext
from domain.session_metrics import CognitiveLevel, RawSessionData, SessionMetrics


class SessionRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_patient_history(
        self,
        patient_id: int,
        limit: int,
    ) -> list[HistoricalSession]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT sps
                FROM schema_telemetria.metricas_sesion
                WHERE patient_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                patient_id,
                limit,
            )
        return [HistoricalSession(sps=float(row["sps"])) for row in rows]

    async def insert_session(
        self,
        raw: RawSessionData,
        metrics: SessionMetrics,
        prediction: CognitiveLevel,
        recommendation: DifficultyRecommendation,
        context: PatientContext,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO schema_telemetria.metricas_sesion (
                    patient_id, created_at,
                    total_objects, correct_objects, total_events, correct_events,
                    comprehension_score, response_times, total_questions,
                    incorrect_answers, interaction_events, expected_interactions,
                    ors, ers, scs, rta, ats, er, sps,
                    prediction, recommendation,
                    baseline_sps, trend, delta_sps, session_count
                ) VALUES (
                    $1, $2,
                    $3, $4, $5, $6,
                    $7, $8, $9,
                    $10, $11, $12,
                    $13, $14, $15, $16, $17, $18, $19,
                    $20, $21,
                    $22, $23, $24, $25
                )
                """,
                raw.patient_id,
                datetime.now(timezone.utc),
                raw.total_objects,
                raw.correct_objects,
                raw.total_events,
                raw.correct_events,
                raw.comprehension_score,
                list(raw.response_times),
                raw.total_questions,
                raw.incorrect_answers,
                raw.interaction_events,
                raw.expected_interactions,
                metrics.ors,
                metrics.ers,
                metrics.scs,
                metrics.rta,
                metrics.ats,
                metrics.er,
                metrics.sps,
                prediction.value,
                recommendation.value,
                context.baseline_sps,
                context.trend.value,
                context.delta_sps,
                context.session_count,
            )
