"""Session repository: reads patient history (used by the stateful ML) and persists completed sessions."""

import asyncpg

from domain.patient_context import HistoricalSession, PatientContext
from domain.session_metrics import CognitiveLevel, RawSessionData, SessionMetrics
from infrastructure.ml.onnx_classifier import ClassificationResult


class SessionRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_patient_history(
        self,
        patient_id: str,
        limit: int,
    ) -> list[HistoricalSession]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT sps, ors, ers, er, rta
                FROM telemetry.sessions
                WHERE patient_id = $1::uuid
                ORDER BY created_at DESC
                LIMIT $2
                """,
                patient_id,
                limit,
            )
        return [
            HistoricalSession(
                sps=float(row["sps"]),
                ors=float(row["ors"]),
                ers=float(row["ers"]),
                er=float(row["er"]),
                rta=float(row["rta"]),
            )
            for row in rows
        ]

    async def insert_session(
        self,
        raw: RawSessionData,
        metrics: SessionMetrics,
        context: PatientContext,
        cognitive_level: CognitiveLevel,
        classification: ClassificationResult,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO telemetry.sessions (
                    patient_id, user_id,
                    level, variation, difficulty, duration_min,
                    correct_key_objects, correct_secondary_objects, incorrect_objects,
                    total_key_objects, total_secondary_objects,
                    total_events, correct_events,
                    comprehension_score, response_times,
                    total_questions, incorrect_answers,
                    ors, ers, scs, rta, er, sps,
                    baseline_sps, slope_sps, delta_sps,
                    mean_ors, mean_ers, mean_er, mean_rta,
                    std_sps, session_count, cold_start,
                    cognitive_level, recommendation,
                    prob_decrease, prob_maintain, prob_increase
                ) VALUES (
                    $1::uuid, $2::uuid,
                    $3, $4, $5, $6,
                    $7, $8, $9,
                    $10, $11,
                    $12, $13,
                    $14, $15,
                    $16, $17,
                    $18, $19, $20, $21, $22, $23,
                    $24, $25, $26,
                    $27, $28, $29, $30,
                    $31, $32, $33,
                    $34, $35,
                    $36, $37, $38
                )
                """,
                raw.patient_id,
                raw.user_id,
                raw.level,
                raw.variation,
                raw.difficulty,
                raw.duration_min,
                raw.correct_key_objects,
                raw.correct_secondary_objects,
                raw.incorrect_objects,
                raw.total_key_objects,
                raw.total_secondary_objects,
                raw.total_events,
                raw.correct_events,
                raw.comprehension_score,
                list(raw.response_times),
                raw.total_questions,
                raw.incorrect_answers,
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
                context.session_count,
                context.cold_start,
                cognitive_level.value,
                classification.recommendation.value,
                classification.prob_decrease,
                classification.prob_maintain,
                classification.prob_increase,
            )
