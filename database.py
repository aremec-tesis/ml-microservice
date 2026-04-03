"""Async PostgreSQL connection pool and queries for schema_telemetria."""

import os
import socket
from datetime import datetime, timezone

import asyncpg

pool: asyncpg.Pool | None = None

DDL = """
CREATE SCHEMA IF NOT EXISTS schema_telemetria;

CREATE TABLE IF NOT EXISTS schema_telemetria.metricas_sesion (
    id BIGSERIAL,
    patient_id INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    total_objects INTEGER NOT NULL,
    correct_objects INTEGER NOT NULL,
    total_events INTEGER NOT NULL,
    correct_events INTEGER NOT NULL,
    comprehension_score INTEGER NOT NULL,
    response_times REAL[] NOT NULL,
    total_questions INTEGER NOT NULL,
    incorrect_answers INTEGER NOT NULL,
    interaction_events INTEGER NOT NULL,
    expected_interactions INTEGER NOT NULL,

    ors NUMERIC(5,4) NOT NULL,
    ers NUMERIC(5,4) NOT NULL,
    scs NUMERIC(5,4) NOT NULL,
    rta NUMERIC(8,4) NOT NULL,
    ats NUMERIC(5,4) NOT NULL,
    er  NUMERIC(5,4) NOT NULL,
    sps NUMERIC(5,4) NOT NULL,

    prediction VARCHAR(10) NOT NULL,
    recommendation VARCHAR(25) NOT NULL,

    PRIMARY KEY (id, created_at)
);
"""


async def init_pool() -> None:
    global pool
    host = os.getenv('DATABASE_HOST', '')
    port = int(os.getenv('DATABASE_PORT', '5432'))

    # Python 3.14 on Windows has a bug where loop.create_connection fails to
    # resolve hostnames via DNS. Pre-resolve to an IP to work around it.
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        resolved_host = infos[0][4][0]
    except socket.gaierror:
        resolved_host = host

    pool = await asyncpg.create_pool(
        host=resolved_host,
        port=port,
        user=os.getenv('DATABASE_USER'),
        password=os.getenv('DATABASE_PASSWORD'),
        database=os.getenv('DATABASE_NAME'),
        min_size=2,
        max_size=10,
        ssl='require',
    )
    async with pool.acquire() as conn:
        await conn.execute(DDL)


async def close_pool() -> None:
    global pool
    if pool is not None:
        await pool.close()
        pool = None


async def get_patient_history(patient_id: int) -> list[asyncpg.Record]:
    if pool is None:
        raise RuntimeError("Database pool not initialized.")
    async with pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT * FROM schema_telemetria.metricas_sesion
            WHERE patient_id = $1
            ORDER BY created_at DESC
            LIMIT 10
            """,
            patient_id,
        )


async def insert_session(
    patient_id: int,
    total_objects: int,
    correct_objects: int,
    total_events: int,
    correct_events: int,
    comprehension_score: int,
    response_times: list[float],
    total_questions: int,
    incorrect_answers: int,
    interaction_events: int,
    expected_interactions: int,
    ors: float,
    ers: float,
    scs: float,
    rta: float,
    ats: float,
    er: float,
    sps: float,
    prediction: str,
    recommendation: str,
) -> None:
    if pool is None:
        raise RuntimeError("Database pool not initialized.")
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO schema_telemetria.metricas_sesion (
                patient_id, created_at,
                total_objects, correct_objects, total_events, correct_events,
                comprehension_score, response_times, total_questions,
                incorrect_answers, interaction_events, expected_interactions,
                ors, ers, scs, rta, ats, er, sps,
                prediction, recommendation
            ) VALUES (
                $1, $2,
                $3, $4, $5, $6,
                $7, $8, $9,
                $10, $11, $12,
                $13, $14, $15, $16, $17, $18, $19,
                $20, $21
            )
            """,
            patient_id,
            datetime.now(timezone.utc),
            total_objects,
            correct_objects,
            total_events,
            correct_events,
            comprehension_score,
            response_times,
            total_questions,
            incorrect_answers,
            interaction_events,
            expected_interactions,
            ors,
            ers,
            scs,
            rta,
            ats,
            er,
            sps,
            prediction,
            recommendation,
        )
