"""Async PostgreSQL/TimescaleDB pool lifecycle and schema DDL."""

import socket

import asyncpg

from infrastructure.config import DatabaseSettings

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

    baseline_sps NUMERIC(5,4) NOT NULL,
    trend VARCHAR(15) NOT NULL,
    delta_sps NUMERIC(6,4) NOT NULL,
    session_count INTEGER NOT NULL,

    PRIMARY KEY (id, created_at)
);
"""


async def create_pool(settings: DatabaseSettings) -> asyncpg.Pool:
    # Python 3.14 on Windows has a bug where loop.create_connection fails to
    # resolve hostnames via DNS. Pre-resolve to an IP to work around it.
    try:
        infos = socket.getaddrinfo(settings.host, settings.port, type=socket.SOCK_STREAM)
        resolved_host = infos[0][4][0]
    except socket.gaierror:
        resolved_host = settings.host

    pool = await asyncpg.create_pool(
        host=resolved_host,
        port=settings.port,
        user=settings.user,
        password=settings.password,
        database=settings.database,
        min_size=2,
        max_size=10,
        ssl="require",
    )
    async with pool.acquire() as conn:
        await conn.execute(DDL)
    return pool


async def close_pool(pool: asyncpg.Pool) -> None:
    await pool.close()
