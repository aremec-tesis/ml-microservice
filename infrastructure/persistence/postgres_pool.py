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

    correct_key_objects INTEGER NOT NULL,
    correct_secondary_objects INTEGER NOT NULL,
    incorrect_objects INTEGER NOT NULL,
    total_key_objects INTEGER NOT NULL,
    total_secondary_objects INTEGER NOT NULL,
    total_events INTEGER NOT NULL,
    correct_events INTEGER NOT NULL,
    comprehension_score INTEGER NOT NULL,
    response_times REAL[] NOT NULL,
    total_questions INTEGER NOT NULL,
    incorrect_answers INTEGER NOT NULL,
    interaction_events INTEGER NOT NULL,
    expected_interactions INTEGER NOT NULL,

    ors NUMERIC(6,4) NOT NULL,
    ers NUMERIC(5,4) NOT NULL,
    scs NUMERIC(5,4) NOT NULL,
    rta NUMERIC(8,4) NOT NULL,
    ats NUMERIC(6,4) NOT NULL,
    er  NUMERIC(5,4) NOT NULL,
    sps NUMERIC(6,4) NOT NULL,

    baseline_sps NUMERIC(6,4) NOT NULL,
    slope_sps NUMERIC(7,5) NOT NULL,
    delta_sps NUMERIC(6,4) NOT NULL,
    mean_ors NUMERIC(6,4) NOT NULL,
    mean_ers NUMERIC(5,4) NOT NULL,
    mean_er NUMERIC(5,4) NOT NULL,
    mean_rta NUMERIC(8,4) NOT NULL,
    std_sps NUMERIC(6,4) NOT NULL,
    session_count INTEGER NOT NULL,
    cold_start BOOLEAN NOT NULL,

    cognitive_level VARCHAR(10) NOT NULL,
    recommendation VARCHAR(25) NOT NULL,
    prob_decrease NUMERIC(5,4) NOT NULL,
    prob_maintain NUMERIC(5,4) NOT NULL,
    prob_increase NUMERIC(5,4) NOT NULL,

    PRIMARY KEY (id, created_at)
);
"""


async def create_pool(settings: DatabaseSettings) -> asyncpg.Pool:
    # Pre-resolve to an IPv4 address. Two reasons:
    #   1. Python 3.14 on Windows has a DNS bug in loop.create_connection.
    #   2. Hosts like Render Free are IPv4-only outbound, so we must avoid
    #      AAAA records (which getaddrinfo without family=AF_INET may prefer).
    try:
        infos = socket.getaddrinfo(
            settings.host,
            settings.port,
            family=socket.AF_INET,
            type=socket.SOCK_STREAM,
        )
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
