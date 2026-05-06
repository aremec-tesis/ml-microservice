"""Async PostgreSQL/TimescaleDB pool lifecycle and schema DDL."""

import socket

import asyncpg

from infrastructure.config import DatabaseSettings

DDL = """
CREATE SCHEMA IF NOT EXISTS telemetry;

CREATE TABLE IF NOT EXISTS telemetry.sessions (
    -- Identifiers and Clinical Relations
    id BIGSERIAL,
    patient_id UUID NOT NULL REFERENCES clinical.patients(id) ON DELETE RESTRICT,
    user_id UUID NOT NULL REFERENCES clinical.users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    date DATE NOT NULL DEFAULT CURRENT_DATE,

    -- VR Environment Context
    level telemetry.vr_level NOT NULL,
    variation telemetry.narrative_variation NOT NULL,
    difficulty telemetry.vr_difficulty NOT NULL,
    duration_min SMALLINT NOT NULL DEFAULT 0,

    -- Raw Metrics (Sent by Unity)
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

    -- Calculated Features (ML Features)
    ors NUMERIC(6,4) NOT NULL,
    ers NUMERIC(5,4) NOT NULL,
    scs NUMERIC(5,4) NOT NULL,
    rta NUMERIC(8,4) NOT NULL,
    er  NUMERIC(5,4) NOT NULL,
    sps NUMERIC(6,4) NOT NULL,

    -- Historical and Trend Data
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

    -- Inference Results (SVM)
    cognitive_level telemetry.cognitive_level NOT NULL,
    recommendation telemetry.difficulty_recommendation NOT NULL,
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
