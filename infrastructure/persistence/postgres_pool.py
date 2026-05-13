"""Async PostgreSQL connection pool lifecycle. Schema is managed by external migration."""

import socket

import asyncpg

from infrastructure.config import DatabaseSettings


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

    return await asyncpg.create_pool(
        host=resolved_host,
        port=settings.port,
        user=settings.user,
        password=settings.password,
        database=settings.database,
        min_size=2,
        max_size=10,
        ssl="require",
    )


async def close_pool(pool: asyncpg.Pool) -> None:
    await pool.close()
