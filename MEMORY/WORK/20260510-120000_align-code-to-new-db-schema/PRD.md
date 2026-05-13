---
task: align codebase to updated PostgreSQL schema
slug: 20260510-120000_align-code-to-new-db-schema
effort: standard
phase: complete
progress: 8/8
mode: interactive
started: 2026-05-10T12:00:00Z
updated: 2026-05-10T12:05:00Z
---

## Context

Schema replaced custom PostgreSQL enum types (telemetry.vr_level, narrative_variation, vr_difficulty, cognitive_level, difficulty_recommendation) with plain column types (SMALLINT, CHAR(1), VARCHAR). Also, telemetry.sessions is now a partitioned table with FKs into clinical schema. DDL startup removed — schema managed externally.

## Criteria

- [x] ISC-1: `RawSessionData.level` typed as `int` in domain model
- [x] ISC-2: `SessionInput.level` typed as `int` in HTTP schema
- [x] ISC-3: INSERT drops `::telemetry.vr_level`, `::narrative_variation`, `::vr_difficulty` casts
- [x] ISC-4: INSERT drops `::telemetry.cognitive_level`, `::difficulty_recommendation` casts
- [x] ISC-5: `DDL` variable removed from `postgres_pool.py`
- [x] ISC-6: `create_pool` DDL execute block removed; pool still created and returned
- [x] ISC-7: `REQUEST.md` sample payload shows `"level": 1` (integer)
- [x] ISC-8: No source file references old enum type names (only PRD context text)

## Decisions

- DDL removed from startup: partitioned table + clinical FK dependencies make startup DDL untenable; schema owned by external migration script.
