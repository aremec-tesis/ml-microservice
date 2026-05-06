"""Query: fetch recent sessions of a patient to derive longitudinal context for the stateful ML."""

from dataclasses import dataclass

from domain.patient_context import HistoricalSession
from infrastructure.persistence.session_repository import SessionRepository


@dataclass(frozen=True)
class GetPatientHistoryQuery:
    patient_id: str
    limit: int


class GetPatientHistoryHandler:
    def __init__(self, repository: SessionRepository) -> None:
        self._repository = repository

    async def handle(self, query: GetPatientHistoryQuery) -> list[HistoricalSession]:
        return await self._repository.get_patient_history(
            patient_id=query.patient_id,
            limit=query.limit,
        )
