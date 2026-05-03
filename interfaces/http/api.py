"""FastAPI application wiring: lifespan bootstraps dependencies and exposes /predict."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.commands.predict_session import (
    PredictSessionCommand,
    PredictSessionHandler,
)
from app.queries.get_patient_history import GetPatientHistoryHandler
from infrastructure.config import MODEL_PATH, SCALER_PATH, DatabaseSettings
from infrastructure.ml.onnx_classifier import OnnxClassifier
from infrastructure.persistence.postgres_pool import close_pool, create_pool
from infrastructure.persistence.session_repository import SessionRepository
from interfaces.http.schemas import (
    PatientContextOut,
    PredictionResponse,
    ProbabilitiesOut,
    SessionInput,
    SessionMetricsOut,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier = OnnxClassifier(MODEL_PATH, SCALER_PATH)
    logger.info("ONNX model and scaler loaded.")

    pool = await create_pool(DatabaseSettings.from_env())
    logger.info("Database pool initialized.")

    repository = SessionRepository(pool)
    history_handler = GetPatientHistoryHandler(repository)
    predict_handler = PredictSessionHandler(
        classifier=classifier,
        repository=repository,
        history_handler=history_handler,
    )

    app.state.pool = pool
    app.state.predict_handler = predict_handler

    yield

    await close_pool(pool)
    logger.info("Database pool closed.")


app = FastAPI(
    title="Cognitive Performance API",
    description=(
        "Microservice that recommends a personalized difficulty adjustment for patients in a "
        "VR cognitive stimulation application using a stateful ML model that consumes both the "
        "current-session metrics and the patient's longitudinal session history."
    ),
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: SessionInput):
    handler: PredictSessionHandler = app.state.predict_handler
    try:
        result = await handler.handle(PredictSessionCommand(raw=data.to_domain()))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return PredictionResponse(
        metrics=SessionMetricsOut(**result.metrics.__dict__),
        cognitive_level=result.cognitive_level.value,
        recommendation=result.classification.recommendation.value,
        probabilities=ProbabilitiesOut(
            decrease_difficulty=result.classification.prob_decrease,
            maintain_difficulty=result.classification.prob_maintain,
            increase_difficulty=result.classification.prob_increase,
        ),
        context=PatientContextOut(
            baseline_sps=result.context.baseline_sps,
            slope_sps=result.context.slope_sps,
            delta_sps=result.context.delta_sps,
            mean_ors=result.context.mean_ors,
            mean_ers=result.context.mean_ers,
            mean_er=result.context.mean_er,
            mean_rta=result.context.mean_rta,
            std_sps=result.context.std_sps,
            session_count=result.context.session_count,
            cold_start=result.context.cold_start,
        ),
    )
