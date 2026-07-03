"""FastAPI application wiring: lifespan loads the ONNX model and exposes /predict."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.commands.predict_session import (
    PredictSessionCommand,
    PredictSessionHandler,
)
from infrastructure.config import CENTRAL_API_ORIGIN, MODEL_PATH, SCALER_PATH
from infrastructure.ml.onnx_classifier import OnnxClassifier
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

    app.state.predict_handler = PredictSessionHandler(classifier=classifier)
    yield


app = FastAPI(
    title="Cognitive Performance API",
    description=(
        "Stateless ML microservice that recommends a personalized difficulty adjustment "
        "for VR cognitive stimulation patients. Consumes the 6 pre-computed session metrics "
        "and the 9 aggregated history features supplied by the Central API."
    ),
    version="5.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CENTRAL_API_ORIGIN],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: SessionInput):
    handler: PredictSessionHandler = app.state.predict_handler
    try:
        result = handler.handle(
            PredictSessionCommand(
                metrics=data.session_metrics.to_domain(),
                context=data.patient_history.to_domain(),
            )
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return PredictionResponse(
        patient_id=data.patient_id,
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
