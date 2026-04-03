"""FastAPI microservice for cognitive performance prediction via ONNX inference."""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import database
import model_handler
from metrics import (
    PredictionResponse,
    SessionInput,
    calculate_metrics,
    get_recommendation,
    metrics_to_feature_vector,
)

load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_handler.load_model()
    logger.info("ONNX model and scaler loaded.")

    await database.init_pool()
    logger.info("Database pool initialized.")

    yield

    await database.close_pool()
    logger.info("Database pool closed.")


app = FastAPI(
    title="Cognitive Performance API",
    description=(
        "Microservice that predicts cognitive performance level "
        "for patients in a VR cognitive stimulation application."
    ),
    version="2.0.0",
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
    if model_handler.ort_session is None:
        raise HTTPException(status_code=503, detail="Model not available.")

    try:
        metrics = calculate_metrics(data)
        features = metrics_to_feature_vector(metrics)
        prediction = model_handler.predict(features)
        recommendation = get_recommendation(metrics.sps)

        await database.insert_session(
            patient_id=data.patient_id,
            total_objects=data.total_objects,
            correct_objects=data.correct_objects,
            total_events=data.total_events,
            correct_events=data.correct_events,
            comprehension_score=data.comprehension_score,
            response_times=data.response_times,
            total_questions=data.total_questions,
            incorrect_answers=data.incorrect_answers,
            interaction_events=data.interaction_events,
            expected_interactions=data.expected_interactions,
            ors=metrics.ors,
            ers=metrics.ers,
            scs=metrics.scs,
            rta=metrics.rta,
            ats=metrics.ats,
            er=metrics.er,
            sps=metrics.sps,
            prediction=prediction,
            recommendation=recommendation,
        )

        return PredictionResponse(
            metrics=metrics,
            prediction=prediction,
            recommendation=recommendation,
        )

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000)
