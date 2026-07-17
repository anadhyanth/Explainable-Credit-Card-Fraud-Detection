"""
Production FastAPI service for the credit card fraud detection model.

Endpoints:
  GET  /health        -> liveness/readiness probe
  POST /predict        -> single transaction prediction + SHAP explanation
  POST /predict/batch   -> batch prediction

Run locally:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

Run in production (see Dockerfile):
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
from src.explain import explain_transaction, load_or_build_explainer
from src.utils import get_logger, load_artifact
from api.schemas import (
    BatchPredictionResponse,
    BatchTransactionRequest,
    HealthResponse,
    PredictionResponse,
    TransactionRequest,
)

logger = get_logger(__name__)

# Holds model/explainer instances loaded once at startup and reused per request.
ml_state = {"model": None, "explainer": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model and SHAP explainer...")
    try:
        ml_state["model"] = load_artifact(config.MODEL_PATH)
        ml_state["explainer"] = load_or_build_explainer()
        logger.info("Model and explainer loaded successfully.")
    except FileNotFoundError as e:
        logger.error("Startup failed: %s", e)
        ml_state["model"] = None
        ml_state["explainer"] = None
    yield
    logger.info("Shutting down fraud detection API.")


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Explainable fraud detection using XGBoost and SHAP.",
    version="1.0.0",
    lifespan=lifespan,
)

# Restrict allow_origins to known frontends in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_model_loaded():
    if ml_state["model"] is None or ml_state["explainer"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded. Train the model before serving.",
        )


def _predict_single(transaction: TransactionRequest) -> PredictionResponse:
    row = pd.DataFrame([transaction.to_ordered_dict()])[config.FEATURE_COLUMNS]

    proba = float(ml_state["model"].predict_proba(row)[:, 1][0])
    is_fraud = proba >= config.DECISION_THRESHOLD

    contributions = explain_transaction(ml_state["explainer"], row, top_n=5)

    return PredictionResponse(
        fraud_probability=round(proba, 5),
        is_fraud=is_fraud,
        decision_threshold=config.DECISION_THRESHOLD,
        top_contributing_features=contributions,
        model_version=config.MODEL_PATH.stem,
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    return HealthResponse(
        status="ok" if ml_state["model"] is not None else "degraded",
        model_loaded=ml_state["model"] is not None,
        explainer_loaded=ml_state["explainer"] is not None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(transaction: TransactionRequest):
    _require_model_loaded()
    try:
        return _predict_single(transaction)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch(payload: BatchTransactionRequest):
    _require_model_loaded()
    try:
        results = [_predict_single(t) for t in payload.transactions]
        return BatchPredictionResponse(results=results)
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host=config.API_HOST, port=config.API_PORT, reload=False)
