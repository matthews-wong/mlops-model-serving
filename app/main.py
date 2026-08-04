"""FastAPI application exposing the Iris classifier.

Endpoints:
    GET  /health   -> liveness: process is up and serving.
    GET  /ready    -> readiness: model artifact is loaded and usable.
    POST /predict   -> classify a single sample -> class + probabilities.
    GET  /metrics   -> Prometheus exposition of request/inference metrics.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from app import __version__
from app import model as model_module
from app.model import N_FEATURES, ModelNotLoadedError

app = FastAPI(
    title="mlops-model-serving",
    description="Production-shaped REST API serving an Iris classifier.",
    version=__version__,
)

# --- Prometheus metrics -----------------------------------------------------
PREDICTION_COUNTER = Counter(
    "model_predictions_total",
    "Total number of prediction requests, labelled by predicted class.",
    labelnames=("predicted_class",),
)
PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total",
    "Total number of prediction requests that failed validation or inference.",
)
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Latency of the inference call for /predict.",
)


# --- Request / response schemas --------------------------------------------
class PredictRequest(BaseModel):
    """Input payload for /predict."""

    features: list[float] = Field(
        ...,
        description=(
            "Four Iris measurements in centimetres: "
            "[sepal length, sepal width, petal length, petal width]."
        ),
        json_schema_extra={"example": [5.1, 3.5, 1.4, 0.2]},
    )


class PredictResponse(BaseModel):
    """Output payload for /predict."""

    class_id: int = Field(..., description="Zero-based predicted class index.")
    class_name: str = Field(..., description="Predicted Iris species name.")
    probabilities: list[float] = Field(
        ..., description="Per-class probabilities aligned with model classes."
    )


class HealthResponse(BaseModel):
    """Output payload for /health and /ready."""

    status: str


# --- Endpoints --------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe: returns 200 as long as the process is serving."""
    return HealthResponse(status="ok")


@app.get("/ready", response_model=HealthResponse, tags=["ops"])
def ready() -> HealthResponse:
    """Readiness probe: 200 only when the model artifact is loadable."""
    if not model_module.is_ready():
        raise HTTPException(status_code=503, detail="model not loaded")
    return HealthResponse(status="ready")


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest) -> PredictResponse:
    """Classify a single Iris sample and return class + probabilities."""
    if len(request.features) != N_FEATURES:
        PREDICTION_ERRORS.inc()
        raise HTTPException(
            status_code=422,
            detail=f"Expected {N_FEATURES} features, got {len(request.features)}.",
        )

    try:
        start = time.perf_counter()
        result = model_module.predict(request.features)
        PREDICTION_LATENCY.observe(time.perf_counter() - start)
    except ModelNotLoadedError as exc:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    PREDICTION_COUNTER.labels(predicted_class=result.class_name).inc()
    return PredictResponse(
        class_id=result.class_id,
        class_name=result.class_name,
        probabilities=result.probabilities,
    )


@app.get("/metrics", tags=["ops"])
def metrics() -> Response:
    """Expose metrics in Prometheus text exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
