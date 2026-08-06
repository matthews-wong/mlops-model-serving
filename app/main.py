"""FastAPI application exposing the Iris classifier.

Endpoints:
    GET  /health   -> liveness: process is up and serving.
    GET  /ready    -> readiness: model artifact is loaded and usable.
    GET  /model    -> model metadata: version, algorithm, classes, trained-at.
    POST /predict   -> classify a single sample -> class + probabilities.
    POST /predict/batch -> classify many samples in one vectorised call.
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
BATCH_SIZE = Histogram(
    "model_batch_size",
    "Number of samples per /predict/batch request.",
    buckets=(1, 2, 5, 10, 25, 50, 100),
)

# Cap batch requests so a single call cannot exhaust memory or block the
# event loop for an unbounded amount of time.
MAX_BATCH_SIZE = 100


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


class BatchPredictRequest(BaseModel):
    """Input payload for /predict/batch."""

    samples: list[list[float]] = Field(
        ...,
        description=(
            "A non-empty list of Iris samples, each being four measurements "
            "in centimetres: [sepal length, sepal width, petal length, "
            "petal width]."
        ),
        json_schema_extra={"example": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]},
    )


class BatchPredictResponse(BaseModel):
    """Output payload for /predict/batch."""

    predictions: list[PredictResponse] = Field(
        ..., description="One prediction per input sample, in request order."
    )


class HealthResponse(BaseModel):
    """Output payload for /health and /ready."""

    status: str


class ModelInfoResponse(BaseModel):
    """Output payload for /model."""

    version: str = Field(..., description="Serving application version.")
    algorithm: str = Field(
        ..., description="Description of the served estimator / pipeline."
    )
    n_features: int = Field(
        ..., description="Number of input features expected per sample."
    )
    classes: list[str] = Field(
        ..., description="Ordered class names aligned with model class indices."
    )
    trained_at: str | None = Field(
        None,
        description="UTC ISO-8601 timestamp the model artifact was written, if known.",
    )
    artifact_path: str = Field(
        ..., description="Filesystem path the model artifact was loaded from."
    )


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


@app.get("/model", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    """Return metadata about the model currently being served.

    Useful for provenance and debugging: which artifact, which algorithm, what
    classes it predicts, and when it was trained. Returns 503 (mirroring
    /ready) when no artifact is loadable.
    """
    try:
        meta = model_module.metadata()
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ModelInfoResponse(
        version=__version__,
        algorithm=meta.algorithm,
        n_features=meta.n_features,
        classes=meta.classes,
        trained_at=meta.trained_at,
        artifact_path=meta.artifact_path,
    )


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


@app.post(
    "/predict/batch", response_model=BatchPredictResponse, tags=["inference"]
)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Classify many Iris samples in a single vectorised inference call."""
    n_samples = len(request.samples)
    if n_samples == 0:
        PREDICTION_ERRORS.inc()
        raise HTTPException(
            status_code=422, detail="Expected at least one sample, got none."
        )
    if n_samples > MAX_BATCH_SIZE:
        PREDICTION_ERRORS.inc()
        raise HTTPException(
            status_code=422,
            detail=f"Batch too large: {n_samples} > {MAX_BATCH_SIZE} samples.",
        )
    for index, features in enumerate(request.samples):
        if len(features) != N_FEATURES:
            PREDICTION_ERRORS.inc()
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Sample {index}: expected {N_FEATURES} features, "
                    f"got {len(features)}."
                ),
            )

    try:
        start = time.perf_counter()
        results = model_module.predict_batch(request.samples)
        PREDICTION_LATENCY.observe(time.perf_counter() - start)
    except ModelNotLoadedError as exc:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    BATCH_SIZE.observe(n_samples)
    for result in results:
        PREDICTION_COUNTER.labels(predicted_class=result.class_name).inc()

    return BatchPredictResponse(
        predictions=[
            PredictResponse(
                class_id=result.class_id,
                class_name=result.class_name,
                probabilities=result.probabilities,
            )
            for result in results
        ]
    )


@app.get("/metrics", tags=["ops"])
def metrics() -> Response:
    """Expose metrics in Prometheus text exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
