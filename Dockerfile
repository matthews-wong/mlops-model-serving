# syntax=docker/dockerfile:1

# --- Build stage: install dependencies into a wheel cache ------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt


# --- Runtime stage ---------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/model.joblib \
    PORT=8000

# Install curl for the container HEALTHCHECK, then clean apt lists.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Create an unprivileged user to run the service.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

# Install dependencies from pre-built wheels.
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Copy source and train the model at build time so the image is self-contained.
COPY app ./app
COPY train.py ./train.py
RUN python train.py

# Drop privileges.
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Liveness check baked into the image; K8s probes still apply independently.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
