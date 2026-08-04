"""Model loading and inference.

Wraps the trained scikit-learn pipeline so the API layer never touches
joblib or numpy directly. Loading is lazy and cached: the artifact is read
from disk once on first use and reused for the process lifetime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

# The Iris dataset has four numeric features per sample.
N_FEATURES = 4

# Class index -> human-readable species name. Order matches the target
# encoding produced by sklearn.datasets.load_iris.
CLASS_NAMES = ("setosa", "versicolor", "virginica")

# Location of the serialized model. Overridable so containers/K8s can mount
# the artifact at a different path via the MODEL_PATH environment variable.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"


def model_path() -> Path:
    """Return the configured model artifact path."""
    return Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))


@dataclass(frozen=True)
class Prediction:
    """Result of a single classification.

    Attributes:
        class_id: Zero-based class index predicted by the model.
        class_name: Human-readable species name for ``class_id``.
        probabilities: Per-class probabilities, aligned with ``CLASS_NAMES``.
    """

    class_id: int
    class_name: str
    probabilities: list[float]


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is attempted before the model is available."""


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the trained model from disk.

    Returns:
        The deserialized scikit-learn estimator/pipeline.

    Raises:
        ModelNotLoadedError: If the artifact file does not exist.
    """
    path = model_path()
    if not path.exists():
        raise ModelNotLoadedError(
            f"Model artifact not found at '{path}'. Run `python train.py` first."
        )
    return joblib.load(path)


def is_ready() -> bool:
    """Return True if the model artifact can be loaded for inference."""
    try:
        load_model()
        return True
    except Exception:  # noqa: BLE001 - readiness must never raise
        return False


def predict(features: list[float]) -> Prediction:
    """Classify a single Iris sample.

    Args:
        features: Exactly ``N_FEATURES`` numeric measurements in the order
            [sepal length, sepal width, petal length, petal width] (cm).

    Returns:
        A :class:`Prediction` with the predicted class and probabilities.

    Raises:
        ValueError: If ``features`` does not have length ``N_FEATURES``.
        ModelNotLoadedError: If the model artifact is unavailable.
    """
    if len(features) != N_FEATURES:
        raise ValueError(
            f"Expected {N_FEATURES} features, received {len(features)}."
        )

    return predict_batch([features])[0]


def predict_batch(samples: list[list[float]]) -> list[Prediction]:
    """Classify several Iris samples in a single vectorised model call.

    Batching amortises the per-call overhead of ``predict``/``predict_proba``
    across the whole request instead of paying it once per sample.

    Args:
        samples: A non-empty list of samples, each with exactly ``N_FEATURES``
            measurements in the order
            [sepal length, sepal width, petal length, petal width] (cm).

    Returns:
        One :class:`Prediction` per input sample, in the same order.

    Raises:
        ValueError: If ``samples`` is empty or any sample does not have
            length ``N_FEATURES``.
        ModelNotLoadedError: If the model artifact is unavailable.
    """
    if not samples:
        raise ValueError("Expected at least one sample, received none.")
    for index, features in enumerate(samples):
        if len(features) != N_FEATURES:
            raise ValueError(
                f"Sample {index}: expected {N_FEATURES} features, "
                f"received {len(features)}."
            )

    model = load_model()
    matrix = np.asarray(samples, dtype=float).reshape(len(samples), N_FEATURES)

    class_ids = model.predict(matrix).astype(int).tolist()
    probabilities = model.predict_proba(matrix).astype(float).tolist()

    return [
        Prediction(
            class_id=class_id,
            class_name=CLASS_NAMES[class_id],
            probabilities=proba,
        )
        for class_id, proba in zip(class_ids, probabilities, strict=True)
    ]
