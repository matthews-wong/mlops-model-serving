"""API tests using FastAPI's TestClient.

The suite trains a fresh model into a temporary location before importing the
app, so tests are self-contained and do not depend on a pre-existing artifact.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# A known setosa sample (small petals) — the canonical easy case for Iris.
KNOWN_SETOSA = [5.1, 3.5, 1.4, 0.2]
VALID_CLASSES = {"setosa", "versicolor", "virginica"}


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module) -> TestClient:
    """Train a model into a temp path and return a TestClient for the app."""
    model_file = tmp_path_factory.mktemp("model") / "model.joblib"
    monkeypatch_module.setenv("MODEL_PATH", str(model_file))

    # Train directly into the temp artifact location.
    import train

    monkeypatch_module.setattr(train, "MODEL_PATH", model_file)
    monkeypatch_module.setattr(
        train, "MODEL_CARD_PATH", model_file.parent / "model_card.md"
    )
    train.main()

    # model.py reads MODEL_PATH lazily on load; clear the cache so the freshly
    # trained temp artifact is picked up instead of any earlier default path.
    from app import model as model_module

    model_module.load_model.cache_clear()

    import app.main as main_module

    return TestClient(main_module.app)


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch (the built-in fixture is function-scoped)."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_health_ok(client: TestClient) -> None:
    """/health returns 200 and a status of ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_ok(client: TestClient) -> None:
    """/ready returns 200 once the model artifact is loaded."""
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_model_metadata(client: TestClient) -> None:
    """/model reports the served model's version, classes, and provenance."""
    response = client.get("/model")
    assert response.status_code == 200

    body = response.json()
    assert body["version"]  # non-empty serving version
    assert body["n_features"] == 4
    assert body["classes"] == ["setosa", "versicolor", "virginica"]
    # The training pipeline is StandardScaler + LogisticRegression.
    assert "LogisticRegression" in body["algorithm"]
    # trained_at is derived from the freshly trained artifact's mtime.
    assert body["trained_at"] is not None
    assert body["artifact_path"].endswith("model.joblib")


def test_predict_known_input(client: TestClient) -> None:
    """/predict returns a valid class + probabilities for a known sample."""
    response = client.post("/predict", json={"features": KNOWN_SETOSA})
    assert response.status_code == 200

    body = response.json()
    assert body["class_name"] in VALID_CLASSES
    assert 0 <= body["class_id"] <= 2
    assert len(body["probabilities"]) == 3
    assert body["probabilities"][body["class_id"]] == max(body["probabilities"])
    # A textbook setosa sample should be classified as setosa.
    assert body["class_name"] == "setosa"


def test_predict_wrong_feature_count(client: TestClient) -> None:
    """/predict rejects payloads without exactly four features."""
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code == 422


def test_metrics_exposition(client: TestClient) -> None:
    """/metrics returns Prometheus text exposition with our custom metric."""
    client.post("/predict", json={"features": KNOWN_SETOSA})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "model_predictions_total" in response.text
