
from fastapi.testclient import TestClient

from src.deployment.api import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["project"] == "MOSAIC"
    assert data["status"] == "running"
    assert data["model"] == "HistGradientBoosting"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_valid_prediction():
    # A valid sequence: 100 observations × 144 features
    sequence = [
        [0.1] * 144
        for _ in range(100)
    ]

    response = client.post(
        "/predict",
        json={"sequence": sequence},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["project"] == "MOSAIC"
    assert data["prediction"] in [
        "DOWN",
        "STATIONARY",
        "UP",
    ]

    assert data["class_id"] in [0, 1, 2]

    probabilities = data["probabilities"]

    assert set(probabilities.keys()) == {
        "DOWN",
        "STATIONARY",
        "UP",
    }

    assert abs(sum(probabilities.values()) - 1.0) < 1e-6


def test_invalid_sequence_length():
    # Only 99 observations instead of 100
    sequence = [
        [0.1] * 144
        for _ in range(99)
    ]

    response = client.post(
        "/predict",
        json={"sequence": sequence},
    )

    assert response.status_code == 422


def test_invalid_feature_count():
    # 143 features instead of 144
    sequence = [
        [0.1] * 143
        for _ in range(100)
    ]

    response = client.post(
        "/predict",
        json={"sequence": sequence},
    )

    assert response.status_code == 422
