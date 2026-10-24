"""Tests for the API application."""

from fastapi.testclient import TestClient

from ml_observability.app import app

client = TestClient(app)


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_schema_is_available() -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"]["title"] == "ML Observability"


def test_validate_endpoint_accepts_valid_datasets() -> None:
    response = client.post(
        "/validate",
        json={
            "reference": [{"age": 25, "score": 0.4}],
            "current": [{"age": 26, "score": 0.5}],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "is_valid": True,
        "missing_columns": [],
        "unexpected_columns": [],
        "missing_values": {},
    }


def test_validate_endpoint_reports_schema_and_null_findings() -> None:
    response = client.post(
        "/validate",
        json={
            "reference": [{"age": 25, "score": 0.4}],
            "current": [{"age": None, "region": "north"}],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "is_valid": False,
        "missing_columns": ["score"],
        "unexpected_columns": ["region"],
        "missing_values": {"age": 1},
    }


def test_validate_endpoint_rejects_inconsistent_record_columns() -> None:
    response = client.post(
        "/validate",
        json={
            "reference": [{"age": 25}, {"score": 0.4}],
            "current": [{"age": 26}],
        },
    )

    assert response.status_code == 422


def test_drift_endpoint_detects_distribution_shift() -> None:
    reference = [{"value": float(index)} for index in range(100)]
    current = [{"value": float(index + 1000)} for index in range(100)]

    response = client.post(
        "/drift",
        json={"reference": reference, "current": current},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["multiple_testing_correction"] == "benjamini_hochberg"
    assert len(body["features"]) == 1
    assert body["features"][0]["feature"] == "value"
    assert body["features"][0]["status"] == "evaluated"
    assert body["features"][0]["drift_detected"] is True


def test_drift_endpoint_marks_small_samples_not_evaluated() -> None:
    response = client.post(
        "/drift",
        json={
            "reference": [{"value": 1.0}, {"value": 2.0}],
            "current": [{"value": 100.0}, {"value": 200.0}],
        },
    )

    assert response.status_code == 200
    feature = response.json()["features"][0]
    assert feature["status"] == "not_evaluated"
    assert feature["drift_detected"] is None


def test_drift_endpoint_rejects_invalid_alpha() -> None:
    response = client.post(
        "/drift",
        json={
            "reference": [{"value": 1.0}],
            "current": [{"value": 2.0}],
            "alpha": 1.0,
        },
    )

    assert response.status_code == 422
