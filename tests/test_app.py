"""Tests for the API application."""

import pytest
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


def test_validate_endpoint_rejects_more_than_100_columns() -> None:
    record = {f"column_{index}": index for index in range(101)}

    response = client.post("/validate", json={"reference": [record], "current": [record]})

    assert response.status_code == 422
    assert "at most 100 columns" in response.text


def test_validate_endpoint_accepts_100_columns() -> None:
    record = {f"column_{index}": index for index in range(100)}

    response = client.post("/validate", json={"reference": [record], "current": [record]})

    assert response.status_code == 200


def test_validate_endpoint_rejects_column_names_over_128_characters() -> None:
    record = {"x" * 129: 1}

    response = client.post("/validate", json={"reference": [record], "current": [record]})

    assert response.status_code == 422
    assert "at most 128 characters" in response.text


def test_validate_endpoint_accepts_column_names_at_128_characters() -> None:
    record = {"x" * 128: 1}

    response = client.post("/validate", json={"reference": [record], "current": [record]})

    assert response.status_code == 200


def test_validate_endpoint_accepts_string_values_at_the_limit() -> None:
    record = {"value": "x" * 10_000}

    response = client.post("/validate", json={"reference": [record], "current": [record]})

    assert response.status_code == 200


@pytest.mark.parametrize(("record_count", "expected_status"), [(10_000, 200), (10_001, 422)])
def test_validate_endpoint_enforces_record_count_limit(
    record_count: int, expected_status: int
) -> None:
    records = [{"value": 1}] * record_count

    response = client.post("/validate", json={"reference": records, "current": records})

    assert response.status_code == expected_status
    if expected_status == 422:
        assert "value" not in response.text


def test_validate_endpoint_rejects_oversized_string_values() -> None:
    record = {"value": "x" * 10_001}

    response = client.post("/validate", json={"reference": [record], "current": [record]})

    assert response.status_code == 422
    assert "at most 10000 characters" in response.text
    assert "x" * 10_001 not in response.text


def test_request_body_limit_returns_413() -> None:
    response = client.post(
        "/validate",
        content=b" " * 2_000_001,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body exceeds the configured size limit."}


def test_request_body_limit_accepts_exact_limit() -> None:
    body = b'{"reference":[{}],"current":[{}]}'
    body += b" " * (2_000_000 - len(body))

    response = client.post(
        "/validate",
        content=body,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200


def test_request_body_limit_rejects_streamed_chunked_payload() -> None:
    chunks = iter((b" " * 1_100_000, b" " * 1_100_000))

    response = client.post(
        "/validate",
        content=chunks,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413


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


def test_drift_endpoint_returns_incident_for_detected_drift() -> None:
    reference = [{"value": float(index)} for index in range(100)]
    current = [{"value": float(index + 1000)} for index in range(100)]

    response = client.post(
        "/drift",
        json={"reference": reference, "current": current},
    )

    assert response.status_code == 200
    body = response.json()
    assert "incidents" in body
    assert len(body["incidents"]) == 1

    incident = body["incidents"][0]
    assert incident["incident_id"].startswith("inc_")
    assert incident["detector"] == "ks_test"
    assert incident["category"] == "feature_drift"
    assert incident["scope"] == "value"
    assert incident["severity"] in {"info", "low", "medium", "high", "critical"}
    assert incident["status"] == "open"
    assert incident["observed_metrics"]["adjusted_p_value"] < body["alpha"]
    assert incident["thresholds"]["alpha"] == body["alpha"]
    assert incident["evidence"]
    assert incident["recommended_steps"]
    assert incident["limitations"]


def test_drift_endpoint_does_not_emit_incident_for_unevaluated_feature() -> None:
    response = client.post(
        "/drift",
        json={
            "reference": [{"value": 1.0}, {"value": 2.0}],
            "current": [{"value": 100.0}, {"value": 200.0}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["features"][0]["status"] == "not_evaluated"
    assert body["incidents"] == []
