"""Tests for the incident domain contract."""

from typing import Any

import pytest

from ml_observability.incidents import build_incident, create_incident_id


def make_incident(**overrides: Any):
    arguments: dict[str, Any] = {
        "detector": "ks_test",
        "category": "feature_drift",
        "scope": "dataset_a.feature_x",
        "severity": "medium",
        "observed_metrics": {"p_value": 0.01, "statistic": 0.4},
        "thresholds": {"alpha": 0.05},
        "evidence": ("KS test detected a distribution difference.",),
        "recommended_steps": ("Inspect recent upstream data changes.",),
        "limitations": ("Statistical significance does not establish root cause.",),
    }
    arguments.update(overrides)
    return build_incident(**arguments)


def test_incident_id_is_deterministic() -> None:
    arguments = {
        "detector": "ks_test",
        "category": "feature_drift",
        "scope": "dataset_a.feature_x",
    }
    assert create_incident_id(**arguments) == create_incident_id(**arguments)
    assert create_incident_id(**arguments).startswith("inc_")


def test_incident_id_changes_when_scope_changes() -> None:
    original = create_incident_id(
        detector="ks_test", category="feature_drift", scope="dataset_a.feature_x"
    )
    changed = create_incident_id(
        detector="ks_test", category="feature_drift", scope="dataset_a.feature_y"
    )
    assert original != changed


def test_build_incident_sets_open_status_and_stable_id() -> None:
    first = make_incident()
    second = make_incident()

    assert first.incident_id == second.incident_id
    assert first.status == "open"
    assert first.severity == "medium"
    assert first.evidence
    assert first.recommended_steps
    assert first.limitations


@pytest.mark.parametrize(
    ("field", "value"),
    [("detector", ""), ("category", "  "), ("scope", "")],
)
def test_build_incident_rejects_empty_identity_fields(field: str, value: str) -> None:
    with pytest.raises(ValueError):
        make_incident(**{field: value})


def test_build_incident_rejects_invalid_severity() -> None:
    with pytest.raises(ValueError, match="Invalid severity"):
        make_incident(severity="urgent")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_build_incident_rejects_non_finite_metrics(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        make_incident(observed_metrics={"statistic": value})


def test_build_incident_rejects_non_finite_thresholds() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        make_incident(thresholds={"alpha": float("nan")})


def test_build_incident_metric_mappings_are_read_only() -> None:
    incident = make_incident()

    with pytest.raises(TypeError):
        incident.observed_metrics["new_metric"] = 1.0

    with pytest.raises(TypeError):
        incident.thresholds["alpha"] = 0.1


def test_build_incident_copies_input_mappings() -> None:
    metrics = {"statistic": 0.4}
    thresholds = {"alpha": 0.05}

    incident = make_incident(observed_metrics=metrics, thresholds=thresholds)
    metrics["statistic"] = 99.0
    thresholds["alpha"] = 0.9

    assert incident.observed_metrics["statistic"] == 0.4
    assert incident.thresholds["alpha"] == 0.05


def test_build_incident_requires_evidence() -> None:
    with pytest.raises(ValueError, match="at least one evidence item"):
        make_incident(evidence=())


def test_build_incident_requires_recommended_steps() -> None:
    with pytest.raises(ValueError, match="at least one recommended step"):
        make_incident(recommended_steps=())


def test_build_incident_rejects_empty_evidence_or_recommendations() -> None:
    with pytest.raises(ValueError, match="must not contain empty items"):
        make_incident(evidence=(" ",))

    with pytest.raises(ValueError, match="must not contain empty items"):
        make_incident(recommended_steps=("",))
