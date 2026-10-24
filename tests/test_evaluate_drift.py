"""Tests for the reproducible synthetic drift evaluation."""

import pytest
from scripts.evaluate_drift import run_evaluation


def test_evaluation_is_reproducible() -> None:
    first = run_evaluation(seed=42, trials=5, sample_size=100)
    second = run_evaluation(seed=42, trials=5, sample_size=100)

    assert first == second


def test_evaluation_reports_valid_rate_bounds() -> None:
    result = run_evaluation(seed=42, trials=5, sample_size=100)

    assert 0.0 <= result["shifted_feature_detection_rate"] <= 1.0
    assert 0.0 <= result["unchanged_feature_false_positive_rate"] <= 1.0
    assert result["trials"] == 5


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"trials": 0}, "trials"),
        ({"sample_size": 19}, "sample_size"),
        ({"feature_count": 1}, "feature_count"),
        ({"shift": 0.0}, "shift"),
    ],
)
def test_evaluation_rejects_invalid_configuration(
    overrides: dict[str, int | float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_evaluation(**overrides)
