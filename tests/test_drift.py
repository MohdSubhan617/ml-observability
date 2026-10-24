"""Tests for numerical feature drift detection."""

import numpy as np
import pandas as pd
import pytest

from ml_observability.drift import detect_numerical_drift


def test_detects_a_large_distribution_shift() -> None:
    reference = pd.DataFrame({"value": np.arange(100, dtype=float)})
    current = pd.DataFrame({"value": np.arange(100, dtype=float) + 1000})

    result = detect_numerical_drift(reference, current)

    assert len(result) == 1
    assert result[0].status == "evaluated"
    assert result[0].drift_detected is True
    assert result[0].statistic is not None
    assert result[0].adjusted_p_value is not None


def test_identical_samples_do_not_trigger_drift() -> None:
    values = np.arange(100, dtype=float)
    reference = pd.DataFrame({"value": values})
    current = pd.DataFrame({"value": values.copy()})

    result = detect_numerical_drift(reference, current)

    assert result[0].status == "evaluated"
    assert result[0].drift_detected is False


def test_insufficient_samples_are_not_evaluated() -> None:
    reference = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    current = pd.DataFrame({"value": [100.0, 200.0, 300.0]})

    result = detect_numerical_drift(reference, current, min_samples=20)

    assert result[0].status == "not_evaluated"
    assert result[0].drift_detected is None
    assert result[0].reason is not None


def test_non_finite_values_are_not_evaluated() -> None:
    reference = pd.DataFrame({"value": [float("inf")] * 25})
    current = pd.DataFrame({"value": np.arange(25, dtype=float)})

    result = detect_numerical_drift(reference, current)

    assert result[0].status == "not_evaluated"
    assert result[0].drift_detected is None


def test_non_numeric_and_unshared_features_are_ignored() -> None:
    reference = pd.DataFrame({"value": np.arange(25), "category": ["a"] * 25, "old": np.arange(25)})
    current = pd.DataFrame({"value": np.arange(25), "category": ["b"] * 25, "new": np.arange(25)})

    result = detect_numerical_drift(reference, current)

    assert [item.feature for item in result] == ["value"]


@pytest.mark.parametrize(
    ("min_samples", "alpha"),
    [(1, 0.05), (20, 0.0), (20, 1.0)],
)
def test_invalid_configuration_is_rejected(min_samples: int, alpha: float) -> None:
    with pytest.raises(ValueError):
        detect_numerical_drift(
            pd.DataFrame({"value": np.arange(25)}),
            pd.DataFrame({"value": np.arange(25)}),
            min_samples=min_samples,
            alpha=alpha,
        )
