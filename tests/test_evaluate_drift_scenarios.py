"""Tests for synthetic distribution-change scenario evaluation."""

from __future__ import annotations

import math

import pytest
from scripts.evaluate_drift_scenarios import (
    SCENARIOS,
    evaluate_scenario,
)


def run_scenario(
    *,
    scenario_name: str = "location_shift",
    seed: int = 42,
    trials: int = 5,
    sample_size: int = 100,
    feature_count: int = 5,
    alpha: float = 0.05,
) -> dict[str, str | int | float | None]:
    """Run a named scenario using deterministic parameters."""
    scenario = next(scenario for scenario in SCENARIOS if scenario.name == scenario_name)
    return evaluate_scenario(
        scenario=scenario,
        seed=seed,
        trials=trials,
        sample_size=sample_size,
        feature_count=feature_count,
        alpha=alpha,
    )


def test_scenario_names_are_unique() -> None:
    names = [scenario.name for scenario in SCENARIOS]

    assert len(names) == len(set(names))
    assert "no_drift" in names
    assert len(names) >= 5


def test_evaluation_is_reproducible() -> None:
    first = run_scenario()
    second = run_scenario()

    assert first == second


def test_confusion_counts_cover_all_feature_trial_pairs() -> None:
    result = run_scenario(
        scenario_name="variance_increase",
        trials=4,
        sample_size=100,
        feature_count=5,
    )

    total = (
        result["true_positives"]
        + result["false_positives"]
        + result["true_negatives"]
        + result["false_negatives"]
    )

    assert total == 4 * 5


def test_no_drift_scenario_has_no_positive_ground_truth() -> None:
    result = run_scenario(
        scenario_name="no_drift",
        trials=5,
        sample_size=100,
        feature_count=4,
    )

    assert result["true_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["true_negatives"] + result["false_positives"] == 20
    assert result["recall"] is None
    assert result["f1"] is None


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("trials", 0, "trials must be >= 1"),
        ("sample_size", 19, "sample_size must be >= 20"),
        ("feature_count", 1, "feature_count must be >= 2"),
        ("alpha", 0.0, "alpha must be finite"),
        ("alpha", 1.0, "alpha must be finite"),
        ("alpha", math.nan, "alpha must be finite"),
        ("alpha", math.inf, "alpha must be finite"),
    ],
)
def test_invalid_parameters_raise_value_error(
    parameter: str,
    value: int | float,
    message: str,
) -> None:
    kwargs: dict[str, int | float] = {
        "seed": 42,
        "trials": 5,
        "sample_size": 100,
        "feature_count": 5,
        "alpha": 0.05,
    }
    kwargs[parameter] = value

    scenario = SCENARIOS[0]

    with pytest.raises(ValueError, match=message):
        evaluate_scenario(scenario=scenario, **kwargs)
