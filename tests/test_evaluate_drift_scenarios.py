"""Tests for synthetic distribution-change scenario evaluation."""

from __future__ import annotations

import math

import pytest
from scripts.evaluate_drift_scenarios import (
    SCENARIOS,
    aggregate_scenario_results,
    evaluate_scenario,
    wilson_interval,
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
    scenario = next(item for item in SCENARIOS if item.name == scenario_name)
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
    assert run_scenario() == run_scenario()


def test_confusion_counts_cover_all_feature_trial_pairs() -> None:
    trials = 4
    feature_count = 5
    result = run_scenario(
        scenario_name="variance_increase",
        trials=trials,
        sample_size=100,
        feature_count=feature_count,
    )

    total = sum(
        result[key]
        for key in (
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
        )
    )

    assert total == trials * feature_count


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
    assert result["target_trial_count"] == 0


def test_wilson_interval_for_zero_successes_is_bounded() -> None:
    interval = wilson_interval(0, 20)

    assert interval is not None
    assert interval[0] == 0.0
    assert 0.0 < interval[1] < 1.0


def test_wilson_interval_for_all_successes_is_bounded() -> None:
    interval = wilson_interval(20, 20)

    assert interval is not None
    assert 0.0 < interval[0] < 1.0
    assert interval[1] == 1.0


def test_wilson_interval_returns_none_for_zero_trials() -> None:
    assert wilson_interval(0, 0) is None


@pytest.mark.parametrize(
    ("successes", "trials", "message"),
    [
        (-1, 10, "successes must be between"),
        (11, 10, "successes must be between"),
        (0, -1, "trials must be >= 0"),
    ],
)
def test_wilson_interval_rejects_invalid_counts(
    successes: int,
    trials: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        wilson_interval(successes, trials)


@pytest.mark.parametrize("z", [0.0, -1.0, math.nan, math.inf])
def test_wilson_interval_rejects_invalid_z(z: float) -> None:
    with pytest.raises(ValueError, match="z must be finite"):
        wilson_interval(1, 10, z=z)


def test_aggregate_results_combines_seeds_and_reports_intervals() -> None:
    scenario = next(item for item in SCENARIOS if item.name == "location_shift")
    results = [
        evaluate_scenario(
            scenario=scenario,
            seed=seed,
            trials=5,
            sample_size=100,
            feature_count=5,
            alpha=0.05,
        )
        for seed in (42, 123)
    ]

    aggregate = aggregate_scenario_results(
        scenario=scenario,
        results=results,
    )

    assert aggregate["seeds"] == [42, 123]
    assert aggregate["total_trials"] == 10
    assert aggregate["target_trial_count"] == 10
    assert aggregate["recall_ci_95"] is not None
    assert aggregate["familywise_false_positive_rate_ci_95"] is not None


def test_aggregate_rejects_empty_results() -> None:
    scenario = SCENARIOS[0]

    with pytest.raises(ValueError, match="at least one run"):
        aggregate_scenario_results(scenario=scenario, results=[])


def test_aggregate_rejects_mismatched_scenario_names() -> None:
    scenario = SCENARIOS[0]
    result = run_scenario(scenario_name="location_shift")

    with pytest.raises(ValueError, match="requested scenario"):
        aggregate_scenario_results(scenario=scenario, results=[result])


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
