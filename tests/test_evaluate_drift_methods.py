"""Regression tests for the variance-sensitive evaluation baseline."""

import numpy as np
import pandas as pd
from scripts.evaluate_drift_methods import (
    ComparisonScenario,
    detect_with_brown_forsythe,
    evaluate_method,
)


def test_brown_forsythe_detects_large_variance_change_with_unchanged_controls() -> None:
    rng = np.random.default_rng(42)
    reference = pd.DataFrame({f"feature_{i}": rng.normal(size=1_000) for i in range(5)})
    current = pd.DataFrame(
        {
            feature: rng.normal(loc=0.0, scale=2.0 if index == 0 else 1.0, size=1_000)
            for index, feature in enumerate(reference.columns)
        }
    )

    predictions = detect_with_brown_forsythe(reference, current, alpha=0.05)

    assert predictions == [True, False, False, False, False]


def test_brown_forsythe_does_not_flag_identical_samples() -> None:
    rng = np.random.default_rng(123)
    reference = pd.DataFrame({f"feature_{i}": rng.normal(size=100) for i in range(5)})

    predictions = detect_with_brown_forsythe(reference, reference.copy(), alpha=0.05)

    assert predictions == [False, False, False, False, False]


def test_brown_forsythe_evaluation_labels_location_shift_as_no_variance_change() -> None:
    def sample_shifted_normal(generator: np.random.Generator, size: int) -> np.ndarray:
        return generator.normal(loc=1.0, scale=1.0, size=size)

    scenario = ComparisonScenario("location_shift", True, False, sample_shifted_normal)
    result = evaluate_method(
        scenario=scenario,
        method="brown_forsythe",
        seed=7,
        trials=1,
        sample_size=200,
        feature_count=5,
        alpha=0.05,
    )

    assert result["target_trial_count"] == 0
    assert result["recall"] is None
    assert result["true_negatives"] + result["false_positives"] == 5
