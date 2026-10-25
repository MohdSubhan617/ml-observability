"""Tests for the labeled synthetic drift evaluation."""

import pytest
from scripts.evaluate_scenarios import run_evaluation


def test_evaluation_is_reproducible() -> None:
    first = run_evaluation(seed=42, trials=5, sample_size=100)
    second = run_evaluation(seed=42, trials=5, sample_size=100)

    assert first == second


def test_confusion_matrix_counts_match_labeled_scenarios() -> None:
    result = run_evaluation(
        seed=42,
        trials=5,
        sample_size=100,
        feature_count=4,
    )

    assert result["true_positives"] + result["false_negatives"] == 5
    assert result["true_negatives"] + result["false_positives"] == 15
    assert result["evaluated_features"] == 20


def test_metrics_are_bounded() -> None:
    result = run_evaluation(seed=42, trials=5, sample_size=100)

    for metric in ("precision", "recall", "f1", "false_positive_rate"):
        assert 0.0 <= result[metric] <= 1.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"trials": 0}, "trials"),
        ({"sample_size": 19}, "sample_size"),
        ({"feature_count": 1}, "feature_count"),
        ({"shift": 0.0}, "shift"),
        ({"shift": float("nan")}, "shift"),
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.0}, "alpha"),
        ({"alpha": float("nan")}, "alpha"),
    ],
)
def test_evaluation_rejects_invalid_configuration(
    overrides: dict[str, int | float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_evaluation(**overrides)
