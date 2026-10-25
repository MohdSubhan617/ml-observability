"""Regression tests for the drift evaluation matrix."""

import pytest
from scripts.evaluate_drift_matrix import evaluate_configuration, run_matrix


def test_matrix_covers_cartesian_product() -> None:
    results = run_matrix(
        shifts=(0.0, 0.5),
        sample_sizes=(30, 50),
        seeds=(1, 2),
        trials=2,
        feature_count=3,
    )

    assert len(results) == 8
    assert {(row["shift"], row["sample_size"], row["seed"]) for row in results} == {
        (shift, sample_size, seed)
        for shift in (0.0, 0.5)
        for sample_size in (30, 50)
        for seed in (1, 2)
    }


def test_evaluation_is_reproducible() -> None:
    kwargs = {
        "seed": 42,
        "trials": 3,
        "sample_size": 50,
        "feature_count": 3,
        "shift": 0.5,
        "alpha": 0.05,
    }

    assert evaluate_configuration(**kwargs) == evaluate_configuration(**kwargs)


def test_confusion_matrix_counts_are_consistent() -> None:
    trials = 3
    feature_count = 4

    for shift in (0.0, 0.5):
        result = evaluate_configuration(
            seed=42,
            trials=trials,
            sample_size=50,
            feature_count=feature_count,
            shift=shift,
            alpha=0.05,
        )

        if shift == 0.0:
            assert result["true_positives"] == 0
            assert result["false_negatives"] == 0
            assert result["true_negatives"] + result["false_positives"] == trials * feature_count
        else:
            assert result["true_positives"] + result["false_negatives"] == trials
            assert result["true_negatives"] + result["false_positives"] == trials * (
                feature_count - 1
            )


def test_metrics_are_bounded() -> None:
    result = evaluate_configuration(
        seed=42,
        trials=3,
        sample_size=50,
        feature_count=3,
        shift=0.5,
        alpha=0.05,
    )

    for metric in ("precision", "recall", "f1", "false_positive_rate"):
        assert 0.0 <= result[metric] <= 1.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"trials": 0}, "trials"),
        ({"sample_size": 19}, "sample_size"),
        ({"feature_count": 1}, "feature_count"),
        ({"shift": -0.1}, "shift"),
        ({"shift": float("nan")}, "shift"),
        ({"shift": float("inf")}, "shift"),
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.0}, "alpha"),
        ({"alpha": float("nan")}, "alpha"),
    ],
)
def test_evaluation_rejects_invalid_configuration(
    overrides: dict[str, int | float],
    message: str,
) -> None:
    kwargs: dict[str, int | float] = {
        "seed": 42,
        "trials": 2,
        "sample_size": 30,
        "feature_count": 3,
        "shift": 0.5,
        "alpha": 0.05,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        evaluate_configuration(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"shifts": ()}, "must not be empty"),
        ({"sample_sizes": ()}, "must not be empty"),
        ({"seeds": ()}, "must not be empty"),
    ],
)
def test_matrix_rejects_empty_dimensions(
    overrides: dict[str, tuple[()]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_matrix(**overrides)


def test_zero_shift_returns_none_for_undefined_metrics() -> None:
    from scripts.evaluate_drift_matrix import evaluate_configuration

    result = evaluate_configuration(
        seed=42,
        trials=5,
        sample_size=50,
        feature_count=5,
        shift=0.0,
        alpha=0.05,
    )

    assert result["precision"] is None
    assert result["recall"] is None
    assert result["f1"] is None
    assert isinstance(result["false_positive_rate"], float)
