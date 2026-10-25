"""Evaluate numerical drift detection across shifts, sample sizes, and seeds."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ml_observability.drift import detect_numerical_drift

DEFAULT_SHIFTS = (0.0, 0.2, 0.5, 0.8, 1.2)
DEFAULT_SAMPLE_SIZES = (50, 100, 200, 500)
DEFAULT_SEEDS = (42, 123, 2026)

MetricRow = dict[str, int | float | None]


def evaluate_configuration(
    *,
    seed: int,
    trials: int,
    sample_size: int,
    feature_count: int,
    shift: float,
    alpha: float,
) -> MetricRow:
    """Measure detection metrics for one synthetic configuration."""
    if trials < 1:
        raise ValueError("trials must be >= 1.")
    if sample_size < 20:
        raise ValueError("sample_size must be >= 20.")
    if feature_count < 2:
        raise ValueError("feature_count must be >= 2.")
    if not math.isfinite(shift) or shift < 0:
        raise ValueError("shift must be finite and >= 0.")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and between 0 and 1.")

    rng = np.random.default_rng(seed)
    tp = fp = tn = fn = 0

    for _ in range(trials):
        reference = pd.DataFrame(
            {f"feature_{index}": rng.normal(size=sample_size) for index in range(feature_count)}
        )
        current = reference.copy(deep=True)
        target = "feature_0"

        # Independent draws prevent accidental pairing from hiding distribution shift.
        current[target] = rng.normal(
            loc=shift,
            scale=1.0,
            size=sample_size,
        )

        results = detect_numerical_drift(
            reference,
            current,
            min_samples=20,
            alpha=alpha,
        )
        by_feature = {result.feature: result for result in results}

        if set(by_feature) != set(reference.columns):
            raise RuntimeError("Detector results do not match the expected features.")

        for feature in reference.columns:
            result = by_feature[feature]
            actual_positive = shift > 0 and feature == target
            predicted_positive = result.status == "evaluated" and result.drift_detected is True

            if actual_positive and predicted_positive:
                tp += 1
            elif actual_positive:
                fn += 1
            elif predicted_positive:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    false_positive_rate = fp / (fp + tn) if fp + tn else None

    return {
        "seed": seed,
        "trials": trials,
        "sample_size": sample_size,
        "feature_count": feature_count,
        "shift": shift,
        "alpha": alpha,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
    }


def run_matrix(
    *,
    shifts: Sequence[float] = DEFAULT_SHIFTS,
    sample_sizes: Sequence[int] = DEFAULT_SAMPLE_SIZES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    trials: int = 100,
    feature_count: int = 5,
    alpha: float = 0.05,
) -> list[MetricRow]:
    """Evaluate every combination of shift, sample size, and random seed."""
    if not shifts or not sample_sizes or not seeds:
        raise ValueError("shifts, sample_sizes, and seeds must not be empty.")

    rows: list[MetricRow] = []
    for shift in shifts:
        for sample_size in sample_sizes:
            for seed in seeds:
                rows.append(
                    evaluate_configuration(
                        seed=seed,
                        trials=trials,
                        sample_size=sample_size,
                        feature_count=feature_count,
                        shift=shift,
                        alpha=alpha,
                    )
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--feature-count", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    results = run_matrix(
        trials=args.trials,
        feature_count=args.feature_count,
        alpha=args.alpha,
    )
    print(json.dumps(results, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
