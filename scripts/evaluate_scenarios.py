"""Seeded, labeled evaluation of numerical drift detection."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

from ml_observability.drift import detect_numerical_drift


def run_evaluation(
    *,
    seed: int = 42,
    trials: int = 100,
    sample_size: int = 200,
    feature_count: int = 5,
    shift: float = 0.8,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Evaluate drift detection against known shifted and unchanged features."""
    if trials < 1:
        raise ValueError("trials must be >= 1.")
    if sample_size < 20:
        raise ValueError("sample_size must be >= 20.")
    if feature_count < 2:
        raise ValueError("feature_count must be >= 2.")
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("shift must be finite and > 0.")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and between 0 and 1.")

    rng = np.random.default_rng(seed)
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    evaluated_features = 0

    for _ in range(trials):
        reference = pd.DataFrame(
            {f"feature_{index}": rng.normal(size=sample_size) for index in range(feature_count)}
        )

        current = reference.copy(deep=True)
        target = "feature_0"
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
        results_by_feature = {result.feature: result for result in results}

        expected_features = set(reference.columns)
        if set(results_by_feature) != expected_features:
            raise RuntimeError("Detector results do not match the expected feature set.")

        for feature in reference.columns:
            result = results_by_feature[feature]

            # Ground truth is defined by the synthetic data construction.
            actual_drift = feature == target

            # Only an evaluated, positive detector result counts as detected.
            predicted_drift = result.status == "evaluated" and result.drift_detected is True

            evaluated_features += 1
            if actual_drift and predicted_drift:
                true_positives += 1
            elif actual_drift:
                false_negatives += 1
            elif predicted_drift:
                false_positives += 1
            else:
                true_negatives += 1

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    negative_denominator = true_negatives + false_positives

    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    f1 = 2 * precision * recall / f1_denominator if f1_denominator else 0.0
    false_positive_rate = false_positives / negative_denominator if negative_denominator else 0.0

    return {
        "seed": seed,
        "trials": trials,
        "sample_size": sample_size,
        "feature_count": feature_count,
        "shift": shift,
        "alpha": alpha,
        "evaluated_features": evaluated_features,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--feature-count", type=int, default=5)
    parser.add_argument("--shift", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    result = run_evaluation(
        seed=args.seed,
        trials=args.trials,
        sample_size=args.sample_size,
        feature_count=args.feature_count,
        shift=args.shift,
        alpha=args.alpha,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
