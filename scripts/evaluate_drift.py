"""Seeded synthetic evaluation of numerical drift detection."""

import argparse
import json

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
    """Measure shifted-feature detection and unchanged-feature false positives."""
    if trials < 1 or sample_size < 20 or feature_count < 2 or shift <= 0:
        raise ValueError(
            "Require trials >= 1, sample_size >= 20, feature_count >= 2, and shift > 0."
        )

    rng = np.random.default_rng(seed)
    null_feature_tests = 0
    false_positives = 0
    shifted_detections = 0

    for _ in range(trials):
        reference = pd.DataFrame(
            {f"feature_{index}": rng.normal(size=sample_size) for index in range(feature_count)}
        )

        unchanged = {
            f"feature_{index}": rng.normal(size=sample_size) for index in range(feature_count)
        }
        current_null = pd.DataFrame(unchanged)

        null_results = detect_numerical_drift(
            reference,
            current_null,
            min_samples=20,
            alpha=alpha,
        )
        null_feature_tests += len(null_results)
        false_positives += sum(result.drift_detected is True for result in null_results)

        current_shifted = current_null.copy()
        target = "feature_0"
        current_shifted[target] = rng.normal(
            loc=shift,
            scale=1.0,
            size=sample_size,
        )

        shifted_results = detect_numerical_drift(
            reference,
            current_shifted,
            min_samples=20,
            alpha=alpha,
        )
        shifted_by_name = {result.feature: result for result in shifted_results}
        target_result = shifted_by_name[target]
        shifted_detections += target_result.drift_detected is True

        null_feature_tests += feature_count - 1
        false_positives += sum(
            shifted_by_name[f"feature_{index}"].drift_detected is True
            for index in range(1, feature_count)
        )

    return {
        "seed": seed,
        "trials": trials,
        "sample_size": sample_size,
        "feature_count": feature_count,
        "shift": shift,
        "alpha": alpha,
        "shifted_feature_detection_rate": shifted_detections / trials,
        "unchanged_feature_false_positive_rate": (false_positives / null_feature_tests),
        "null_feature_tests": null_feature_tests,
        "false_positives": false_positives,
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
