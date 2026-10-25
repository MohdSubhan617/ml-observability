"""Evaluate numerical drift detection across distribution-change scenarios."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ml_observability.drift import detect_numerical_drift

FloatArray = NDArray[np.float64]
DistributionSampler = Callable[[np.random.Generator, int], FloatArray]


@dataclass(frozen=True)
class Scenario:
    """Synthetic distribution-change scenario."""

    name: str
    changed: bool
    sampler: DistributionSampler


def sample_normal(rng: np.random.Generator, size: int) -> FloatArray:
    """Sample from a standard normal distribution."""
    return rng.normal(loc=0.0, scale=1.0, size=size)


def sample_location_shift(rng: np.random.Generator, size: int) -> FloatArray:
    """Sample from a normal distribution with a shifted mean."""
    return rng.normal(loc=0.8, scale=1.0, size=size)


def sample_variance_increase(rng: np.random.Generator, size: int) -> FloatArray:
    """Sample from a normal distribution with increased variance."""
    return rng.normal(loc=0.0, scale=1.5, size=size)


def sample_heavy_tailed(rng: np.random.Generator, size: int) -> FloatArray:
    """Sample from a variance-normalized Student's t distribution."""
    degrees_of_freedom = 3
    variance = degrees_of_freedom / (degrees_of_freedom - 2)
    return rng.standard_t(df=degrees_of_freedom, size=size) / math.sqrt(variance)


def sample_skewed(rng: np.random.Generator, size: int) -> FloatArray:
    """Sample from a centered, right-skewed exponential distribution."""
    return rng.exponential(scale=1.0, size=size) - 1.0


SCENARIOS = (
    Scenario("no_drift", False, sample_normal),
    Scenario("location_shift", True, sample_location_shift),
    Scenario("variance_increase", True, sample_variance_increase),
    Scenario("heavy_tailed", True, sample_heavy_tailed),
    Scenario("skewed", True, sample_skewed),
)


def evaluate_scenario(
    *,
    scenario: Scenario,
    seed: int,
    trials: int,
    sample_size: int,
    feature_count: int,
    alpha: float,
) -> dict[str, str | int | float | None]:
    """Measure feature-level detection metrics against synthetic labels."""
    if trials < 1:
        raise ValueError("trials must be >= 1.")
    if sample_size < 20:
        raise ValueError("sample_size must be >= 20.")
    if feature_count < 2:
        raise ValueError("feature_count must be >= 2.")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and between 0 and 1.")

    rng = np.random.default_rng(seed)
    tp = fp = tn = fn = 0

    for _ in range(trials):
        reference = pd.DataFrame(
            {f"feature_{index}": sample_normal(rng, sample_size) for index in range(feature_count)}
        )

        current = pd.DataFrame(
            {
                f"feature_{index}": (
                    scenario.sampler(rng, sample_size)
                    if index == 0 and scenario.changed
                    else sample_normal(rng, sample_size)
                )
                for index in range(feature_count)
            }
        )

        results = detect_numerical_drift(
            reference,
            current,
            min_samples=20,
            alpha=alpha,
        )
        by_feature = {result.feature: result for result in results}

        if set(by_feature) != set(reference.columns):
            raise RuntimeError("Detector results do not match expected features.")

        for feature in reference.columns:
            result = by_feature[feature]
            actual_positive = scenario.changed and feature == "feature_0"
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

    fpr = fp / (fp + tn) if fp + tn else None

    return {
        "scenario": scenario.name,
        "seed": seed,
        "trials": trials,
        "sample_size": sample_size,
        "feature_count": feature_count,
        "alpha": alpha,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
    }


def main() -> None:
    """Run all scenarios and print their metrics as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--feature-count", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    results = [
        evaluate_scenario(
            scenario=scenario,
            seed=args.seed,
            trials=args.trials,
            sample_size=args.sample_size,
            feature_count=args.feature_count,
            alpha=args.alpha,
        )
        for scenario in SCENARIOS
    ]

    print(json.dumps(results, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
