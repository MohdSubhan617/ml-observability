"""Evaluate numerical drift detection across synthetic distribution changes."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ml_observability.drift import detect_numerical_drift

FloatArray = NDArray[np.float64]
DistributionSampler = Callable[[np.random.Generator, int], FloatArray]
DEFAULT_SEEDS = (42, 123, 2026)
WILSON_Z_95 = 1.959963984540054


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


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = WILSON_Z_95,
) -> tuple[float, float] | None:
    """Return a Wilson score confidence interval for a binomial proportion."""
    if trials < 0:
        raise ValueError("trials must be >= 0.")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials.")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and > 0.")
    if trials == 0:
        return None

    proportion = successes / trials
    z_squared = z * z
    denominator = 1 + z_squared / trials
    center = (proportion + z_squared / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z_squared / (4 * trials * trials))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def evaluate_scenario(
    *,
    scenario: Scenario,
    seed: int,
    trials: int,
    sample_size: int,
    feature_count: int,
    alpha: float,
) -> dict[str, str | int | float | None]:
    """Measure detection metrics for one scenario and random seed."""
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
    target_detected_trials = 0
    target_trial_count = trials if scenario.changed else 0
    control_trials_with_false_positive = 0

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

        control_false_positive_this_trial = False

        for feature in reference.columns:
            result = by_feature[feature]
            actual_positive = scenario.changed and feature == "feature_0"
            predicted_positive = result.status == "evaluated" and result.drift_detected is True

            if actual_positive and predicted_positive:
                tp += 1
                target_detected_trials += 1
            elif actual_positive:
                fn += 1
            elif predicted_positive:
                fp += 1
                control_false_positive_this_trial = True
            else:
                tn += 1

        if control_false_positive_this_trial:
            control_trials_with_false_positive += 1

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
        "target_detected_trials": target_detected_trials,
        "target_trial_count": target_trial_count,
        "control_trials_with_false_positive": control_trials_with_false_positive,
        "control_trial_count": trials,
    }


def aggregate_scenario_results(
    *,
    scenario: Scenario,
    results: Sequence[dict[str, str | int | float | None]],
) -> dict[str, object]:
    """Aggregate independent seeded runs and compute Wilson confidence intervals."""
    if not results:
        raise ValueError("results must contain at least one run.")

    expected_name = scenario.name
    for result in results:
        if result["scenario"] != expected_name:
            raise ValueError("All results must belong to the requested scenario.")

    count_keys = (
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "target_detected_trials",
        "target_trial_count",
        "control_trials_with_false_positive",
        "control_trial_count",
    )
    totals = {key: sum(int(result[key]) for result in results) for key in count_keys}

    tp = totals["true_positives"]
    fp = totals["false_positives"]
    tn = totals["true_negatives"]
    fn = totals["false_negatives"]

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    fpr = fp / (fp + tn) if fp + tn else None

    recall_ci = wilson_interval(
        totals["target_detected_trials"],
        totals["target_trial_count"],
    )
    familywise_fpr_ci = wilson_interval(
        totals["control_trials_with_false_positive"],
        totals["control_trial_count"],
    )
    familywise_fpr = (
        totals["control_trials_with_false_positive"] / totals["control_trial_count"]
        if totals["control_trial_count"]
        else None
    )

    first = results[0]
    return {
        "scenario": expected_name,
        "seeds": [int(result["seed"]) for result in results],
        "trials_per_seed": int(first["trials"]),
        "total_trials": sum(int(result["trials"]) for result in results),
        "sample_size": int(first["sample_size"]),
        "feature_count": int(first["feature_count"]),
        "alpha": float(first["alpha"]),
        **totals,
        "precision": precision,
        "recall": recall,
        "recall_ci_95": list(recall_ci) if recall_ci is not None else None,
        "f1": f1,
        "false_positive_rate": fpr,
        "familywise_false_positive_rate": familywise_fpr,
        "familywise_false_positive_rate_ci_95": (
            list(familywise_fpr_ci) if familywise_fpr_ci is not None else None
        ),
    }


def main() -> None:
    """Run scenarios across multiple seeds and print JSON metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed",
        type=int,
        help="Run one seed instead of the default multi-seed evaluation.",
    )
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Explicit list of independent random seeds.",
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--feature-count", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    seeds = (
        [args.seed]
        if args.seed is not None
        else args.seeds
        if args.seeds is not None
        else list(DEFAULT_SEEDS)
    )

    all_results = []
    for scenario in SCENARIOS:
        seeded_results = [
            evaluate_scenario(
                scenario=scenario,
                seed=seed,
                trials=args.trials,
                sample_size=args.sample_size,
                feature_count=args.feature_count,
                alpha=args.alpha,
            )
            for seed in seeds
        ]
        all_results.append(
            aggregate_scenario_results(
                scenario=scenario,
                results=seeded_results,
            )
        )

    print(json.dumps(all_results, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
