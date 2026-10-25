"""Compare KS distribution drift with a variance-focused Brown–Forsythe baseline."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import false_discovery_control, levene

from ml_observability.drift import detect_numerical_drift
from ml_observability.evaluation import wilson_interval

FloatArray = NDArray[np.float64]
Sampler = Callable[[np.random.Generator, int], FloatArray]
MethodName = Literal["ks", "brown_forsythe"]


@dataclass(frozen=True)
class ComparisonScenario:
    """Synthetic case with distinct general-drift and variance-drift labels."""

    name: str
    distribution_changed: bool
    variance_changed: bool
    sampler: Sampler


class RunCounts(TypedDict):
    """Classification counts from one seeded method/scenario run."""

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    target_detected_trials: int
    target_trial_count: int
    control_trials_with_false_positive: int
    control_trial_count: int


class MethodRunResult(RunCounts):
    """Configuration and metrics for a single run."""

    method: MethodName
    scenario: str
    seed: int
    trials: int
    sample_size: int
    feature_count: int
    alpha: float
    recall: float | None
    feature_false_positive_rate: float | None


def sample_normal(rng: np.random.Generator, size: int) -> FloatArray:
    """Sample a standard normal distribution."""
    return rng.normal(loc=0.0, scale=1.0, size=size)


def location_sampler(shift: float) -> Sampler:
    """Build a normal-location alternative sampler."""

    def sample(rng: np.random.Generator, size: int) -> FloatArray:
        return rng.normal(loc=shift, scale=1.0, size=size)

    return sample


def variance_sampler(scale: float) -> Sampler:
    """Build a normal-scale alternative sampler; variance changes by scale squared."""

    def sample(rng: np.random.Generator, size: int) -> FloatArray:
        return rng.normal(loc=0.0, scale=scale, size=size)

    return sample


def sample_heavy_tailed(rng: np.random.Generator, size: int) -> FloatArray:
    """Sample variance-normalized Student t with three degrees of freedom."""
    return rng.standard_t(df=3, size=size) / math.sqrt(3.0)


def sample_skewed(rng: np.random.Generator, size: int) -> FloatArray:
    """Sample a centered exponential distribution with unit variance."""
    return rng.exponential(scale=1.0, size=size) - 1.0


def build_scenarios(
    *, location_shifts: Sequence[float], variance_scales: Sequence[float]
) -> tuple[ComparisonScenario, ...]:
    """Create controls and configured distribution alternatives."""
    scenarios = [ComparisonScenario("no_drift", False, False, sample_normal)]
    scenarios.extend(
        ComparisonScenario(f"location_shift_{shift:g}", True, False, location_sampler(shift))
        for shift in location_shifts
    )
    scenarios.extend(
        ComparisonScenario(f"variance_scale_{scale:g}", True, True, variance_sampler(scale))
        for scale in variance_scales
    )
    scenarios.extend(
        (
            ComparisonScenario("heavy_tailed", True, False, sample_heavy_tailed),
            ComparisonScenario("skewed", True, False, sample_skewed),
        )
    )
    return tuple(scenarios)


def detect_with_brown_forsythe(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    alpha: float,
) -> list[bool]:
    """Apply median-centered Levene tests with BH correction across features."""
    p_values = [
        float(
            levene(
                reference[feature].to_numpy(),
                current[feature].to_numpy(),
                center="median",
            ).pvalue
        )
        for feature in reference.columns
    ]
    adjusted = false_discovery_control(p_values, method="bh")
    return [bool(p_value < alpha) for p_value in adjusted]


def evaluate_method(
    *,
    scenario: ComparisonScenario,
    method: MethodName,
    seed: int,
    trials: int,
    sample_size: int,
    feature_count: int,
    alpha: float,
) -> MethodRunResult:
    """Evaluate a detector using its intended ground-truth target."""
    if trials < 1:
        raise ValueError("trials must be >= 1.")
    if sample_size < 20:
        raise ValueError("sample_size must be >= 20.")
    if feature_count < 2:
        raise ValueError("feature_count must be >= 2.")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and between 0 and 1.")

    rng = np.random.default_rng(seed)
    target_positive = scenario.distribution_changed if method == "ks" else scenario.variance_changed
    tp = fp = tn = fn = 0
    target_detected_trials = 0
    target_trial_count = trials if target_positive else 0
    control_trials_with_false_positive = 0

    for _ in range(trials):
        columns = [f"feature_{index}" for index in range(feature_count)]
        reference = pd.DataFrame({feature: sample_normal(rng, sample_size) for feature in columns})
        current = pd.DataFrame(
            {
                feature: (
                    scenario.sampler(rng, sample_size)
                    if index == 0 and (scenario.distribution_changed or scenario.variance_changed)
                    else sample_normal(rng, sample_size)
                )
                for index, feature in enumerate(columns)
            }
        )

        if method == "ks":
            by_feature = {
                result.feature: result.drift_detected is True
                for result in detect_numerical_drift(reference, current, alpha=alpha)
            }
            predictions = [by_feature[feature] for feature in columns]
        else:
            predictions = detect_with_brown_forsythe(reference, current, alpha=alpha)

        false_positive_this_trial = False
        for index, predicted_positive in enumerate(predictions):
            actual_positive = target_positive and index == 0
            if actual_positive and predicted_positive:
                tp += 1
                target_detected_trials += 1
            elif actual_positive:
                fn += 1
            elif predicted_positive:
                fp += 1
                false_positive_this_trial = True
            else:
                tn += 1

        if false_positive_this_trial:
            control_trials_with_false_positive += 1

    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    return {
        "method": method,
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
        "target_detected_trials": target_detected_trials,
        "target_trial_count": target_trial_count,
        "control_trials_with_false_positive": control_trials_with_false_positive,
        "control_trial_count": trials,
        "recall": recall,
        "feature_false_positive_rate": fpr,
    }


def aggregate_method_results(
    *, runs: Sequence[MethodRunResult], target_definition: str
) -> dict[str, object]:
    """Aggregate independent seeds and report recall and false-positive intervals."""
    if not runs:
        raise ValueError("runs must contain at least one result.")

    totals: RunCounts = {
        "true_positives": sum(run["true_positives"] for run in runs),
        "false_positives": sum(run["false_positives"] for run in runs),
        "true_negatives": sum(run["true_negatives"] for run in runs),
        "false_negatives": sum(run["false_negatives"] for run in runs),
        "target_detected_trials": sum(run["target_detected_trials"] for run in runs),
        "target_trial_count": sum(run["target_trial_count"] for run in runs),
        "control_trials_with_false_positive": sum(
            run["control_trials_with_false_positive"] for run in runs
        ),
        "control_trial_count": sum(run["control_trial_count"] for run in runs),
    }
    tp = totals["true_positives"]
    fp = totals["false_positives"]
    tn = totals["true_negatives"]
    fn = totals["false_negatives"]
    recall = tp / (tp + fn) if tp + fn else None
    feature_fpr = fp / (fp + tn) if fp + tn else None
    familywise_fpr = (
        totals["control_trials_with_false_positive"] / totals["control_trial_count"]
        if totals["control_trial_count"]
        else None
    )
    recall_ci = wilson_interval(totals["target_detected_trials"], totals["target_trial_count"])
    familywise_ci = wilson_interval(
        totals["control_trials_with_false_positive"], totals["control_trial_count"]
    )
    first = runs[0]
    return {
        "method": first["method"],
        "scenario": first["scenario"],
        "target_definition": target_definition,
        "seeds": [run["seed"] for run in runs],
        "trials_per_seed": first["trials"],
        "total_trials": sum(run["trials"] for run in runs),
        "sample_size": first["sample_size"],
        "feature_count": first["feature_count"],
        "alpha": first["alpha"],
        **totals,
        "recall": recall,
        "recall_ci_95": list(recall_ci) if recall_ci is not None else None,
        "feature_false_positive_rate": feature_fpr,
        "familywise_false_positive_rate": familywise_fpr,
        "familywise_false_positive_rate_ci_95": (
            list(familywise_ci) if familywise_ci is not None else None
        ),
    }


def main() -> None:
    """Run the bounded KS versus Brown–Forsythe comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[50, 200])
    parser.add_argument("--location-shifts", type=float, nargs="+", default=[0.5, 0.8])
    parser.add_argument("--variance-scales", type=float, nargs="+", default=[1.5, 2.0])
    parser.add_argument("--feature-count", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    if any(sample_size < 20 for sample_size in args.sample_sizes):
        parser.error("all sample sizes must be >= 20")
    if any(not math.isfinite(shift) or shift <= 0 for shift in args.location_shifts):
        parser.error("location shifts must be finite and > 0")
    if any(not math.isfinite(scale) or scale <= 1 for scale in args.variance_scales):
        parser.error("variance scales must be finite and > 1")

    scenarios = build_scenarios(
        location_shifts=args.location_shifts,
        variance_scales=args.variance_scales,
    )
    summaries: list[dict[str, object]] = []
    methods: tuple[MethodName, ...] = ("ks", "brown_forsythe")
    for sample_size in args.sample_sizes:
        for scenario in scenarios:
            for method in methods:
                runs = [
                    evaluate_method(
                        scenario=scenario,
                        method=method,
                        seed=seed,
                        trials=args.trials,
                        sample_size=sample_size,
                        feature_count=args.feature_count,
                        alpha=args.alpha,
                    )
                    for seed in args.seeds
                ]
                target = "any_distribution_change" if method == "ks" else "variance_change_only"
                summaries.append(aggregate_method_results(runs=runs, target_definition=target))

    print(json.dumps(summaries, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
