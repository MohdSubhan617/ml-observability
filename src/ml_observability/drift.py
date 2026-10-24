"""Numerical feature drift detection using KS tests and FDR correction."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, ks_2samp


@dataclass(frozen=True)
class FeatureDriftResult:
    """Statistical drift result for one feature."""

    feature: str
    status: str
    reference_count: int
    current_count: int
    statistic: float | None
    p_value: float | None
    adjusted_p_value: float | None
    drift_detected: bool | None
    reason: str | None = None


def detect_numerical_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    min_samples: int = 20,
    alpha: float = 0.05,
) -> list[FeatureDriftResult]:
    """Compare shared numerical features and adjust p-values for multiple tests."""
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    for name, frame in (("reference", reference), ("current", current)):
        if not frame.columns.is_unique:
            raise ValueError(f"{name} dataset contains duplicate column names.")

    shared_numeric = [
        column
        for column in reference.columns
        if column in current.columns
        and pd.api.types.is_numeric_dtype(reference[column])
        and pd.api.types.is_numeric_dtype(current[column])
        and not pd.api.types.is_bool_dtype(reference[column])
        and not pd.api.types.is_bool_dtype(current[column])
    ]

    results: list[FeatureDriftResult] = []
    evaluated_indices: list[int] = []
    raw_p_values: list[float] = []

    for feature in shared_numeric:
        ref = reference[feature].dropna().to_numpy(dtype=float)
        cur = current[feature].dropna().to_numpy(dtype=float)

        if not np.isfinite(ref).all() or not np.isfinite(cur).all():
            results.append(
                FeatureDriftResult(
                    feature=feature,
                    status="not_evaluated",
                    reference_count=len(ref),
                    current_count=len(cur),
                    statistic=None,
                    p_value=None,
                    adjusted_p_value=None,
                    drift_detected=None,
                    reason="Feature contains non-finite values.",
                )
            )
            continue

        if len(ref) < min_samples or len(cur) < min_samples:
            results.append(
                FeatureDriftResult(
                    feature=feature,
                    status="not_evaluated",
                    reference_count=len(ref),
                    current_count=len(cur),
                    statistic=None,
                    p_value=None,
                    adjusted_p_value=None,
                    drift_detected=None,
                    reason=f"Each dataset requires at least {min_samples} valid samples.",
                )
            )
            continue

        test = ks_2samp(ref, cur, alternative="two-sided", method="auto")
        evaluated_indices.append(len(results))
        raw_p_values.append(float(test.pvalue))
        results.append(
            FeatureDriftResult(
                feature=feature,
                status="evaluated",
                reference_count=len(ref),
                current_count=len(cur),
                statistic=float(test.statistic),
                p_value=float(test.pvalue),
                adjusted_p_value=None,
                drift_detected=None,
            )
        )

    if raw_p_values:
        adjusted = false_discovery_control(raw_p_values, method="bh")
        for index, adjusted_p in zip(evaluated_indices, adjusted, strict=True):
            result = results[index]
            results[index] = FeatureDriftResult(
                feature=result.feature,
                status=result.status,
                reference_count=result.reference_count,
                current_count=result.current_count,
                statistic=result.statistic,
                p_value=result.p_value,
                adjusted_p_value=float(adjusted_p),
                drift_detected=bool(adjusted_p < alpha),
                reason=result.reason,
            )

    return results
