"""Regression tests for diagnostic evaluation metrics and repeatability."""

from scripts.evaluate_investigator import _ratio_or_none, evaluate


def test_synthetic_investigation_evaluation_is_reproducible() -> None:
    first = evaluate(seed=42)
    second = evaluate(seed=42)

    assert first == second
    assert first["scenario_count"] == 10
    metrics = first["metrics"]
    assert metrics["single_target_denominator"] == 3
    assert metrics["multiple_target_denominator"] == 2
    assert metrics["mrr_denominator_rankable_scenarios"] == 5
    assert metrics["normal_scenario_denominator"] == 1
    assert metrics["evidence_grounding_violations"] == 0


def test_evaluation_reports_abstention_and_unsupported_case_metrics() -> None:
    report = evaluate(seed=42)
    metrics = report["metrics"]

    assert metrics["abstention_expected_denominator"] == 5
    assert metrics["false_positive_diagnosis_rate_normal"] == 0.0
    assert metrics["unsupported_category_diagnoses"] == 0


def test_ineligible_metric_denominator_returns_none() -> None:
    assert _ratio_or_none(0, 0) is None
    assert _ratio_or_none(1, 2) == 0.5
