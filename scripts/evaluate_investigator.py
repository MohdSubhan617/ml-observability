"""Reproducibly evaluate the investigator against small synthetic observations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ml_observability.drift import detect_numerical_drift
from ml_observability.investigator import (
    DriftEvidence,
    InvestigationRequest,
    InvestigationResult,
    SchemaEvidence,
    investigate,
)
from ml_observability.monitoring import validate_schema

Mode = Literal["single", "multiple", "abstain"]


@dataclass(frozen=True)
class Scenario:
    name: str
    category: str
    request: InvestigationRequest
    expected: tuple[str, ...]
    mode: Mode


def _ratio_or_none(numerator: float, denominator: int) -> float | None:
    """Return an undefined metric as None when its denominator is empty."""
    return numerator / denominator if denominator else None


def _observations(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    prefix: str,
    *,
    seed: int,
) -> list[DriftEvidence | SchemaEvidence]:
    findings: list[DriftEvidence | SchemaEvidence] = []
    schema = validate_schema(reference, current)
    findings.append(
        SchemaEvidence(
            kind="schema_validation",
            evidence_id=f"{prefix}-schema-{seed}",
            missing_columns=[str(value) for value in schema.missing_columns],
            unexpected_columns=[str(value) for value in schema.unexpected_columns],
            missing_values={str(key): value for key, value in schema.missing_values.items()},
        )
    )
    for index, result in enumerate(
        detect_numerical_drift(reference, current, min_samples=20, alpha=0.05)
    ):
        findings.append(
            DriftEvidence(
                kind="feature_drift",
                evidence_id=f"{prefix}-drift-{index}-{seed}",
                feature=result.feature,
                status=result.status,
                drift_detected=result.drift_detected,
                adjusted_p_value=result.adjusted_p_value,
                alpha=0.05,
            )
        )
    return findings


def _paired(
    seed: int, *, features: int = 3, shift: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_rng = np.random.default_rng(seed)
    current_rng = np.random.default_rng(seed + 1000)
    columns = [f"feature_{index}" for index in range(features)]
    reference = pd.DataFrame(reference_rng.normal(size=(120, features)), columns=columns)
    current = pd.DataFrame(current_rng.normal(size=(120, features)), columns=columns)
    if shift:
        current["feature_0"] += shift
    return reference, current


def build_scenarios(seed: int = 42) -> list[Scenario]:
    """Build named scenarios from generated tabular data and real project detectors."""
    scenarios: list[Scenario] = []

    reference, current = _paired(seed, shift=1.25)
    findings = _observations(reference, current, "shift", seed=seed)
    scenarios.append(
        Scenario(
            "clear_feature_shift",
            "feature_distribution_shift",
            InvestigationRequest(evidence=findings),
            ("feature_distribution_shift",),
            "single",
        )
    )

    reference, current = _paired(seed + 1)
    current.loc[current.index[::3], "feature_0"] = np.nan
    findings = _observations(reference, current, "missing", seed=seed)
    scenarios.append(
        Scenario(
            "missingness",
            "missingness",
            InvestigationRequest(evidence=findings),
            ("missingness",),
            "single",
        )
    )

    reference, current = _paired(seed + 2, features=2)
    current = current.drop(columns="feature_1")
    findings = _observations(reference, current, "schema", seed=seed)
    scenarios.append(
        Scenario(
            "schema_contract_failure",
            "schema_contract_violation",
            InvestigationRequest(evidence=findings),
            ("schema_contract_violation",),
            "single",
        )
    )

    # This service has no prediction distribution, version, pipeline, or label signals.
    scenarios.append(
        Scenario(
            "unsupported_prediction_anomaly",
            "unsupported_prediction_anomaly",
            InvestigationRequest(),
            (),
            "abstain",
        )
    )
    scenarios.append(
        Scenario(
            "unsupported_model_version_change",
            "unsupported_model_version_change",
            InvestigationRequest(),
            (),
            "abstain",
        )
    )

    reference, current = _paired(seed + 3, features=3, shift=1.25)
    current = current.drop(columns="feature_1")
    current.loc[current.index[::3], "feature_2"] = np.nan
    findings = _observations(reference, current, "multi", seed=seed)
    scenarios.append(
        Scenario(
            "multiple_simultaneous_findings",
            "multiple_simultaneous_findings",
            InvestigationRequest(evidence=findings),
            ("feature_distribution_shift", "schema_contract_violation", "missingness"),
            "multiple",
        )
    )

    reference, current = _paired(seed + 4, features=2, shift=1.25)
    findings = _observations(reference, current, "ambiguous", seed=seed)
    findings = [item for item in findings if isinstance(item, DriftEvidence)]
    findings.append(
        SchemaEvidence(
            kind="schema_validation",
            evidence_id=f"ambiguous-null-{seed}",
            missing_values={"feature_1": 3},
        )
    )
    scenarios.append(
        Scenario(
            "equal_score_cooccurring_findings",
            "equal_score_cooccurring_findings",
            InvestigationRequest(evidence=findings),
            ("feature_distribution_shift", "missingness"),
            "multiple",
        )
    )

    reference, current = _paired(seed + 5, features=1, shift=1.25)
    findings = _observations(reference, current, "conflict", seed=seed)
    detected = next(item for item in findings if isinstance(item, DriftEvidence))
    findings.append(
        DriftEvidence(
            kind="feature_drift",
            evidence_id=f"conflict-opposite-{seed}",
            feature=detected.feature,
            status="evaluated",
            drift_detected=not detected.drift_detected,
            adjusted_p_value=0.99 if detected.drift_detected else 0.001,
            alpha=0.05,
        )
    )
    scenarios.append(
        Scenario(
            "conflicting_same_feature_results",
            "conflicting_same_feature_results",
            InvestigationRequest(evidence=findings),
            ("feature_distribution_shift",),
            "abstain",
        )
    )

    reference, current = _paired(seed + 6)
    reference = reference.iloc[:10]
    current = current.iloc[:10]
    findings = _observations(reference, current, "insufficient", seed=seed)
    scenarios.append(
        Scenario(
            "insufficient_samples",
            "insufficient_evidence",
            InvestigationRequest(evidence=findings),
            (),
            "abstain",
        )
    )

    reference, current = _paired(seed + 7)
    findings = _observations(reference, current, "normal", seed=seed)
    scenarios.append(
        Scenario(
            "normal_operation",
            "normal_operation",
            InvestigationRequest(evidence=findings),
            (),
            "abstain",
        )
    )
    return scenarios


def evaluate(seed: int = 42) -> dict[str, Any]:
    """Compute diagnostic and transparent first-supported-category baseline metrics."""
    scenarios = build_scenarios(seed)
    rows: list[dict[str, Any]] = []
    grounding_violations = 0
    evidence_reference_count = 0
    for scenario in scenarios:
        result: InvestigationResult = investigate(scenario.request)
        supplied_ids = {item.evidence_id for item in scenario.request.evidence}
        references = [
            ref
            for hypothesis in result.hypotheses
            for ref in (*hypothesis.supporting_evidence, *hypothesis.contradictory_evidence)
        ]
        evidence_reference_count += len(references)
        grounding_violations += sum(ref.evidence_id not in supplied_ids for ref in references)
        top = result.hypotheses[0].category if result.hypotheses else None
        baseline = next(
            (
                hypothesis.category
                for hypothesis in sorted(result.hypotheses, key=lambda item: item.category)
                if hypothesis.supporting_evidence
            ),
            None,
        )
        baseline_ranked = sorted(result.hypotheses, key=lambda item: item.category)
        baseline_reciprocal_rank = next(
            (
                1 / rank
                for rank, hypothesis in enumerate(baseline_ranked, start=1)
                if hypothesis.category in scenario.expected
            ),
            0.0,
        )
        reciprocal_rank = next(
            (
                1 / rank
                for rank, hypothesis in enumerate(result.hypotheses, start=1)
                if hypothesis.category in scenario.expected
            ),
            0.0,
        )
        rows.append(
            {
                "scenario": scenario.name,
                "category": scenario.category,
                "expected_mode": scenario.mode,
                "expected_hypotheses": list(scenario.expected),
                "predicted_top1": top,
                "predicted_hypotheses": [item.category for item in result.hypotheses],
                "predicted_abstain": result.abstained,
                "abstention_reason": result.abstention_reason,
                "baseline_top1": baseline,
                "baseline_hit_at_3": any(
                    item.category in scenario.expected for item in baseline_ranked[:3]
                ),
                "baseline_reciprocal_rank": baseline_reciprocal_rank,
                "hit_at_3": any(
                    item.category in scenario.expected for item in result.hypotheses[:3]
                ),
                "reciprocal_rank": reciprocal_rank,
                "grounded": all(
                    ref.evidence_id in supplied_ids
                    for hypothesis in result.hypotheses
                    for ref in (*hypothesis.supporting_evidence, *hypothesis.contradictory_evidence)
                ),
            }
        )

    single = [
        row for row, scenario in zip(rows, scenarios, strict=True) if scenario.mode == "single"
    ]
    multiple = [
        row for row, scenario in zip(rows, scenarios, strict=True) if scenario.mode == "multiple"
    ]
    rankable = [
        row for row, scenario in zip(rows, scenarios, strict=True) if scenario.mode != "abstain"
    ]
    abstain_expected = [
        row for row, scenario in zip(rows, scenarios, strict=True) if scenario.mode == "abstain"
    ]
    abstain_tp = sum(bool(row["predicted_abstain"]) for row in abstain_expected)
    abstain_fp = sum(
        bool(row["predicted_abstain"])
        for row, scenario in zip(rows, scenarios, strict=True)
        if scenario.mode != "abstain"
    )
    abstain_fn = len(abstain_expected) - abstain_tp
    normal = [
        row
        for row, scenario in zip(rows, scenarios, strict=True)
        if scenario.name == "normal_operation"
    ]
    unsupported = [
        row
        for row, scenario in zip(rows, scenarios, strict=True)
        if scenario.category.startswith("unsupported_")
    ]
    per_category: dict[str, dict[str, int]] = {}
    for category in sorted({scenario.category for scenario in scenarios}):
        selected = [
            row
            for row, scenario in zip(rows, scenarios, strict=True)
            if scenario.category == category
        ]
        per_category[category] = {
            "scenarios": len(selected),
            "top1_expected_hit": sum(
                row["predicted_top1"] in row["expected_hypotheses"] for row in selected
            ),
            "abstained": sum(bool(row["predicted_abstain"]) for row in selected),
        }

    return {
        "seed": seed,
        "scenario_count": len(rows),
        "metrics": {
            "single_target_top1_accuracy": _ratio_or_none(
                sum(row["predicted_top1"] in row["expected_hypotheses"] for row in single),
                len(single),
            ),
            "single_target_denominator": len(single),
            "multiple_target_hit_at_3": _ratio_or_none(
                sum(bool(row["hit_at_3"]) for row in multiple), len(multiple)
            ),
            "multiple_target_denominator": len(multiple),
            "mean_reciprocal_rank": _ratio_or_none(
                sum(float(row["reciprocal_rank"]) for row in rankable), len(rankable)
            ),
            "mrr_denominator_rankable_scenarios": len(rankable),
            "abstention_precision": _ratio_or_none(abstain_tp, abstain_tp + abstain_fp),
            "abstention_precision_denominator": abstain_tp + abstain_fp,
            "abstention_recall": _ratio_or_none(abstain_tp, abstain_tp + abstain_fn),
            "abstention_recall_denominator": abstain_tp + abstain_fn,
            "abstention_expected_denominator": len(abstain_expected),
            "false_positive_diagnosis_rate_normal": _ratio_or_none(
                sum(row["predicted_top1"] is not None for row in normal), len(normal)
            ),
            "normal_scenario_denominator": len(normal),
            "evidence_grounding_violations": grounding_violations,
            "evidence_reference_denominator": evidence_reference_count,
            "unsupported_category_diagnoses": sum(
                row["predicted_top1"] is not None for row in unsupported
            ),
            "unsupported_category_scenario_denominator": len(unsupported),
            "baseline_single_target_top1_accuracy": _ratio_or_none(
                sum(row["baseline_top1"] in row["expected_hypotheses"] for row in single),
                len(single),
            ),
            "baseline_multiple_target_hit_at_3": _ratio_or_none(
                sum(bool(row["baseline_hit_at_3"]) for row in multiple), len(multiple)
            ),
            "baseline_mrr": _ratio_or_none(
                sum(float(row["baseline_reciprocal_rank"]) for row in rankable),
                len(rankable),
            ),
        },
        "per_category": per_category,
        "scenarios": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-output", type=str)
    args = parser.parse_args()
    report = evaluate(args.seed)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as output:
            output.write(encoded + "\n")
    print("ML Failure Investigator synthetic evaluation")
    for name, value in report["metrics"].items():
        print(f"{name}: {value}")
    print(encoded)


if __name__ == "__main__":
    main()
