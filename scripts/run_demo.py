"""Run a deterministic synthetic monitoring-to-investigation walkthrough."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_observability.drift import detect_numerical_drift
from ml_observability.incidents import build_incident
from ml_observability.investigator import (
    DriftEvidence,
    InvestigationRequest,
    SchemaEvidence,
    investigate,
)
from ml_observability.monitoring import validate_schema


def run_demo(seed: int = 42) -> dict[str, Any]:
    """Generate synthetic telemetry and return the complete monitoring analysis."""
    rng = np.random.default_rng(seed)
    sample_size = 200
    reference = pd.DataFrame(
        {
            "account_age_days": rng.normal(loc=30, scale=8, size=sample_size),
            "balance": rng.normal(loc=500, scale=100, size=sample_size),
        }
    )
    current = pd.DataFrame(
        {
            "account_age_days": rng.normal(loc=40, scale=8, size=sample_size),
            "balance": rng.normal(loc=500, scale=100, size=sample_size),
            "campaign_source": ["synthetic-campaign"] * sample_size,
        }
    )
    current.loc[:11, "balance"] = np.nan

    schema_result = validate_schema(reference, current)
    drift_results = detect_numerical_drift(reference, current, min_samples=20, alpha=0.05)
    schema_evidence = SchemaEvidence(
        kind="schema_validation",
        evidence_id="demo-schema-001",
        is_valid=schema_result.is_valid,
        missing_columns=[str(column) for column in schema_result.missing_columns],
        unexpected_columns=[str(column) for column in schema_result.unexpected_columns],
        missing_values={
            str(column): count for column, count in schema_result.missing_values.items()
        },
    )
    drift_evidence = [
        DriftEvidence(
            kind="feature_drift",
            evidence_id=f"demo-drift-{result.feature}",
            feature=result.feature,
            status=result.status,
            drift_detected=result.drift_detected,
            adjusted_p_value=result.adjusted_p_value,
            alpha=0.05,
            reference_count=result.reference_count,
            current_count=result.current_count,
            statistic=result.statistic,
            p_value=result.p_value,
            reason=result.reason,
        )
        for result in drift_results
    ]
    investigation = investigate(InvestigationRequest(evidence=[schema_evidence, *drift_evidence]))

    incidents = []
    for result in drift_results:
        if result.status != "evaluated" or result.drift_detected is not True:
            continue
        if result.statistic is None or result.p_value is None or result.adjusted_p_value is None:
            raise RuntimeError("An evaluated positive drift result omitted a required metric.")
        incident = build_incident(
            detector="ks_test",
            category="feature_drift",
            scope=result.feature,
            severity="info",
            observed_metrics={
                "reference_count": result.reference_count,
                "current_count": result.current_count,
                "statistic": result.statistic,
                "p_value": result.p_value,
                "adjusted_p_value": result.adjusted_p_value,
            },
            thresholds={"alpha": 0.05},
            evidence=(
                f"Feature {result.feature!r} differs between synthetic reference and current data.",
            ),
            recommended_steps=(
                "Inspect the synthetic feature-generation change.",
                "Assess model performance before assigning operational impact.",
            ),
            limitations=(
                "Distributional difference does not establish root cause or model-performance "
                "impact.",
            ),
        )
        incidents.append(
            {
                "incident_id": incident.incident_id,
                "category": incident.category,
                "scope": incident.scope,
                "severity": incident.severity,
                "observed_metrics": dict(incident.observed_metrics),
                "thresholds": dict(incident.thresholds),
                "evidence": list(incident.evidence),
                "recommended_steps": list(incident.recommended_steps),
                "limitations": list(incident.limitations),
            }
        )

    expected = {
        "schema_contract_violation",
        "missingness",
        "feature_distribution_shift",
    }
    observed = {item.category for item in investigation.hypotheses}
    if not expected.issubset(observed):
        raise RuntimeError(
            "The injected schema, missingness, and location-shift signals were not all detected."
        )

    return {
        "seed": seed,
        "sample_size_per_dataset": sample_size,
        "injected_changes": {
            "account_age_days_location_shift": 10.0,
            "balance_null_count": 12,
            "unexpected_column": "campaign_source",
        },
        "schema_validation": {
            "is_valid": schema_result.is_valid,
            "missing_columns": list(schema_result.missing_columns),
            "unexpected_columns": list(schema_result.unexpected_columns),
            "missing_values": schema_result.missing_values,
        },
        "drift_results": [
            {
                "feature": result.feature,
                "status": result.status,
                "reference_count": result.reference_count,
                "current_count": result.current_count,
                "statistic": result.statistic,
                "p_value": result.p_value,
                "adjusted_p_value": result.adjusted_p_value,
                "drift_detected": result.drift_detected,
                "reason": result.reason,
            }
            for result in drift_results
        ],
        "incidents": incidents,
        "investigation": investigation.model_dump(mode="json"),
    }


def render_report(result: dict[str, Any]) -> str:
    """Render the machine-readable run as a short human-readable report."""
    schema = result["schema_validation"]
    lines = [
        "# Synthetic ML observability demo",
        "",
        f"Seed: `{result['seed']}`  ",
        f"Rows per dataset: `{result['sample_size_per_dataset']}`",
        "",
        "## Injected signals",
        "",
        "- Account age location shift: "
        f"`{result['injected_changes']['account_age_days_location_shift']}`",
        f"- Null balance values: `{result['injected_changes']['balance_null_count']}`",
        f"- Unexpected column: `{result['injected_changes']['unexpected_column']}`",
        "",
        "## Monitoring results",
        "",
        f"- Schema valid: `{schema['is_valid']}`",
        f"- Missing columns: `{schema['missing_columns']}`",
        f"- Unexpected columns: `{schema['unexpected_columns']}`",
        f"- Null counts: `{schema['missing_values']}`",
        "",
        "| Feature | Status | Adjusted p-value | Drift detected |",
        "|---|---|---:|:---:|",
    ]
    for item in result["drift_results"]:
        adjusted_p = item["adjusted_p_value"]
        rendered_p = "N/A" if adjusted_p is None else f"{adjusted_p:.6g}"
        lines.append(
            f"| {item['feature']} | {item['status']} | {rendered_p} | {item['drift_detected']} |"
        )
    lines.extend(["", "## Incidents", ""])
    if result["incidents"]:
        lines.extend(
            f"- `{item['category']}` on `{item['scope']}` (severity `{item['severity']}`)"
            for item in result["incidents"]
        )
    else:
        lines.append("- None")
    investigation = result["investigation"]
    lines.extend(
        [
            "",
            "## Investigation",
            "",
            f"- Analysis fingerprint: `{investigation['analysis_id']}`",
            f"- Abstained: `{investigation['abstained']}`",
        ]
    )
    if investigation["abstention_reason"]:
        lines.append(f"- Abstention reason: {investigation['abstention_reason']}")
    for hypothesis in investigation["hypotheses"]:
        reference_ids = [
            reference["evidence_id"] for reference in hypothesis["supporting_evidence"]
        ]
        lines.append(
            f"- `{hypothesis['category']}`: {hypothesis['explanation']} "
            f"(heuristic points: {hypothesis['score']}; evidence: {reference_ids})"
        )
    lines.extend(
        [
            "",
            "> Synthetic contract walkthrough only. It does not demonstrate production "
            "diagnostic accuracy or model-performance impact.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optionally save the complete machine-readable report to this path.",
    )
    args = parser.parse_args()

    result = run_demo(seed=args.seed)
    if args.json_output is not None:
        args.json_output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(render_report(result), end="")


if __name__ == "__main__":
    main()
