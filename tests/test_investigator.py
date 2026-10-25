"""Unit tests for deterministic evidence-grounded diagnosis."""

import pytest
from pydantic import ValidationError

from ml_observability.investigator import (
    DriftEvidence,
    InvestigationRequest,
    SchemaEvidence,
    investigate,
)


def drift(evidence_id: str, *, feature: str = "income", detected: bool = True) -> DriftEvidence:
    return DriftEvidence(
        kind="feature_drift",
        evidence_id=evidence_id,
        feature=feature,
        status="evaluated",
        drift_detected=detected,
        adjusted_p_value=0.001 if detected else 0.9,
        alpha=0.05,
    )


def schema(
    evidence_id: str,
    *,
    missing_columns: list[str] | None = None,
    unexpected_columns: list[str] | None = None,
    missing_values: dict[str, int] | None = None,
) -> SchemaEvidence:
    return SchemaEvidence(
        kind="schema_validation",
        evidence_id=evidence_id,
        missing_columns=missing_columns or [],
        unexpected_columns=unexpected_columns or [],
        missing_values=missing_values or {},
    )


def test_feature_drift_hypothesis_cites_actual_drift_result() -> None:
    result = investigate(InvestigationRequest(evidence=[drift("d1")]))

    assert result.hypotheses[0].category == "feature_distribution_shift"
    assert result.hypotheses[0].supporting_evidence[0].evidence_id == "d1"
    assert result.hypotheses[0].score == 4
    assert result.hypotheses[0].score_type == "heuristic_points"
    assert not result.abstained


@pytest.mark.parametrize(
    ("finding", "category"),
    [
        (schema("s1", missing_columns=["age"]), "schema_contract_violation"),
        (schema("s2", unexpected_columns=["region"]), "schema_contract_violation"),
        (schema("s3", missing_values={"age": 2}), "missingness"),
    ],
)
def test_schema_and_missingness_categories_are_supported(
    finding: SchemaEvidence, category: str
) -> None:
    result = investigate(InvestigationRequest(evidence=[finding]))

    assert [item.category for item in result.hypotheses] == [category]
    assert result.hypotheses[0].supporting_evidence[0].evidence_id == finding.evidence_id


def test_multiple_causes_are_returned_and_ranked_deterministically() -> None:
    evidence = [
        drift("d1", feature="a"),
        drift("d2", feature="b"),
        schema("s1", missing_values={"c": 3}),
    ]

    first = investigate(InvestigationRequest(evidence=evidence))
    second = investigate(InvestigationRequest(evidence=list(reversed(evidence))))

    assert [item.category for item in first.hypotheses] == [
        "feature_distribution_shift",
        "missingness",
    ]
    assert first.hypotheses[0].score == 5
    assert first.model_dump() == second.model_dump()
    assert not first.abstained


def test_one_large_signal_cannot_receive_strong_evidence_label() -> None:
    finding = schema("one-record", missing_columns=[f"column_{index}" for index in range(8)])

    result = investigate(InvestigationRequest(evidence=[finding]))

    assert result.hypotheses[0].score == 10
    assert result.hypotheses[0].evidence_strength == "moderate"


def test_strong_evidence_label_requires_multiple_evidence_records() -> None:
    evidence = [
        schema(
            f"record-{index}",
            missing_columns=[f"column_{index * 4 + offset}" for offset in range(4)],
        )
        for index in range(2)
    ]

    result = investigate(InvestigationRequest(evidence=evidence))

    assert result.hypotheses[0].score == 10
    assert result.hypotheses[0].evidence_strength == "strong"


def test_equal_score_hypotheses_use_stable_category_tie_break_without_dropping_results() -> None:
    result = investigate(
        InvestigationRequest(evidence=[drift("d1"), schema("s1", missing_values={"age": 1})])
    )

    assert not result.abstained
    assert result.abstention_reason is None
    assert [item.category for item in result.hypotheses] == [
        "feature_distribution_shift",
        "missingness",
    ]


def test_same_feature_conflicting_results_are_cited_and_abstain() -> None:
    result = investigate(
        InvestigationRequest(evidence=[drift("positive"), drift("negative", detected=False)])
    )
    hypothesis = result.hypotheses[0]

    assert result.abstained
    assert hypothesis.score == 1
    assert [item.evidence_id for item in hypothesis.contradictory_evidence] == ["negative"]
    assert hypothesis.evidence_strength == "limited"


def test_clean_observations_and_unevaluated_drift_produce_no_diagnosis() -> None:
    unevaluated = DriftEvidence(
        kind="feature_drift",
        evidence_id="d1",
        feature="age",
        status="not_evaluated",
        drift_detected=None,
        adjusted_p_value=None,
        alpha=0.05,
    )
    result = investigate(InvestigationRequest(evidence=[schema("s1"), unevaluated]))

    assert result.hypotheses == []
    assert result.abstained
    assert result.abstention_reason


def test_analysis_id_is_stable_for_same_evidence_in_any_input_order() -> None:
    evidence = [drift("d1"), schema("s1", missing_columns=["x"])]

    first = investigate(InvestigationRequest(evidence=evidence))
    second = investigate(InvestigationRequest(evidence=evidence[::-1]))

    assert first.analysis_id == second.analysis_id
    assert first.diagnostic_version == "1"


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        InvestigationRequest(evidence=[drift("same"), schema("same", missing_columns=["x"])])


def test_inconsistent_evaluated_drift_payload_is_rejected() -> None:
    with pytest.raises(ValidationError, match="requires a decision"):
        DriftEvidence(
            kind="feature_drift",
            evidence_id="d1",
            feature="x",
            status="evaluated",
            drift_detected=None,
            adjusted_p_value=None,
            alpha=0.05,
        )


def test_evaluated_decision_must_match_adjusted_p_value_threshold() -> None:
    with pytest.raises(ValidationError, match="must match adjusted p-value"):
        DriftEvidence(
            kind="feature_drift",
            evidence_id="d1",
            feature="x",
            status="evaluated",
            drift_detected=True,
            adjusted_p_value=0.5,
            alpha=0.05,
        )


def test_empty_and_oversized_evidence_are_bounded() -> None:
    assert investigate(InvestigationRequest()).abstained
    with pytest.raises(ValidationError):
        InvestigationRequest(evidence=[drift(f"id-{index}") for index in range(201)])


def test_feature_and_evidence_id_length_boundaries() -> None:
    accepted = drift("e" * 64, feature="f" * 128)
    assert accepted.feature == "f" * 128

    with pytest.raises(ValidationError):
        drift("e" * 65)
    with pytest.raises(ValidationError):
        drift("ok", feature="f" * 129)


def test_column_name_collection_accepts_100_and_rejects_101() -> None:
    accepted = schema("s1", missing_columns=[f"c{index}" for index in range(100)])
    assert len(accepted.missing_columns) == 100
    with pytest.raises(ValidationError):
        schema("s2", missing_columns=[f"c{index}" for index in range(101)])


def test_numeric_fields_reject_string_coercion() -> None:
    with pytest.raises(ValidationError):
        DriftEvidence(
            kind="feature_drift",
            evidence_id="d1",
            feature="x",
            status="evaluated",
            drift_detected=True,
            adjusted_p_value="0.01",  # type: ignore[arg-type]
            alpha=0.05,
        )


def test_copied_schema_summary_must_match_findings() -> None:
    with pytest.raises(ValidationError, match="is_valid must match"):
        SchemaEvidence(
            kind="schema_validation",
            evidence_id="s1",
            is_valid=True,
            missing_columns=["age"],
        )


def test_unsupported_output_categories_cannot_be_emitted() -> None:
    result = investigate(InvestigationRequest())

    assert all(
        item.category in {"feature_distribution_shift", "schema_contract_violation", "missingness"}
        for item in result.hypotheses
    )


def test_hypothesis_evidence_references_are_a_subset_of_input_ids() -> None:
    request = InvestigationRequest(evidence=[drift("d1"), schema("s1", missing_values={"x": 2})])
    result = investigate(request)
    ids = {item.evidence_id for item in request.evidence}

    assert all(
        reference.evidence_id in ids
        for hypothesis in result.hypotheses
        for reference in (*hypothesis.supporting_evidence, *hypothesis.contradictory_evidence)
    )
