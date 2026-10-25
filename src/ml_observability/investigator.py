"""Deterministic, evidence-grounded diagnosis over existing monitoring signals."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EvidenceId = Annotated[str, Field(min_length=1, max_length=64)]
FeatureName = Annotated[str, Field(min_length=1, max_length=128)]
ColumnNames = Annotated[list[FeatureName], Field(max_length=100)]
EvidenceList = Annotated[list["Evidence"], Field(max_length=200)]


class ContractModel(BaseModel):
    """Strict model shared by the transport contract and diagnostic result."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)


class SchemaEvidence(ContractModel):
    """Schema validation output matching the existing `/validate` endpoint."""

    kind: Literal["schema_validation"]
    evidence_id: EvidenceId
    is_valid: bool | None = None
    missing_columns: ColumnNames = Field(default_factory=list)
    unexpected_columns: ColumnNames = Field(default_factory=list)
    missing_values: dict[FeatureName, Annotated[int, Field(ge=1)]] = Field(
        default_factory=dict, max_length=100
    )

    @model_validator(mode="after")
    def validate_reported_validity(self) -> SchemaEvidence:
        """If copied from `/validate`, require its summary flag to match findings."""
        findings_are_clean = not (
            self.missing_columns or self.unexpected_columns or self.missing_values
        )
        if self.is_valid is not None and self.is_valid != findings_are_clean:
            raise ValueError("is_valid must match the supplied schema findings.")
        return self


class DriftEvidence(ContractModel):
    """One feature result matching the existing `/drift` endpoint."""

    kind: Literal["feature_drift"]
    evidence_id: EvidenceId
    feature: FeatureName
    status: Literal["evaluated", "not_evaluated"]
    drift_detected: bool | None
    adjusted_p_value: Annotated[float, Field(ge=0, le=1)] | None
    alpha: Annotated[float, Field(gt=0, lt=1)]
    reference_count: Annotated[int, Field(ge=0)] | None = None
    current_count: Annotated[int, Field(ge=0)] | None = None
    statistic: Annotated[float, Field(ge=0, le=1)] | None = None
    p_value: Annotated[float, Field(ge=0, le=1)] | None = None
    reason: Annotated[str, Field(max_length=256)] | None = None

    @model_validator(mode="after")
    def validate_result_consistency(self) -> DriftEvidence:
        """Reject impossible combinations instead of inferring from malformed results."""
        if self.status == "not_evaluated":
            if self.drift_detected is not None or self.adjusted_p_value is not None:
                raise ValueError(
                    "not_evaluated drift evidence must not include a decision or p-value."
                )
        elif self.drift_detected is None or self.adjusted_p_value is None:
            raise ValueError("evaluated drift evidence requires a decision and adjusted p-value.")
        elif self.drift_detected != (self.adjusted_p_value < self.alpha):
            raise ValueError("drift decision must match adjusted p-value and alpha.")
        return self


Evidence = Annotated[SchemaEvidence | DriftEvidence, Field(discriminator="kind")]


class InvestigationRequest(ContractModel):
    """Bounded collection of outputs from the existing validation and drift checks."""

    evidence: EvidenceList = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence_ids(self) -> InvestigationRequest:
        ids = [item.evidence_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence_id values must be unique within an investigation.")
        return self


class EvidenceReference(ContractModel):
    """Traceable, bounded observation derived from a supplied evidence item."""

    evidence_id: EvidenceId
    observation: Annotated[str, Field(min_length=1, max_length=256)]


class Hypothesis(ContractModel):
    """Ranked diagnostic explanation and its evidence trace."""

    category: Literal["feature_distribution_shift", "schema_contract_violation", "missingness"]
    explanation: Annotated[str, Field(min_length=1, max_length=256)]
    score: Annotated[int, Field(ge=0, le=10)]
    score_type: Literal["heuristic_points"] = "heuristic_points"
    evidence_strength: Literal["limited", "moderate", "strong"]
    supporting_evidence: Annotated[list[EvidenceReference], Field(max_length=200)]
    contradictory_evidence: Annotated[list[EvidenceReference], Field(max_length=200)]


class InvestigationResult(ContractModel):
    """Stable analysis result with explicit abstention and follow-up guidance."""

    diagnostic_version: Literal["1"] = "1"
    analysis_id: Annotated[str, Field(pattern=r"^analysis_[0-9a-f]{24}$")]
    hypotheses: Annotated[list[Hypothesis], Field(max_length=3)]
    abstained: bool
    abstention_reason: Annotated[str, Field(max_length=256)] | None
    missing_signals: Annotated[list[str], Field(max_length=8)]


_CATEGORY_ORDER = {
    "feature_distribution_shift": 0,
    "schema_contract_violation": 1,
    "missingness": 2,
}


def _observation(item: Evidence, text: str) -> EvidenceReference:
    return EvidenceReference(evidence_id=item.evidence_id, observation=text[:256])


def _strength(
    score: int,
    contradictions: int,
    supporting_record_count: int,
) -> Literal["limited", "moderate", "strong"]:
    """Describe rule support, not probability or calibrated statistical confidence."""
    if contradictions or score < 4:
        return "limited"
    if score < 8 or supporting_record_count < 2:
        return "moderate"
    return "strong"


def _hypothesis(
    category: Literal["feature_distribution_shift", "schema_contract_violation", "missingness"],
    explanation: str,
    supporting: list[EvidenceReference],
    contradictory: list[EvidenceReference],
    scopes: set[str],
) -> Hypothesis:
    # One point per distinct feature/column scope after the four-point initial finding;
    # repeated evidence about the same scope cannot inflate this heuristic score.
    score = min(10, 4 + max(0, len(scopes) - 1))
    score = max(0, score - 3 * len(contradictory))
    return Hypothesis(
        category=category,
        explanation=explanation,
        score=score,
        evidence_strength=_strength(
            score,
            len(contradictory),
            len({item.evidence_id for item in supporting}),
        ),
        supporting_evidence=sorted(supporting, key=lambda item: item.evidence_id),
        contradictory_evidence=sorted(contradictory, key=lambda item: item.evidence_id),
    )


def investigate(request: InvestigationRequest) -> InvestigationResult:
    """Rank only diagnoses directly supported by schema and KS detector outputs."""
    drift = [item for item in request.evidence if isinstance(item, DriftEvidence)]
    schemas = [item for item in request.evidence if isinstance(item, SchemaEvidence)]

    positive_drift = [item for item in drift if item.status == "evaluated" and item.drift_detected]
    positive_features = {item.feature for item in positive_drift}
    negative_drift = [
        item
        for item in drift
        if item.status == "evaluated"
        and item.drift_detected is False
        and item.feature in positive_features
    ]
    hypotheses: list[Hypothesis] = []
    if positive_drift:
        hypotheses.append(
            _hypothesis(
                "feature_distribution_shift",
                "The supplied KS results report statistically significant feature "
                "distribution differences.",
                [
                    _observation(
                        item,
                        f"Feature {item.feature!r}: adjusted p-value "
                        f"{item.adjusted_p_value:g} is below alpha {item.alpha:g}.",
                    )
                    for item in positive_drift
                ],
                [
                    _observation(
                        item,
                        f"Feature {item.feature!r}: evaluated with no detected drift "
                        f"(adjusted p-value {item.adjusted_p_value:g}).",
                    )
                    for item in negative_drift
                ],
                {item.feature for item in positive_drift},
            )
        )

    schema_support = [item for item in schemas if item.missing_columns or item.unexpected_columns]
    schema_clean = [
        item for item in schemas if not item.missing_columns and not item.unexpected_columns
    ]
    if schema_support:
        references = [
            _observation(
                item,
                "Schema mismatch: "
                + "; ".join(
                    part
                    for part in (
                        f"missing columns {item.missing_columns!r}" if item.missing_columns else "",
                        f"unexpected columns {item.unexpected_columns!r}"
                        if item.unexpected_columns
                        else "",
                    )
                    if part
                ),
            )
            for item in schema_support
        ]
        hypotheses.append(
            _hypothesis(
                "schema_contract_violation",
                "The supplied schema validation results report missing or unexpected columns.",
                references,
                [
                    _observation(item, "Schema validation found no missing or unexpected columns.")
                    for item in schema_clean
                ],
                {
                    column
                    for item in schema_support
                    for column in (*item.missing_columns, *item.unexpected_columns)
                },
            )
        )

    missing_support = [item for item in schemas if item.missing_values]
    missing_clean = [item for item in schemas if not item.missing_values]
    if missing_support:
        hypotheses.append(
            _hypothesis(
                "missingness",
                "The supplied schema validation results report null values in current data.",
                [
                    _observation(
                        item,
                        "Null counts: "
                        + ", ".join(
                            f"{column!r}={count}"
                            for column, count in sorted(item.missing_values.items())
                        ),
                    )
                    for item in missing_support
                ],
                [
                    _observation(item, "Schema validation found no null values in current data.")
                    for item in missing_clean
                ],
                {column for item in missing_support for column in item.missing_values},
            )
        )

    hypotheses.sort(key=lambda item: (-item.score, _CATEGORY_ORDER[item.category]))
    canonical_evidence = sorted(
        (item.model_dump(mode="json") for item in request.evidence),
        key=lambda item: (str(item["kind"]), str(item["evidence_id"])),
    )
    payload = json.dumps(canonical_evidence, sort_keys=True, separators=(",", ":"))
    analysis_id = f"analysis_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"

    has_conflict = any(item.contradictory_evidence for item in hypotheses)
    if not hypotheses:
        abstention_reason = "No supported schema or evaluated feature-drift findings were supplied."
    elif has_conflict:
        abstention_reason = (
            "Supplied observations conflict; review the supporting and contradictory evidence."
        )
    else:
        abstention_reason = None

    return InvestigationResult(
        analysis_id=analysis_id,
        hypotheses=hypotheses,
        abstained=abstention_reason is not None,
        abstention_reason=abstention_reason,
        missing_signals=[
            "prediction distribution and performance metrics",
            "model and model-version metadata",
            "upstream pipeline health and change history",
            "outcomes or labels for model-quality diagnosis",
        ],
    )
