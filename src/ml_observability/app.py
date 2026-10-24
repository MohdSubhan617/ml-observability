"""FastAPI application entry point."""

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from ml_observability.drift import detect_numerical_drift
from ml_observability.monitoring import validate_schema

app = FastAPI(
    title="ML Observability",
    description="ML data drift detection and evidence-based incident triage.",
    version="0.1.0",
)


class ValidationRequest(BaseModel):
    """Reference and current datasets represented as JSON records."""

    reference: list[dict[str, Any]] = Field(min_length=1, max_length=10_000)
    current: list[dict[str, Any]] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_record_columns(self) -> "ValidationRequest":
        """Require consistent columns within each dataset."""
        for dataset_name, records in (
            ("reference", self.reference),
            ("current", self.current),
        ):
            expected = set(records[0])
            for index, record in enumerate(records):
                if set(record) != expected:
                    raise ValueError(f"{dataset_name} record {index} has inconsistent columns.")
        return self


class DriftRequest(ValidationRequest):
    """Dataset comparison settings for numerical drift detection."""

    min_samples: int = Field(default=20, ge=2, le=100_000)
    alpha: float = Field(default=0.05, gt=0, lt=1)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Report whether the API process is responding."""
    return {"status": "ok"}


@app.post("/validate", tags=["monitoring"])
def validate_datasets(request: ValidationRequest) -> dict[str, Any]:
    """Validate current dataset schema and missing values against reference."""
    try:
        result = validate_schema(
            pd.DataFrame(request.reference),
            pd.DataFrame(request.current),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "is_valid": result.is_valid,
        "missing_columns": list(result.missing_columns),
        "unexpected_columns": list(result.unexpected_columns),
        "missing_values": result.missing_values,
    }


@app.post("/drift", tags=["monitoring"])
def detect_drift(request: DriftRequest) -> dict[str, Any]:
    """Evaluate numerical feature drift and return adjusted statistical results."""
    try:
        results = detect_numerical_drift(
            pd.DataFrame(request.reference),
            pd.DataFrame(request.current),
            min_samples=request.min_samples,
            alpha=request.alpha,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "alpha": request.alpha,
        "multiple_testing_correction": "benjamini_hochberg",
        "features": [
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
            for result in results
        ],
    }
