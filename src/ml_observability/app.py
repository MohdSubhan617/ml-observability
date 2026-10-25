"""FastAPI application entry point."""

import os
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ml_observability.drift import detect_numerical_drift
from ml_observability.incidents import build_incident
from ml_observability.monitoring import validate_schema

app = FastAPI(
    title="ML Observability",
    description="ML data drift detection and evidence-based incident triage.",
    version="0.1.0",
)


class RequestBodyLimitMiddleware:
    """Reject HTTP request bodies larger than the configured byte limit."""

    def __init__(self, app: ASGIApp, max_bytes: int | None = None) -> None:
        self.app = app
        raw_limit = os.getenv("ML_OBSERVABILITY_MAX_REQUEST_BYTES", "2000000")
        try:
            configured_limit = max_bytes if max_bytes is not None else int(raw_limit)
        except ValueError as exc:
            raise ValueError(
                "ML_OBSERVABILITY_MAX_REQUEST_BYTES must be a positive integer."
            ) from exc
        if configured_limit < 1:
            raise ValueError("ML_OBSERVABILITY_MAX_REQUEST_BYTES must be a positive integer.")
        self.max_bytes = configured_limit

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_lengths = [
            value for key, value in scope.get("headers", []) if key.lower() == b"content-length"
        ]
        has_transfer_encoding = any(
            key.lower() == b"transfer-encoding" for key, _ in scope.get("headers", [])
        )
        if content_lengths and has_transfer_encoding:
            await self._send_bad_request(
                send, "Content-Length and Transfer-Encoding cannot be combined."
            )
            return
        if len(content_lengths) > 1:
            await self._send_bad_request(send, "Multiple Content-Length headers are not allowed.")
            return
        declared_length: int | None = None
        if content_lengths:
            raw_content_length = content_lengths[0]
            if not raw_content_length.isdigit():
                await self._send_bad_request(send, "Invalid Content-Length header.")
                return
            declared_length = int(raw_content_length)
            if declared_length > self.max_bytes:
                await self._send_too_large(send)
                return

        messages: list[Message] = []
        total_bytes = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            total_bytes += len(message.get("body", b""))
            if total_bytes > self.max_bytes:
                await self._send_too_large(send)
                return
            messages.append(message)
            if not message.get("more_body", False):
                break

        if declared_length is not None and total_bytes != declared_length:
            await self._send_bad_request(send, "Request body length does not match Content-Length.")
            return

        replay = iter(messages)

        async def replay_receive() -> Message:
            try:
                return next(replay)
            except StopIteration:
                return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)

    async def _send_too_large(self, send: Send) -> None:
        body = b'{"detail":"Request body exceeds the configured size limit."}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def _send_bad_request(self, send: Send, detail: str) -> None:
        body = JSONResponse(status_code=400, content={"detail": detail}).body
        await send(
            {
                "type": "http.response.start",
                "status": 400,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


app.add_middleware(RequestBodyLimitMiddleware)


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return validation details without echoing untrusted request values."""
    details = [
        {key: error[key] for key in ("loc", "msg", "type") if key in error}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": details})


class ValidationRequest(BaseModel):
    """Reference and current datasets represented as JSON records."""

    reference: list[dict[str, Any]] = Field(min_length=1, max_length=10_000)
    current: list[dict[str, Any]] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_record_columns(self) -> "ValidationRequest":
        """Require consistent columns and bound per-record dimensions and values."""
        for dataset_name, records in (
            ("reference", self.reference),
            ("current", self.current),
        ):
            expected = set(records[0])
            if len(expected) > 100:
                raise ValueError(f"{dataset_name} records may contain at most 100 columns.")
            if any(len(column) > 128 for column in expected):
                raise ValueError(f"{dataset_name} column names may contain at most 128 characters.")
            for index, record in enumerate(records):
                if set(record) != expected:
                    raise ValueError(f"{dataset_name} record {index} has inconsistent columns.")
                if any(isinstance(value, str) and len(value) > 10_000 for value in record.values()):
                    raise ValueError(
                        f"{dataset_name} string values may contain at most 10000 characters."
                    )
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
    """Evaluate numerical drift and emit evidence-backed incidents."""
    try:
        results = detect_numerical_drift(
            pd.DataFrame(request.reference),
            pd.DataFrame(request.current),
            min_samples=request.min_samples,
            alpha=request.alpha,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    incidents = []
    for result in results:
        if result.status != "evaluated" or result.drift_detected is not True:
            continue

        incidents.append(
            build_incident(
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
                thresholds={"alpha": request.alpha},
                evidence=(
                    f"Feature {result.feature!r} differs between reference "
                    f"and current distributions (KS statistic={result.statistic}, "
                    f"adjusted p-value={result.adjusted_p_value}).",
                ),
                recommended_steps=(
                    "Inspect recent upstream data and feature-generation changes.",
                    "Assess whether the observed shift affects model performance.",
                ),
                limitations=(
                    "Distributional difference alone does not establish root cause "
                    "or model-performance impact.",
                    "Severity is informational until an operational impact policy is defined.",
                ),
            )
        )

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
        "incidents": [
            {
                "incident_id": incident.incident_id,
                "detector": incident.detector,
                "category": incident.category,
                "scope": incident.scope,
                "severity": incident.severity,
                "status": incident.status,
                "observed_metrics": dict(incident.observed_metrics),
                "thresholds": dict(incident.thresholds),
                "evidence": list(incident.evidence),
                "recommended_steps": list(incident.recommended_steps),
                "limitations": list(incident.limitations),
            }
            for incident in incidents
        ],
    }
