"""Domain contract for evidence-backed ML monitoring incidents."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

Severity = Literal["info", "low", "medium", "high", "critical"]
IncidentStatus = Literal["open", "acknowledged", "resolved"]

type MetricValue = float | int | str | bool | None
type ThresholdValue = float | int | str | None

SEVERITIES = frozenset({"info", "low", "medium", "high", "critical"})


@dataclass(frozen=True, slots=True)
class Incident:
    """Immutable incident with traceable evidence and explicit uncertainty."""

    incident_id: str
    detector: str
    category: str
    scope: str
    severity: Severity
    status: IncidentStatus
    observed_metrics: Mapping[str, MetricValue]
    thresholds: Mapping[str, ThresholdValue]
    evidence: tuple[str, ...]
    recommended_steps: tuple[str, ...]
    limitations: tuple[str, ...]


def create_incident_id(
    *,
    detector: str,
    category: str,
    scope: str,
) -> str:
    """Create a deterministic fingerprint for an incident identity."""
    identity = {
        "category": category,
        "detector": detector,
        "scope": scope,
    }
    canonical = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"inc_{fingerprint[:24]}"


def _validate_metrics(
    name: str,
    values: Mapping[str, MetricValue | ThresholdValue],
) -> None:
    for key, value in values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be non-empty strings.")

        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name}[{key!r}] must be finite.")

        if not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError(f"{name}[{key!r}] has an unsupported value type.")


def build_incident(
    *,
    detector: str,
    category: str,
    scope: str,
    severity: Severity,
    observed_metrics: dict[str, MetricValue],
    thresholds: dict[str, ThresholdValue],
    evidence: tuple[str, ...],
    recommended_steps: tuple[str, ...],
    limitations: tuple[str, ...],
) -> Incident:
    """Construct a validated, open incident with a stable identity."""
    for name, value in (
        ("detector", detector),
        ("category", category),
        ("scope", scope),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must not be empty.")

    if severity not in SEVERITIES:
        raise ValueError(f"Invalid severity: {severity!r}.")

    if not evidence:
        raise ValueError("An incident must include at least one evidence item.")

    if not recommended_steps:
        raise ValueError("An incident must include at least one recommended step.")

    if any(
        not isinstance(item, str) or not item.strip()
        for item in (*evidence, *recommended_steps, *limitations)
    ):
        raise ValueError("Evidence, recommendations, and limitations must not contain empty items.")

    _validate_metrics("observed_metrics", observed_metrics)
    _validate_metrics("thresholds", thresholds)

    return Incident(
        incident_id=create_incident_id(
            detector=detector,
            category=category,
            scope=scope,
        ),
        detector=detector,
        category=category,
        scope=scope,
        severity=cast(Severity, severity),
        status="open",
        observed_metrics=MappingProxyType(observed_metrics.copy()),
        thresholds=MappingProxyType(thresholds.copy()),
        evidence=tuple(evidence),
        recommended_steps=tuple(recommended_steps),
        limitations=tuple(limitations),
    )
