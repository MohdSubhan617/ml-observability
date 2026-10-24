"""Deterministic dataset validation for ML monitoring."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SchemaValidationResult:
    """Schema and missing-value findings for a current dataset."""

    missing_columns: tuple[str, ...]
    unexpected_columns: tuple[str, ...]
    missing_values: dict[str, int]

    @property
    def is_valid(self) -> bool:
        """Return whether the dataset passes the current validation policy."""
        return not self.missing_columns and not self.unexpected_columns and not self.missing_values


def validate_schema(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> SchemaValidationResult:
    """Compare dataset columns and count null values in the current dataset."""
    reference_columns = list(reference.columns)
    current_columns = list(current.columns)

    if len(reference_columns) != len(set(reference_columns)):
        raise ValueError("Reference dataset contains duplicate column names.")

    if len(current_columns) != len(set(current_columns)):
        raise ValueError("Current dataset contains duplicate column names.")

    reference_set = set(reference_columns)
    current_set = set(current_columns)

    missing_columns = tuple(column for column in reference_columns if column not in current_set)
    unexpected_columns = tuple(column for column in current_columns if column not in reference_set)
    missing_values = {
        column: int(count) for column, count in current.isna().sum().items() if count > 0
    }

    return SchemaValidationResult(
        missing_columns=missing_columns,
        unexpected_columns=unexpected_columns,
        missing_values=missing_values,
    )
