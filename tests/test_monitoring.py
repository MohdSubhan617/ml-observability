"""Tests for dataset schema validation."""

import pandas as pd
import pytest

from ml_observability.monitoring import validate_schema


def test_matching_schema_without_nulls_is_valid() -> None:
    reference = pd.DataFrame({"age": [25, 30], "score": [0.4, 0.8]})
    current = pd.DataFrame({"age": [26, 31], "score": [0.5, 0.7]})

    result = validate_schema(reference, current)

    assert result.is_valid
    assert result.missing_columns == ()
    assert result.unexpected_columns == ()
    assert result.missing_values == {}


def test_missing_and_unexpected_columns_are_reported() -> None:
    reference = pd.DataFrame({"age": [25], "score": [0.4]})
    current = pd.DataFrame({"age": [26], "region": ["north"]})

    result = validate_schema(reference, current)

    assert not result.is_valid
    assert result.missing_columns == ("score",)
    assert result.unexpected_columns == ("region",)


def test_missing_values_are_counted_per_column() -> None:
    reference = pd.DataFrame({"age": [25, 30], "score": [0.4, 0.8]})
    current = pd.DataFrame({"age": [None, 31], "score": [None, None]})

    result = validate_schema(reference, current)

    assert not result.is_valid
    assert result.missing_values == {"age": 1, "score": 2}


@pytest.mark.parametrize(
    ("reference", "current"),
    [
        (
            pd.DataFrame(columns=["age", "age"]),
            pd.DataFrame(columns=["age"]),
        ),
        (
            pd.DataFrame(columns=["age"]),
            pd.DataFrame(columns=["age", "age"]),
        ),
    ],
)
def test_duplicate_column_names_are_rejected(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="duplicate column names"):
        validate_schema(reference, current)
