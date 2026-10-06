from __future__ import annotations

from typing import Any

import pandas as pd


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    *,
    id_columns: list[str] | None = None,
    date_columns: list[str] | None = None,
    numeric_columns: list[str] | None = None,
) -> dict[str, list[str]]:
    """Return a simple validation summary with readable errors and warnings."""
    errors: list[str] = []
    warnings: list[str] = []

    if df.empty:
        warnings.append("DataFrame is empty.")

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")

    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        if missing_count:
            warnings.append(f"Column '{column}' contains {missing_count} missing value(s).")

    if id_columns:
        for column in id_columns:
            if column in df.columns:
                duplicate_count = int(df[column].duplicated().sum())
                if duplicate_count:
                    errors.append(
                        f"Column '{column}' contains {duplicate_count} duplicate value(s)."
                    )

    if date_columns:
        for column in date_columns:
            if column in df.columns:
                parsed = pd.to_datetime(df[column], errors="coerce")
                invalid_count = int(parsed.isna().sum())
                if invalid_count:
                    errors.append(
                        f"Column '{column}' contains {invalid_count} invalid date value(s)."
                    )

    if numeric_columns:
        for column in numeric_columns:
            if column in df.columns:
                converted = pd.to_numeric(df[column], errors="coerce")
                invalid_count = int((converted.isna() & df[column].notna()).sum())
                if invalid_count:
                    errors.append(
                        f"Column '{column}' contains {invalid_count} non-numeric value(s)."
                    )

    return {"errors": errors, "warnings": warnings}


def validate_business_table(
    table_name: str,
    df: pd.DataFrame,
    *,
    required_columns: list[str],
    id_columns: list[str] | None = None,
    date_columns: list[str] | None = None,
    numeric_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Validate a business data table and include table context in the response."""
    result = validate_dataframe(
        df,
        required_columns,
        id_columns=id_columns,
        date_columns=date_columns,
        numeric_columns=numeric_columns,
    )
    return {"table_name": table_name, **result}
