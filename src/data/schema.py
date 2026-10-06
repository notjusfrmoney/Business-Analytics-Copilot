from __future__ import annotations

from typing import Any

import pandas as pd


def profile_dataframe(table_name: str, df: pd.DataFrame) -> dict[str, Any]:
    """Return column-level metadata and row counts for a DataFrame."""
    columns = []
    for column in df.columns:
        series = df[column]
        columns.append(
            {
                "table_name": table_name,
                "column_name": column,
                "dtype": str(series.dtype),
                "row_count": int(len(df)),
                "missing_values": int(series.isna().sum()),
                "sample_values": series.dropna().head(5).astype(str).tolist(),
                "unique_values": int(series.nunique(dropna=True)),
            }
        )

    return {"table_name": table_name, "row_count": int(len(df)), "columns": columns}


def profile_table(table_name: str, df: pd.DataFrame) -> dict[str, Any]:
    """Alias for profile_dataframe for readability in later phases."""
    return profile_dataframe(table_name, df)


def get_expected_relationships() -> dict[str, str]:
    """Return the expected relationships used by the sample dataset."""
    return {
        "orders.customer_id": "customers.customer_id",
        "orders.product_id": "products.product_id",
    }
