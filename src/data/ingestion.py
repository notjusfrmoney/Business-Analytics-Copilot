from __future__ import annotations

from typing import IO
from pathlib import Path

import pandas as pd

from src.data.validation import validate_business_table


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase snake_case style."""
    df = df.copy()
    df.columns = [
        str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for column in df.columns
    ]
    return df


def read_csv_file(
    file_path: str | Path | IO[bytes],
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    """Read a CSV path or binary file-like object and return a DataFrame."""
    source = Path(file_path) if isinstance(file_path, (str, Path)) else file_path
    df = pd.read_csv(source)
    return normalize_columns(df) if normalize else df


def read_excel_file(
    file_path: str | Path | IO[bytes],
    sheet_name: int | str | None = 0,
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    """Read an Excel path or binary file-like object and return a DataFrame."""
    source = Path(file_path) if isinstance(file_path, (str, Path)) else file_path
    df = pd.read_excel(source, sheet_name=sheet_name)
    return normalize_columns(df) if normalize else df


def load_data_with_validation(
    file_path: str | Path,
    *,
    table_name: str,
    required_columns: list[str],
    id_columns: list[str] | None = None,
    date_columns: list[str] | None = None,
    numeric_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Read a dataset file, normalize columns, and run the basic validation checks."""
    df = read_csv_file(file_path)
    validation_result = validate_business_table(
        table_name,
        df,
        required_columns=required_columns,
        id_columns=id_columns,
        date_columns=date_columns,
        numeric_columns=numeric_columns,
    )
    return df, validation_result
