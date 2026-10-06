from __future__ import annotations

import re
from pathlib import Path
from typing import BinaryIO, Sequence
from zipfile import BadZipFile

import pandas as pd

from src.data.ingestion import read_csv_file, read_excel_file
from src.data.schema import profile_dataframe
from src.data.validation import validate_business_table
from src.database.duckdb_manager import DuckDBManager

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}


class UploadDataError(ValueError):
    """Raised when uploaded data cannot be safely loaded for analysis."""


def load_uploaded_files(
    uploaded_files: Sequence[BinaryIO],
    db_path: str | Path,
) -> list[dict[str, object]]:
    """Validate uploaded CSV/XLSX files and load them into a fresh DuckDB."""
    if not uploaded_files:
        raise UploadDataError("Choose at least one CSV or XLSX file.")

    prepared: list[tuple[str, str, pd.DataFrame, dict[str, object]]] = []
    used_names: set[str] = set()
    for uploaded_file in uploaded_files:
        file_name = str(getattr(uploaded_file, "name", ""))
        extension = _file_extension(file_name)
        if extension not in SUPPORTED_EXTENSIONS:
            raise UploadDataError(
                f"'{_display_file_name(file_name)}' is not a supported CSV or XLSX file."
            )

        table_name = _unique_table_name(file_name, used_names)
        try:
            uploaded_file.seek(0)
            dataframe = (
                read_csv_file(uploaded_file)
                if extension == ".csv"
                else read_excel_file(uploaded_file)
            )
        except (
            ImportError,
            BadZipFile,
            OSError,
            UnicodeDecodeError,
            ValueError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ) as exc:
            raise UploadDataError(
                f"Could not read '{_display_file_name(file_name)}': {exc}"
            ) from exc
        finally:
            uploaded_file.seek(0)

        if dataframe.empty:
            raise UploadDataError(
                f"'{_display_file_name(file_name)}' has no data rows."
            )
        if len(dataframe.columns) == 0:
            raise UploadDataError(
                f"'{_display_file_name(file_name)}' has no columns."
            )
        duplicate_columns = dataframe.columns[dataframe.columns.duplicated()].tolist()
        if duplicate_columns:
            duplicates = ", ".join(map(str, duplicate_columns))
            raise UploadDataError(
                f"'{_display_file_name(file_name)}' has duplicate column names "
                f"after normalization: {duplicates}."
            )

        validation = validate_business_table(
            table_name,
            dataframe,
            required_columns=[],
        )
        if validation["errors"]:
            raise UploadDataError(
                f"Validation failed for '{_display_file_name(file_name)}': "
                + "; ".join(validation["errors"])
            )
        prepared.append(
            (
                table_name,
                _display_file_name(file_name),
                dataframe,
                profile_dataframe(table_name, dataframe),
            )
        )

    tables: list[dict[str, object]] = []
    with DuckDBManager(db_path) as database:
        for table_name, file_name, dataframe, profile in prepared:
            schema = database.load_dataframe(dataframe, table_name)
            tables.append(
                {
                    "table_name": table_name,
                    "source_name": file_name,
                    "row_count": int(profile["row_count"]),
                    "column_count": len(schema),
                    "schema": [
                        {
                            "column_name": str(row["column_name"]),
                            "column_type": str(row["column_type"]),
                        }
                        for _, row in schema.iterrows()
                    ],
                    "preview": dataframe.head(10).copy(),
                    "warnings": validate_business_table(
                        table_name,
                        dataframe,
                        required_columns=[],
                    )["warnings"],
                }
            )
    return tables


def _file_extension(file_name: str) -> str:
    return Path(file_name.replace("\\", "/")).suffix.casefold()


def _display_file_name(file_name: str) -> str:
    return Path(file_name.replace("\\", "/")).name or "uploaded file"


def _unique_table_name(file_name: str, used_names: set[str]) -> str:
    stem = Path(file_name.replace("\\", "/")).stem.casefold()
    normalized = re.sub(r"[^a-z0-9_]+", "_", stem).strip("_")
    normalized = re.sub(r"_+", "_", normalized) or "uploaded_table"
    if normalized[0].isdigit():
        normalized = f"table_{normalized}"

    candidate = normalized
    suffix = 2
    while candidate in used_names:
        candidate = f"{normalized}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate
