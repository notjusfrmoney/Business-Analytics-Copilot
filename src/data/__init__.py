"""Data loading and profiling helpers for the analytics foundation."""

from .ingestion import read_csv_file, read_excel_file, normalize_columns
from .schema import get_expected_relationships, profile_dataframe, profile_table
from .validation import validate_dataframe
