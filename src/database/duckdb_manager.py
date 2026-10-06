from __future__ import annotations

from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd


class DuckDBManager:
    """A lightweight wrapper around a local DuckDB database."""

    def __init__(self, db_path: str | Path = "data/processed/business_analytics.duckdb") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(self.db_path))

    def __enter__(self) -> "DuckDBManager":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _quote_identifier(self, identifier: str) -> str:
        return f'"{str(identifier).replace("\"", "\"\"")}"'

    def list_tables(self) -> list[str]:
        """Return the names of tables in the main schema."""
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND table_name != '_tmp_df' ORDER BY table_name"
        rows = self.connection.execute(query).fetchall()
        return [row[0] for row in rows]

    def get_schema(self, table_name: str) -> pd.DataFrame:
        """Describe the columns for a table."""
        query = f"DESCRIBE {self._quote_identifier(table_name)}"
        return self.connection.execute(query).df()

    def load_dataframe(self, df: pd.DataFrame, table_name: str, *, if_exists: str = "replace") -> pd.DataFrame:
        """Register a pandas DataFrame as a DuckDB table."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame input.")

        quoted_table_name = self._quote_identifier(table_name)

        if if_exists == "replace":
            self.connection.execute(f"DROP TABLE IF EXISTS {quoted_table_name}")
        elif if_exists == "fail":
            existing_tables = self.list_tables()
            if table_name in existing_tables:
                raise ValueError(f"Table '{table_name}' already exists.")
        elif if_exists != "append":
            raise ValueError("if_exists must be one of: 'replace', 'fail', or 'append'.")

        self.connection.register("_tmp_df", df)
        if if_exists == "append":
            self.connection.execute(
                f"CREATE TABLE {quoted_table_name} AS SELECT * FROM _tmp_df"
            )
        else:
            self.connection.execute(
                f"CREATE TABLE {quoted_table_name} AS SELECT * FROM _tmp_df"
            )

        self.connection.unregister("_tmp_df")
        return self.get_schema(table_name)

    def load_csv(self, csv_path: str | Path, table_name: str | None = None, *, if_exists: str = "replace") -> pd.DataFrame:
        """Load a CSV file into DuckDB as a table."""
        path = Path(csv_path)
        table = table_name or path.stem
        df = pd.read_csv(path)
        return self.load_dataframe(df, table, if_exists=if_exists)

    def execute_query(self, query: str) -> pd.DataFrame:
        """Run a SQL query and return the results as a pandas DataFrame."""
        return self.connection.execute(query).df()

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None
