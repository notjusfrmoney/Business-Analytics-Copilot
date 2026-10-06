from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import duckdb

from src.analytics.sql_generator import SQLGenerator, TextGenerator
from src.analytics.sql_validator import SQLValidator
from src.data.schema import get_expected_relationships
from src.database.duckdb_manager import DuckDBManager
from src.llm.client import LLMError
from src.utils.config import get_database_path


@dataclass
class QueryResult:
    """Structured outcome from a natural-language analytics request."""

    success: bool
    question: str
    sql: str = ""
    explanation: str = ""
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    error: str | None = None
    trace: list[str] = field(default_factory=list)


class QueryService:
    """Coordinate schema retrieval, SQL generation, validation, and execution."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        sql_generator: SQLGenerator | None = None,
        llm_client: TextGenerator | None = None,
        debug: bool = True,
    ) -> None:
        if sql_generator is not None and llm_client is not None:
            raise ValueError("Pass either sql_generator or llm_client, not both.")
        self.db_path = Path(db_path) if db_path is not None else get_database_path()
        self.sql_generator = sql_generator or SQLGenerator(llm_client)
        self.validator = SQLValidator()
        self.debug = debug

    def get_schema_context(self) -> dict[str, list[dict[str, str]]]:
        """Return live DuckDB table and column metadata without reading table rows."""
        with DuckDBManager(self.db_path) as database:
            return {
                table_name: [
                    {
                        "column_name": str(row["column_name"]),
                        "column_type": str(row["column_type"]),
                    }
                    for _, row in database.get_schema(table_name).iterrows()
                ]
                for table_name in database.list_tables()
            }

    def answer_question(
        self,
        question: str,
        *,
        objective: str | None = None,
        prior_results: list[dict[str, Any]] | None = None,
        validation_feedback: str | None = None,
    ) -> QueryResult:
        trace = [f"Question:\n{question}"]
        if not isinstance(question, str) or not question.strip():
            return self._fail(question, "Question is empty.", trace)

        try:
            with DuckDBManager(self.db_path) as database:
                tables = database.list_tables()
                if not tables:
                    return self._fail(
                        question,
                        "The database contains no available tables.",
                        trace,
                    )
                schema = {
                    table_name: [
                        {
                            "column_name": str(row["column_name"]),
                            "column_type": str(row["column_type"]),
                        }
                        for _, row in database.get_schema(table_name).iterrows()
                    ]
                    for table_name in tables
                }
                metadata = {
                    "row_counts": {
                        table_name: int(
                            database.execute_query(
                                f'SELECT COUNT(*) AS row_count FROM "{table_name}"'
                            ).iloc[0]["row_count"]
                        )
                        for table_name in tables
                    }
                }
                relationships = {
                    source: target
                    for source, target in get_expected_relationships().items()
                    if _relationship_exists(source, schema)
                    and _relationship_exists(target, schema)
                }
                proposal = self.sql_generator.generate(
                    question,
                    schema,
                    relationships,
                    metadata,
                    objective=objective,
                    prior_results=prior_results,
                    validation_feedback=validation_feedback,
                )
                if not proposal["can_answer"]:
                    explanation = proposal["explanation"] or (
                        "The available data does not contain the information needed."
                    )
                    return self._fail(question, explanation, trace)

                sql = proposal["sql"]
                trace.append(f"Generated SQL:\n{sql}")
                validation = self.validator.validate(
                    sql,
                    set(tables),
                    connection=database.connection,
                )
                if not validation.valid:
                    trace.append(f"Validation:\nFAILED - {validation.error}")
                    return QueryResult(
                        False,
                        question,
                        sql=sql,
                        explanation=proposal["explanation"],
                        error=validation.error,
                        trace=self._emit(trace),
                    )
                trace.append("Validation:\nPASSED")

                try:
                    data = database.execute_query(sql)
                except duckdb.Error as exc:
                    error = f"SQL execution failed: {exc}"
                    trace.append(f"Execution:\nFAILED - {error}")
                    return QueryResult(
                        False,
                        question,
                        sql=sql,
                        explanation=proposal["explanation"],
                        error=error,
                        trace=self._emit(trace),
                    )
                trace.extend(
                    [
                        "Execution:\nSUCCESS",
                        f"Rows returned:\n{len(data)}",
                    ]
                )
                return QueryResult(
                    True,
                    question,
                    sql=sql,
                    explanation=proposal["explanation"],
                    data=data,
                    trace=self._emit(trace),
                )
        except (duckdb.Error, OSError, ValueError, LLMError) as exc:
            return self._fail(question, str(exc), trace)

    def _fail(self, question: str, error: str, trace: list[str]) -> QueryResult:
        trace.append(f"Execution:\nFAILED - {error}")
        return QueryResult(
            False,
            question,
            error=error,
            trace=self._emit(trace),
        )

    def _emit(self, trace: list[str]) -> list[str]:
        if self.debug:
            print("\n".join(trace))
        return trace


def answer_question(question: str) -> QueryResult:
    """Convenience entry point for the default configured database and LLM."""
    return QueryService().answer_question(question)


def _relationship_exists(
    reference: str,
    schema: dict[str, list[dict[str, str]]],
) -> bool:
    table, separator, column = reference.partition(".")
    return bool(
        separator
        and any(item["column_name"] == column for item in schema.get(table, []))
    )
