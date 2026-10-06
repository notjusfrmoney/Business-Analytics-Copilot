from __future__ import annotations

import re
from dataclasses import dataclass

import duckdb


_FORBIDDEN_KEYWORDS = {
    "ALTER",
    "ATTACH",
    "CALL",
    "COPY",
    "CREATE",
    "DELETE",
    "DETACH",
    "DROP",
    "EXPORT",
    "IMPORT",
    "INSERT",
    "INSTALL",
    "LOAD",
    "PRAGMA",
    "REPLACE",
    "SET",
    "TRUNCATE",
    "UPDATE",
    "USE",
    "VACUUM",
}
_TABLE_REFERENCE = re.compile(
    r"\b(?:FROM|JOIN)\s+((?:\"(?:[^\"]|\"\")+\"|[A-Za-z_][\w$]*)(?:\s*\.\s*(?:\"(?:[^\"]|\"\")+\"|[A-Za-z_][\w$]*))?)",
    re.IGNORECASE,
)
_CTE_NAME = re.compile(
    r"(?:\bWITH\b|,)\s*(?:RECURSIVE\s+)?(\"(?:[^\"]|\"\")+\"|[A-Za-z_][\w$]*)\s+AS\s*\(",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"\b[A-Za-z_][A-Za-z_0-9$]*\b")


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    error: str | None = None


class SQLValidator:
    """Perform basic read-only SQL validation against known table names."""

    def validate(
        self,
        sql: str,
        known_tables: set[str] | None = None,
        *,
        connection: duckdb.DuckDBPyConnection | None = None,
    ) -> ValidationResult:
        if not isinstance(sql, str) or not sql.strip():
            return ValidationResult(False, "SQL query is empty.")

        cleaned = self._remove_comments_and_literals(sql).strip()
        statement = sql.strip()
        if statement.endswith(";"):
            statement = statement[:-1].rstrip()
        if ";" in cleaned.strip().removesuffix(";"):
            return ValidationResult(False, "Only one SQL statement is allowed.")

        tokens = [token.upper() for token in _TOKEN.findall(cleaned)]
        if not tokens or tokens[0] not in {"SELECT", "WITH"}:
            return ValidationResult(False, "Only SELECT queries are allowed.")

        forbidden = sorted(_FORBIDDEN_KEYWORDS.intersection(tokens))
        if forbidden:
            return ValidationResult(
                False,
                f"Disallowed SQL operation detected: {', '.join(forbidden)}.",
            )

        if known_tables is not None:
            normalized_tables = {table.casefold() for table in known_tables}
            cte_names = {
                self._unquote(match.group(1)).casefold()
                for match in _CTE_NAME.finditer(cleaned)
            }
            table_scan_sql = re.sub(
                r"\bEXTRACT\s*\(\s*[A-Za-z_]+\s+FROM\b",
                "EXTRACT(",
                cleaned,
                flags=re.IGNORECASE,
            )
            for match in _TABLE_REFERENCE.finditer(table_scan_sql):
                reference = match.group(1).strip()
                components = [
                    self._unquote(part.strip())
                    for part in reference.split(".")
                ]
                table_name = components[-1].casefold()
                if table_name not in normalized_tables and table_name not in cte_names:
                    return ValidationResult(
                        False,
                        f"Unknown table referenced in SQL: '{components[-1]}'.",
                    )

        if connection is not None:
            try:
                connection.execute(f"EXPLAIN {statement}")
            except duckdb.Error as exc:
                return ValidationResult(False, f"Invalid SQL: {exc}")

        return ValidationResult(True)

    @staticmethod
    def _unquote(identifier: str) -> str:
        if identifier.startswith('"') and identifier.endswith('"'):
            return identifier[1:-1].replace('""', '"')
        return identifier

    @staticmethod
    def _remove_comments_and_literals(sql: str) -> str:
        output: list[str] = []
        index = 0
        while index < len(sql):
            if sql.startswith("--", index):
                end = sql.find("\n", index)
                index = len(sql) if end == -1 else end
                output.append(" ")
            elif sql.startswith("/*", index):
                end = sql.find("*/", index + 2)
                if end == -1:
                    return ""
                index = end + 2
                output.append(" ")
            elif sql[index] == "'":
                index += 1
                while index < len(sql):
                    if sql[index] == "'" and index + 1 < len(sql) and sql[index + 1] == "'":
                        index += 2
                    elif sql[index] == "'":
                        index += 1
                        break
                    else:
                        index += 1
                output.append(" ")
            else:
                output.append(sql[index])
                index += 1
        return "".join(output)
