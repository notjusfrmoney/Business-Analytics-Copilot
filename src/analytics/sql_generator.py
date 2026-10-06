from __future__ import annotations

import json
import re
from typing import Any, Protocol

from src.llm.client import LLMClient, LLMError


class TextGenerator(Protocol):
    def generate_text(self, prompt: str) -> str: ...


class SQLGenerator:
    """Generate one read-only DuckDB query from a question and live schema."""

    def __init__(self, llm_client: TextGenerator | None = None) -> None:
        self.llm_client = llm_client

    def generate(
        self,
        question: str,
        schema: dict[str, list[dict[str, str]]],
        relationships: dict[str, str],
        metadata: dict[str, Any],
        *,
        objective: str | None = None,
        prior_results: list[dict[str, Any]] | None = None,
        validation_feedback: str | None = None,
    ) -> dict[str, Any]:
        """Return a normalized SQL proposal from plain SQL or legacy JSON output."""
        prompt = self._build_prompt(
            question,
            schema,
            relationships,
            metadata,
            objective=objective,
            prior_results=prior_results,
            validation_feedback=validation_feedback,
        )
        if self.llm_client is None:
            self.llm_client = LLMClient()
        response = self.llm_client.generate_text(prompt)
        proposal = self._parse_response(response)
        for retry_number in range(2):
            if proposal["can_answer"] or self._is_explicit_refusal(response):
                break
            retry_prompt = (
                f"{prompt}\n\n"
                f"Retry {retry_number + 1}: the previous response was not usable SQL. "
                "Return exactly one DuckDB SELECT or WITH ... SELECT query, optionally "
                "inside a ```sql code block. Do not return a status, safety assessment, "
                "or explanation. If and only if the schema cannot answer the question, "
                "return only -- CANNOT_ANSWER: <brief reason>.\n"
                f"Previous response:\n{response}"
            )
            response = self.llm_client.generate_text(retry_prompt)
            proposal = self._parse_response(response)
        return proposal

    @staticmethod
    def _is_explicit_refusal(response: str) -> bool:
        text = response.strip()
        if text.casefold().startswith("-- cannot_answer:"):
            return True
        fenced = re.fullmatch(
            r"```json\s*(.*?)\s*```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced:
            text = fenced.group(1).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return False
        return (
            isinstance(payload, dict)
            and payload.get("can_answer") is False
            and payload.get("sql") == ""
        )

    @classmethod
    def _parse_response(cls, response: str) -> dict[str, Any]:
        if not isinstance(response, str) or not response.strip():
            raise LLMError("The model returned an empty response.")

        text = response.strip()
        structured = cls._parse_legacy_json(text)
        if structured is not None:
            return structured

        refusal = re.match(r"(?is)^\s*--\s*CANNOT_ANSWER\s*:\s*(.+?)\s*$", text)
        if refusal:
            return {
                "can_answer": False,
                "sql": "",
                "explanation": refusal.group(1).strip(),
            }

        sql = cls._extract_sql(text)
        if sql:
            return {
                "can_answer": True,
                "sql": sql,
                "explanation": "",
            }

        return {
            "can_answer": False,
            "sql": "",
            "explanation": text,
        }

    @classmethod
    def _parse_legacy_json(cls, text: str) -> dict[str, Any] | None:
        json_text = text
        fenced_json = re.fullmatch(
            r"```json\s*(.*?)\s*```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced_json:
            json_text = fenced_json.group(1).strip()
        if not json_text.startswith("{"):
            return None
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None

        can_answer = payload.get("can_answer")
        sql = payload.get("sql")
        explanation = payload.get("explanation", "")
        if not isinstance(can_answer, bool):
            raise LLMError("The model JSON response must include boolean 'can_answer'.")
        if not isinstance(sql, str):
            raise LLMError("The model JSON response must include string 'sql'.")
        if not isinstance(explanation, str):
            raise LLMError("The model JSON response 'explanation' must be a string.")
        if can_answer and not sql.strip():
            raise LLMError("The model JSON response has no SQL for an answerable question.")
        return {
            "can_answer": can_answer,
            "sql": sql.strip(),
            "explanation": explanation.strip(),
        }

    @staticmethod
    def _extract_sql(text: str) -> str:
        fenced_blocks = re.findall(
            r"```(?:sql)?\s*(.*?)```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for block in fenced_blocks:
            candidate = block.strip()
            if re.match(r"(?is)^(SELECT|WITH)\b", candidate):
                return candidate

        query_start = re.search(r"(?im)(?:^|\n|:\s*)(SELECT|WITH)\b", text)
        if query_start:
            candidate = text[query_start.start(1):].strip()
        elif re.match(r"(?is)^(SELECT|WITH)\b", text):
            candidate = text
        else:
            return ""
        candidate = re.sub(r"\s*```\s*$", "", candidate).strip()

        semicolon = candidate.find(";")
        if semicolon >= 0:
            candidate = candidate[: semicolon + 1].strip()
        return candidate if re.match(r"(?is)^(SELECT|WITH)\b", candidate) else ""

    @staticmethod
    def _build_prompt(
        question: str,
        schema: dict[str, list[dict[str, str]]],
        relationships: dict[str, str],
        metadata: dict[str, Any],
        *,
        objective: str | None = None,
        prior_results: list[dict[str, Any]] | None = None,
        validation_feedback: str | None = None,
    ) -> str:
        schema_lines = []
        for table_name, columns in schema.items():
            schema_lines.append(f"TABLE {table_name}")
            schema_lines.extend(
                f"- {column['column_name']}: {column['column_type']}"
                for column in columns
            )
        relationship_lines = [
            f"- {source} -> {target}" for source, target in relationships.items()
        ]
        return (
            "Return the DuckDB SQL query required to answer the current analysis "
            "objective in the context of the user's original question. Use prior "
            "query evidence to make this a useful follow-up, without claiming "
            "unsupported findings. "
            "Return exactly one read-only SQL query, optionally wrapped in a "
            "```sql code block. The query must be DuckDB-compatible and use only "
            "the tables and columns listed below. Do not invent schema or data. "
            "Do not modify data or database structure; do not use INSERT, UPDATE, "
            "DELETE, DROP, ALTER, CREATE, COPY, or other mutating statements. "
            "Do not return multiple queries. A SELECT query or WITH ... SELECT "
            "query is allowed. If the question cannot be answered from the "
            "provided schema, return only a SQL comment in this exact form: "
            "-- CANNOT_ANSWER: <brief reason>\n\n"
            f"AVAILABLE SCHEMA:\n{chr(10).join(schema_lines)}\n\n"
            "TABLE RELATIONSHIPS:\n"
            f"{chr(10).join(relationship_lines) or '- None provided'}\n\n"
            "DATABASE METADATA:\n"
            f"{json.dumps(metadata, ensure_ascii=True, sort_keys=True)}\n\n"
            f"ANALYSIS OBJECTIVE:\n{objective or question}\n\n"
            "PRIOR ANALYSIS RESULTS (metadata and compact result samples only):\n"
            f"{json.dumps(prior_results or [], ensure_ascii=True, sort_keys=True, default=str)}\n\n"
            "SQL VALIDATION FEEDBACK:\n"
            f"{validation_feedback or 'No previous SQL validation failure.'}\n\n"
            f"USER QUESTION:\n{question}"
        )
