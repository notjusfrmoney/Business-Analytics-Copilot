from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from src.analytics.planner import AnalysisPlan
from src.llm.client import LLMClient, LLMError


class TextGenerator(Protocol):
    def generate_text(self, prompt: str) -> str: ...


@dataclass
class AnswerStep:
    goal: str
    sql: str
    data: pd.DataFrame
    error: str | None = None


class AnswerGenerator:
    """Summarize query evidence compactly and generate a fact-led answer."""

    MAX_RESULT_ROWS = 8

    def __init__(self, llm_client: TextGenerator | None = None) -> None:
        self.llm_client = llm_client

    def generate(
        self,
        question: str,
        plan: AnalysisPlan,
        steps: list[AnswerStep],
        trace: list[str],
    ) -> str:
        if self.llm_client is None:
            self.llm_client = LLMClient()

        evidence = [
            {
                "goal": step.goal,
                "sql": step.sql,
                "error": step.error,
                "summary": summarize_dataframe(step.data, max_rows=self.MAX_RESULT_ROWS),
            }
            for step in steps
        ]
        prompt = (
            "Answer the business question using only the supplied analysis plan, "
            "query results, and trace. Do not invent facts, causes, or numbers. "
            "Every numerical claim must be directly present in the supplied query "
            "results or their Python-computed summaries; do not do arithmetic "
            "mentally. A normal concise response is acceptable; FACT and "
            "INTERPRETATION labels are optional. Refer to specific result columns "
            "or categories to make the answer clearly evidence-grounded. "
            "If evidence is insufficient, say so. Be concise and identify the "
            "relevant analysis steps.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"PLAN:\n{json.dumps(plan.as_dict(), ensure_ascii=True)}\n\n"
            f"ANALYSIS TRACE:\n{json.dumps(trace, ensure_ascii=True)}\n\n"
            f"QUERY EVIDENCE:\n{json.dumps(evidence, ensure_ascii=True, default=str)}"
        )
        answer = self.llm_client.generate_text(prompt).strip()
        for attempt in range(2):
            error = self._validate_answer(answer, evidence)
            if error is None:
                return answer
            if attempt == 1:
                raise LLMError(error)
            prompt = (
                f"{prompt}\n\nYour previous answer was rejected: {error} "
                "Try once more. Use normal concise prose, mention relevant "
                "evidence columns or result labels, and use only supported numbers."
            )
            answer = self.llm_client.generate_text(prompt).strip()
        raise LLMError("The answer generator could not produce a valid answer.")

    @staticmethod
    def _validate_answer(answer: str, evidence: list[dict[str, Any]]) -> str | None:
        if not answer:
            return "The answer generator returned an empty response."
        unsupported = _unsupported_numeric_values(answer, evidence)
        if unsupported:
            values = ", ".join(f"{value:g}" for value in unsupported)
            return f"The answer contains unsupported numerical claims: {values}."
        if not _references_result_content(answer, evidence):
            return "The answer does not reference any supplied query-result evidence."
        return None


def summarize_dataframe(
    dataframe: pd.DataFrame,
    *,
    max_rows: int = 8,
) -> dict[str, Any]:
    """Build a JSON-safe compact result summary, retaining the full frame elsewhere."""
    numeric_summary: dict[str, dict[str, float | int]] = {}
    for column in dataframe.select_dtypes(include="number").columns:
        values = dataframe[column].dropna()
        if values.empty:
            continue
        numeric_summary[str(column)] = {
            "min": _json_number(values.min()),
            "max": _json_number(values.max()),
            "sum": _json_number(values.sum()),
            "mean": _json_number(values.mean()),
        }

    rows = dataframe.head(max_rows).to_dict(orient="records")
    quarter_pairs: list[dict[str, Any]] = []
    normalized_columns = {str(column).casefold(): column for column in dataframe.columns}
    for q2_name, q2_column in normalized_columns.items():
        if not q2_name.startswith("q2_"):
            continue
        q3_column = normalized_columns.get(f"q3_{q2_name[3:]}")
        if q3_column is None:
            continue
        for _, row in dataframe.head(max_rows).iterrows():
            q2_value = pd.to_numeric(pd.Series([row[q2_column]]), errors="coerce").iloc[0]
            q3_value = pd.to_numeric(pd.Series([row[q3_column]]), errors="coerce").iloc[0]
            if pd.isna(q2_value) or pd.isna(q3_value):
                continue
            labels = {
                str(column): _json_safe(row[column])
                for column in dataframe.columns
                if column not in {q2_column, q3_column}
                and not pd.api.types.is_numeric_dtype(dataframe[column])
            }
            delta = float(q3_value - q2_value)
            quarter_pairs.append(
                {
                    "labels": labels,
                    "comparison": f"{q3_column} vs {q2_column}",
                    "absolute_change": _json_number(delta),
                    "percent_change": (
                        _json_number(delta / abs(float(q2_value)) * 100)
                        if float(q2_value) != 0
                        else None
                    ),
                }
            )

    derived_changes: list[dict[str, Any]] = []
    numeric_columns = {
        str(column).casefold(): column
        for column in dataframe.select_dtypes(include="number").columns
    }
    for change_name, change_column in numeric_columns.items():
        if not change_name.endswith("_change"):
            continue
        metric = change_name[: -len("_change")]
        previous_column = numeric_columns.get(f"previous_{metric}") or numeric_columns.get(
            f"prev_{metric}"
        )
        if previous_column is None:
            continue
        for _, row in dataframe.head(max_rows).iterrows():
            previous_value = float(row[previous_column])
            change_value = float(row[change_column])
            if not math.isfinite(previous_value) or not math.isfinite(change_value):
                continue
            derived_changes.append(
                {
                    "metric": metric,
                    "previous_value": _json_number(previous_value),
                    "absolute_change": _json_number(change_value),
                    "percent_change": (
                        _json_number(change_value / abs(previous_value) * 100)
                        if previous_value != 0
                        else None
                    ),
                }
            )

    return {
        "columns": [str(column) for column in dataframe.columns],
        "row_count": int(len(dataframe)),
        "sample_rows": _json_safe(rows),
        "numeric_summary": numeric_summary,
        "derived_quarter_comparisons": quarter_pairs,
        "derived_changes": derived_changes,
        "truncated": len(dataframe) > max_rows,
    }


def _json_number(value: Any) -> float | int:
    number = float(value)
    if not math.isfinite(number):
        return 0
    rounded = round(number, 4)
    return int(rounded) if rounded.is_integer() else rounded


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return None
        return _json_number(value)
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _contains_only_supported_numbers(answer: str, evidence: list[dict[str, Any]]) -> bool:
    return not _unsupported_numeric_values(answer, evidence)


def _unsupported_numeric_values(
    answer: str,
    evidence: list[dict[str, Any]],
) -> list[float]:
    supported: set[float] = set()

    def collect(value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            supported.add(float(value))
        elif isinstance(value, dict):
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)
        elif isinstance(value, str):
            supported.update(number for number, _ in _numeric_values(value))

    for item in evidence:
        collect(item.get("summary", {}))
    answer_without_step_numbers = re.sub(
        r"\b(?:step|q)\s*\d+\b",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    decline_language = bool(
        re.search(
            r"\b(?:fell|fall|declined?|decreased?|dropped?|lower|reduced?|loss)\b",
            answer_without_step_numbers,
            flags=re.IGNORECASE,
        )
    )
    unsupported: list[float] = []
    for number, tolerance in _numeric_values(answer_without_step_numbers):
        if not any(
            min(
                abs(number - allowed),
                abs(abs(number) - abs(allowed))
                if decline_language
                else float("inf"),
            )
            <= max(tolerance, 1e-4, abs(allowed) * 1e-6)
            for allowed in supported
        ):
            unsupported.append(number)
    return unsupported


def _numeric_values(text: str) -> list[tuple[float, float]]:
    values: list[tuple[float, float]] = []
    pattern = re.compile(
        r"(?<![A-Za-z])\$?[-+]?\d[\d,]*(?:\.\d+)?\s*[kKmMbB]?%?"
    )
    for match in pattern.finditer(text):
        token = match.group(0).replace("$", "").replace(",", "").strip()
        scale = 1.0
        display_precision = 0
        if token and token[-1].casefold() in {"k", "m", "b"}:
            scale = {"k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}[
                token[-1].casefold()
            ]
            token = token[:-1].strip()
            decimal = token.partition(".")[2]
            display_precision = len(decimal)
        is_percent = token.endswith("%")
        token = token.rstrip("%").strip()
        try:
            number = float(token) * scale
            tolerance = (
                0.5 * (10 ** -display_precision) * scale
                if scale != 1.0
                else 1e-4
            )
            values.append((number, tolerance))
            if is_percent:
                values.append((number / 100, tolerance / 100))
        except ValueError:
            continue
    return values


def _references_result_content(answer: str, evidence: list[dict[str, Any]]) -> bool:
    stop_words = {
        "actual", "analysis", "answer", "change", "data", "evidence", "fact",
        "interpretation", "query", "result", "results", "step", "the", "this",
    }
    evidence_terms: set[str] = set()

    def collect_terms(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                collect_terms(nested)
        elif isinstance(value, list):
            for nested in value:
                collect_terms(nested)
        elif isinstance(value, str):
            evidence_terms.update(
                token.casefold()
                for token in re.findall(r"[A-Za-z][A-Za-z_-]{2,}", value)
                if token.casefold() not in stop_words
            )

    for item in evidence:
        collect_terms(item.get("summary", {}))
    answer_terms = {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z_-]{2,}", answer)
        if token.casefold() not in stop_words
    }
    return bool(answer_terms & evidence_terms)


def verify_answer(answer: str, steps: list[AnswerStep]) -> bool:
    """Verify all explicit numeric claims against actual compact query-result summaries."""
    evidence = [
        {
            "summary": summarize_dataframe(step.data, max_rows=AnswerGenerator.MAX_RESULT_ROWS)
        }
        for step in steps
    ]
    return AnswerGenerator._validate_answer(answer, evidence) is None
