from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from src.llm.client import LLMClient, LLMError

SUPPORTED_INTENTS = {
    "simple_lookup",
    "aggregation",
    "comparison",
    "trend_analysis",
    "ranking",
    "root_cause_analysis",
}


class TextGenerator(Protocol):
    def generate_text(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class AnalysisPlan:
    intent: str
    metric: str
    time_period: str
    steps: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "metric": self.metric,
            "time_period": self.time_period,
            "steps": self.steps,
        }


class AnalysisPlanner:
    """Ask the configured LLM for a small, bounded analysis plan."""

    def __init__(self, llm_client: TextGenerator | None = None) -> None:
        self.llm_client = llm_client

    def create_plan(
        self,
        question: str,
        *,
        schema: dict[str, list[dict[str, str]]] | None = None,
        max_steps: int = 4,
    ) -> AnalysisPlan:
        if self.llm_client is None:
            self.llm_client = LLMClient()
        prompt = self._build_prompt(question, max_steps, schema or {})
        response = self.llm_client.generate_text(prompt)
        try:
            return self.parse_plan(response, max_steps=max_steps)
        except LLMError:
            repair_prompt = (
                f"{prompt}\n\nYour previous response was not a valid plan JSON. "
                "Return only one valid JSON object with the required fields and "
                "supported intent. Do not include markdown or commentary.\n"
                f"Previous response:\n{response}"
            )
            repaired = self.llm_client.generate_text(repair_prompt)
            return self.parse_plan(repaired, max_steps=max_steps)

    @staticmethod
    def _build_prompt(
        question: str,
        max_steps: int,
        schema: dict[str, list[dict[str, str]]],
    ) -> str:
        schema_lines = [
            f"TABLE {table}\n"
            + "\n".join(
                f"- {column['column_name']}: {column['column_type']}"
                for column in columns
            )
            for table, columns in schema.items()
        ]
        return (
            "Create a concise analysis plan for a business analytics question. "
            "Return only a JSON object with fields: intent, metric, time_period, steps. "
            "intent must be one of: simple_lookup, aggregation, comparison, "
            "trend_analysis, ranking, root_cause_analysis. metric and time_period "
            "must be strings (use an empty string if unspecified). steps must be an "
            f"array of 1 to {max_steps} short, actionable SQL-analysis objectives. "
            "Use the supplied dataset schema; do not propose analysis requiring "
            "missing data. Use multiple distinct steps for root-cause questions "
            "when useful, such as comparison, then breakdown by region/category. "
            "Do not write SQL in the plan. Only plan queries answerable from this "
            "schema.\n\n"
            f"AVAILABLE SCHEMA:\n{chr(10).join(schema_lines) or '- Schema unavailable'}\n\n"
            f"USER QUESTION:\n{question}"
        )

    @staticmethod
    def parse_plan(response: str, *, max_steps: int = 4) -> AnalysisPlan:
        if not isinstance(response, str) or not response.strip():
            raise LLMError("The planner returned an empty response.")

        text = response.strip()
        fence = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fence:
            text = fence.group(1).strip()
        if not text.startswith("{"):
            object_start = text.find("{")
            object_end = text.rfind("}")
            if object_start >= 0 and object_end > object_start:
                text = text[object_start : object_end + 1]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError("The planner returned malformed JSON.") from exc
        if not isinstance(payload, dict):
            raise LLMError("The planner response must be a JSON object.")

        intent = payload.get("intent")
        metric = payload.get("metric")
        time_period = payload.get("time_period", "")
        steps = payload.get("steps")
        if not isinstance(intent, str) or intent not in SUPPORTED_INTENTS:
            raise LLMError("The planner returned an unsupported analysis intent.")
        if not isinstance(metric, str) or not isinstance(time_period, str):
            raise LLMError("The planner's metric and time_period must be strings.")
        if (
            not isinstance(steps, list)
            or not steps
            or any(not isinstance(step, str) or not step.strip() for step in steps)
        ):
            raise LLMError("The planner must return at least one non-empty analysis step.")
        normalized_steps = [step.strip() for step in steps[:max_steps]]
        return AnalysisPlan(
            intent=intent,
            metric=metric.strip(),
            time_period=time_period.strip(),
            steps=normalized_steps,
        )
