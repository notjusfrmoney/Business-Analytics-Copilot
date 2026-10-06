from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import duckdb

from src.analytics.answer_generator import (
    AnswerGenerator,
    AnswerStep,
    summarize_dataframe,
    verify_answer,
)
from src.analytics.planner import AnalysisPlan, AnalysisPlanner
from src.analytics.query_service import QueryResult, QueryService
from src.analytics.sql_generator import TextGenerator
from src.llm.client import LLMError
from src.utils.config import get_database_path

MAX_ANALYSIS_STEPS = 4
MAX_CONTEXT_ROWS = 8


class Planner(Protocol):
    def create_plan(
        self,
        question: str,
        *,
        schema: dict[str, list[dict[str, str]]] | None = None,
        max_steps: int = 4,
    ) -> AnalysisPlan: ...


class FinalAnswerer(Protocol):
    def generate(
        self,
        question: str,
        plan: AnalysisPlan,
        steps: list[AnswerStep],
        trace: list[str],
    ) -> str: ...


@dataclass
class AnalysisStep:
    number: int
    goal: str
    sql: str
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    validation: str = "NOT RUN"

    def as_evidence(self) -> dict[str, Any]:
        return {
            "step": self.number,
            "goal": self.goal,
            "sql": self.sql,
            "error": self.error,
            "summary": self.summary,
        }


@dataclass
class AnalysisResult:
    success: bool
    question: str
    plan: AnalysisPlan | None = None
    steps: list[AnalysisStep] = field(default_factory=list)
    answer: str = ""
    error: str | None = None
    trace: list[str] = field(default_factory=list)
    limited: bool = False
    answer_verified: bool = False


class AnalysisAgent:
    """Run a short plan-driven sequence of validated DuckDB analysis queries."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        planner: Planner | None = None,
        sql_generator: TextGenerator | None = None,
        answer_generator: FinalAnswerer | None = None,
        query_service: QueryService | None = None,
        max_analysis_steps: int = MAX_ANALYSIS_STEPS,
        debug: bool = True,
    ) -> None:
        if max_analysis_steps < 1:
            raise ValueError("max_analysis_steps must be at least 1.")
        if query_service is not None and (db_path is not None or sql_generator is not None):
            raise ValueError(
                "Pass query_service or db_path/sql_generator, not both."
            )
        self.max_analysis_steps = max_analysis_steps
        self.planner = planner or AnalysisPlanner()
        self.answer_generator = answer_generator or AnswerGenerator()
        self.query_service = query_service or QueryService(
            Path(db_path) if db_path is not None else get_database_path(),
            llm_client=sql_generator,
            debug=False,
        )
        self.debug = debug

    def answer_question(self, question: str) -> AnalysisResult:
        if not isinstance(question, str) or not question.strip():
            return AnalysisResult(
                success=False,
                question=question,
                error="Question is empty.",
                trace=["Question is empty."],
            )

        trace = [f"Question: {question}"]
        try:
            schema = self.query_service.get_schema_context()
            plan = self.planner.create_plan(
                question,
                schema=schema,
                max_steps=self.max_analysis_steps,
            )
        except (duckdb.Error, LLMError, OSError, ValueError, TypeError) as exc:
            return self._failure(question, f"Analysis planning failed: {exc}", trace)

        trace.append(f"Plan: {json.dumps(plan.as_dict(), ensure_ascii=True)}")
        steps: list[AnalysisStep] = []
        analysis_steps = plan.steps[: self.max_analysis_steps]
        limited = len(plan.steps) > self.max_analysis_steps

        for number, goal in enumerate(analysis_steps, start=1):
            trace.append(f"Step {number} goal: {goal}")
            prior_results = [step.as_evidence() for step in steps if step.error is None]
            result: QueryResult = self.query_service.answer_question(
                question,
                objective=goal,
                prior_results=prior_results,
            )
            trace.extend(result.trace[1:])
            if not result.success and result.sql.strip():
                trace.append(
                    f"Step {number}: retrying once with SQL validation/execution feedback."
                )
                retry_context = [
                    *prior_results,
                    {
                        "step": number,
                        "goal": goal,
                        "sql": result.sql,
                        "error": result.error,
                    },
                ]
                retry_result = self.query_service.answer_question(
                    question,
                    objective=goal,
                    prior_results=retry_context,
                    validation_feedback=result.error,
                )
                trace.extend(retry_result.trace[1:])
                result = retry_result

            step = AnalysisStep(
                number=number,
                goal=goal,
                sql=result.sql,
                data=result.data,
                summary=summarize_dataframe(
                    result.data,
                    max_rows=MAX_CONTEXT_ROWS,
                ),
                error=result.error if not result.success else None,
                validation=self._validation_status(result.trace, result.success),
            )
            steps.append(step)
            trace.append(f"Step {number} SQL: {result.sql or '(not generated)'}")
            if step.error:
                trace.append(f"Step {number} error: {step.error}")
                if not any(item.error is None and not item.data.empty for item in steps[:-1]):
                    return AnalysisResult(
                        success=False,
                        question=question,
                        plan=plan,
                        steps=steps,
                        error=step.error,
                        trace=self._emit(trace),
                        limited=limited,
                    )
                trace.append("Stopping further queries after a follow-up failure.")
                break

            trace.append(
                f"Step {number} result: {step.summary['row_count']} row(s); "
                f"columns={step.summary['columns']}"
            )
            if step.data.empty:
                trace.append(
                    f"Step {number} returned no rows; remaining steps will be skipped."
                )
                break

        if limited:
            trace.append(
                f"Analysis limited to {self.max_analysis_steps} steps."
            )
        if not any(step.error is None and not step.data.empty for step in steps):
            return AnalysisResult(
                success=False,
                question=question,
                plan=plan,
                steps=steps,
                error="No analysis step returned usable evidence.",
                trace=self._emit(trace),
                limited=limited,
            )

        answer_steps = [
            AnswerStep(goal=step.goal, sql=step.sql, data=step.data, error=step.error)
            for step in steps
        ]
        try:
            answer = self.answer_generator.generate(
                question,
                plan,
                answer_steps,
                trace,
            )
        except (LLMError, ValueError, TypeError) as exc:
            error = f"Final answer generation failed: {exc}"
            trace.append(error)
            fallback_answer = (
                "FACT: The executed query evidence is retained in the analysis trace.\n"
                "INTERPRETATION: The final synthesis could not be verified, so no "
                "causal conclusion is asserted."
            )
            trace.append(f"Conservative fallback: {fallback_answer}")
            return AnalysisResult(
                success=False,
                question=question,
                plan=plan,
                steps=steps,
                answer=fallback_answer,
                error=error,
                trace=self._emit(trace),
                limited=limited,
            )

        answer_verified = verify_answer(answer, answer_steps)
        if not answer_verified:
            error = "Final answer failed evidence verification."
            trace.append(error)
            fallback_answer = (
                "FACT: Query results were produced and are available in the analysis trace.\n"
                "INTERPRETATION: The supplied evidence did not support a verified "
                "conclusion, so no causal claim is made."
            )
            trace.append(f"Conservative fallback: {fallback_answer}")
            return AnalysisResult(
                success=False,
                question=question,
                plan=plan,
                steps=steps,
                answer=fallback_answer,
                error=error,
                trace=self._emit(trace),
                limited=limited,
                answer_verified=False,
            )

        trace.append(f"Final answer: {answer}")
        return AnalysisResult(
            success=True,
            question=question,
            plan=plan,
            steps=steps,
            answer=answer,
            trace=self._emit(trace),
            limited=limited,
            answer_verified=answer_verified,
        )

    @staticmethod
    def _validation_status(trace: list[str], success: bool) -> str:
        for item in trace:
            if item.startswith("Validation:\n"):
                return item.partition("\n")[2]
        return "PASSED" if success else "FAILED or not reached"

    def _failure(
        self,
        question: str,
        error: str,
        trace: list[str],
    ) -> AnalysisResult:
        trace.append(error)
        return AnalysisResult(
            success=False,
            question=question,
            error=error,
            trace=self._emit(trace),
        )

    def _emit(self, trace: list[str]) -> list[str]:
        if self.debug:
            print("\n".join(trace))
        return trace
