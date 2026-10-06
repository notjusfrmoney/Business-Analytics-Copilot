from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluator import (  # noqa: E402
    DEFAULT_DATABASE_PATH,
    EXPECTED_RESULTS_PATH,
    compare_dataframes,
    generate_expected_results,
    load_questions,
)
from src.analytics import AnalysisAgent, AnalysisPlan  # noqa: E402
from src.analytics.answer_generator import AnswerGenerator  # noqa: E402
from src.analytics.planner import AnalysisPlanner  # noqa: E402
from src.llm.client import LLMError  # noqa: E402

RESULTS_DIR = ROOT / "evaluation" / "results"


class DeterministicEvaluationLLM:
    """Return reproducible, benchmark-specific responses without external API calls."""

    def __init__(self, question_item: dict[str, Any]) -> None:
        self.item = question_item
        self.calls = 0
        self.calls_by_kind = {"planner": 0, "sql": 0, "answer": 0}
        self.sql_calls = 0
        self._step_index = 0

    def generate_text(self, prompt: str) -> str:
        self.calls += 1
        if prompt.startswith("Create a concise analysis plan"):
            self.calls_by_kind["planner"] += 1
            return json.dumps(self.item["plan"])

        if prompt.startswith("Return the DuckDB SQL query"):
            self.calls_by_kind["sql"] += 1
            self.sql_calls += 1
            if not self.item["answerable"]:
                reason = self.item["expected_reason"]
                return f"-- CANNOT_ANSWER: {reason}"
            step_sql = self.item.get("mock_step_sql")
            if step_sql:
                sql = step_sql[min(self._step_index, len(step_sql) - 1)]
                self._step_index += 1
                return sql
            return self.item["mock_sql"]

        if prompt.startswith("Answer the business question"):
            self.calls_by_kind["answer"] += 1
            return self._grounded_response(prompt)

        raise LLMError("Evaluation fixture received an unrecognized prompt type.")

    @staticmethod
    def _grounded_response(prompt: str) -> str:
        marker = "QUERY EVIDENCE:\n"
        try:
            evidence = json.loads(prompt.split(marker, maxsplit=1)[1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise LLMError("Could not read evidence from the deterministic answer prompt.") from exc

        for step in reversed(evidence):
            summary = step.get("summary", {})
            columns = summary.get("columns", [])
            rows = summary.get("sample_rows", [])
            first_text = next(
                (
                    value
                    for row in rows
                    for value in row.values()
                    if isinstance(value, str) and value.strip()
                ),
                None,
            )
            if first_text and columns:
                return (
                    f"The query results include {first_text} in the "
                    f"{columns[0]} field."
                )
            if columns:
                return f"The query results report the {columns[0]} field."
        raise LLMError("No result evidence was available to ground a deterministic answer.")


def run_evaluation() -> dict[str, Any]:
    questions = load_questions()
    expected_results = generate_expected_results(
        database_path=DEFAULT_DATABASE_PATH,
    )
    rows: list[dict[str, Any]] = []
    answerable_count = sum(bool(item["answerable"]) for item in questions)
    unsupported_count = len(questions) - answerable_count
    agentic_questions = [
        item
        for item in questions
        if item["answerable"] and item["category"] == "root_cause_analysis"
    ]

    for item in questions:
        llm = DeterministicEvaluationLLM(item)
        agent = AnalysisAgent(
            db_path=DEFAULT_DATABASE_PATH,
            planner=AnalysisPlanner(llm),
            sql_generator=llm,
            answer_generator=AnswerGenerator(llm),
            debug=False,
        )
        started = time.perf_counter()
        result = agent.answer_question(item["question"])
        elapsed = time.perf_counter() - started

        generated_sql_count = sum(
            entry.startswith("Generated SQL:\n") for entry in result.trace
        )
        validated_count = sum(
            entry == "Validation:\nPASSED" for entry in result.trace
        )
        executed_count = sum(
            entry == "Execution:\nSUCCESS" for entry in result.trace
        )
        sql_generated = generated_sql_count > 0
        sql_validated = validated_count > 0 and validated_count == generated_sql_count
        sql_executed = executed_count > 0 and executed_count == validated_count
        result_matches = False
        comparison_reason = "No actual DataFrame was returned."
        if item["answerable"] and result.steps:
            final_evidence = next(
                (step.data for step in reversed(result.steps) if step.error is None),
                None,
            )
            if final_evidence is not None:
                expected = expected_results[item["id"]]
                expected_frame = pd.DataFrame(
                    expected["rows"],
                    columns=expected["columns"],
                )
                comparison = compare_dataframes(expected_frame, final_evidence)
                result_matches = comparison.matches
                comparison_reason = comparison.reason

        if item["answerable"]:
            query_correct = bool(sql_executed and result_matches)
            unsupported_refusal = None
        else:
            reason = item["expected_reason"].casefold()
            error = (result.error or "").casefold()
            unsupported_refusal = (
                bool(not result.success and error and any(
                    term in error for term in _reason_terms(reason)
                ))
            )
            query_correct = unsupported_refusal

        all_steps_valid = bool(result.steps) and all(
            step.validation == "PASSED" and step.error is None
            for step in result.steps
        )
        is_agentic = item in agentic_questions
        agentic_success = (
            bool(
                result.success
                and result.plan is not None
                and len(result.steps) > 1
                and all_steps_valid
                and result.answer_verified
            )
            if is_agentic
            else None
        )
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "category": item["category"],
                "answerable": item["answerable"],
                "sql_generated": sql_generated,
                "sql_validated": sql_validated,
                "sql_executed": sql_executed,
                "sql_pipeline": (
                    bool(sql_generated and sql_validated and sql_executed)
                    if item["answerable"]
                    else None
                ),
                "result_matches": result_matches,
                "query_correct": query_correct,
                "answer_grounded": (
                    bool(result.success and result.answer_verified)
                    if item["answerable"]
                    else None
                ),
                "unsupported_refusal": unsupported_refusal,
                "planner_succeeded": bool(result.plan is not None),
                "analysis_steps": len(result.steps),
                "agentic": is_agentic,
                "agentic_success": agentic_success,
                "llm_calls": llm.calls,
                "llm_calls_by_kind": dict(llm.calls_by_kind),
                "sql_queries": generated_sql_count,
                "validated_queries": validated_count,
                "executed_queries": executed_count,
                "latency_seconds": round(elapsed, 6),
                "comparison_reason": comparison_reason,
                "answer": result.answer,
                "error": result.error,
                "trace": result.trace,
            }
        )

    summary = _summarize(rows, answerable_count, unsupported_count, len(agentic_questions))
    report_text = format_report(summary, rows)
    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "deterministic",
        "database": str(DEFAULT_DATABASE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "question_count": len(questions),
        "summary": summary,
        "questions": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "latest_results.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (RESULTS_DIR / "latest_report.txt").write_text(report_text, encoding="utf-8")
    print(report_text)
    return output


def _reason_terms(reason: str) -> list[str]:
    words = [word for word in reason.split() if len(word) >= 5]
    return words[-4:]


def _summarize(
    rows: list[dict[str, Any]],
    answerable_count: int,
    unsupported_count: int,
    agentic_count: int,
) -> dict[str, Any]:
    supported = [row for row in rows if row["answerable"]]
    unsupported = [row for row in rows if not row["answerable"]]
    agentic_rows = [row for row in rows if row["agentic"]]
    total = len(rows)
    return {
        "total_questions": total,
        "answerable_questions": answerable_count,
        "unsupported_questions": unsupported_count,
        "sql_execution_success": _rate(
            sum(row["sql_executed"] for row in supported), answerable_count
        ),
        "result_accuracy": _rate(
            sum(row["result_matches"] for row in supported), answerable_count
        ),
        "answer_grounding": _rate(
            sum(row["answer_grounded"] is True for row in supported),
            answerable_count,
        ),
        "query_success_rate": _rate(
            sum(row["query_correct"] for row in rows), total
        ),
        "unsupported_refusal_rate": _rate(
            sum(row["unsupported_refusal"] is True for row in unsupported),
            unsupported_count,
        ),
        "agentic_question_count": agentic_count,
        "successful_multi_step_analyses": sum(
            row["agentic_success"] is True for row in agentic_rows
        ),
        "agentic_analysis_rate": _rate(
            sum(row["agentic_success"] is True for row in agentic_rows),
            agentic_count,
        ),
        "average_latency_seconds": _mean(row["latency_seconds"] for row in rows),
        "average_llm_calls": _mean(row["llm_calls"] for row in rows),
        "average_sql_queries": _mean(row["sql_queries"] for row in rows),
        "average_analysis_steps": _mean(row["analysis_steps"] for row in rows),
        "total_llm_calls": sum(row["llm_calls"] for row in rows),
        "total_sql_queries": sum(row["sql_queries"] for row in rows),
    }


def _rate(numerator: int, denominator: int) -> dict[str, float | int]:
    return {
        "passed": numerator,
        "total": denominator,
        "percent": round(100 * numerator / denominator, 2) if denominator else 0.0,
    }


def _mean(values: Any) -> float:
    samples = list(values)
    return round(sum(samples) / len(samples), 6) if samples else 0.0


def format_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    def metric(name: str) -> str:
        value = summary[name]
        return f"{value['passed']}/{value['total']} ({value['percent']:.1f}%)"

    lines = [
        "=" * 88,
        "BUSINESS ANALYTICS COPILOT EVALUATION (DETERMINISTIC)",
        f"Total questions: {summary['total_questions']}",
        f"Answerable: {summary['answerable_questions']}",
        f"Unsupported: {summary['unsupported_questions']}",
        f"SQL execution success: {metric('sql_execution_success')}",
        f"SQL/result accuracy: {metric('result_accuracy')}",
        f"Answer grounding: {metric('answer_grounding')}",
        f"Query success rate (including expected unsupported refusals): {metric('query_success_rate')}",
        f"Unsupported refusal rate: {metric('unsupported_refusal_rate')}",
        f"Agentic questions: {summary['agentic_question_count']}",
        f"Successful multi-step analyses: {summary['successful_multi_step_analyses']}/{summary['agentic_question_count']}",
        f"Average latency: {summary['average_latency_seconds']:.4f}s",
        f"Average mock LLM calls: {summary['average_llm_calls']:.2f}",
        f"Average SQL queries: {summary['average_sql_queries']:.2f}",
        f"Total LLM calls / SQL queries: {summary['total_llm_calls']} / {summary['total_sql_queries']}",
        "",
        "ID   | Category            | SQL | Result | Answer | Steps | Calls | Queries | Time(s) | Question",
        "-" * 150,
    ]
    for row in rows:
        lines.append(
            f"{row['id']:<4} | {row['category']:<19} | "
            f"{_status(row['sql_pipeline']):<3} | "
            f"{_status(row['result_matches'] if row['answerable'] else row['unsupported_refusal']):<6} | "
            f"{_status(row['answer_grounded'] if row['answerable'] else None):<6} | "
            f"{row['analysis_steps']:<5} | {row['llm_calls']:<5} | "
            f"{row['sql_queries']:<7} | {row['latency_seconds']:<7.3f} | "
            f"{row['question']}"
        )
        if row["error"]:
            lines.append(f"     Error: {row['error']}")
        elif row["answerable"] and not row["result_matches"]:
            lines.append(f"     Result mismatch: {row['comparison_reason']}")
    lines.extend(
        [
            "",
            "PASS / FAIL uses the deterministic response fixtures, not a live model.",
            "See evaluation/results/latest_results.json for full traces and per-question metrics.",
            "=" * 88,
        ]
    )
    return "\n".join(lines) + "\n"


def _status(value: bool | None) -> str:
    if value is None:
        return "N/A"
    return "PASS" if value else "FAIL"


if __name__ == "__main__":
    run_evaluation()
