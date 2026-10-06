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
    compare_dataframes,
    generate_expected_results,
    load_questions,
)
from src.analytics import AnalysisAgent  # noqa: E402
from src.analytics.answer_generator import AnswerGenerator  # noqa: E402
from src.analytics.planner import AnalysisPlanner  # noqa: E402
from src.analytics.sql_generator import SQLGenerator  # noqa: E402
from src.llm.client import LLMClient, LLMError  # noqa: E402
from src.utils.config import get_openrouter_settings  # noqa: E402

LIVE_IDS = ("q03", "q06", "q07", "q14", "q18")
RESULTS_DIR = ROOT / "evaluation" / "results"


class CountingLiveClient:
    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.calls = 0

    def generate_text(self, prompt: str) -> str:
        self.calls += 1
        return self.client.generate_text(prompt)


def run_live_evaluation() -> dict[str, Any]:
    try:
        _, _, _ = get_openrouter_settings()
    except ValueError as exc:
        raise SystemExit(f"Live evaluation requires OPENROUTER_API_KEY and model settings: {exc}") from exc

    questions_by_id = {item["id"]: item for item in load_questions()}
    expected_results = generate_expected_results(
        database_path=DEFAULT_DATABASE_PATH,
    )
    live_items = [questions_by_id[identifier] for identifier in LIVE_IDS]
    rows: list[dict[str, Any]] = []
    rate_limited = False

    for item in live_items:
        if rate_limited:
            rows.append(_skipped_row(item, "Skipped after an OpenRouter HTTP 429 response."))
            continue

        client = CountingLiveClient(LLMClient())
        agent = AnalysisAgent(
            db_path=DEFAULT_DATABASE_PATH,
            planner=AnalysisPlanner(client),
            sql_generator=client,
            answer_generator=AnswerGenerator(client),
            debug=False,
        )
        started = time.perf_counter()
        try:
            result = agent.answer_question(item["question"])
        except LLMError as exc:
            elapsed = time.perf_counter() - started
            error = str(exc)
            rate_limited = "HTTP 429" in error
            rows.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "latency_seconds": round(elapsed, 6),
                    "llm_calls": client.calls,
                    "success": False,
                    "sql_queries": 0,
                    "result_matches": False,
                    "answer_grounded": False,
                    "error": error,
                    "answer": "",
                    "trace": [],
                }
            )
            continue
        elapsed = time.perf_counter() - started
        if result.error and "HTTP 429" in result.error:
            rate_limited = True

        final_data = next(
            (step.data for step in reversed(result.steps) if step.error is None),
            pd.DataFrame(),
        )
        expected = expected_results[item["id"]]
        comparison = compare_dataframes(
            pd.DataFrame(expected["rows"], columns=expected["columns"]),
            final_data,
        )
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "latency_seconds": round(elapsed, 6),
                "llm_calls": client.calls,
                "success": result.success,
                "sql_queries": sum(
                    entry.startswith("Generated SQL:\n") for entry in result.trace
                ),
                "result_matches": comparison.matches,
                "comparison_reason": comparison.reason,
                "answer_grounded": result.answer_verified,
                "error": result.error,
                "answer": result.answer,
                "trace": result.trace,
            }
        )

    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "live",
        "question_count": len(rows),
        "rate_limited": rate_limited,
        "questions": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "latest_live_results.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = format_live_report(rows, rate_limited)
    (RESULTS_DIR / "latest_live_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return output


def _skipped_row(item: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "id": item["id"],
        "question": item["question"],
        "latency_seconds": 0.0,
        "llm_calls": 0,
        "success": False,
        "sql_queries": 0,
        "result_matches": False,
        "answer_grounded": False,
        "error": reason,
        "answer": "",
        "trace": [],
    }


def format_live_report(rows: list[dict[str, Any]], rate_limited: bool) -> str:
    completed = [row for row in rows if row["llm_calls"] > 0]
    lines = [
        "BUSINESS ANALYTICS COPILOT LIVE EVALUATION",
        f"Questions selected: {len(rows)}",
        f"Questions started: {len(completed)}",
        f"Provider rate-limited: {'yes' if rate_limited else 'no'}",
        f"Successful analyses: {sum(row['success'] for row in completed)}/{len(completed)}",
        f"Matching results: {sum(row['result_matches'] for row in completed)}/{len(completed)}",
        f"Evidence-grounded answers: {sum(row['answer_grounded'] for row in completed)}/{len(completed)}",
        f"Average latency: {sum(row['latency_seconds'] for row in completed) / len(completed) if completed else 0:.3f}s",
        "",
    ]
    for row in rows:
        status = "PASS" if row["success"] and row["result_matches"] and row["answer_grounded"] else "FAIL"
        lines.append(
            f"{row['id']} | {status} | {row['latency_seconds']:.3f}s | "
            f"{row['llm_calls']} LLM calls | {row['question']}"
        )
        if row["error"]:
            lines.append(f"  {row['error']}")
        if row["answer"]:
            lines.append(f"  {row['answer']}")
    lines.append("Secrets are not recorded or printed.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    run_live_evaluation()
