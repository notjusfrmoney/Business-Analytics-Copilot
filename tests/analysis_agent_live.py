from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics import AnalysisAgent
from src.analytics.answer_generator import summarize_dataframe
from src.utils.config import get_openrouter_settings


def main() -> int:
    question = "Why did profit decline in Q3?"
    try:
        get_openrouter_settings()
    except ValueError as exc:
        print(f"Live analysis not run: {exc}")
        return 1

    result = AnalysisAgent(debug=False).answer_question(question)
    print(f"Question: {question}")
    if result.plan is not None:
        print(f"Plan: {result.plan.as_dict()}")
    for step in result.steps:
        print(f"\nStep {step.number}: {step.goal}")
        print(f"SQL: {step.sql or '(not generated)'}")
        print(f"Validation: {step.validation}")
        if step.error:
            print(f"Error: {step.error}")
        else:
            print(
                "Result summary: "
                + json.dumps(
                    summarize_dataframe(step.data, max_rows=8),
                    ensure_ascii=True,
                    default=str,
                )
            )
    print(f"\nAnalysis steps: {len(result.steps)}")
    print(
        "Queries executed: "
        + str(sum(step.validation == "PASSED" for step in result.steps))
    )
    print(f"Workflow successful: {result.success}")
    if result.error:
        print(f"Workflow note: {result.error}")
    print(f"Final answer passed evidence verification: {result.answer_verified}")
    print(f"Final answer: {result.answer or result.error}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
