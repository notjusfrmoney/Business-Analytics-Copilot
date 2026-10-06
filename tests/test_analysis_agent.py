from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics import AnalysisAgent, AnalysisPlan, AnalysisPlanner
from src.analytics.answer_generator import AnswerGenerator, AnswerStep
from src.analytics.sql_generator import SQLGenerator
from src.database.duckdb_manager import DuckDBManager
from src.llm.client import LLMError

SAMPLE_DIR = ROOT / "data" / "sample"

ROOT_CAUSE_QUESTION = "Why did profit decline in Q3?"

PLANS = {
    "Which region generated the highest revenue?": AnalysisPlan(
        "ranking",
        "revenue",
        "",
        ["Rank regions by total revenue"],
    ),
    "Which product category generates the highest revenue?": AnalysisPlan(
        "ranking",
        "revenue",
        "",
        ["Rank product categories by revenue"],
    ),
    ROOT_CAUSE_QUESTION: AnalysisPlan(
        "root_cause_analysis",
        "profit",
        "Q3",
        [
            "Compare quarterly profit over time",
            "Break profit down by region",
            "Break profit down by product category",
        ],
    ),
    "Which regions contributed most to the profit decline?": AnalysisPlan(
        "root_cause_analysis",
        "profit",
        "Q3",
        [
            "Compare profit by region across quarters",
            "Rank regions by quarterly profit change",
        ],
    ),
    "What explains the change in revenue over the last few months?": AnalysisPlan(
        "trend_analysis",
        "revenue",
        "recent months",
        [
            "Aggregate revenue by month",
            "Compare monthly revenue changes",
        ],
    ),
    "Which product categories are responsible for the decline in profit?": AnalysisPlan(
        "root_cause_analysis",
        "profit",
        "",
        [
            "Compare profit across product categories",
            "Rank product categories by profit decline",
        ],
    ),
}

OBJECTIVE_SQL = {
    "Rank regions by total revenue": (
        "SELECT region, SUM(revenue) AS revenue FROM orders "
        "GROUP BY region ORDER BY revenue DESC LIMIT 1"
    ),
    "Rank product categories by revenue": (
        "SELECT p.category, SUM(o.revenue) AS revenue FROM orders o "
        "JOIN products p ON o.product_id = p.product_id "
        "GROUP BY p.category ORDER BY revenue DESC LIMIT 1"
    ),
    "Compare quarterly profit over time": (
        "SELECT date_part('year', CAST(order_date AS DATE)) AS year, "
        "date_part('quarter', CAST(order_date AS DATE)) AS quarter, "
        "SUM(revenue - cost) AS profit FROM orders "
        "GROUP BY year, quarter ORDER BY year, quarter"
    ),
    "Break profit down by region": (
        "SELECT region, SUM(revenue - cost) AS profit FROM orders "
        "GROUP BY region ORDER BY profit"
    ),
    "Break profit down by product category": (
        "SELECT p.category, SUM(o.revenue - o.cost) AS profit FROM orders o "
        "JOIN products p ON o.product_id = p.product_id "
        "GROUP BY p.category ORDER BY profit"
    ),
    "Compare profit by region across quarters": (
        "SELECT region, date_part('quarter', CAST(order_date AS DATE)) AS quarter, "
        "SUM(revenue - cost) AS profit FROM orders "
        "GROUP BY region, quarter ORDER BY region, quarter"
    ),
    "Rank regions by quarterly profit change": (
        "SELECT region, SUM(revenue - cost) AS profit FROM orders "
        "GROUP BY region ORDER BY profit"
    ),
    "Aggregate revenue by month": (
        "SELECT strftime(CAST(order_date AS DATE), '%Y-%m') AS month, "
        "SUM(revenue) AS revenue FROM orders GROUP BY month ORDER BY month"
    ),
    "Compare monthly revenue changes": (
        "SELECT strftime(CAST(order_date AS DATE), '%Y-%m') AS month, "
        "SUM(revenue) AS revenue FROM orders GROUP BY month ORDER BY month"
    ),
    "Compare profit across product categories": (
        "SELECT p.category, SUM(o.revenue - o.cost) AS profit FROM orders o "
        "JOIN products p ON o.product_id = p.product_id "
        "GROUP BY p.category ORDER BY profit"
    ),
    "Rank product categories by profit decline": (
        "SELECT p.category, SUM(o.revenue - o.cost) AS profit FROM orders o "
        "JOIN products p ON o.product_id = p.product_id "
        "GROUP BY p.category ORDER BY profit"
    ),
}


class FakePlanner:
    def __init__(self, plans: dict[str, AnalysisPlan] | None = None) -> None:
        self.plans = plans or PLANS
        self.questions: list[str] = []

    def create_plan(
        self,
        question: str,
        *,
        schema: dict[str, list[dict[str, str]]] | None = None,
        max_steps: int = 4,
    ) -> AnalysisPlan:
        self.questions.append(question)
        if schema is not None:
            assert "orders" in schema
            assert "revenue" in {
                column["column_name"] for column in schema["orders"]
            }
        plan = self.plans[question]
        return AnalysisPlan(
            intent=plan.intent,
            metric=plan.metric,
            time_period=plan.time_period,
            steps=plan.steps,
        )


class FakeSQLTextGenerator:
    def __init__(self, fail_objective: str | None = None) -> None:
        self.objectives: list[str] = []
        self.prior_contexts: list[list[dict[str, Any]]] = []
        self.fail_objective = fail_objective
        self.feedbacks: list[str | None] = []
        self.always_fail = False

    def generate_text(self, prompt: str) -> str:
        objective = prompt.split("ANALYSIS OBJECTIVE:\n", maxsplit=1)[1]
        objective = objective.split("\n\nPRIOR ANALYSIS RESULTS", maxsplit=1)[0].strip()
        prior_text = prompt.split(
            "PRIOR ANALYSIS RESULTS (metadata and compact result samples only):\n",
            maxsplit=1,
        )[1]
        prior_text = prior_text.split(        "\n\nSQL VALIDATION FEEDBACK:",
        maxsplit=1,
        )[0]
        prior_text = prior_text.split(
        "\n\nUSER QUESTION:", maxsplit=1)[0].strip()
        self.objectives.append(objective)
        self.prior_contexts.append(json.loads(prior_text))
        feedback = prompt.split("SQL VALIDATION FEEDBACK:\n", maxsplit=1)[1]
        feedback = feedback.split("\n\nUSER QUESTION:", maxsplit=1)[0].strip()
        self.feedbacks.append(feedback)
        if objective == self.fail_objective:
            if self.always_fail or self.feedbacks[-1] == "No previous SQL validation failure.":
                return "SELECT unknown_column FROM orders"
            return OBJECTIVE_SQL[objective]
        return OBJECTIVE_SQL[objective]


class FakeAnswerGenerator:
    def __init__(self) -> None:
        self.received_steps: list[AnswerStep] = []

    def generate(
        self,
        question: str,
        plan: AnalysisPlan,
        steps: list[AnswerStep],
        trace: list[str],
    ) -> str:
        self.received_steps = steps
        return (
            "The profit and revenue query results show the completed regional and "
            "category breakdowns. These results identify the dimensions available "
            "for further investigation."
        )


class AnalysisAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not all(
            (SAMPLE_DIR / f"{table}.csv").exists()
            for table in ("orders", "customers", "products", "targets")
        ):
            raise RuntimeError("Phase 1 sample CSV files are missing.")

    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_directory.name) / "phase3.duckdb"
        with DuckDBManager(self.db_path) as database:
            for table in ("orders", "customers", "products", "targets"):
                database.load_csv(SAMPLE_DIR / f"{table}.csv", table_name=table)
        self.planner = FakePlanner()
        self.sql_generator = FakeSQLTextGenerator()
        self.answer_generator = FakeAnswerGenerator()

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def _agent(self, *, max_steps: int = 4) -> AnalysisAgent:
        return AnalysisAgent(
            self.db_path,
            planner=self.planner,
            sql_generator=self.sql_generator,
            answer_generator=self.answer_generator,
            max_analysis_steps=max_steps,
            debug=False,
        )

    def test_simple_ranking_questions_use_one_validated_query(self) -> None:
        for question in (
            "Which region generated the highest revenue?",
            "Which product category generates the highest revenue?",
        ):
            with self.subTest(question=question):
                result = self._agent().answer_question(question)
                self.assertTrue(result.success, result.error)
                self.assertEqual(len(result.steps), 1)
                self.assertFalse(result.steps[0].data.empty)
                self.assertIn("Validation:\nPASSED", "\n".join(result.trace))

    def test_multi_step_questions_execute_more_than_one_query(self) -> None:
        questions = (
            ROOT_CAUSE_QUESTION,
            "Which regions contributed most to the profit decline?",
            "What explains the change in revenue over the last few months?",
            "Which product categories are responsible for the decline in profit?",
        )
        for question in questions:
            with self.subTest(question=question):
                agent = self._agent()
                result = agent.answer_question(question)
                self.assertTrue(result.success, result.error)
                self.assertGreater(len(result.steps), 1)
                self.assertTrue(all(not step.data.empty for step in result.steps))
                self.assertIn("profit", result.answer)
                self.assertTrue(result.answer_verified)
                self.assertEqual(result.steps[0].validation, "PASSED")
                self.assertIn("Validation:\nPASSED", "\n".join(result.trace) if result.trace else "")

    def test_prior_step_summary_is_passed_to_follow_up_sql_generation(self) -> None:
        result = self._agent().answer_question(ROOT_CAUSE_QUESTION)
        self.assertTrue(result.success, result.error)
        self.assertGreater(len(self.sql_generator.prior_contexts[1]), 0)
        prior = self.sql_generator.prior_contexts[1][0]
        self.assertEqual(prior["goal"], "Compare quarterly profit over time")
        self.assertIn("sample_rows", prior["summary"])
        self.assertLessEqual(len(prior["summary"]["sample_rows"]), 8)

    def test_maximum_steps_is_enforced_and_reported(self) -> None:
        result = self._agent(max_steps=2).answer_question(ROOT_CAUSE_QUESTION)
        self.assertTrue(result.success, result.error)
        self.assertEqual(len(result.steps), 2)
        self.assertTrue(result.limited)
        self.assertTrue(any("limited to 2 steps" in item for item in result.trace))

    def test_failed_follow_up_preserves_earlier_evidence(self) -> None:
        self.sql_generator.fail_objective = "Break profit down by region"
        self.sql_generator.always_fail = True
        result = self._agent().answer_question(ROOT_CAUSE_QUESTION)
        self.assertTrue(result.success, result.error)
        self.assertEqual(len(result.steps), 2)
        self.assertIsNotNone(result.steps[1].error)
        self.assertIn("Stopping further queries", "\n".join(result.trace))

    def test_invalid_query_is_retried_once_with_validator_feedback(self) -> None:
        self.sql_generator.fail_objective = "Break profit down by region"
        result = self._agent().answer_question(ROOT_CAUSE_QUESTION)
        self.assertTrue(result.success, result.error)
        self.assertIsNone(result.steps[1].error)
        self.assertEqual(len(result.steps[1].data), 4)
        self.assertTrue(
            any("Invalid SQL" in feedback for feedback in self.sql_generator.feedbacks)
        )
        self.assertIn("retrying once", "\n".join(result.trace))

    def test_invalid_final_synthesis_returns_conservative_fallback(self) -> None:
        class BrokenAnswerTextGenerator:
            def generate_text(self, prompt: str) -> str:
                return "User Safety: safe"

        result = AnalysisAgent(
            self.db_path,
            planner=self.planner,
            sql_generator=self.sql_generator,
            answer_generator=AnswerGenerator(BrokenAnswerTextGenerator()),
            debug=False,
        ).answer_question(ROOT_CAUSE_QUESTION)
        self.assertFalse(result.success)
        self.assertIn("FACT:", result.answer)
        self.assertIn("INTERPRETATION:", result.answer)
        self.assertIn("no causal conclusion", result.answer)

    def test_final_answer_allows_normal_grounded_prose_and_verifies_numbers(self) -> None:
        class GroundedAnswerTextGenerator:
            def generate_text(self, prompt: str) -> str:
                return (
                    "Profit fell by about $33.4K quarter over quarter, while "
                    "revenue and Electronics results provide additional context."
                )

        plan = PLANS[ROOT_CAUSE_QUESTION]
        data = pd.DataFrame(
            [
                {
                    "category": "Electronics",
                    "q2_profit": 259505.23,
                    "q3_profit": 226080.86,
                    "profit_diff": -33424.37,
                    "revenue": 100000.0,
                }
            ]
        )
        answerer = AnswerGenerator(GroundedAnswerTextGenerator())
        self.assertEqual(
            answerer.generate(
                ROOT_CAUSE_QUESTION,
                plan,
                [AnswerStep("Profit by category", "SELECT ...", data)],
                [],
            ),
            "Profit fell by about $33.4K quarter over quarter, while "
            "revenue and Electronics results provide additional context.",
        )

        class UnsupportedNumberTextGenerator:
            def generate_text(self, prompt: str) -> str:
                return "Profit was 999999, led by Electronics."

        unsupported_answerer = AnswerGenerator(UnsupportedNumberTextGenerator())
        with self.assertRaisesRegex(LLMError, "unsupported numerical claims"):
            unsupported_answerer.generate(
                ROOT_CAUSE_QUESTION,
                plan,
                [AnswerStep("Profit by category", "SELECT ...", data)],
                [],
            )

    def test_planner_failure_returns_structured_error(self) -> None:
        class BrokenPlanner:
            def create_plan(
                self,
                question: str,
                *,
                schema: dict[str, list[dict[str, str]]] | None = None,
                max_steps: int = 4,
            ) -> AnalysisPlan:
                raise LLMError("Planner response was malformed.")

        result = AnalysisAgent(
            self.db_path,
            planner=BrokenPlanner(),
            sql_generator=self.sql_generator,
            answer_generator=self.answer_generator,
            debug=False,
        ).answer_question(ROOT_CAUSE_QUESTION)
        self.assertFalse(result.success)
        self.assertIn("Analysis planning failed", result.error or "")

    def test_planner_rejects_malformed_or_unsupported_plans(self) -> None:
        with self.assertRaisesRegex(LLMError, "malformed JSON"):
            AnalysisPlanner.parse_plan("not-json")
        with self.assertRaisesRegex(LLMError, "unsupported analysis intent"):
            AnalysisPlanner.parse_plan(
                '{"intent":"unknown","metric":"profit","steps":["analyze"]}'
            )

    def test_planner_repairs_malformed_json_once(self) -> None:
        class RepairClient:
            def __init__(self) -> None:
                self.responses = iter(
                    (
                        "I will analyze the question.",
                        '```json\n{"intent":"ranking","metric":"revenue",'
                        '"time_period":"","steps":["Rank regions by revenue"]}\n```',
                    )
                )
                self.calls = 0

            def generate_text(self, prompt: str) -> str:
                self.calls += 1
                return next(self.responses)

        client = RepairClient()
        plan = AnalysisPlanner(client).create_plan("Top region?", max_steps=3)
        self.assertEqual(plan.intent, "ranking")
        self.assertEqual(client.calls, 2)

    def test_result_summary_is_compact_and_computes_statistics(self) -> None:
        from src.analytics.answer_generator import summarize_dataframe

        dataframe = pd.DataFrame(
            {
                "month": [f"m{i}" for i in range(12)],
                "revenue": range(12),
                "q2_profit": [100.0] * 12,
                "q3_profit": [90.0] * 12,
            }
        )
        summary = summarize_dataframe(dataframe, max_rows=8)
        self.assertEqual(summary["row_count"], 12)
        self.assertEqual(len(summary["sample_rows"]), 8)
        self.assertTrue(summary["truncated"])
        self.assertEqual(summary["numeric_summary"]["revenue"]["sum"], 66)
        self.assertEqual(len(summary["derived_quarter_comparisons"]), 8)
        self.assertEqual(
            summary["derived_quarter_comparisons"][0]["percent_change"],
            -10,
        )

    def test_percentage_change_is_computed_in_python_and_available_as_evidence(self) -> None:
        from src.analytics.answer_generator import summarize_dataframe

        summary = summarize_dataframe(
            pd.DataFrame(
                [
                    {
                        "region": "North",
                        "profit_change": -10.0,
                        "previous_profit": 100.0,
                    }
                ]
            )
        )
        self.assertEqual(summary["derived_changes"][0]["percent_change"], -10)


if __name__ == "__main__":
    unittest.main()
