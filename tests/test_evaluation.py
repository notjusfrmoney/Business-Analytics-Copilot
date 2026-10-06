from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evaluation.evaluator import (
    DEFAULT_DATABASE_PATH,
    compare_dataframes,
    generate_expected_results,
    load_expected_results,
    load_questions,
)
from evaluation.run_evaluation import DeterministicEvaluationLLM, _summarize
from src.analytics import AnalysisAgent, AnalysisPlan
from src.analytics.answer_generator import AnswerGenerator
from src.analytics.planner import AnalysisPlanner
from src.analytics.query_service import QueryService
from src.analytics.sql_generator import SQLGenerator
from src.database.duckdb_manager import DuckDBManager
from src.llm.client import LLMClient, LLMError

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample"


class StaticTextGenerator:
    def __init__(self, response: str | None = None, error: Exception | None = None) -> None:
        self.response = response or ""
        self.error = error
        self.calls = 0

    def generate_text(self, prompt: str) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.response


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.database_path = self.directory / "test.duckdb"
        with DuckDBManager(self.database_path) as database:
            for table in ("orders", "customers", "products", "targets"):
                database.load_csv(SAMPLE_DIR / f"{table}.csv", table_name=table)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_benchmark_has_representative_categories_and_supported_questions(self) -> None:
        questions = load_questions()
        self.assertGreaterEqual(len(questions), 25)
        self.assertLessEqual(len(questions), 30)
        self.assertEqual(sum(item["answerable"] for item in questions), 22)
        self.assertEqual(sum(not item["answerable"] for item in questions), 5)
        self.assertTrue(
            {
                "aggregation",
                "simple_lookup",
                "ranking",
                "comparison",
                "trend_analysis",
                "multi_table_join",
                "target_vs_actual",
                "root_cause_analysis",
                "unsupported",
            }.issubset({item["category"] for item in questions})
        )

    def test_expected_results_are_generated_by_executing_question_sql(self) -> None:
        question = next(item for item in load_questions() if item["id"] == "q01")
        question_path = self.directory / "questions.json"
        question_path.write_text(json.dumps([question]), encoding="utf-8")
        output_path = self.directory / "expected.json"

        results = generate_expected_results(
            database_path=self.database_path,
            questions_path=question_path,
            output_path=output_path,
        )

        with DuckDBManager(self.database_path) as database:
            direct = database.execute_query(question["expected_sql"])
        saved = json.loads(output_path.read_text(encoding="utf-8"))["results"]["q01"]
        self.assertEqual(results["q01"]["columns"], ["total_revenue"])
        self.assertEqual(saved["rows"][0]["total_revenue"], direct.iloc[0, 0])

    def test_sample_expected_results_file_covers_every_answerable_question(self) -> None:
        expected = load_expected_results()
        answerable_ids = {
            item["id"] for item in load_questions() if item["answerable"]
        }
        self.assertEqual(set(expected), answerable_ids)

    def test_evaluation_metrics_use_defined_denominators_and_latency_means(self) -> None:
        summary = _summarize(
            [
                {
                    "answerable": True,
                    "sql_executed": True,
                    "result_matches": True,
                    "answer_grounded": True,
                    "query_correct": True,
                    "unsupported_refusal": None,
                    "agentic": True,
                    "agentic_success": True,
                    "latency_seconds": 0.2,
                    "llm_calls": 3,
                    "sql_queries": 2,
                    "analysis_steps": 2,
                },
                {
                    "answerable": False,
                    "sql_executed": False,
                    "result_matches": False,
                    "answer_grounded": None,
                    "query_correct": True,
                    "unsupported_refusal": True,
                    "agentic": False,
                    "agentic_success": None,
                    "latency_seconds": 0.1,
                    "llm_calls": 2,
                    "sql_queries": 0,
                    "analysis_steps": 1,
                },
            ],
            answerable_count=1,
            unsupported_count=1,
            agentic_count=1,
        )
        self.assertEqual(summary["result_accuracy"]["percent"], 100.0)
        self.assertEqual(summary["query_success_rate"]["passed"], 2)
        self.assertEqual(summary["agentic_analysis_rate"]["percent"], 100.0)
        self.assertEqual(summary["average_latency_seconds"], 0.15)

    def test_dataframe_comparison_tolerates_float_noise_and_irrelevant_order(self) -> None:
        expected = pd.DataFrame(
            {"region": ["North", "South"], "revenue": [100.0, 200.0]}
        )
        actual = pd.DataFrame(
            {"revenue": [200.05, 100.02], "region": ["South", "North"]}
        )
        comparison = compare_dataframes(expected, actual)
        self.assertTrue(comparison.matches, comparison.reason)

    def test_dataframe_comparison_rejects_wrong_values_and_handles_empty_results(self) -> None:
        expected = pd.DataFrame({"revenue": [100.0]})
        wrong = pd.DataFrame({"revenue": [105.0]})
        self.assertFalse(compare_dataframes(expected, wrong).matches)
        empty = pd.DataFrame({"region": pd.Series(dtype=str)})
        self.assertTrue(compare_dataframes(empty, empty.copy()).matches)
        self.assertFalse(
            compare_dataframes(empty, pd.DataFrame({"region": ["North"]})).matches
        )

    def test_empty_question_fails_without_calling_the_planner(self) -> None:
        class NoCallPlanner:
            called = False

            def create_plan(self, *args: object, **kwargs: object) -> AnalysisPlan:
                self.called = True
                raise AssertionError("Empty questions must not invoke the planner.")

        planner = NoCallPlanner()
        result = AnalysisAgent(self.database_path, planner=planner).answer_question(" ")
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Question is empty.")
        self.assertFalse(planner.called)

    def test_unknown_column_and_malformed_sql_fail_gracefully(self) -> None:
        for sql in ("SELECT employee_salary FROM orders", "SELECT FROM orders"):
            with self.subTest(sql=sql):
                client = StaticTextGenerator(sql)
                result = QueryService(
                    self.database_path,
                    llm_client=client,
                    debug=False,
                ).answer_question("Run this benchmark test query.")
                self.assertFalse(result.success)
                self.assertIn("Invalid SQL", result.error or "")
                self.assertEqual(client.calls, 1)

    def test_empty_sql_result_is_a_successful_empty_dataframe(self) -> None:
        client = StaticTextGenerator(
            "SELECT region FROM orders WHERE 1 = 0"
        )
        result = QueryService(
            self.database_path,
            llm_client=client,
            debug=False,
        ).answer_question("Return no rows.")
        self.assertTrue(result.success, result.error)
        self.assertTrue(result.data.empty)
        self.assertIn("Execution:\nSUCCESS", result.trace)

    def test_llm_api_failure_returns_a_structured_query_error(self) -> None:
        client = StaticTextGenerator(error=LLMError("OpenRouter HTTP 429 rate limit."))
        result = QueryService(
            self.database_path,
            llm_client=client,
            debug=False,
        ).answer_question("What is total revenue?")
        self.assertFalse(result.success)
        self.assertIn("HTTP 429", result.error or "")

    def test_openrouter_rate_limit_message_is_concise_and_redacted(self) -> None:
        message = LLMClient._format_api_error(
            429,
            {"error": {"message": "secret token"}},
            api_key="secret token",
        )
        self.assertIn("HTTP 429", message)
        self.assertNotIn("secret token", message)

    def test_malformed_planner_output_returns_analysis_error_without_crashing(self) -> None:
        client = StaticTextGenerator("not a plan", error=None)
        result = AnalysisAgent(
            self.database_path,
            planner=AnalysisPlanner(client),
            debug=False,
        ).answer_question("Explain the regional revenue result.")
        self.assertFalse(result.success)
        self.assertIn("planning failed", result.error or "")
        self.assertEqual(client.calls, 2)

    def test_unsupported_benchmark_question_is_refused_without_sql(self) -> None:
        item = next(question for question in load_questions() if question["id"] == "q22")
        client = DeterministicEvaluationLLM(item)
        result = AnalysisAgent(
            self.database_path,
            planner=AnalysisPlanner(client),
            sql_generator=client,
            answer_generator=AnswerGenerator(client),
            debug=False,
        ).answer_question(item["question"])
        self.assertFalse(result.success)
        self.assertIn("employee or salary", result.error or "")
        self.assertEqual(sum(line.startswith("Generated SQL:") for line in result.trace), 0)

    def test_failed_follow_up_preserves_prior_query_evidence(self) -> None:
        plan = AnalysisPlan(
            "root_cause_analysis",
            "revenue",
            "",
            ["Aggregate revenue by region", "Run a failing follow-up"],
        )

        class FixedPlanner:
            def create_plan(self, *args: object, **kwargs: object) -> AnalysisPlan:
                return plan

        class FollowUpFailureGenerator:
            def generate_text(self, prompt: str) -> str:
                objective = prompt.split("ANALYSIS OBJECTIVE:\n", maxsplit=1)[1]
                objective = objective.split("\n\nPRIOR ANALYSIS RESULTS", maxsplit=1)[0]
                if objective == plan.steps[0]:
                    return (
                        "SELECT region, SUM(revenue) AS revenue "
                        "FROM orders GROUP BY region"
                    )
                return "SELECT missing_column FROM orders"

        class GroundedAnswerClient:
            def generate_text(self, prompt: str) -> str:
                return "The query results contain a region field and revenue values."

        result = AnalysisAgent(
            self.database_path,
            planner=FixedPlanner(),
            sql_generator=FollowUpFailureGenerator(),
            answer_generator=AnswerGenerator(GroundedAnswerClient()),
            debug=False,
        ).answer_question("Why did regional revenue change?")
        self.assertTrue(result.success, result.error)
        self.assertEqual(len(result.steps), 2)
        self.assertFalse(result.steps[0].data.empty)
        self.assertIsNotNone(result.steps[1].error)
        self.assertTrue(result.answer_verified)


if __name__ == "__main__":
    unittest.main()
