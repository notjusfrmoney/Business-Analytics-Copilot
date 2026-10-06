from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.query_service import QueryService
from src.analytics.sql_generator import SQLGenerator
from src.analytics.sql_validator import SQLValidator
from src.database.duckdb_manager import DuckDBManager
from src.llm.client import LLMClient, LLMError


SAMPLE_DIR = ROOT / "data" / "sample"

QUESTION_SQL = {
    "What is the total revenue?": (
        "SELECT ROUND(SUM(revenue), 2) AS total_revenue FROM orders"
    ),
    "Which region generated the highest revenue?": (
        "SELECT region, ROUND(SUM(revenue), 2) AS revenue "
        "FROM orders GROUP BY region ORDER BY revenue DESC LIMIT 1"
    ),
    "Which region has the highest profit?": (
        "SELECT region, ROUND(SUM(revenue - cost), 2) AS profit "
        "FROM orders GROUP BY region ORDER BY profit DESC LIMIT 1"
    ),
    "What were the monthly revenues?": (
        "SELECT strftime(CAST(order_date AS DATE), '%Y-%m') AS month, "
        "ROUND(SUM(revenue), 2) AS revenue FROM orders "
        "GROUP BY month ORDER BY month"
    ),
    "What are the top 10 products by revenue?": (
        "SELECT p.product_name, ROUND(SUM(o.revenue), 2) AS revenue "
        "FROM orders o JOIN products p ON o.product_id = p.product_id "
        "GROUP BY p.product_name ORDER BY revenue DESC LIMIT 10"
    ),
    "Which product category generates the highest revenue?": (
        "SELECT p.category, ROUND(SUM(o.revenue), 2) AS revenue "
        "FROM orders o JOIN products p ON o.product_id = p.product_id "
        "GROUP BY p.category ORDER BY revenue DESC LIMIT 1"
    ),
    "Which customer segment generates the most revenue?": (
        "SELECT c.segment, ROUND(SUM(o.revenue), 2) AS revenue "
        "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
        "GROUP BY c.segment ORDER BY revenue DESC LIMIT 1"
    ),
    "Which regions missed their revenue target?": (
        "WITH actual AS ("
        "SELECT region, SUM(revenue) AS actual_revenue FROM orders GROUP BY region"
        "), targets_by_region AS ("
        "SELECT region, SUM(revenue_target) AS target_revenue "
        "FROM targets GROUP BY region"
        ") SELECT a.region, ROUND(a.actual_revenue, 2) AS actual_revenue, "
        "ROUND(t.target_revenue, 2) AS target_revenue "
        "FROM actual a JOIN targets_by_region t ON a.region = t.region "
        "WHERE a.actual_revenue < t.target_revenue ORDER BY a.region"
    ),
}


class FakeLLMClient:
    def __init__(self, response_for: dict[str, dict[str, Any] | str] | None = None) -> None:
        self.response_for = response_for or {}
        self.prompts: list[str] = []

    def generate_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        question = prompt.split("USER QUESTION:\n", maxsplit=1)[-1].strip()
        response = self.response_for.get(question)
        if response is not None:
            return json.dumps(response) if isinstance(response, dict) else response
        return json.dumps(
            {
                "can_answer": True,
                "sql": QUESTION_SQL[question],
                "explanation": "Answers the requested business question from the available data.",
            }
        )


class NaturalLanguageSQLPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not all((SAMPLE_DIR / f"{name}.csv").exists() for name in ("orders", "customers", "products", "targets")):
            raise RuntimeError("Phase 1 sample CSV files are missing; generate them before testing.")

    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_directory.name) / "test.duckdb"
        with DuckDBManager(self.db_path) as database:
            for table_name in ("orders", "customers", "products", "targets"):
                database.load_csv(
                    SAMPLE_DIR / f"{table_name}.csv",
                    table_name=table_name,
                )
        self.llm = FakeLLMClient()
        self.service = QueryService(
            self.db_path,
            llm_client=self.llm,
            debug=False,
        )

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_eight_business_questions_generate_valid_sql_and_results(self) -> None:
        for question in QUESTION_SQL:
            with self.subTest(question=question):
                result = self.service.answer_question(question)
                self.assertTrue(result.success, result.error)
                self.assertTrue(result.sql)
                self.assertFalse(result.data.empty)
                self.assertIn("Validation:\nPASSED", result.trace)
                self.assertIn("Execution:\nSUCCESS", result.trace)

        prompt = self.llm.prompts[0]
        self.assertIn("TABLE orders", prompt)
        self.assertIn("TABLE targets", prompt)
        self.assertIn("orders.customer_id -> customers.customer_id", prompt)
        self.assertIn('"orders": 2200', prompt)
        self.assertNotIn("Customer 1", prompt)
        self.assertIn("Return exactly one read-only SQL query", prompt)
        self.assertIn("Do not return multiple queries", prompt)

    def test_unknown_table_is_rejected_before_execution(self) -> None:
        self._set_response("unknown table", "SELECT name FROM employees")
        result = self.service.answer_question("unknown table")
        self.assertFalse(result.success)
        self.assertIn("Unknown table", result.error or "")

    def test_sql_code_fence_flows_through_validation_and_execution(self) -> None:
        self._set_response(
            "fenced sql",
            "The query is:\n```sql\n"
            "SELECT region, SUM(revenue) AS revenue FROM orders "
            "GROUP BY region ORDER BY revenue DESC LIMIT 1;\n```",
        )
        result = self.service.answer_question("fenced sql")
        self.assertTrue(result.success, result.error)
        self.assertIn("Validation:\nPASSED", result.trace)
        self.assertFalse(result.data.empty)

    def test_unknown_column_is_rejected_by_duckdb_explain(self) -> None:
        self._set_response("unknown column", "SELECT salary FROM orders")
        result = self.service.answer_question("unknown column")
        self.assertFalse(result.success)
        self.assertIn("Invalid SQL", result.error or "")

    def test_malformed_sql_is_rejected_before_execution(self) -> None:
        self._set_response("malformed", "SELECT FROM orders")
        result = self.service.answer_question("malformed")
        self.assertFalse(result.success)
        self.assertIn("Invalid SQL", result.error or "")

    def test_empty_question_does_not_call_llm(self) -> None:
        result = self.service.answer_question("  ")
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Question is empty.")
        self.assertEqual(self.llm.prompts, [])

    def test_unanswerable_question_returns_clear_refusal(self) -> None:
        self._set_response(
            "employee salaries",
            {
                "can_answer": False,
                "sql": "",
                "explanation": (
                    "I couldn't answer this question because the available data "
                    "does not contain employee or salary information."
                ),
            },
        )
        result = self.service.answer_question("employee salaries")
        self.assertFalse(result.success)
        self.assertIn("does not contain employee or salary", result.error or "")

    def test_missing_openrouter_configuration_returns_structured_error(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "",
                "OPENROUTER_MODEL": "",
                "OPENROUTER_BASE_URL": "",
            },
        ):
            result = QueryService(self.db_path, debug=False).answer_question(
                "What is the total revenue?"
            )
        self.assertFalse(result.success)
        self.assertIn("OPENROUTER_API_KEY", result.error or "")
        self.assertIn("OPENROUTER_MODEL", result.error or "")

    def test_validator_rejects_empty_destructive_and_multi_statement_sql(self) -> None:
        validator = SQLValidator()
        self.assertFalse(validator.validate("").valid)
        self.assertFalse(validator.validate("DELETE FROM orders").valid)
        self.assertFalse(validator.validate("SELECT 1; SELECT 2").valid)
        self.assertTrue(validator.validate("SELECT 'DROP'; -- read only").valid)
        self.assertTrue(
            validator.validate(
                "SELECT EXTRACT(MONTH FROM CAST(order_date AS DATE)) FROM orders",
                {"orders"},
            ).valid
        )
        with DuckDBManager(self.db_path) as database:
            validation = validator.validate(
                "SELECT EXTRACT(MONTH FROM CAST(order_date AS DATE)) FROM orders",
                {"orders"},
                connection=database.connection,
            )
        self.assertTrue(validation.valid, validation.error)

    def _set_response(self, question: str, sql_or_response: str | dict[str, Any]) -> None:
        self.llm.response_for[question] = sql_or_response


class SQLOutputParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = SQLGenerator(llm_client=Mock())

    def test_parses_plain_sql(self) -> None:
        result = self.generator._parse_response(
            "SELECT region, SUM(revenue) AS revenue\n"
            "FROM orders\n"
            "GROUP BY region\n"
            "ORDER BY revenue DESC\n"
            "LIMIT 1;"
        )
        self.assertTrue(result["can_answer"])
        self.assertEqual(result["sql"], "SELECT region, SUM(revenue) AS revenue\nFROM orders\nGROUP BY region\nORDER BY revenue DESC\nLIMIT 1;")

    def test_parses_sql_markdown_fence(self) -> None:
        result = self.generator._parse_response(
            "```sql\nSELECT region, SUM(revenue) AS revenue "
            "FROM orders GROUP BY region ORDER BY revenue DESC LIMIT 1;\n```"
        )
        self.assertEqual(result["sql"], "SELECT region, SUM(revenue) AS revenue FROM orders GROUP BY region ORDER BY revenue DESC LIMIT 1;")

    def test_extracts_sql_with_explanatory_text_around_it(self) -> None:
        result = self.generator._parse_response(
            "Here is the SQL query:\n"
            "SELECT region, SUM(revenue) AS revenue FROM orders "
            "GROUP BY region ORDER BY revenue DESC LIMIT 1;\n"
            "This ranks regions by revenue."
        )
        self.assertTrue(result["can_answer"])
        self.assertTrue(result["sql"].startswith("SELECT"))
        self.assertTrue(result["sql"].endswith("LIMIT 1;"))
        self.assertNotIn("Here is the SQL", result["sql"])

    def test_preserves_legacy_structured_json_and_refusal(self) -> None:
        answer = self.generator._parse_response(
            '{"can_answer": true, "sql": "SELECT 1", "explanation": "test"}'
        )
        self.assertTrue(answer["can_answer"])
        self.assertEqual(answer["sql"], "SELECT 1")

        refusal = self.generator._parse_response(
            "-- CANNOT_ANSWER: The schema has no employee salary data."
        )
        self.assertFalse(refusal["can_answer"])
        self.assertIn("no employee salary", refusal["explanation"])

    def test_rejects_empty_output_and_does_not_fabricate_sql(self) -> None:
        with self.assertRaisesRegex(LLMError, "empty response"):
            self.generator._parse_response("  ")

        result = self.generator._parse_response(
            "The model could not identify a query."
        )
        self.assertFalse(result["can_answer"])
        self.assertEqual(result["sql"], "")

    def test_sql_generator_retries_once_for_non_sql_model_text(self) -> None:
        class RetryClient:
            def __init__(self) -> None:
                self.responses = iter(
                    (
                        "User Safety: safe",
                        "```sql\nSELECT 1;\n```",
                    )
                )
                self.prompts: list[str] = []

            def generate_text(self, prompt: str) -> str:
                self.prompts.append(prompt)
                return next(self.responses)

        client = RetryClient()
        generator = SQLGenerator(client)
        proposal = generator.generate(
            "Simple lookup",
            {"items": [{"column_name": "id", "column_type": "INTEGER"}]},
            {},
            {},
        )
        self.assertTrue(proposal["can_answer"])
        self.assertEqual(proposal["sql"], "SELECT 1;")
        self.assertEqual(len(client.prompts), 2)
        self.assertIn("previous response was not usable SQL", client.prompts[1])

    def test_explicit_cannot_answer_response_does_not_retry(self) -> None:
        class RefusalClient:
            def __init__(self) -> None:
                self.calls = 0

            def generate_text(self, prompt: str) -> str:
                self.calls += 1
                return "-- CANNOT_ANSWER: This schema has no employee data."

        client = RefusalClient()
        proposal = SQLGenerator(client).generate(
            "Employee salaries",
            {},
            {},
            {},
        )
        self.assertFalse(proposal["can_answer"])
        self.assertEqual(client.calls, 1)


class OpenRouterClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key-never-print",
                "OPENROUTER_MODEL": "qwen/qwen3.8-27b:free",
                "OPENROUTER_BASE_URL": "https://openrouter.example/api/v1/",
            },
        )
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.client = LLMClient()

    def _response(self, status_code: int, body: Any) -> Mock:
        response = Mock()
        response.status_code = status_code
        response.ok = 200 <= status_code < 400
        response.json.return_value = body
        return response

    def test_generate_text_sends_openrouter_chat_completion_request(self) -> None:
        content = "SELECT 1"
        response = self._response(
            200,
            {"choices": [{"message": {"content": content}}]},
        )
        with patch("src.llm.client.requests.post", return_value=response) as post:
            result = self.client.generate_text("Return a SQL query.")

        self.assertEqual(result, content)
        args, kwargs = post.call_args
        self.assertEqual(
            args[0],
            "https://openrouter.example/api/v1/chat/completions",
        )
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key-never-print")
        self.assertEqual(kwargs["json"]["model"], "qwen/qwen3.8-27b:free")
        self.assertNotIn("response_format", kwargs["json"])

    def test_api_auth_rate_limit_and_model_errors_are_concise(self) -> None:
        cases = (
            (401, "Invalid", "API key"),
            (429, "Rate limited", "quota"),
            (404, "Model unavailable", "OPENROUTER_MODEL"),
            (503, "test-key-never-print provider failure", "OpenRouter API error"),
        )
        for status, message, expected in cases:
            with self.subTest(status=status):
                response = self._response(
                    status,
                    {"error": {"message": message}},
                )
                with patch("src.llm.client.requests.post", return_value=response):
                    with self.assertRaises(LLMError) as caught:
                        self.client.generate_text("Prompt")
                self.assertIn(expected, str(caught.exception))
                self.assertNotIn("test-key-never-print", str(caught.exception))

    def test_timeout_and_network_failure_are_reported_without_secrets(self) -> None:
        cases = (
            (requests.Timeout, "timed out"),
            (requests.ConnectionError, "connect"),
        )
        for exception, expected in cases:
            with self.subTest(exception=exception.__name__):
                with patch(
                    "src.llm.client.requests.post",
                    side_effect=exception("test-key-never-print"),
                ):
                    with self.assertRaises(LLMError) as caught:
                        self.client.generate_text("Prompt")
                self.assertIn(expected, str(caught.exception))
                self.assertNotIn("test-key-never-print", str(caught.exception))

    def test_invalid_response_and_missing_fields_are_errors(self) -> None:
        bad_body = self._response(200, {"unexpected": "response"})
        with patch("src.llm.client.requests.post", return_value=bad_body):
            with self.assertRaisesRegex(LLMError, "did not contain a model message"):
                self.client.generate_text("Prompt")

    def test_generate_text_rejects_empty_content(self) -> None:
        empty = self._response(200, {"choices": [{"message": {"content": "  "}}]})
        with patch("src.llm.client.requests.post", return_value=empty):
            with self.assertRaisesRegex(LLMError, "empty model response"):
                self.client.generate_text("Prompt")


if __name__ == "__main__":
    unittest.main()
