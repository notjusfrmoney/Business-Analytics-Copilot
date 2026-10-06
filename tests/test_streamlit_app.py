from __future__ import annotations

import importlib
import io
import tempfile
import unittest
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd

from app import _build_analysis_question, _markdown_safe_answer
from src.analytics.query_service import QueryService
from src.database.duckdb_manager import DuckDBManager
from src.ui.charts import create_auto_chart
from src.ui.data_loader import UploadDataError, load_uploaded_files


class NamedBytesIO(io.BytesIO):
    def __init__(self, name: str, content: bytes) -> None:
        super().__init__(content)
        self.name = name


class StreamlitAppSmokeTests(unittest.TestCase):
    def test_app_and_backend_modules_import(self) -> None:
        for module_name in (
            "app",
            "src.ui.components",
            "src.ui.charts",
            "src.ui.data_loader",
            "src.analytics.analysis_agent",
            "src.analytics.query_service",
            "src.database.duckdb_manager",
        ):
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_uploads_load_csv_and_xlsx_into_safe_tables(self) -> None:
        csv_content = b"Customer ID,Revenue\n1,120\n2,90\n"
        excel_buffer = io.BytesIO()
        pd.DataFrame({"category": ["Hardware", "Software"], "profit": [12, 18]}).to_excel(
            excel_buffer,
            index=False,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "session.duckdb"
            tables = load_uploaded_files(
                [
                    NamedBytesIO("Orders.csv", csv_content),
                    NamedBytesIO("orders.xlsx", excel_buffer.getvalue()),
                ],
                database_path,
            )

        self.assertEqual([table["table_name"] for table in tables], ["orders", "orders_2"])
        self.assertEqual(tables[0]["row_count"], 2)
        self.assertEqual(tables[0]["schema"][0]["column_name"], "customer_id")

    def test_upload_rejects_empty_files_and_normalized_duplicate_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "session.duckdb"
            with self.assertRaisesRegex(UploadDataError, "no data rows"):
                load_uploaded_files(
                    [NamedBytesIO("empty.csv", b"column\n")],
                    database_path,
                )
            with self.assertRaisesRegex(UploadDataError, "duplicate column names"):
                load_uploaded_files(
                    [NamedBytesIO("duplicate.csv", b"Customer ID,customer-id\n1,2\n")],
                    database_path,
                )
            with self.assertRaises(UploadDataError):
                load_uploaded_files(
                    [NamedBytesIO("corrupt.xlsx", b"not an Excel workbook")],
                    database_path,
                )

    def test_query_service_only_shares_relationships_present_in_live_schema(self) -> None:
        class CapturingGenerator:
            relationships: dict[str, str] | None = None

            def generate(self, question: str, schema: object, relationships: object, metadata: object, **kwargs: object) -> dict[str, object]:
                self.relationships = relationships
                return {
                    "can_answer": False,
                    "sql": "",
                    "explanation": "Insufficient schema for this question.",
                }

        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "orders_only.duckdb"
            with DuckDBManager(database_path) as database:
                database.load_dataframe(pd.DataFrame({"customer_id": [1]}), "orders")
            generator = CapturingGenerator()
            QueryService(
                database_path,
                sql_generator=generator,
                debug=False,
            ).answer_question("Join orders to customers.")

        self.assertEqual(generator.relationships, {})

    def test_follow_up_uses_verified_chat_context_but_runs_as_a_new_question(self) -> None:
        question = _build_analysis_question(
            "What about profit?",
            [
                {"role": "user", "content": "Which region generated the highest revenue?"},
                {
                    "role": "assistant",
                    "content": "South had the highest revenue.",
                    "verification": True,
                },
            ],
        )
        self.assertIn("South had the highest revenue.", question)
        self.assertIn("Run fresh analysis against the active database", question)
        self.assertTrue(question.endswith("Current follow-up question: What about profit?"))

    def test_currency_markers_do_not_trigger_markdown_math_rendering(self) -> None:
        self.assertEqual(
            _markdown_safe_answer("Revenue was $975,951.11."),
            "Revenue was &#36;975,951.11.",
        )

    def test_chart_selector_handles_time_ranking_categories_scatter_and_unsuitable_data(self) -> None:
        time_result = create_auto_chart(
            pd.DataFrame({"month": ["2026-01", "2026-02"], "revenue": [10, 15]})
        )
        ranking_result = create_auto_chart(
            pd.DataFrame({"region": ["North", "South"], "revenue": [20, 10]}),
            question="Which region generated the highest revenue?",
        )
        category_result = create_auto_chart(
            pd.DataFrame({"category": ["A", "B"], "profit": [2, 3]})
        )
        scatter_result = create_auto_chart(
            pd.DataFrame({"revenue": [10, 20, 30], "profit": [1, 3, 2]})
        )
        unsuitable_result = create_auto_chart(pd.DataFrame({"description": ["one", "two"]}))

        self.assertEqual(time_result.chart_type, "line")
        self.assertEqual(ranking_result.chart_type, "horizontal_bar")
        self.assertEqual(category_result.chart_type, "bar")
        self.assertEqual(scatter_result.chart_type, "scatter")
        self.assertIsNone(unsuitable_result.figure)


if __name__ == "__main__":
    unittest.main()
