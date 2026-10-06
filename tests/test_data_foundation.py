from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_sample_data import generate_sample_data
from src.database.duckdb_manager import DuckDBManager


def test_csv_to_duckdb_sql_pipeline() -> None:
    sample_files = generate_sample_data(Path(__file__).resolve().parents[1])

    required_files = ["orders", "customers", "products", "targets"]
    for name in required_files:
        assert sample_files[name].exists(), f"Missing sample file: {sample_files[name]}"

    db_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "business_analytics.duckdb"
    db_manager = DuckDBManager(db_path)

    try:
        for table_name in required_files:
            db_manager.load_csv(sample_files[table_name], table_name=table_name, if_exists="replace")

        tables = db_manager.list_tables()
        assert set(required_files).issubset(tables), f"Expected tables missing: {tables}"

        orders_schema = db_manager.get_schema("orders")
        assert "order_id" in orders_schema["column_name"].tolist()

        total_revenue = db_manager.execute_query("SELECT SUM(revenue) AS total_revenue FROM orders").iloc[0, 0]
        assert total_revenue is not None
        assert float(total_revenue) > 0

        revenue_by_region = db_manager.execute_query(
            "SELECT region, ROUND(SUM(revenue), 2) AS revenue FROM orders GROUP BY region ORDER BY revenue DESC"
        )
        assert not revenue_by_region.empty

        profit_by_region = db_manager.execute_query(
            "SELECT region, ROUND(SUM(revenue - cost), 2) AS profit FROM orders GROUP BY region ORDER BY profit DESC"
        )
        assert not profit_by_region.empty

        monthly_revenue = db_manager.execute_query(
            "SELECT strftime(CAST(order_date AS DATE), '%Y-%m') AS month, ROUND(SUM(revenue), 2) AS revenue FROM orders GROUP BY 1 ORDER BY 1 LIMIT 12"
        )
        assert not monthly_revenue.empty

        top_products = db_manager.execute_query(
            "SELECT product_id, ROUND(SUM(revenue), 2) AS revenue FROM orders GROUP BY product_id ORDER BY revenue DESC LIMIT 10"
        )
        assert not top_products.empty

    finally:
        db_manager.close()


if __name__ == "__main__":
    test_csv_to_duckdb_sql_pipeline()
    print("CSV -> DuckDB -> SQL -> Pandas DataFrame pipeline is working correctly.")
