from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_data(base_dir: str | Path | None = None) -> dict[str, Path]:
    """Generate a small, reproducible business dataset inside data/sample/."""
    root_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[1]
    sample_dir = root_dir / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for file_path in sample_dir.glob("*.csv"):
        file_path.unlink()

    rng = np.random.default_rng(42)
    regions = ["North", "South", "East", "West"]
    segments = ["Enterprise", "SMB", "Consumer", "Public Sector"]

    product_specs = {
        "P001": {"product_name": "Laptop Pro", "category": "Electronics", "subcategory": "Laptops", "base_price": 1200, "cost_ratio": 0.72},
        "P002": {"product_name": "Laptop Air", "category": "Electronics", "subcategory": "Laptops", "base_price": 980, "cost_ratio": 0.70},
        "P003": {"product_name": "Ultra Monitor", "category": "Electronics", "subcategory": "Monitors", "base_price": 420, "cost_ratio": 0.63},
        "P004": {"product_name": "Docking Station", "category": "Electronics", "subcategory": "Accessories", "base_price": 180, "cost_ratio": 0.50},
        "P005": {"product_name": "Wireless Mouse", "category": "Accessories", "subcategory": "Peripherals", "base_price": 55, "cost_ratio": 0.45},
        "P006": {"product_name": "Mechanical Keyboard", "category": "Accessories", "subcategory": "Peripherals", "base_price": 90, "cost_ratio": 0.52},
        "P007": {"product_name": "Office Chair", "category": "Furniture", "subcategory": "Seating", "base_price": 260, "cost_ratio": 0.58},
        "P008": {"product_name": "Standing Desk", "category": "Furniture", "subcategory": "Desks", "base_price": 500, "cost_ratio": 0.60},
        "P009": {"product_name": "Filing Cabinet", "category": "Furniture", "subcategory": "Storage", "base_price": 310, "cost_ratio": 0.57},
        "P010": {"product_name": "Analytics Suite", "category": "Software", "subcategory": "Analytics", "base_price": 240, "cost_ratio": 0.42},
        "P011": {"product_name": "CRM License", "category": "Software", "subcategory": "CRM", "base_price": 120, "cost_ratio": 0.38},
        "P012": {"product_name": "Video Conferencing", "category": "Software", "subcategory": "Collaboration", "base_price": 150, "cost_ratio": 0.40},
    }

    products_df = pd.DataFrame(
        [
            {
                "product_id": product_id,
                "product_name": values["product_name"],
                "category": values["category"],
                "subcategory": values["subcategory"],
            }
            for product_id, values in product_specs.items()
        ]
    )

    customers = []
    for customer_number in range(1, 151):
        region = regions[(customer_number - 1) % len(regions)]
        segment = segments[(customer_number - 1) % len(segments)]
        signup_date = pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 2000), unit="D")
        customers.append(
            {
                "customer_id": f"C{customer_number:04d}",
                "customer_name": f"Customer {customer_number}",
                "segment": segment,
                "region": region,
                "signup_date": signup_date.strftime("%Y-%m-%d"),
            }
        )
    customers_df = pd.DataFrame(customers)

    order_rows = []
    for order_number in range(1, 2201):
        customer = customers_df.iloc[order_number % len(customers_df)]
        product_id = list(product_specs)[(order_number - 1) % len(product_specs)]
        product = product_specs[product_id]
        region = customer["region"]
        quantity = int(rng.integers(1, 10))
        unit_price = round(float(product["base_price"] * rng.uniform(0.85, 1.25)), 2)
        discount = round(float(rng.uniform(0.04, 0.24)), 4)
        revenue = round(quantity * unit_price * (1 - discount), 2)
        unit_cost = round(float(unit_price * product["cost_ratio"] * rng.uniform(0.94, 1.08)), 2)
        cost = round(quantity * unit_cost, 2)
        order_date = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 730), unit="D")

        order_rows.append(
            {
                "order_id": f"O{order_number:05d}",
                "order_date": order_date.strftime("%Y-%m-%d"),
                "customer_id": customer["customer_id"],
                "product_id": product_id,
                "region": region,
                "quantity": quantity,
                "unit_price": unit_price,
                "discount": discount,
                "revenue": revenue,
                "cost": cost,
            }
        )

    orders_df = pd.DataFrame(order_rows)

    target_rows = []
    month_starts = pd.period_range(start="2023-01-01", end="2024-12-01", freq="M")
    revenue_factor = {"North": 1.18, "South": 0.95, "East": 1.05, "West": 1.12}
    monthly_pattern = [0.92, 0.94, 1.02, 1.05, 1.07, 1.10, 1.12, 1.18, 1.15, 1.20, 1.30, 1.42] * 2

    for index, month in enumerate(month_starts):
        month_date = month.to_timestamp()
        for region in regions:
            revenue_target = round(130000 * monthly_pattern[index] * revenue_factor[region], 2)
            profit_target = round(revenue_target * 0.18, 2)
            target_rows.append(
                {
                    "month": month_date.strftime("%Y-%m-%d"),
                    "region": region,
                    "revenue_target": revenue_target,
                    "profit_target": profit_target,
                }
            )

    targets_df = pd.DataFrame(target_rows)

    file_map = {
        "orders": sample_dir / "orders.csv",
        "customers": sample_dir / "customers.csv",
        "products": sample_dir / "products.csv",
        "targets": sample_dir / "targets.csv",
    }

    orders_df.to_csv(file_map["orders"], index=False)
    customers_df.to_csv(file_map["customers"], index=False)
    products_df.to_csv(file_map["products"], index=False)
    targets_df.to_csv(file_map["targets"], index=False)

    return file_map


if __name__ == "__main__":
    generate_sample_data()
    print("Sample business data generated successfully.")
