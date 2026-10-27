
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Sales_Data.csv"


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["SALES_WEEK"] = pd.to_datetime(df["SALES_WEEK"], errors="coerce")

    print("\n=== TARGET VALUE VALIDATION ===")
    non_integer_actual = ~np.isclose(
        df["ACTUAL"],
        np.round(df["ACTUAL"]),
    )

    print(f"Zero ACTUAL values: {(df['ACTUAL'] == 0).sum():,}")
    print(f"Negative ACTUAL values: {(df['ACTUAL'] < 0).sum():,}")
    print(f"Non-integer ACTUAL values: {non_integer_actual.sum():,}")

    if non_integer_actual.any():
        print(df.loc[non_integer_actual, ["PRODUCT_CODE", "SALES_WEEK", "ACTUAL"]].head(10))

    print("\n=== PRODUCT-CATEGORY INCONSISTENCIES ===")
    category_counts = df.groupby("PRODUCT_CODE")["CATEGORY_CODE"].nunique()
    multi_category_products = category_counts[category_counts > 1].index

    print(f"Products with multiple categories: {len(multi_category_products):,}")

    if len(multi_category_products) > 0:
        inconsistent_rows = df.loc[
            df["PRODUCT_CODE"].isin(multi_category_products),
            [
                "PRODUCT_CODE",
                "CATEGORY_CODE",
                "SALES_WEEK",
                "ACTUAL",
                "AVG_UNIT_PRICE",
            ],
        ].sort_values(["PRODUCT_CODE", "SALES_WEEK"])

        print(inconsistent_rows.to_string(index=False))

    print("\n=== ZERO OR NEGATIVE PRICES ===")
    nonpositive_price = df["AVG_UNIT_PRICE"] <= 0
    zero_price_rows = df.loc[nonpositive_price]

    print(f"Rows with nonpositive prices: {len(zero_price_rows):,}")
    print(f"Distinct affected products: {zero_price_rows['PRODUCT_CODE'].nunique():,}")
    print(f"Distinct affected weeks: {zero_price_rows['SALES_WEEK'].nunique():,}")

    if not zero_price_rows.empty:
        print("\nAffected products and price counts:")
        print(
            zero_price_rows.groupby("PRODUCT_CODE")
            .agg(
                rows=("AVG_UNIT_PRICE", "size"),
                categories=("CATEGORY_CODE", "nunique"),
                weeks=("SALES_WEEK", "nunique"),
            )
            .sort_values("rows", ascending=False)
            .head(15)
            .to_string()
        )

    print("\n=== EXTREME PRICE INVESTIGATION ===")
    price_99th = df["AVG_UNIT_PRICE"].quantile(0.99)
    price_999th = df["AVG_UNIT_PRICE"].quantile(0.999)

    print(f"99th percentile price: {price_99th:.2f}")
    print(f"99.9th percentile price: {price_999th:.2f}")
    print(f"Rows above 99th percentile: {(df['AVG_UNIT_PRICE'] > price_99th).sum():,}")
    print(f"Rows above 99.9th percentile: {(df['AVG_UNIT_PRICE'] > price_999th).sum():,}")

    extreme_rows = df.nlargest(15, "AVG_UNIT_PRICE")[
        [
            "PRODUCT_CODE",
            "CATEGORY_CODE",
            "SALES_WEEK",
            "ACTUAL",
            "AVG_UNIT_PRICE",
        ]
    ]

    print("\n15 highest-price records:")
    print(extreme_rows.to_string(index=False))

    extreme_products = extreme_rows["PRODUCT_CODE"].unique()
    product_price_history = (
        df[df["PRODUCT_CODE"].isin(extreme_products)]
        .groupby("PRODUCT_CODE")
        .agg(
            rows=("AVG_UNIT_PRICE", "size"),
            categories=("CATEGORY_CODE", "nunique"),
            first_week=("SALES_WEEK", "min"),
            last_week=("SALES_WEEK", "max"),
            minimum_price=("AVG_UNIT_PRICE", "min"),
            median_price=("AVG_UNIT_PRICE", "median"),
            maximum_price=("AVG_UNIT_PRICE", "max"),
        )
        .sort_values("maximum_price", ascending=False)
    )

    print("\nPrice history for products in the extreme-price records:")
    print(product_price_history.to_string())


if __name__ == "__main__":
    main()
