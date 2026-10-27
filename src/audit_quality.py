
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Sales_Data.csv"

KEY_COLUMNS = ["STORE_CODE", "PRODUCT_CODE", "SALES_WEEK"]


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    df["SALES_WEEK_PARSED"] = pd.to_datetime(
        df["SALES_WEEK"],
        errors="coerce",
    )

    print("\n=== DATE COVERAGE ===")
    valid_dates = (
        df["SALES_WEEK_PARSED"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    print(f"Earliest week: {valid_dates.min()}")
    print(f"Latest week:   {valid_dates.max()}")
    print(f"Distinct valid weeks: {len(valid_dates)}")
    print(
        "Unparseable dates: "
        f"{df['SALES_WEEK_PARSED'].isna().sum():,}"
    )

    date_gaps = valid_dates.diff().dt.days.dropna()

    print("\nDays between consecutive observed dates:")
    print(date_gaps.value_counts().sort_index().to_string())

    print("\n=== DUPLICATE STORE-PRODUCT-WEEK RECORDS ===")
    key_columns = [
        "STORE_CODE",
        "PRODUCT_CODE",
        "SALES_WEEK_PARSED",
    ]

    key_counts = df.groupby(key_columns, dropna=False).size()
    duplicate_keys = key_counts[key_counts > 1]

    print(f"Repeated key combinations: {len(duplicate_keys):,}")
    print(
        "Rows belonging to repeated key combinations: "
        f"{duplicate_keys.sum():,}"
    )

    if not duplicate_keys.empty:
        repeated_rows = df.merge(
            duplicate_keys.rename("key_count").reset_index(),
            on=key_columns,
            how="inner",
        )
        print("\nExample repeated records:")
        print(
            repeated_rows[
                key_columns
                + ["ACTUAL", "AVG_UNIT_PRICE", "key_count"]
            ].head(10).to_string(index=False)
        )

    print("\n=== WEEKLY DATA COVERAGE ===")
    weekly_summary = (
        df.groupby("SALES_WEEK_PARSED")
        .agg(
            rows=("PRODUCT_CODE", "size"),
            unique_products=("PRODUCT_CODE", "nunique"),
            unique_categories=("CATEGORY_CODE", "nunique"),
        )
        .sort_index()
    )
    print(weekly_summary.to_string())

    print("\n=== PRODUCT HISTORY LENGTH ===")
    product_weeks = df.groupby("PRODUCT_CODE")[
        "SALES_WEEK_PARSED"
    ].nunique()

    print("Number of distinct weeks per product:")
    print(product_weeks.value_counts().sort_index().to_string())

    print("\n=== PRODUCT CATEGORY CONSISTENCY ===")
    categories_per_product = df.groupby("PRODUCT_CODE")[
        "CATEGORY_CODE"
    ].nunique()

    inconsistent_products = categories_per_product[
        categories_per_product > 1
    ]

    print(
        "Products associated with multiple categories: "
        f"{len(inconsistent_products):,}"
    )
    if not inconsistent_products.empty:
        print("Example product codes:")
        print(inconsistent_products.head(10).to_string())

    print("\n=== ACTUAL DISTRIBUTION ===")
    print(
        df["ACTUAL"]
        .describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.999])
        .to_string()
    )

    print("\n=== PRICE DISTRIBUTION ===")
    print(
        df["AVG_UNIT_PRICE"]
        .describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.999])
        .to_string()
    )

    nonpositive_prices = df["AVG_UNIT_PRICE"] <= 0
    print(
        "Rows with zero or negative price: "
        f"{nonpositive_prices.sum():,}"
    )

    if nonpositive_prices.any():
        print("\nExamples with zero or negative price:")
        print(
            df.loc[
                nonpositive_prices,
                [
                    "STORE_CODE",
                    "PRODUCT_CODE",
                    "CATEGORY_CODE",
                    "SALES_WEEK",
                    "ACTUAL",
                    "AVG_UNIT_PRICE",
                ],
            ].head(10).to_string(index=False)
        )

    print("\nHighest recorded prices:")
    print(
        df.nlargest(10, "AVG_UNIT_PRICE")[
            [
                "PRODUCT_CODE",
                "CATEGORY_CODE",
                "SALES_WEEK",
                "ACTUAL",
                "AVG_UNIT_PRICE",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
