
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Sales_Data.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["SALES_WEEK"] = pd.to_datetime(df["SALES_WEEK"], errors="raise")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    weekly = (
        df.groupby("SALES_WEEK", as_index=False)
        .agg(
            total_actual=("ACTUAL", "sum"),
            mean_actual=("ACTUAL", "mean"),
            median_actual=("ACTUAL", "median"),
            records=("PRODUCT_CODE", "size"),
            products_present=("PRODUCT_CODE", "nunique"),
        )
        .sort_values("SALES_WEEK")
    )

    summary_path = REPORTS_DIR / "weekly_summary.csv"
    weekly.to_csv(summary_path, index=False)

    print("\n=== WEEKLY SUMMARY ===")
    print(weekly.to_string(index=False))

    print("\n=== ADDITIONAL EDA COUNTS ===")
    print(f"Rows in dataset: {len(df):,}")
    print(f"Distinct products overall: {df['PRODUCT_CODE'].nunique():,}")
    print(f"Zero-price rows: {(df['AVG_UNIT_PRICE'] == 0).sum():,}")
    print(f"Positive-price rows: {(df['AVG_UNIT_PRICE'] > 0).sum():,}")

    # 1. Sales quantity distribution.
    actual = df["ACTUAL"]
    bins = np.geomspace(actual.min(), actual.max(), 45)

    plt.figure(figsize=(9, 5))
    plt.hist(actual, bins=bins, edgecolor="white")
    plt.xscale("log")
    plt.xlabel("ACTUAL quantity (logarithmic scale)")
    plt.ylabel("Number of product-week records")
    plt.title("Distribution of Recorded Sales Quantity")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "actual_distribution.png", dpi=150)
    plt.close()

    # 2. Total recorded quantity by week.
    plt.figure(figsize=(10, 5))
    plt.plot(
        weekly["SALES_WEEK"],
        weekly["total_actual"],
        marker="o",
    )
    plt.xlabel("Sales week")
    plt.ylabel("Sum of ACTUAL")
    plt.title("Total Recorded Sales Quantity by Week")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "weekly_recorded_sales.png", dpi=150)
    plt.close()

    # 3. Number of products represented each week.
    plt.figure(figsize=(10, 5))
    plt.plot(
        weekly["SALES_WEEK"],
        weekly["products_present"],
        marker="o",
    )
    plt.xlabel("Sales week")
    plt.ylabel("Distinct products with records")
    plt.title("Product Coverage by Week")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "weekly_product_coverage.png", dpi=150)
    plt.close()

    # 4. Positive prices only; zero prices are counted separately above.
    positive_prices = df.loc[
        df["AVG_UNIT_PRICE"] > 0, "AVG_UNIT_PRICE"
    ]
    price_bins = np.geomspace(
        positive_prices.min(),
        positive_prices.max(),
        50,
    )

    plt.figure(figsize=(9, 5))
    plt.hist(positive_prices, bins=price_bins, edgecolor="white")
    plt.xscale("log")
    plt.xlabel("AVG_UNIT_PRICE (logarithmic scale)")
    plt.ylabel("Number of records")
    plt.title("Distribution of Positive Unit Prices")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "positive_price_distribution.png", dpi=150)
    plt.close()

    print(f"\nWeekly summary saved to: {summary_path}")
    print(f"Charts saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
