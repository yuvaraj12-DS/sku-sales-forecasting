
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Sales_Data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

KEY_COLUMNS = ["STORE_CODE", "PRODUCT_CODE", "SALES_WEEK"]
REQUIRED_COLUMNS = [
    "STORE_CODE",
    "PRODUCT_CODE",
    "CATEGORY_CODE",
    "SALES_WEEK",
    "ACTUAL",
    "AVG_UNIT_PRICE",
]


def build_weekly_panel(df):
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    df["SALES_WEEK"] = pd.to_datetime(
        df["SALES_WEEK"],
        errors="raise",
    )

    if df["SALES_WEEK"].isna().any():
        raise ValueError("SALES_WEEK contains missing dates.")

    for column in ["STORE_CODE", "PRODUCT_CODE"]:
        if df[column].isna().any():
            raise ValueError(f"{column} contains missing values.")

    df["ACTUAL"] = pd.to_numeric(
        df["ACTUAL"],
        errors="raise",
    )
    df["AVG_UNIT_PRICE"] = pd.to_numeric(
        df["AVG_UNIT_PRICE"],
        errors="raise",
    )

    if not np.isfinite(df["ACTUAL"].to_numpy(dtype=float)).all():
        raise ValueError("ACTUAL contains non-finite values.")

    if df["ACTUAL"].lt(0).any():
        raise ValueError("ACTUAL contains negative sales quantities.")

    if df.duplicated(KEY_COLUMNS).any():
        raise ValueError(
            "Duplicate store-product-week records found."
        )

    observed_weeks = pd.DatetimeIndex(
        df["SALES_WEEK"].drop_duplicates().sort_values()
    )

    if len(observed_weeks) == 0:
        raise ValueError("No valid sales weeks found.")

    # Build a complete weekly calendar, including globally missing weeks.
    weeks = pd.date_range(
        start=observed_weeks.min(),
        end=observed_weeks.max(),
        freq="7D",
    )

    weeks_set = set(weeks)
    off_calendar_weeks = [
        week for week in observed_weeks if week not in weeks_set
    ]

    if off_calendar_weeks:
        raise ValueError(
            "Sales dates do not follow a consistent seven-day cadence. "
            f"Examples: {off_calendar_weeks[:5]}"
        )

    # Keep at least one training week, three validation weeks,
    # and three test weeks.
    if len(weeks) < 7:
        raise ValueError(
            f"Expected at least 7 distinct weekly dates, found {len(weeks)}. "
            "Need at least one training week, three validation weeks, "
            "and three test weeks."
        )

    train_end = len(weeks) - 6
    validation_end = len(weeks) - 3

    split_by_week = {}
    split_by_week.update(
        dict.fromkeys(weeks[:train_end], "train")
    )
    split_by_week.update(
        dict.fromkeys(weeks[train_end:validation_end], "validation")
    )
    split_by_week.update(
        dict.fromkeys(weeks[validation_end:], "test")
    )

    # Identify when each store-product combination first appeared.
    first_seen = (
        df.groupby(
            ["STORE_CODE", "PRODUCT_CODE"],
            as_index=False,
        )
        .agg(FIRST_SEEN_WEEK=("SALES_WEEK", "min"))
    )

    # Generate one row per store-product-week, starting at first appearance.
    calendar = pd.DataFrame({"SALES_WEEK": weeks})
    first_seen["_JOIN_KEY"] = 1
    calendar["_JOIN_KEY"] = 1

    grid = first_seen.merge(
        calendar,
        on="_JOIN_KEY",
        how="inner",
    ).drop(columns="_JOIN_KEY")

    grid = grid.loc[
        grid["SALES_WEEK"] >= grid["FIRST_SEEN_WEEK"]
    ].copy()

    # Merge source records to retain the distinction between recorded
    # sales and zero-filled product-week combinations.
    panel = grid.merge(
        df,
        on=["STORE_CODE", "PRODUCT_CODE", "SALES_WEEK"],
        how="left",
        validate="one_to_one",
        indicator="_SOURCE_MATCH",
    )

    panel["WAS_RECORDED"] = panel["_SOURCE_MATCH"].eq("both")
    panel = panel.drop(columns="_SOURCE_MATCH")

    # Zero-fill only product-week combinations absent from the source.
    panel["ACTUAL"] = panel["ACTUAL"].fillna(0)

    # Use the most recent recorded category for unrecorded weeks.
    panel = panel.sort_values(
        ["STORE_CODE", "PRODUCT_CODE", "SALES_WEEK"]
    ).reset_index(drop=True)

    panel["CATEGORY_CODE"] = (
        panel.groupby(
            ["STORE_CODE", "PRODUCT_CODE"],
            sort=False,
        )["CATEGORY_CODE"]
        .ffill()
    )

    panel["SPLIT"] = panel["SALES_WEEK"].map(split_by_week)

    if panel["SPLIT"].isna().any():
        raise ValueError(
            "Some panel weeks were not assigned to a data split."
        )

    # A product-week can be evaluated only if an earlier panel row exists.
    prior_observation_exists = (
        panel.groupby(
            ["STORE_CODE", "PRODUCT_CODE"],
            sort=False,
        ).cumcount().gt(0)
    )

    panel["VALIDATION_EVALUABLE"] = (
        panel["SPLIT"].eq("validation")
        & prior_observation_exists
    )

    panel["TEST_EVALUABLE"] = (
        panel["SPLIT"].eq("test")
        & prior_observation_exists
    )

    panel_columns = [
        "STORE_CODE",
        "PRODUCT_CODE",
        "CATEGORY_CODE",
        "SALES_WEEK",
        "ACTUAL",
        "AVG_UNIT_PRICE",
        "FIRST_SEEN_WEEK",
        "WAS_RECORDED",
        "SPLIT",
        "VALIDATION_EVALUABLE",
        "TEST_EVALUABLE",
    ]

    panel = panel[panel_columns].sort_values(
        ["PRODUCT_CODE", "SALES_WEEK"]
    ).reset_index(drop=True)

    # Summarize panel coverage for each split and week.
    coverage = (
        panel.groupby(
            ["SPLIT", "SALES_WEEK"],
            as_index=False,
        )
        .agg(
            panel_rows=("PRODUCT_CODE", "size"),
            unique_products=("PRODUCT_CODE", "nunique"),
            unique_categories=("CATEGORY_CODE", "nunique"),
            source_rows=("WAS_RECORDED", "sum"),
            zero_filled_rows=(
                "WAS_RECORDED",
                lambda values: (~values).sum(),
            ),
            validation_evaluable_rows=(
                "VALIDATION_EVALUABLE",
                "sum",
            ),
            test_evaluable_rows=("TEST_EVALUABLE", "sum"),
            total_actual=("ACTUAL", "sum"),
        )
        .sort_values(["SALES_WEEK", "SPLIT"])
        .reset_index(drop=True)
    )

    # Count newly observed products by their first sales week.
    new_products = (
        first_seen.groupby(
            "FIRST_SEEN_WEEK",
            as_index=False,
        )
        .agg(
            new_products=("PRODUCT_CODE", "nunique"),
            new_store_product_pairs=("PRODUCT_CODE", "size"),
        )
        .rename(columns={"FIRST_SEEN_WEEK": "SALES_WEEK"})
        .sort_values("SALES_WEEK")
        .reset_index(drop=True)
    )

    return panel, coverage, new_products, weeks, split_by_week


def main():
    if not RAW_PATH.is_file():
        raise FileNotFoundError(
            f"Raw sales file not found: {RAW_PATH}"
        )

    df = pd.read_csv(RAW_PATH, low_memory=False)

    panel, coverage, new_products, weeks, split_by_week = (
        build_weekly_panel(df)
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    panel_path = PROCESSED_DIR / "sales_weekly_panel.csv"
    coverage_path = REPORTS_DIR / "panel_coverage_by_week.csv"
    new_products_path = REPORTS_DIR / "new_products_by_week.csv"

    panel.to_csv(panel_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    new_products.to_csv(new_products_path, index=False)

    print("\n=== WEEKLY PANEL SUMMARY ===")
    print(f"Source rows: {len(df):,}")
    print(f"Panel rows: {len(panel):,}")
    print(f"Unique products: {panel['PRODUCT_CODE'].nunique():,}")
    print(f"Weekly periods: {len(weeks):,}")
    print(f"First week: {weeks.min().date()}")
    print(f"Last week: {weeks.max().date()}")
    print(f"Train weeks: {sum(v == 'train' for v in split_by_week.values())}")
    print(
        "Validation weeks: "
        f"{sum(v == 'validation' for v in split_by_week.values())}"
    )
    print(f"Test weeks: {sum(v == 'test' for v in split_by_week.values())}")
    print(f"Recorded source rows: {int(panel['WAS_RECORDED'].sum()):,}")
    print(
        "Zero-filled panel rows: "
        f"{int((~panel['WAS_RECORDED']).sum()):,}"
    )

    print("\n=== PANEL COVERAGE ===")
    print(coverage.to_string(index=False))

    print("\n=== NEW PRODUCTS BY WEEK ===")
    print(new_products.to_string(index=False))

    print(f"\nPanel saved to: {panel_path}")
    print(f"Coverage report saved to: {coverage_path}")
    print(f"New-product report saved to: {new_products_path}")


if __name__ == "__main__":
    main()
