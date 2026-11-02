
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Sales_Data.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

REQUIRED_COLUMNS = [
    "SALES_WEEK",
    "PRODUCT_CODE",
    "ACTUAL",
]


def build_time_splits(df):
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df.copy()

    df["SALES_WEEK"] = pd.to_datetime(
        df["SALES_WEEK"],
        errors="raise",
    )
    df["ACTUAL"] = pd.to_numeric(
        df["ACTUAL"],
        errors="raise",
    )

    if df["SALES_WEEK"].isna().any():
        raise ValueError("SALES_WEEK contains missing dates.")

    if df["ACTUAL"].isna().any():
        raise ValueError("ACTUAL contains missing values.")

    observed_weeks = pd.DatetimeIndex(
        df["SALES_WEEK"].drop_duplicates().sort_values()
    )

    if observed_weeks.empty:
        raise ValueError("No sales weeks found.")

    # Include calendar weeks even if a week has no source records.
    weeks = pd.date_range(
        start=observed_weeks.min(),
        end=observed_weeks.max(),
        freq="7D",
    )

    calendar_weeks = set(weeks)
    off_calendar_weeks = [
        week for week in observed_weeks
        if week not in calendar_weeks
    ]

    if off_calendar_weeks:
        raise ValueError(
            "Sales dates do not follow a consistent weekly cadence. "
            f"Examples: {off_calendar_weeks[:5]}"
        )

    if len(weeks) < 7:
        raise ValueError(
            f"Expected at least 7 weekly periods, found {len(weeks)}. "
            "Need at least one training week, three validation weeks, "
            "and three test weeks."
        )

    # Keep the final three weeks for testing and the previous
    # three weeks for validation.
    train_end = len(weeks) - 6
    validation_end = len(weeks) - 3

    split_by_week = {}
    split_by_week.update(
        dict.fromkeys(weeks[:train_end], "train")
    )
    split_by_week.update(
        dict.fromkeys(
            weeks[train_end:validation_end],
            "validation",
        )
    )
    split_by_week.update(
        dict.fromkeys(weeks[validation_end:], "test")
    )

    df["SPLIT"] = df["SALES_WEEK"].map(split_by_week)

    if df["SPLIT"].isna().any():
        raise ValueError("Some source rows could not be assigned a split.")

    week_plan = pd.DataFrame({
        "SALES_WEEK": weeks,
        "SPLIT": [split_by_week[week] for week in weeks],
    })

    split_summary = (
        df.groupby("SPLIT")
        .agg(
            start_week=("SALES_WEEK", "min"),
            end_week=("SALES_WEEK", "max"),
            weeks=("SALES_WEEK", "nunique"),
            rows=("SALES_WEEK", "size"),
            products=("PRODUCT_CODE", "nunique"),
            total_actual=("ACTUAL", "sum"),
            mean_actual=("ACTUAL", "mean"),
        )
        .reindex(["train", "validation", "test"])
        .reset_index()
    )

    return week_plan, split_summary


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Raw sales data not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)
    week_plan, split_summary = build_time_splits(df)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    week_plan_path = REPORTS_DIR / "time_split_by_week.csv"
    summary_path = REPORTS_DIR / "time_split_summary.csv"

    week_plan.to_csv(week_plan_path, index=False)
    split_summary.to_csv(summary_path, index=False)

    print("\n=== WEEK-LEVEL SPLIT PLAN ===")
    print(week_plan.to_string(index=False))

    print("\n=== SPLIT SUMMARY ===")
    print(
        split_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nSaved: {week_plan_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
