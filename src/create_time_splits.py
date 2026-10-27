
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Sales_Data.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["SALES_WEEK"] = pd.to_datetime(
        df["SALES_WEEK"],
        errors="raise",
    )

    weeks = sorted(df["SALES_WEEK"].drop_duplicates().tolist())

    if len(weeks) != 18:
        raise ValueError(
            f"Expected 18 distinct weeks, but found {len(weeks)}. "
            "Review the split design before proceeding."
        )

    split_by_week = {}
    split_by_week.update(dict.fromkeys(weeks[:12], "train"))
    split_by_week.update(dict.fromkeys(weeks[12:15], "validation"))
    split_by_week.update(dict.fromkeys(weeks[15:], "test"))

    df["SPLIT"] = df["SALES_WEEK"].map(split_by_week)

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
            rows=("PRODUCT_CODE", "size"),
            products=("PRODUCT_CODE", "nunique"),
            total_actual=("ACTUAL", "sum"),
            mean_actual=("ACTUAL", "mean"),
        )
        .reindex(["train", "validation", "test"])
        .reset_index()
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    week_plan_path = REPORTS_DIR / "time_split_by_week.csv"
    summary_path = REPORTS_DIR / "time_split_summary.csv"

    week_plan.to_csv(week_plan_path, index=False)
    split_summary.to_csv(summary_path, index=False)

    print("\n=== WEEK-LEVEL SPLIT PLAN ===")
    print(week_plan.to_string(index=False))

    print("\n=== SPLIT SUMMARY ===")
    print(split_summary.to_string(index=False))

    print(f"\nSaved: {week_plan_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
