
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "validation_predictions.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FORECAST_COLUMNS = {
    "Random Forest": "Random Forest",
    "Four-week mean": "Prior four-week mean",
    "Linear Regression": "Linear Regression",
}


def main():
    df = pd.read_csv(PREDICTIONS_PATH)
    df["SALES_WEEK"] = pd.to_datetime(
        df["SALES_WEEK"],
        errors="raise",
    )

    required_columns = [
        "SALES_WEEK",
        "ACTUAL",
        *FORECAST_COLUMNS.values(),
    ]
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in prediction file: {sorted(missing_columns)}"
        )

    weekly = (
        df.groupby("SALES_WEEK", as_index=False)
        .agg(
            evaluated_product_weeks=("ACTUAL", "size"),
            actual_quantity=("ACTUAL", "sum"),
            random_forest_forecast=("Random Forest", "sum"),
            four_week_mean_forecast=("Prior four-week mean", "sum"),
            linear_regression_forecast=("Linear Regression", "sum"),
        )
        .sort_values("SALES_WEEK")
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = REPORTS_DIR / "validation_weekly_forecast_totals.csv"
    chart_path = FIGURES_DIR / "validation_forecasts.png"

    weekly.to_csv(summary_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(
        weekly["SALES_WEEK"],
        weekly["actual_quantity"],
        marker="o",
        linewidth=2,
        label="Actual quantity",
    )
    plt.plot(
        weekly["SALES_WEEK"],
        weekly["random_forest_forecast"],
        marker="o",
        linewidth=2,
        label="Random Forest",
    )
    plt.plot(
        weekly["SALES_WEEK"],
        weekly["four_week_mean_forecast"],
        marker="o",
        linewidth=2,
        label="Four-week mean baseline",
    )
    plt.plot(
        weekly["SALES_WEEK"],
        weekly["linear_regression_forecast"],
        marker="o",
        linewidth=2,
        label="Linear Regression",
    )

    plt.title("Validation: Actual vs Forecast Quantity")
    plt.xlabel("Forecast week")
    plt.ylabel("Total quantity across evaluated product-week rows")
    plt.xticks(
        weekly["SALES_WEEK"],
        weekly["SALES_WEEK"].dt.strftime("%d %b %Y"),
        rotation=30,
    )
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=160)
    plt.close()

    print("\n=== VALIDATION FORECAST TOTALS ===")
    print(weekly.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print(f"\nWeekly summary saved to: {summary_path}")
    print(f"Chart saved to: {chart_path}")


if __name__ == "__main__":
    main()
