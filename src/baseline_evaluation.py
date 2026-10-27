
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "sales_weekly_panel.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"


def calculate_metrics(actual, predicted):
    actual = actual.to_numpy(dtype=float)
    predicted = predicted.to_numpy(dtype=float)

    absolute_errors = np.abs(actual - predicted)
    total_actual = np.abs(actual).sum()
    positive_actual = actual > 0

    mape = np.nan
    if positive_actual.any():
        mape = (
            np.abs(
                (actual[positive_actual] - predicted[positive_actual])
                / actual[positive_actual]
            ).mean()
            * 100
        )

    wape = np.nan
    if total_actual > 0:
        wape = absolute_errors.sum() / total_actual * 100

    return {
        "n_rows": len(actual),
        "positive_actual_rows": int(positive_actual.sum()),
        "MAE_units_per_product_week": mean_absolute_error(
            actual, predicted
        ),
        "RMSE_units_per_product_week": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "MAPE_pct_positive_actual_only": mape,
        "WAPE_pct_all_rows": wape,
        "R2_all_rows": r2_score(actual, predicted),
    }


def main():
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    df["SALES_WEEK"] = pd.to_datetime(
        df["SALES_WEEK"],
        errors="raise",
    )

    df["VALIDATION_EVALUABLE"] = (
        df["VALIDATION_EVALUABLE"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    df = df.sort_values(
        ["PRODUCT_CODE", "SALES_WEEK"]
    ).reset_index(drop=True)

    # Baseline 1: predict this week's quantity using the prior week's quantity.
    df["BASELINE_LAST_WEEK"] = (
        df.groupby("PRODUCT_CODE")["ACTUAL"].shift(1)
    )

    # Baseline 2: average up to the prior four weekly quantities.
    # The current week's ACTUAL is excluded by shift(1).
    df["BASELINE_PAST_4W_MEAN"] = (
        df.groupby("PRODUCT_CODE")["ACTUAL"]
        .transform(
            lambda series: series.shift(1).rolling(
                window=4,
                min_periods=1,
            ).mean()
        )
    )

    validation = df.loc[
        df["SPLIT"].eq("validation")
        & df["VALIDATION_EVALUABLE"]
    ].copy()

    if validation.empty:
        raise ValueError("No eligible validation rows were found.")

    baseline_columns = [
        "BASELINE_LAST_WEEK",
        "BASELINE_PAST_4W_MEAN",
    ]

    if validation[baseline_columns].isna().any().any():
        raise ValueError(
            "Missing baseline predictions found among eligible "
            "validation rows. Review product history and the panel."
        )

    results = []

    for baseline_name, prediction_column in [
        ("Last-week naive", "BASELINE_LAST_WEEK"),
        ("Prior four-week mean", "BASELINE_PAST_4W_MEAN"),
    ]:
        metrics = calculate_metrics(
            validation["ACTUAL"],
            validation[prediction_column],
        )
        metrics["baseline"] = baseline_name
        results.append(metrics)

    results_df = pd.DataFrame(results)

    column_order = [
        "baseline",
        "n_rows",
        "positive_actual_rows",
        "MAE_units_per_product_week",
        "RMSE_units_per_product_week",
        "MAPE_pct_positive_actual_only",
        "WAPE_pct_all_rows",
        "R2_all_rows",
    ]
    results_df = results_df[column_order]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "validation_baseline_metrics.csv"
    results_df.to_csv(output_path, index=False)

    print("\n=== VALIDATION BASELINE EVALUATION ===")
    print(
        f"Validation dates: "
        f"{validation['SALES_WEEK'].min().date()} to "
        f"{validation['SALES_WEEK'].max().date()}"
    )
    print(f"Eligible product-week rows: {len(validation):,}")
    print(
        "Zero-target rows: "
        f"{(validation['ACTUAL'] == 0).sum():,}"
    )
    print("\n=== BASELINE METRICS ===")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
