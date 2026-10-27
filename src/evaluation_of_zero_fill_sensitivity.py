
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "final_test_predictions.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

PREDICTION_COLUMNS = [
    "Random Forest",
    "Last-week naive",
    "Prior four-week mean",
]


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)

    errors = predicted - actual
    absolute_errors = np.abs(errors)
    positive_actual = actual > 0
    total_actual = np.abs(actual).sum()

    mape = np.nan
    if positive_actual.any():
        mape = (
            np.abs(errors[positive_actual] / actual[positive_actual]).mean()
            * 100
        )

    wape = np.nan
    if total_actual > 0:
        wape = absolute_errors.sum() / total_actual * 100

    r2 = np.nan
    if len(actual) > 1 and np.unique(actual).size > 1:
        r2 = r2_score(actual, predicted)

    return {
        "rows": len(actual),
        "actual_positive_rows": int(positive_actual.sum()),
        "actual_total": actual.sum(),
        "forecast_total": predicted.sum(),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MAPE_positive_pct": mape,
        "WAPE_pct": wape,
        "R2": r2,
        "mean_forecast_bias": errors.mean(),
    }


def main():
    df = pd.read_csv(PREDICTIONS_PATH, low_memory=False)

    df["WAS_RECORDED"] = (
        df["WAS_RECORDED"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1"])
    )

    required_columns = [
        "ACTUAL",
        "WAS_RECORDED",
        *PREDICTION_COLUMNS,
    ]

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    subsets = {
        "All eligible rows": df,
        "Original-source rows only": df.loc[df["WAS_RECORDED"]],
        "Assumed zero rows only": df.loc[~df["WAS_RECORDED"]],
    }

    results = []

    for scope, subset in subsets.items():
        for model_name in PREDICTION_COLUMNS:
            metrics = calculate_metrics(
                subset["ACTUAL"],
                subset[model_name],
            )
            results.append({
                "scope": scope,
                "model": model_name,
                **metrics,
            })

    results_df = pd.DataFrame(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "test_zero_fill_sensitivity.csv"
    results_df.to_csv(output_path, index=False)

    print("\n=== ZERO-FILL SENSITIVITY ===")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
