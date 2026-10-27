
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "sales_model_features.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_DIR = PROJECT_ROOT / "models"

FEATURE_COLUMNS = [
    "LAG_1",
    "LAG_2",
    "LAG_4",
    "LAG_8",
    "ROLLING_MEAN_4",
    "ROLLING_MEAN_8",
    "ROLLING_STD_4",
    "PRICE_LOG1P_LAST_KNOWN",
    "PRICE_RECORDED_PREVIOUS_WEEK",
    "WEEK_INDEX",
    "MONTH",
    "WEEKS_SINCE_FIRST_SEEN",
]


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)

    errors = predicted - actual
    absolute_errors = np.abs(errors)
    positive_actual = actual > 0

    mape = np.nan
    if positive_actual.any():
        mape = (
            np.abs(
                errors[positive_actual] / actual[positive_actual]
            ).mean()
            * 100
        )

    wape = np.nan
    total_actual = np.abs(actual).sum()
    if total_actual > 0:
        wape = absolute_errors.sum() / total_actual * 100

    return {
        "n_rows": len(actual),
        "positive_actual_rows": int(positive_actual.sum()),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MAPE_positive_actual_pct": mape,
        "WAPE_all_rows_pct": wape,
        "R2": r2_score(actual, predicted),
        "mean_forecast_bias_units": errors.mean(),
    }


def main():
    df = pd.read_csv(FEATURES_PATH, low_memory=False)
    df["SALES_WEEK"] = pd.to_datetime(
        df["SALES_WEEK"],
        errors="raise",
    )

    df["FEATURE_ELIGIBLE"] = (
        df["FEATURE_ELIGIBLE"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1"])
    )

    df = df.sort_values(
        ["PRODUCT_CODE", "SALES_WEEK"]
    ).reset_index(drop=True)

    # Recompute simple baselines using prior weeks only.
    product_groups = df.groupby("PRODUCT_CODE", sort=False)

    df["BASELINE_LAST_WEEK"] = product_groups["ACTUAL"].shift(1)

    df["BASELINE_PAST_4W_MEAN"] = (
        df.groupby("PRODUCT_CODE", sort=False)["ACTUAL"]
        .transform(
            lambda series: series.shift(1).rolling(
                window=4,
                min_periods=1,
            ).mean()
        )
    )

    # Train on all eligible observations through the validation period.
    train_mask = (
        df["SPLIT"].isin(["train", "validation"])
        & df["FEATURE_ELIGIBLE"]
    )

    # Preserve the final three weeks for evaluation only.
    test_mask = (
        df["SPLIT"].eq("test")
        & df["FEATURE_ELIGIBLE"]
    )

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Training or test data is empty.")

    if train_df["SALES_WEEK"].max() >= test_df["SALES_WEEK"].min():
        raise ValueError("Training data overlaps the final test period.")

    baseline_columns = [
        "BASELINE_LAST_WEEK",
        "BASELINE_PAST_4W_MEAN",
    ]
    if test_df[baseline_columns].isna().any().any():
        raise ValueError("A baseline prediction is missing on test rows.")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["ACTUAL"].to_numpy(dtype=float)
    X_test = test_df[FEATURE_COLUMNS]

    final_model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        RandomForestRegressor(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=10,
            max_features=0.8,
            n_jobs=-1,
            random_state=42,
        ),
    )

    print("Fitting selected Random Forest on training + validation data...")
    final_model.fit(X_train, y_train)

    rf_predictions = np.maximum(final_model.predict(X_test), 0)

    predictions = test_df[
        [
            "STORE_CODE",
            "PRODUCT_CODE",
            "SALES_WEEK",
            "ACTUAL",
            "WAS_RECORDED",
        ]
    ].copy()

    predictions["Random Forest"] = rf_predictions
    predictions["Last-week naive"] = test_df[
        "BASELINE_LAST_WEEK"
    ].to_numpy()
    predictions["Prior four-week mean"] = test_df[
        "BASELINE_PAST_4W_MEAN"
    ].to_numpy()

    model_columns = [
        "Random Forest",
        "Last-week naive",
        "Prior four-week mean",
    ]

    overall_results = []
    weekly_results = []

    for model_name in model_columns:
        overall_results.append({
            "model": model_name,
            **calculate_metrics(
                predictions["ACTUAL"],
                predictions[model_name],
            ),
        })

        for week, week_df in predictions.groupby("SALES_WEEK"):
            weekly_results.append({
                "model": model_name,
                "SALES_WEEK": week,
                **calculate_metrics(
                    week_df["ACTUAL"],
                    week_df[model_name],
                ),
            })

    overall_df = pd.DataFrame(overall_results)
    weekly_df = pd.DataFrame(weekly_results)

    weekly_totals = (
        predictions.groupby("SALES_WEEK", as_index=False)
        .agg(
            evaluated_product_weeks=("ACTUAL", "size"),
            actual_quantity=("ACTUAL", "sum"),
            random_forest_forecast=("Random Forest", "sum"),
            last_week_forecast=("Last-week naive", "sum"),
            four_week_mean_forecast=("Prior four-week mean", "sum"),
        )
        .sort_values("SALES_WEEK")
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    overall_path = REPORTS_DIR / "final_test_metrics.csv"
    weekly_metrics_path = REPORTS_DIR / "final_test_metrics_by_week.csv"
    predictions_path = REPORTS_DIR / "final_test_predictions.csv"
    totals_path = REPORTS_DIR / "final_test_weekly_totals.csv"
    chart_path = FIGURES_DIR / "final_test_forecast.png"
    model_path = MODEL_DIR / "random_forest_final.joblib"

    overall_df.to_csv(overall_path, index=False)
    weekly_df.to_csv(weekly_metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    weekly_totals.to_csv(totals_path, index=False)
    joblib.dump(final_model, model_path)

    plt.figure(figsize=(10, 6))
    plt.plot(
        weekly_totals["SALES_WEEK"],
        weekly_totals["actual_quantity"],
        marker="o",
        linewidth=2,
        label="Actual quantity",
    )
    plt.plot(
        weekly_totals["SALES_WEEK"],
        weekly_totals["random_forest_forecast"],
        marker="o",
        linewidth=2,
        label="Random Forest",
    )
    plt.plot(
        weekly_totals["SALES_WEEK"],
        weekly_totals["four_week_mean_forecast"],
        marker="o",
        linewidth=2,
        label="Four-week mean baseline",
    )
    plt.xlabel("Test week")
    plt.ylabel("Quantity across evaluated product-week rows")
    plt.title("Final Test: Actual vs Forecast Quantity")
    plt.xticks(
        weekly_totals["SALES_WEEK"],
        weekly_totals["SALES_WEEK"].dt.strftime("%d %b %Y"),
        rotation=30,
    )
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=160)
    plt.close()

    print("\n=== FINAL TEST DATA ===")
    print(f"Training rows: {len(train_df):,}")
    print(
        f"Training period: {train_df['SALES_WEEK'].min().date()} "
        f"to {train_df['SALES_WEEK'].max().date()}"
    )
    print(f"Test rows: {len(test_df):,}")
    print(
        f"Test period: {test_df['SALES_WEEK'].min().date()} "
        f"to {test_df['SALES_WEEK'].max().date()}"
    )
    print(f"Features: {len(FEATURE_COLUMNS)}")

    print("\n=== OVERALL FINAL TEST METRICS ===")
    print(
        overall_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n=== FINAL TEST METRICS BY WEEK ===")
    print(
        weekly_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n=== FINAL TEST WEEKLY TOTALS ===")
    print(weekly_totals.to_string(index=False))

    print(f"\nSaved model: {model_path}")
    print(f"Saved chart: {chart_path}")
    print(f"Saved metrics: {overall_path}")
    print(f"Saved weekly metrics: {weekly_metrics_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved weekly totals: {totals_path}")


if __name__ == "__main__":
    main()
