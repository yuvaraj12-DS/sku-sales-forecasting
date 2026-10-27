
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "sales_model_features.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

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


def evaluate_predictions(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)

    absolute_errors = np.abs(actual - predicted)
    positive_actual = actual > 0
    total_actual = np.abs(actual).sum()

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
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MAPE_positive_pct": mape,
        "WAPE_all_pct": wape,
        "R2": r2_score(actual, predicted),
    }


def main():
    df = pd.read_csv(FEATURES_PATH, low_memory=False)
    df["SALES_WEEK"] = pd.to_datetime(df["SALES_WEEK"], errors="raise")

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

    # Recalculate both baselines so all models use identical evaluation rows.
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

    train_mask = (
        df["SPLIT"].eq("train")
        & df["FEATURE_ELIGIBLE"]
    )
    validation_mask = (
        df["SPLIT"].eq("validation")
        & df["FEATURE_ELIGIBLE"]
    )

    train_df = df.loc[train_mask].copy()
    validation_df = df.loc[validation_mask].copy()

    if train_df.empty or validation_df.empty:
        raise ValueError("Training or validation data is empty.")

    if train_df["SALES_WEEK"].max() >= validation_df["SALES_WEEK"].min():
        raise ValueError("Training and validation periods overlap.")

    if validation_df[
        ["BASELINE_LAST_WEEK", "BASELINE_PAST_4W_MEAN"]
    ].isna().any().any():
        raise ValueError(
            "Baseline predictions are missing on validation rows."
        )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["ACTUAL"].to_numpy(dtype=float)

    X_validation = validation_df[FEATURE_COLUMNS]
    y_validation = validation_df["ACTUAL"].to_numpy(dtype=float)

    models = {
        "Linear Regression": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LinearRegression(),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RandomForestRegressor(
                n_estimators=100,
                max_depth=14,
                min_samples_leaf=10,
                max_features=0.8,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        "XGBoost": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=250,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.9,
                reg_lambda=5,
                tree_method="hist",
                n_jobs=6,
                random_state=42,
            ),
        ),
    }

    prediction_table = validation_df[
        [
            "STORE_CODE",
            "PRODUCT_CODE",
            "SALES_WEEK",
            "ACTUAL",
            "WAS_RECORDED",
        ]
    ].copy()

    prediction_table["Last-week naive"] = (
        validation_df["BASELINE_LAST_WEEK"].to_numpy()
    )
    prediction_table["Prior four-week mean"] = (
        validation_df["BASELINE_PAST_4W_MEAN"].to_numpy()
    )

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        predictions = np.maximum(
            model.predict(X_validation),
            0,
        )
        prediction_table[model_name] = predictions

    model_names = [
        "Last-week naive",
        "Prior four-week mean",
        "Linear Regression",
        "Random Forest",
        "XGBoost",
    ]

    overall_results = []
    weekly_results = []

    for model_name in model_names:
        overall_results.append({
            "model": model_name,
            **evaluate_predictions(
                prediction_table["ACTUAL"],
                prediction_table[model_name],
            ),
        })

        for week, week_df in prediction_table.groupby("SALES_WEEK"):
            weekly_results.append({
                "model": model_name,
                "SALES_WEEK": week,
                **evaluate_predictions(
                    week_df["ACTUAL"],
                    week_df[model_name],
                ),
            })

    overall_df = pd.DataFrame(overall_results)
    weekly_df = pd.DataFrame(weekly_results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_path = REPORTS_DIR / "validation_model_comparison.csv"
    weekly_path = REPORTS_DIR / "validation_metrics_by_week.csv"
    predictions_path = REPORTS_DIR / "validation_predictions.csv"

    overall_df.to_csv(comparison_path, index=False)
    weekly_df.to_csv(weekly_path, index=False)
    prediction_table.to_csv(predictions_path, index=False)

    print("\n=== DATA USED ===")
    print(f"Training rows: {len(train_df):,}")
    print(f"Training dates: {train_df['SALES_WEEK'].min().date()} "
          f"to {train_df['SALES_WEEK'].max().date()}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"Validation dates: {validation_df['SALES_WEEK'].min().date()} "
          f"to {validation_df['SALES_WEEK'].max().date()}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    print("\n=== OVERALL VALIDATION MODEL COMPARISON ===")
    print(
        overall_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n=== VALIDATION METRICS BY WEEK ===")
    print(
        weekly_df[
            [
                "model",
                "SALES_WEEK",
                "n_rows",
                "MAE",
                "RMSE",
                "MAPE_positive_pct",
                "WAPE_all_pct",
                "R2",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nSaved comparison: {comparison_path}")
    print(f"Saved weekly metrics: {weekly_path}")
    print(f"Saved predictions: {predictions_path}")


if __name__ == "__main__":
    main()
