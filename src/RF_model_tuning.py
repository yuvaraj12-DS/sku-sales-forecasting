
from pathlib import Path
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
tracking_db = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(
        f"sqlite:///{tracking_db.as_posix()}"
     )
mlflow.set_experiment("sku-sales-forecasting-rf-tuning")

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

FOLD_WEEKS = [
    pd.Timestamp("2024-07-29"),
    pd.Timestamp("2024-08-19"),
]

CANDIDATES = {
    "current_settings": {
        "n_estimators": 100,
        "max_depth": 14,
        "min_samples_leaf": 10,
        "max_features": 0.8,
    },
    "more_conservative": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 20,
        "max_features": 0.8,
    },
    "more_flexible": {
        "n_estimators": 100,
        "max_depth": 18,
        "min_samples_leaf": 5,
        "max_features": 0.8,
    },
}


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)

    absolute_errors = np.abs(actual - predicted)
    wape = absolute_errors.sum() / np.abs(actual).sum() * 100

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "WAPE_pct": wape,
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

    fold_results = []

    for candidate_name, parameters in CANDIDATES.items():
        candidate_results = []

        with mlflow.start_run(run_name=candidate_name):
            mlflow.log_params(parameters)
            mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
            mlflow.log_param("fold_count", len(FOLD_WEEKS))

            mlflow.set_tags({
                "candidate": candidate_name,
                "model_type": "RandomForestRegressor",
                "validation_strategy": "chronological_folds",
                "script": "src/RF_model_tuning.py",
            })

            for fold_index, fold_week in enumerate(
                FOLD_WEEKS, start=1
            ):
                train_mask = (
                    df["SPLIT"].eq("train")
                    & df["FEATURE_ELIGIBLE"]
                    & df["SALES_WEEK"].lt(fold_week)
                )
                validation_mask = (
                    df["SPLIT"].eq("train")
                    & df["FEATURE_ELIGIBLE"]
                    & df["SALES_WEEK"].eq(fold_week)
                )

                train_df = df.loc[train_mask]
                validation_df = df.loc[validation_mask]

                if train_df.empty or validation_df.empty:
                    raise ValueError(
                        "Empty training or validation data for "
                        f"fold {fold_week.date()}."
                    )

                X_train = train_df[FEATURE_COLUMNS]
                y_train = train_df["ACTUAL"].to_numpy(dtype=float)
                X_validation = validation_df[FEATURE_COLUMNS]
                y_validation = validation_df["ACTUAL"].to_numpy(
                    dtype=float
                )

                baseline = validation_df[
                    "ROLLING_MEAN_4"
                ].to_numpy(dtype=float)
                baseline_metrics = calculate_metrics(
                    y_validation, baseline
                )

                model = make_pipeline(
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                    RandomForestRegressor(
                        **parameters,
                        n_jobs=-1,
                        random_state=42,
                    ),
                )

                print(
                    f"\nCandidate: {candidate_name} | "
                    f"Fold: {fold_week.date()} | "
                    f"Training rows: {len(train_df):,} | "
                    f"Validation rows: {len(validation_df):,}"
                )

                model.fit(X_train, y_train)
                predictions = np.maximum(
                    model.predict(X_validation), 0
                )

                model_metrics = calculate_metrics(
                    y_validation, predictions
                )

                fold_result = {
                    "candidate": candidate_name,
                    "fold_week": fold_week.date().isoformat(),
                    "train_rows": len(train_df),
                    "validation_rows": len(validation_df),
                    **model_metrics,
                    "baseline_MAE": baseline_metrics["MAE"],
                    "baseline_WAPE_pct": (
                        baseline_metrics["WAPE_pct"]
                    ),
                }

                fold_results.append(fold_result)
                candidate_results.append(fold_result)

                mlflow.log_metrics({
                    f"fold_{fold_index}_MAE": float(
                        model_metrics["MAE"]
                    ),
                    f"fold_{fold_index}_WAPE_pct": float(
                        model_metrics["WAPE_pct"]
                    ),
                    f"fold_{fold_index}_baseline_MAE": float(
                        baseline_metrics["MAE"]
                    ),
                    f"fold_{fold_index}_baseline_WAPE_pct": float(
                        baseline_metrics["WAPE_pct"]
                    ),
                })

            candidate_df = pd.DataFrame(candidate_results)

            mlflow.log_metrics({
                "mean_fold_MAE": float(
                    candidate_df["MAE"].mean()
                ),
                "mean_fold_WAPE_pct": float(
                    candidate_df["WAPE_pct"].mean()
                ),
                "mean_baseline_MAE": float(
                    candidate_df["baseline_MAE"].mean()
                ),
                "mean_baseline_WAPE_pct": float(
                    candidate_df["baseline_WAPE_pct"].mean()
                ),
            })

            mlflow.set_tag(
                "selection_metric",
                "mean_fold_WAPE_pct",
            )

            print(
                f"Mean validation WAPE: "
                f"{candidate_df['WAPE_pct'].mean():.4f}%"
            )

    results = pd.DataFrame(fold_results)

    summary = (
        results.groupby("candidate", as_index=False)
        .agg(
            mean_fold_MAE=("MAE", "mean"),
            mean_fold_WAPE_pct=("WAPE_pct", "mean"),
            mean_baseline_MAE=("baseline_MAE", "mean"),
            mean_baseline_WAPE_pct=("baseline_WAPE_pct", "mean"),
        )
        .sort_values("mean_fold_WAPE_pct")
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fold_path = REPORTS_DIR / "random_forest_tuning_folds.csv"
    summary_path = REPORTS_DIR / "random_forest_tuning_summary.csv"

    results.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n=== RESULTS BY FOLD ===")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== CANDIDATE SUMMARY (LOWER WAPE IS BETTER) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nSaved fold results: {fold_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
