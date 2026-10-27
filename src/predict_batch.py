from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "sales_model_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_final.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_PATH = REPORTS_DIR / "batch_predictions_test.csv"

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


def generate_batch_predictions(
    df: pd.DataFrame,
    model,
) -> pd.DataFrame:
    required_columns = {
        "STORE_CODE",
        "PRODUCT_CODE",
        "SALES_WEEK",
        "SPLIT",
        "FEATURE_ELIGIBLE",
        "WAS_RECORDED",
        *FEATURE_COLUMNS,
    }
    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df.copy()

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

    batch_df = df.loc[
        df["SPLIT"].eq("test") & df["FEATURE_ELIGIBLE"]
    ].copy()

    if batch_df.empty:
        raise ValueError("No eligible rows found in the test split.")

    predictions = np.asarray(
        model.predict(batch_df[FEATURE_COLUMNS]),
        dtype=float,
    )

    if predictions.ndim != 1 or len(predictions) != len(batch_df):
        raise ValueError(
            "Model returned an invalid number of predictions."
        )

    predictions = np.maximum(predictions, 0)

    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions.")

    output = batch_df[
        [
            "STORE_CODE",
            "PRODUCT_CODE",
            "SALES_WEEK",
            "WAS_RECORDED",
        ]
    ].copy()

    output["FORECAST_QUANTITY"] = predictions

    return output


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    df = pd.read_csv(FEATURES_PATH, low_memory=False)
    model = joblib.load(MODEL_PATH)

    output = generate_batch_predictions(df, model)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("=== BATCH PREDICTION SUMMARY ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Rows predicted: {len(output):,}")
    print(f"First forecast week: {output['SALES_WEEK'].min().date()}")
    print(f"Last forecast week: {output['SALES_WEEK'].max().date()}")
    print(
        f"Mean forecast quantity: "
        f"{output['FORECAST_QUANTITY'].mean():.4f}"
    )
    print(f"Output saved to: {OUTPUT_PATH}")

    print("\n=== SAMPLE PREDICTIONS ===")
    print(output.head(10).to_string(index=False))


if __name__ == "__main__":
    main()