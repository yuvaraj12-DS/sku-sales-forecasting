from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "sales_model_features.csv"
)
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODEL_DIR / "random_forest_forward.joblib"
OUTPUT_PATH = REPORTS_DIR / "next_week_forecast.csv"

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


def to_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1"])
    )


def build_next_week_features(df):
    required_columns = {
    "STORE_CODE",
    "PRODUCT_CODE",
    "SALES_WEEK",
    "FIRST_SEEN_WEEK",
    "ACTUAL",
    "AVG_UNIT_PRICE",
    "FEATURE_ELIGIBLE",
    "WAS_RECORDED",
    "WEEK_INDEX",
    "LAG_1",
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
    df["FIRST_SEEN_WEEK"] = pd.to_datetime(
        df["FIRST_SEEN_WEEK"],
        errors="raise",
    )
    df["ACTUAL"] = pd.to_numeric(
        df["ACTUAL"],
        errors="raise",
    )
    df["AVG_UNIT_PRICE"] = pd.to_numeric(
        df["AVG_UNIT_PRICE"],
        errors="raise",
    )
    df["FEATURE_ELIGIBLE"] = to_bool(df["FEATURE_ELIGIBLE"])
    df["WAS_RECORDED"] = to_bool(df["WAS_RECORDED"])

    df = df.sort_values(
        ["PRODUCT_CODE", "SALES_WEEK"]
    ).reset_index(drop=True)

    groups = df.groupby("PRODUCT_CODE", sort=False)

    df["_ACTUAL_SHIFT_3"] = groups["ACTUAL"].shift(3)
    df["_ACTUAL_SHIFT_7"] = groups["ACTUAL"].shift(7)

    df["_ROLLING_MEAN_4_NEXT"] = groups["ACTUAL"].transform(
        lambda series: series.rolling(
            window=4,
            min_periods=1,
        ).mean()
    )

    df["_ROLLING_MEAN_8_NEXT"] = groups["ACTUAL"].transform(
        lambda series: series.rolling(
            window=8,
            min_periods=1,
        ).mean()
    )

    df["_ROLLING_STD_4_NEXT"] = groups["ACTUAL"].transform(
        lambda series: series.rolling(
            window=4,
            min_periods=2,
        ).std()
    )

    df["_PRICE_LAST_KNOWN_NEXT"] = (
        groups["AVG_UNIT_PRICE"].transform(
            lambda series: series.ffill()
        )
    )

    latest_week = df["SALES_WEEK"].max()
    forecast_week = latest_week + pd.Timedelta(days=7)

    latest = df.loc[
        df["SALES_WEEK"].eq(latest_week)
    ].copy()

    cohort = latest.loc[
        latest["WAS_RECORDED"]
        & latest["FEATURE_ELIGIBLE"]
    ].copy()

    if cohort.empty:
        raise ValueError(
            "No eligible products found for forecasting."
        )

    future = pd.DataFrame(index=cohort.index)

    future["LAG_1"] = cohort["ACTUAL"]
    future["LAG_2"] = cohort["LAG_1"]
    future["LAG_4"] = cohort["_ACTUAL_SHIFT_3"]
    future["LAG_8"] = cohort["_ACTUAL_SHIFT_7"]

    future["ROLLING_MEAN_4"] = cohort["_ROLLING_MEAN_4_NEXT"]
    future["ROLLING_MEAN_8"] = cohort["_ROLLING_MEAN_8_NEXT"]
    future["ROLLING_STD_4"] = cohort["_ROLLING_STD_4_NEXT"]

    future["PRICE_LOG1P_LAST_KNOWN"] = np.log1p(
        cohort["_PRICE_LAST_KNOWN_NEXT"].clip(lower=0)
    )

    future["PRICE_RECORDED_PREVIOUS_WEEK"] = (
        cohort["AVG_UNIT_PRICE"]
        .notna()
        .astype("int8")
    )

    future["WEEK_INDEX"] = cohort["WEEK_INDEX"] + 1
    future["MONTH"] = forecast_week.month

    future["WEEKS_SINCE_FIRST_SEEN"] = (
        (forecast_week - cohort["FIRST_SEEN_WEEK"]).dt.days // 7
    )

    future = future[FEATURE_COLUMNS]

    return cohort, future, latest_week, forecast_week


def main():
    if not FEATURES_PATH.is_file():
        raise FileNotFoundError(
            f"Feature data not found: {FEATURES_PATH}"
        )

    df = pd.read_csv(
        FEATURES_PATH,
        low_memory=False,
    )

    df["SALES_WEEK"] = pd.to_datetime(
        df["SALES_WEEK"],
        errors="raise",
    )
    df["FIRST_SEEN_WEEK"] = pd.to_datetime(
        df["FIRST_SEEN_WEEK"],
        errors="raise",
    )
    df["ACTUAL"] = pd.to_numeric(
        df["ACTUAL"],
        errors="raise",
    )
    df["AVG_UNIT_PRICE"] = pd.to_numeric(
        df["AVG_UNIT_PRICE"],
        errors="raise",
    )
    df["FEATURE_ELIGIBLE"] = to_bool(df["FEATURE_ELIGIBLE"])
    df["WAS_RECORDED"] = to_bool(df["WAS_RECORDED"])

    df = df.sort_values(
        ["PRODUCT_CODE", "SALES_WEEK"]
    ).reset_index(drop=True)

    keys = [
        "STORE_CODE",
        "PRODUCT_CODE",
        "SALES_WEEK",
    ]

    if df.duplicated(keys).any():
        raise ValueError(
            "Duplicate store-product-week records found."
        )

    cohort, future, latest_week, forecast_week = (
        build_next_week_features(df)
    )

    # Retrain a separate model using the full available history.
    training = df.loc[
        df["FEATURE_ELIGIBLE"]
        & df["ACTUAL"].notna()
    ].copy()

    if training.empty:
        raise ValueError(
            "No eligible historical training rows found."
        )

    model = make_pipeline(
        SimpleImputer(
            strategy="median",
            add_indicator=True,
        ),
        RandomForestRegressor(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=10,
            max_features=0.8,
            n_jobs=-1,
            random_state=42,
        ),
    )

    print(
        "Training forward-forecast model on eligible history..."
    )

    model.fit(
        training[FEATURE_COLUMNS],
        training["ACTUAL"].to_numpy(dtype=float),
    )

    predictions = np.asarray(
        model.predict(future),
        dtype=float,
    )

    if predictions.ndim != 1 or len(predictions) != len(cohort):
        raise ValueError(
            "Prediction count does not match the forecast cohort."
        )

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Model produced non-finite predictions."
        )

    predictions = np.maximum(predictions, 0)

    output = cohort[
        ["STORE_CODE", "PRODUCT_CODE"]
    ].copy()

    output["ORIGIN_WEEK"] = latest_week
    output["FORECAST_WEEK"] = forecast_week
    output["ORIGIN_WAS_RECORDED"] = (
        cohort["WAS_RECORDED"].to_numpy()
    )
    output["FORECAST_QUANTITY"] = predictions

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, MODEL_PATH)
    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n=== FORWARD FORECAST SUMMARY ===")
    print(f"Training rows: {len(training):,}")
    print(f"Forecast origin: {latest_week.date()}")
    print(f"Forecast week: {forecast_week.date()}")
    print(f"Products forecast: {len(output):,}")
    print(
        f"Mean forecast quantity: "
        f"{predictions.mean():.4f}"
    )
    print(f"Saved forward model: {MODEL_PATH}")
    print(f"Saved forecasts: {OUTPUT_PATH}")

    print("\n=== SAMPLE FORECASTS ===")
    print(
        output.head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()