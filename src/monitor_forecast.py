
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "sales_model_features.csv"
)
FORECAST_PATH = PROJECT_ROOT / "reports" / "next_week_forecast.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "forecast_monitoring_summary.csv"

FRESHNESS_LIMIT_DAYS = 14


def to_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1"])
    )


def build_monitoring_summary(
    forecasts,
    features,
    as_of_date=None,
    freshness_limit_days=FRESHNESS_LIMIT_DAYS,
):
    required_forecast_columns = {
        "STORE_CODE",
        "PRODUCT_CODE",
        "ORIGIN_WEEK",
        "FORECAST_WEEK",
        "FORECAST_QUANTITY",
    }
    required_feature_columns = {
        "STORE_CODE",
        "PRODUCT_CODE",
        "SALES_WEEK",
        "FEATURE_ELIGIBLE",
        "WAS_RECORDED",
    }

    missing_forecast_columns = sorted(
        required_forecast_columns - set(forecasts.columns)
    )
    missing_feature_columns = sorted(
        required_feature_columns - set(features.columns)
    )

    if missing_forecast_columns:
        raise ValueError(
            f"Forecast file is missing columns: "
            f"{missing_forecast_columns}"
        )

    if missing_feature_columns:
        raise ValueError(
            f"Feature data is missing columns: "
            f"{missing_feature_columns}"
        )

    forecasts = forecasts.copy()
    features = features.copy()

    forecasts["ORIGIN_WEEK"] = pd.to_datetime(
        forecasts["ORIGIN_WEEK"], errors="coerce"
    )
    forecasts["FORECAST_WEEK"] = pd.to_datetime(
        forecasts["FORECAST_WEEK"], errors="coerce"
    )
    features["SALES_WEEK"] = pd.to_datetime(
        features["SALES_WEEK"], errors="coerce"
    )
    forecasts["FORECAST_QUANTITY"] = pd.to_numeric(
        forecasts["FORECAST_QUANTITY"], errors="coerce"
    )

    valid_input_dates = features["SALES_WEEK"].dropna()
    if valid_input_dates.empty:
        raise ValueError("No valid historical sales dates found.")

    latest_input_week = valid_input_dates.max()
    expected_forecast_week = (
        latest_input_week + pd.Timedelta(days=7)
    )

    latest_features = features.loc[
        features["SALES_WEEK"].eq(latest_input_week)
    ].copy()

    expected_cohort = latest_features.loc[
        to_bool(latest_features["WAS_RECORDED"])
        & to_bool(latest_features["FEATURE_ELIGIBLE"])
    ]

    key_columns = ["STORE_CODE", "PRODUCT_CODE"]

    expected_keys = set(
        expected_cohort[key_columns].itertuples(
            index=False, name=None
        )
    )
    forecast_keys = set(
        forecasts[key_columns].itertuples(
            index=False, name=None
        )
    )

    missing_expected_products = len(expected_keys - forecast_keys)
    unexpected_products = len(forecast_keys - expected_keys)

    origin_dates = forecasts["ORIGIN_WEEK"]
    target_dates = forecasts["FORECAST_WEEK"]

    dates_aligned = (
        not forecasts.empty
        and origin_dates.notna().all()
        and target_dates.notna().all()
        and origin_dates.nunique() == 1
        and target_dates.nunique() == 1
        and origin_dates.iloc[0] == latest_input_week
        and target_dates.iloc[0] == expected_forecast_week
        and (target_dates == origin_dates + pd.Timedelta(days=7)).all()
    )

    quantities = forecasts["FORECAST_QUANTITY"]
    quantity_values = quantities.to_numpy(dtype=float)
    finite_mask = np.isfinite(quantity_values)

    missing_forecasts = int(quantities.isna().sum())
    non_finite_forecasts = int(
        ((~finite_mask) & quantities.notna().to_numpy()).sum()
    )
    negative_forecasts = int(
        (finite_mask & (quantity_values < 0)).sum()
    )

    duplicate_keys = int(
        forecasts.duplicated(
            ["STORE_CODE", "PRODUCT_CODE", "FORECAST_WEEK"]
        ).sum()
    )

    if as_of_date is None:
        checked_date = pd.Timestamp.now().normalize()
    else:
        checked_date = pd.Timestamp(as_of_date).normalize()

    input_age_days = int(
        (checked_date - latest_input_week.normalize()).days
    )

    if 0 <= input_age_days <= freshness_limit_days:
        freshness_status = "FRESH"
    elif input_age_days < 0:
        freshness_status = "FUTURE_DATED"
    else:
        freshness_status = "STALE"

    cohort_matches = (
        missing_expected_products == 0
        and unexpected_products == 0
        and len(forecasts) == len(expected_cohort)
    )

    validation_passes = (
        not forecasts.empty
        and missing_forecasts == 0
        and non_finite_forecasts == 0
        and negative_forecasts == 0
        and duplicate_keys == 0
        and dates_aligned
        and cohort_matches
    )

    validation_status = "PASS" if validation_passes else "FAIL"

    if validation_status == "FAIL":
        overall_status = "FAIL"
    elif freshness_status != "FRESH":
        overall_status = "WARNING"
    else:
        overall_status = "PASS"

    return pd.DataFrame([{
        "checked_date": checked_date.date().isoformat(),
        "latest_input_week": latest_input_week.date().isoformat(),
        "expected_forecast_week": (
            expected_forecast_week.date().isoformat()
        ),
        "forecast_rows": len(forecasts),
        "unique_products": forecasts["PRODUCT_CODE"].nunique(),
        "expected_cohort_rows": len(expected_cohort),
        "missing_expected_products": missing_expected_products,
        "unexpected_products": unexpected_products,
        "missing_forecasts": missing_forecasts,
        "non_finite_forecasts": non_finite_forecasts,
        "negative_forecasts": negative_forecasts,
        "duplicate_keys": duplicate_keys,
        "dates_aligned": bool(dates_aligned),
        "cohort_matches": bool(cohort_matches),
        "mean_forecast_quantity": (
            float(quantities.mean()) if len(quantities) else np.nan
        ),
        "median_forecast_quantity": (
            float(quantities.median()) if len(quantities) else np.nan
        ),
        "total_forecast_quantity": (
            float(quantities.sum()) if len(quantities) else np.nan
        ),
        "input_age_days": input_age_days,
        "freshness_limit_days": freshness_limit_days,
        "validation_status": validation_status,
        "freshness_status": freshness_status,
        "overall_status": overall_status,
    }])


def main():
    if not FEATURES_PATH.is_file():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_PATH}"
        )

    if not FORECAST_PATH.is_file():
        raise FileNotFoundError(
            f"Forecast file not found: {FORECAST_PATH}"
        )

    features = pd.read_csv(FEATURES_PATH, low_memory=False)
    forecasts = pd.read_csv(FORECAST_PATH, low_memory=False)

    summary = build_monitoring_summary(forecasts, features)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORT_PATH, index=False)

    print("=== FORECAST MONITORING ===")
    print(summary.to_string(index=False))
    print(f"\nMonitoring report saved to: {REPORT_PATH}")

    if summary.loc[0, "overall_status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
