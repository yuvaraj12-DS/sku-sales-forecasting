
import pandas as pd
import pytest

from src.monitor_forecast import build_monitoring_summary


@pytest.fixture
def monitoring_data():
    features = pd.DataFrame({
        "STORE_CODE": [89888, 89888, 89888, 89888],
        "PRODUCT_CODE": [101, 102, 101, 102],
        "SALES_WEEK": [
            "2024-09-23",
            "2024-09-23",
            "2024-09-30",
            "2024-09-30",
        ],
        "FEATURE_ELIGIBLE": [True, True, True, True],
        "WAS_RECORDED": [True, True, True, True],
    })

    forecasts = pd.DataFrame({
        "STORE_CODE": [89888, 89888],
        "PRODUCT_CODE": [101, 102],
        "ORIGIN_WEEK": ["2024-09-30", "2024-09-30"],
        "FORECAST_WEEK": ["2024-10-07", "2024-10-07"],
        "FORECAST_QUANTITY": [12.0, 8.0],
    })

    return forecasts, features


def test_valid_fresh_forecast_passes(monitoring_data):
    forecasts, features = monitoring_data

    summary = build_monitoring_summary(
        forecasts,
        features,
        as_of_date="2024-10-05",
    )

    assert summary.loc[0, "validation_status"] == "PASS"
    assert summary.loc[0, "freshness_status"] == "FRESH"
    assert summary.loc[0, "overall_status"] == "PASS"
    assert summary.loc[0, "forecast_rows"] == 2
    assert summary.loc[0, "cohort_matches"]


def test_stale_input_produces_warning(monitoring_data):
    forecasts, features = monitoring_data

    summary = build_monitoring_summary(
        forecasts,
        features,
        as_of_date="2026-10-27",
    )

    assert summary.loc[0, "validation_status"] == "PASS"
    assert summary.loc[0, "freshness_status"] == "STALE"
    assert summary.loc[0, "overall_status"] == "WARNING"
    assert summary.loc[0, "input_age_days"] > 14


def test_negative_forecast_fails_validation(monitoring_data):
    forecasts, features = monitoring_data
    forecasts.loc[0, "FORECAST_QUANTITY"] = -1.0

    summary = build_monitoring_summary(
        forecasts,
        features,
        as_of_date="2024-10-05",
    )

    assert summary.loc[0, "negative_forecasts"] == 1
    assert summary.loc[0, "validation_status"] == "FAIL"
    assert summary.loc[0, "overall_status"] == "FAIL"


def test_duplicate_forecast_record_fails_validation(monitoring_data):
    forecasts, features = monitoring_data
    forecasts = pd.concat(
        [forecasts, forecasts.iloc[[0]]],
        ignore_index=True,
    )

    summary = build_monitoring_summary(
        forecasts,
        features,
        as_of_date="2024-10-05",
    )

    assert summary.loc[0, "duplicate_keys"] == 1
    assert summary.loc[0, "validation_status"] == "FAIL"
