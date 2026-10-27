
import numpy as np
import pandas as pd
import pytest

from src.forecast_next_week import (
    FEATURE_COLUMNS,
    build_next_week_features,
)


@pytest.fixture
def forecast_history():
    rows = []
    weeks = pd.date_range("2024-08-05", periods=9, freq="W-MON")

    for product_code in (101, 102):
        actuals = [10 * (i + 1) for i in range(9)]

        for i, week in enumerate(weeks):
            rows.append({
                "STORE_CODE": 89888,
                "PRODUCT_CODE": product_code,
                "SALES_WEEK": week,
                "FIRST_SEEN_WEEK": weeks[0],
                "ACTUAL": actuals[i],
                "AVG_UNIT_PRICE": float(i + 1),
                "FEATURE_ELIGIBLE": i > 0,
                "WAS_RECORDED": not (
                    product_code == 102 and i == 8
                ),
                "WEEK_INDEX": i + 9,
                "LAG_1": actuals[i - 1] if i >= 1 else np.nan,
                "LAG_2": actuals[i - 2] if i >= 2 else np.nan,
            })

    rows.append({
        "STORE_CODE": 89888,
        "PRODUCT_CODE": 103,
        "SALES_WEEK": weeks[-1],
        "FIRST_SEEN_WEEK": weeks[-1],
        "ACTUAL": 5,
        "AVG_UNIT_PRICE": 2.0,
        "FEATURE_ELIGIBLE": False,
        "WAS_RECORDED": True,
        "WEEK_INDEX": 17,
        "LAG_1": np.nan,
        "LAG_2": np.nan,
    })

    return pd.DataFrame(rows)


def test_next_week_dates_and_cohort(forecast_history):
    cohort, future, origin, forecast_week = (
        build_next_week_features(forecast_history)
    )

    assert origin == pd.Timestamp("2024-09-30")
    assert forecast_week == pd.Timestamp("2024-10-07")
    assert cohort["PRODUCT_CODE"].tolist() == [101]
    assert list(future.columns) == FEATURE_COLUMNS


def test_next_week_features_use_historical_values(forecast_history):
    cohort, future, _, _ = build_next_week_features(forecast_history)

    row = future.iloc[0]

    assert row["LAG_1"] == 90
    assert row["LAG_2"] == 80
    assert row["LAG_4"] == 60
    assert row["LAG_8"] == 20
    assert row["ROLLING_MEAN_4"] == pytest.approx(75.0)
    assert row["ROLLING_MEAN_8"] == pytest.approx(55.0)
    assert row["PRICE_LOG1P_LAST_KNOWN"] == pytest.approx(np.log(10))
    assert row["WEEK_INDEX"] == 18
    assert row["MONTH"] == 10
    assert row["WEEKS_SINCE_FIRST_SEEN"] == 9
    assert "ACTUAL" not in future.columns


def test_empty_forecast_cohort_is_rejected(forecast_history):
    latest_week = forecast_history["SALES_WEEK"].max()
    mask = forecast_history["SALES_WEEK"].eq(latest_week)
    forecast_history.loc[mask, "WAS_RECORDED"] = False

    with pytest.raises(ValueError, match="No eligible products"):
        build_next_week_features(forecast_history)
