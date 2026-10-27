
import pandas as pd
import pytest

from src.features_building import add_sales_history_features


@pytest.fixture
def weekly_sales():
    return pd.DataFrame({
        "PRODUCT_CODE": [101] * 6,
        "SALES_WEEK": pd.to_datetime([
            "2024-06-03",
            "2024-06-10",
            "2024-06-17",
            "2024-06-24",
            "2024-07-01",
            "2024-07-08",
        ]),
        "ACTUAL": [10, 20, 30, 40, 50, 60],
    })


def test_lag_features_use_only_prior_sales(weekly_sales):
    features = add_sales_history_features(weekly_sales)

    row = features.loc[
        features["SALES_WEEK"] == pd.Timestamp("2024-07-01")
    ].iloc[0]

    assert row["LAG_1"] == 40
    assert row["LAG_2"] == 30
    assert row["LAG_4"] == 10


def test_rolling_mean_excludes_current_week(weekly_sales):
    features = add_sales_history_features(weekly_sales)

    row = features.loc[
        features["SALES_WEEK"] == pd.Timestamp("2024-07-01")
    ].iloc[0]

    assert row["ACTUAL"] == 50
    assert row["ROLLING_MEAN_4"] == pytest.approx(25.0)
