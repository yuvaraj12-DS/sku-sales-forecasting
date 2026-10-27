import numpy as np
import pandas as pd
import pytest

from src.predict_batch import (
    FEATURE_COLUMNS,
    generate_batch_predictions,
)


class StubModel:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions, dtype=float)
        self.received_columns = None
        self.received_rows = None

    def predict(self, features):
        self.received_columns = list(features.columns)
        self.received_rows = len(features)
        return self.predictions


@pytest.fixture
def feature_data():
    data = {
        "STORE_CODE": [89888] * 4,
        "PRODUCT_CODE": [101, 102, 103, 104],
        "SALES_WEEK": [
            "2024-09-16",
            "2024-09-16",
            "2024-09-16",
            "2024-09-16",
        ],
        "SPLIT": ["test", "test", "train", "test"],
        "FEATURE_ELIGIBLE": ["True", "true", "true", "false"],
        "WAS_RECORDED": [True, True, True, False],
    }

    for column in FEATURE_COLUMNS:
        data[column] = [1.0, 2.0, 3.0, 4.0]

    return pd.DataFrame(data)


def test_only_eligible_test_rows_are_predicted(feature_data):
    model = StubModel([-2.0, 3.5])

    result = generate_batch_predictions(feature_data, model)

    assert result["PRODUCT_CODE"].tolist() == [101, 102]
    assert model.received_rows == 2
    assert model.received_columns == FEATURE_COLUMNS
    assert result["FORECAST_QUANTITY"].tolist() == [0.0, 3.5]


def test_missing_features_are_rejected(feature_data):
    data = feature_data.drop(columns=["LAG_1"])

    with pytest.raises(ValueError, match="Missing required columns"):
        generate_batch_predictions(data, StubModel([1.0, 2.0]))


def test_empty_test_batch_is_rejected(feature_data):
    feature_data["SPLIT"] = "train"

    with pytest.raises(ValueError, match="No eligible rows"):
        generate_batch_predictions(
            feature_data,
            StubModel([]),
        )


def test_non_finite_predictions_are_rejected(feature_data):
    model = StubModel([np.nan, 2.0])

    with pytest.raises(ValueError, match="non-finite"):
        generate_batch_predictions(feature_data, model)


def test_wrong_prediction_count_is_rejected(feature_data):
    model = StubModel([1.0])

    with pytest.raises(ValueError, match="invalid number"):
        generate_batch_predictions(feature_data, model)