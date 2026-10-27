
import numpy as np
import pytest

from src.forecasting_metrics import evaluate_forecast


def test_metrics_for_known_values():
    result = evaluate_forecast(
        actual=[10, 0, 5],
        predicted=[8, 2, 7],
    )

    assert result["n_rows"] == 3
    assert result["positive_actual_rows"] == 2
    assert result["MAE"] == pytest.approx(2.0)
    assert result["RMSE"] == pytest.approx(2.0)
    assert result["MAPE_positive_pct"] == pytest.approx(30.0)
    assert result["WAPE_all_pct"] == pytest.approx(40.0)
    assert result["R2"] == pytest.approx(0.76)
    assert result["mean_forecast_bias_units"] == pytest.approx(2 / 3)


def test_negative_predictions_are_clipped_to_zero():
    result = evaluate_forecast(
        actual=[0, 2],
        predicted=[-5, 3],
    )

    assert result["MAE"] == pytest.approx(0.5)
    assert result["WAPE_all_pct"] == pytest.approx(50.0)


def test_zero_sales_make_mape_and_wape_undefined():
    result = evaluate_forecast(
        actual=[0, 0],
        predicted=[2, 0],
    )

    assert result["MAE"] == pytest.approx(1.0)
    assert np.isnan(result["MAPE_positive_pct"])
    assert np.isnan(result["WAPE_all_pct"])
    assert np.isnan(result["R2"])


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluate_forecast([], [])


def test_different_array_lengths_are_rejected():
    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_forecast([1, 2], [1])


def test_non_finite_values_are_rejected():
    with pytest.raises(ValueError, match="must be finite"):
        evaluate_forecast([1, np.nan], [1, 2])
