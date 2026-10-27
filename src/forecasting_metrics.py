
import numpy as np


def evaluate_forecast(actual, predicted):
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)

    if actual.size == 0:
        raise ValueError("Actual values cannot be empty.")

    if actual.size != predicted.size:
        raise ValueError("Actual and predicted values must have equal lengths.")

    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and predicted values must be finite.")

    # Sales quantities cannot be negative.
    predicted = np.maximum(predicted, 0.0)

    errors = predicted - actual
    absolute_errors = np.abs(errors)
    positive_actual = actual > 0

    mape = np.nan
    if positive_actual.any():
        mape = (
            absolute_errors[positive_actual]
            / actual[positive_actual]
        ).mean() * 100

    actual_total = np.abs(actual).sum()
    wape = np.nan
    if actual_total > 0:
        wape = absolute_errors.sum() / actual_total * 100

    r2 = np.nan
    if actual.size >= 2:
        total_sum_squares = ((actual - actual.mean()) ** 2).sum()
        if total_sum_squares > 0:
            residual_sum_squares = (errors ** 2).sum()
            r2 = 1 - residual_sum_squares / total_sum_squares

    return {
        "n_rows": int(actual.size),
        "positive_actual_rows": int(positive_actual.sum()),
        "MAE": float(absolute_errors.mean()),
        "RMSE": float(np.sqrt((errors ** 2).mean())),
        "MAPE_positive_pct": float(mape),
        "WAPE_all_pct": float(wape),
        "R2": float(r2),
        "mean_forecast_bias_units": float(errors.mean()),
    }
