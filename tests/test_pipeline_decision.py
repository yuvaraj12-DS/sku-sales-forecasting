import pandas as pd

from src.run_forecast_pipeline import should_run_forecast_pipeline


def test_skips_pipeline_when_data_is_unchanged():
    result = should_run_forecast_pipeline(
        latest_source_week=pd.Timestamp("2024-09-30"),
        previous_forecast_week=pd.Timestamp("2024-10-07"),
        source_changed=False,
    )

    assert result is False


def test_runs_pipeline_when_a_new_week_arrives():
    result = should_run_forecast_pipeline(
        latest_source_week=pd.Timestamp("2024-10-07"),
        previous_forecast_week=pd.Timestamp("2024-10-07"),
        source_changed=False,
    )

    assert result is True


def test_runs_pipeline_when_historical_data_changes():
    result = should_run_forecast_pipeline(
        latest_source_week=pd.Timestamp("2024-09-30"),
        previous_forecast_week=pd.Timestamp("2024-10-07"),
        source_changed=True,
    )

    assert result is True


def test_runs_pipeline_when_no_previous_forecast_exists():
    result = should_run_forecast_pipeline(
        latest_source_week=pd.Timestamp("2024-09-30"),
        previous_forecast_week=None,
        source_changed=False,
    )

    assert result is True