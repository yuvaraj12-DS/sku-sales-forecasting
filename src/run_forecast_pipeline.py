import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

# use this code to run manually python src\run_forecast_pipeline.py, automatic sheduling is disabled.

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Sales_Data.csv"
FORECAST_PATH = PROJECT_ROOT / "reports" / "next_week_forecast.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "sales_model_features.csv"
STATE_PATH = PROJECT_ROOT / "reports" / "pipeline_state.json"

PIPELINE_SCRIPTS = [
    "src/build_weekly_panel.py",
    "src/create_time_splits.py",
    "src/features_building.py",
    "src/forecast_next_week.py",
    "src/monitor_forecast.py",
]


def get_file_hash(path: Path) -> str:
    """Return a SHA-256 fingerprint of a file's contents."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_latest_source_week(path: Path) -> pd.Timestamp:
    """Read the latest week present in the source data."""
    df = pd.read_csv(path, usecols=["SALES_WEEK"])

    dates = pd.to_datetime(df["SALES_WEEK"], errors="raise").dropna()

    if dates.empty:
        raise ValueError("The source file contains no valid SALES_WEEK values.")

    return dates.max().normalize()


def get_latest_forecast_week(path: Path) -> pd.Timestamp | None:
    """Read the forecast target week, if a usable forecast exists."""
    if not path.exists():
        return None

    columns = pd.read_csv(path, nrows=0).columns.tolist()

    date_column = next(
        (
            column
            for column in ("SALES_WEEK", "FORECAST_WEEK")
            if column in columns
        ),
        None,
    )

    if date_column is None:
        return None

    dates = pd.to_datetime(
        pd.read_csv(path, usecols=[date_column])[date_column],
        errors="coerce",
    ).dropna()

    if dates.empty:
        return None

    return dates.max().normalize()
def should_run_forecast_pipeline(
    latest_source_week: pd.Timestamp,
    previous_forecast_week: pd.Timestamp | None,
    source_changed: bool,
) -> bool:
    """Determine whether new or changed source data requires a pipeline run."""
    if previous_forecast_week is None:
        return True

    previous_forecast_origin = (
        previous_forecast_week - pd.Timedelta(days=7)
    )

    new_week_available = latest_source_week > previous_forecast_origin

    return new_week_available or source_changed

def load_previous_state() -> dict | None:
    """Load the saved pipeline state, if available."""
    if not STATE_PATH.exists():
        return None

    try:
        with STATE_PATH.open("r", encoding="utf-8") as file:
            state = json.load(file)

        if isinstance(state, dict):
            return state

    except (json.JSONDecodeError, OSError):
        print("Warning: saved pipeline state is unreadable; reinitializing it.")

    return None


def save_state(
    source_hash: str,
    latest_source_week: pd.Timestamp,
    forecast_week: pd.Timestamp | None,
) -> None:
    """Save pipeline state only after the workflow succeeds."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "source_sha256": source_hash,
        "latest_source_week": latest_source_week.strftime("%Y-%m-%d"),
        "forecast_week": (
            forecast_week.strftime("%Y-%m-%d")
            if forecast_week is not None
            else None
        ),
    }

    temporary_path = STATE_PATH.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)

    temporary_path.replace(STATE_PATH)


def run_script(script: str) -> None:
    """Run one pipeline script and stop if it fails."""
    script_path = PROJECT_ROOT / script

    if not script_path.exists():
        raise FileNotFoundError(f"Pipeline script not found: {script_path}")

    print(f"\n{'=' * 60}")
    print(f"Running: {script}")
    print(f"{'=' * 60}")

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Source data not found: {RAW_PATH}")

    latest_source_week = get_latest_source_week(RAW_PATH)
    current_source_hash = get_file_hash(RAW_PATH)

    previous_state = load_previous_state()
    previous_hash = (
        previous_state.get("source_sha256")
        if previous_state is not None
        else None
    )

    source_changed = (
        previous_hash is not None
        and current_source_hash != previous_hash
    )

    previous_forecast_week = get_latest_forecast_week(FORECAST_PATH)

    should_run_pipeline = should_run_forecast_pipeline(
        latest_source_week=latest_source_week,
        previous_forecast_week=previous_forecast_week,
        source_changed=source_changed,
    )

    print(f"Latest source week: {latest_source_week.date()}")
    print(
        "Previous forecast target week: "
        f"{previous_forecast_week.date() if previous_forecast_week is not None else 'Not available'}"
    )
    print(f"Source file changed since last recorded run: {source_changed}")

    if not should_run_pipeline:
        print("\nNo new data detected. Skipping panel rebuild and forecasting.")

        if FEATURES_PATH.exists():
            run_script("src/monitor_forecast.py")
        else:
            print("Monitoring skipped: processed feature file does not exist.")

        save_state(
            source_hash=current_source_hash,
            latest_source_week=latest_source_week,
            forecast_week=previous_forecast_week,
        )

        print("\nPipeline check completed. No new forecast was generated.")
        return

    print("\nNew or changed data detected. Starting the forecast pipeline.")

    for script in PIPELINE_SCRIPTS:
        run_script(script)

    final_forecast_week = get_latest_forecast_week(FORECAST_PATH)

    if final_forecast_week is None:
        raise RuntimeError(
            "The pipeline finished, but no valid forecast target week was found."
        )

    save_state(
        source_hash=get_file_hash(RAW_PATH),
        latest_source_week=get_latest_source_week(RAW_PATH),
        forecast_week=final_forecast_week,
    )

    print("\nForecast pipeline completed successfully.")
    print(f"Forecast output: {FORECAST_PATH}")
    print(
        "Monitoring summary: "
        f"{PROJECT_ROOT / 'reports' / 'forecast_monitoring_summary.csv'}"
    )

if __name__ == "__main__":
    main()