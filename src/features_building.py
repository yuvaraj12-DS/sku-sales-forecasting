from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "sales_weekly_panel.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


def add_sales_history_features(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"PRODUCT_CODE", "SALES_WEEK", "ACTUAL"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    result = (
        df.sort_values(["PRODUCT_CODE", "SALES_WEEK"])
        .reset_index(drop=True)
        .copy()
    )

    product_groups = result.groupby("PRODUCT_CODE", sort=False)

    for lag in (1, 2, 4, 8):
        result[f"LAG_{lag}"] = product_groups["ACTUAL"].shift(lag)

    for window in (4, 8):
        result[f"ROLLING_MEAN_{window}"] = (
            result.groupby("PRODUCT_CODE", sort=False)["ACTUAL"]
            .transform(
                lambda series: series.shift(1).rolling(
                    window=window,
                    min_periods=1,
                ).mean()
            )
        )

    result["ROLLING_STD_4"] = (
        result.groupby("PRODUCT_CODE", sort=False)["ACTUAL"]
        .transform(
            lambda series: series.shift(1).rolling(
                window=4,
                min_periods=2,
            ).std()
        )
    )

    return result


def main():
    df = pd.read_csv(PANEL_PATH, low_memory=False)

    df["SALES_WEEK"] = pd.to_datetime(
        df["SALES_WEEK"],
        errors="raise",
    )
    df["FIRST_SEEN_WEEK"] = pd.to_datetime(
        df["FIRST_SEEN_WEEK"],
        errors="raise",
    )

    df = df.sort_values(
        ["PRODUCT_CODE", "SALES_WEEK"]
    ).reset_index(drop=True)

    keys = ["STORE_CODE", "PRODUCT_CODE", "SALES_WEEK"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate store-product-week records found.")

    df = add_sales_history_features(df)
    # Use the last price and category known before the forecast week.
    df["PRICE_LAST_KNOWN"] = (
        df.groupby("PRODUCT_CODE", sort=False)["AVG_UNIT_PRICE"]
        .transform(lambda series: series.ffill().shift(1))
    )

    df["PRICE_LOG1P_LAST_KNOWN"] = np.log1p(
        df["PRICE_LAST_KNOWN"].clip(lower=0)
    )

    df["PRICE_RECORDED_PREVIOUS_WEEK"] = (
        df.groupby("PRODUCT_CODE", sort=False)["AVG_UNIT_PRICE"]
        .shift(1)
        .notna()
        .astype("int8")
    )

    df["CATEGORY_LAST_KNOWN"] = (
        df.groupby("PRODUCT_CODE", sort=False)["CATEGORY_CODE"]
        .transform(lambda series: series.ffill().shift(1))
    )

    # Calendar and product-history features.
    week_list = sorted(df["SALES_WEEK"].drop_duplicates())
    week_index = {week: index for index, week in enumerate(week_list)}

    df["WEEK_INDEX"] = df["SALES_WEEK"].map(week_index).astype("int8")
    df["MONTH"] = df["SALES_WEEK"].dt.month.astype("int8")

    df["WEEKS_SINCE_FIRST_SEEN"] = (
        (df["SALES_WEEK"] - df["FIRST_SEEN_WEEK"]).dt.days // 7
    ).astype("int16")

    # Exclude the first product-week, which has no prior sales history.
    df["FEATURE_ELIGIBLE"] = df["LAG_1"].notna()

    # Verify that lag-1 refers to the immediately preceding weekly row.
    previous_week = (
        df.groupby("PRODUCT_CODE", sort=False)["SALES_WEEK"].shift(1)
    )
    eligible = df["FEATURE_ELIGIBLE"]

    gaps = (
        df.loc[eligible, "SALES_WEEK"]
        - previous_week.loc[eligible]
    ).dt.days

    if not gaps.eq(7).all():
        raise ValueError(
            "A lag-1 feature does not refer to the previous week."
        )

    feature_report = (
        df.groupby(["SPLIT", "SALES_WEEK"], as_index=False)
        .agg(
            panel_rows=("PRODUCT_CODE", "size"),
            feature_eligible_rows=("FEATURE_ELIGIBLE", "sum"),
            source_rows=("WAS_RECORDED", "sum"),
            zero_targets=("ACTUAL", lambda values: values.eq(0).sum()),
            positive_targets=("ACTUAL", lambda values: values.gt(0).sum()),
        )
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    features_path = PROCESSED_DIR / "sales_model_features.csv"
    report_path = REPORTS_DIR / "feature_availability.csv"

    df.to_csv(features_path, index=False)
    feature_report.to_csv(report_path, index=False)

    print("\n=== FEATURE ENGINEERING SUMMARY ===")
    print(f"Rows in panel: {len(df):,}")
    print(f"Rows with lag-1 available: {eligible.sum():,}")
    print(f"Rows without lag-1: {(~eligible).sum():,}")
    print(f"Unique products: {df['PRODUCT_CODE'].nunique():,}")

    print("\n=== FEATURE AVAILABILITY BY WEEK ===")
    print(feature_report.to_string(index=False))

    print("\n=== FEATURE COLUMNS CREATED ===")
    print(
        "LAG_1, LAG_2, LAG_4, LAG_8, "
        "ROLLING_MEAN_4, ROLLING_MEAN_8, ROLLING_STD_4, "
        "PRICE_LAST_KNOWN, PRICE_LOG1P_LAST_KNOWN, "
        "PRICE_RECORDED_PREVIOUS_WEEK, CATEGORY_LAST_KNOWN, "
        "WEEK_INDEX, MONTH, WEEKS_SINCE_FIRST_SEEN"
    )

    print(f"\nFeature data saved to: {features_path}")
    print(f"Availability report saved to: {report_path}")


if __name__ == "__main__":
    main()
