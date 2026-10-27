
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Sales_Data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main():
    df = pd.read_csv(RAW_PATH, low_memory=False)
    df["SALES_WEEK"] = pd.to_datetime(
        df["SALES_WEEK"],
        errors="raise",
    )

    keys = ["STORE_CODE", "PRODUCT_CODE", "SALES_WEEK"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate store-product-week records found.")

    weeks = sorted(df["SALES_WEEK"].drop_duplicates().tolist())

    if len(weeks) != 18:
        raise ValueError(
            f"Expected 18 weeks, found {len(weeks)}. "
            "Review the split design before proceeding."
        )

    gaps = pd.Series(weeks).diff().dropna().dt.days
    if not gaps.eq(7).all():
        raise ValueError("The observed dates are not all seven days apart.")

    first_seen = (
        df.groupby("PRODUCT_CODE", as_index=False)
        .agg(FIRST_SEEN_WEEK=("SALES_WEEK", "min"))
    )

    calendar = pd.DataFrame({"SALES_WEEK": weeks})

    panel = first_seen.merge(calendar, how="cross")
    panel = panel.loc[
        panel["SALES_WEEK"] >= panel["FIRST_SEEN_WEEK"]
    ].copy()

    panel = panel.merge(
        df,
        on=["PRODUCT_CODE", "SALES_WEEK"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    panel["WAS_RECORDED"] = panel["_merge"].eq("both")
    panel = panel.drop(columns="_merge")

    if panel.loc[panel["WAS_RECORDED"], "ACTUAL"].isna().any():
        raise ValueError("An original source row has a missing ACTUAL value.")

    panel["ACTUAL"] = panel["ACTUAL"].fillna(0)

    panel = panel.sort_values(
        ["PRODUCT_CODE", "SALES_WEEK"]
    ).reset_index(drop=True)

    panel["STORE_CODE"] = (
        panel.groupby("PRODUCT_CODE")["STORE_CODE"].ffill()
    )
    panel["CATEGORY_CODE"] = (
        panel.groupby("PRODUCT_CODE")["CATEGORY_CODE"].ffill()
    )

    split_by_week = {}
    split_by_week.update(dict.fromkeys(weeks[:12], "train"))
    split_by_week.update(dict.fromkeys(weeks[12:15], "validation"))
    split_by_week.update(dict.fromkeys(weeks[15:], "test"))

    panel["SPLIT"] = panel["SALES_WEEK"].map(split_by_week)

    # Rolling one-week-ahead evaluation:
    # a product must have appeared in an earlier week.
    prior_observation_exists = (
        panel["FIRST_SEEN_WEEK"] < panel["SALES_WEEK"]
    )

    panel["VALIDATION_EVALUABLE"] = (
        panel["SPLIT"].eq("validation")
        & prior_observation_exists
    )

    panel["TEST_EVALUABLE"] = (
        panel["SPLIT"].eq("test")
        & prior_observation_exists
    )

    panel_columns = [
        "STORE_CODE",
        "PRODUCT_CODE",
        "CATEGORY_CODE",
        "SALES_WEEK",
        "ACTUAL",
        "AVG_UNIT_PRICE",
        "FIRST_SEEN_WEEK",
        "WAS_RECORDED",
        "SPLIT",
        "VALIDATION_EVALUABLE",
        "TEST_EVALUABLE",
    ]
    panel = panel[panel_columns]

    coverage = (
        panel.groupby(["SPLIT", "SALES_WEEK"], as_index=False)
        .agg(
            products_in_panel=("PRODUCT_CODE", "nunique"),
            panel_rows=("PRODUCT_CODE", "size"),
            source_rows=("WAS_RECORDED", "sum"),
            zero_filled_rows=(
                "WAS_RECORDED",
                lambda values: (~values).sum(),
            ),
            validation_evaluable_rows=("VALIDATION_EVALUABLE", "sum"),
            test_evaluable_rows=("TEST_EVALUABLE", "sum"),
            total_actual=("ACTUAL", "sum"),
        )
    )

    new_products = (
        first_seen.groupby("FIRST_SEEN_WEEK")
        .size()
        .rename("products_first_seen")
        .reset_index()
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    panel_path = PROCESSED_DIR / "sales_weekly_panel.csv"
    coverage_path = REPORTS_DIR / "panel_coverage_by_week.csv"
    new_products_path = REPORTS_DIR / "products_first_seen_by_week.csv"

    panel.to_csv(panel_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    new_products.to_csv(new_products_path, index=False)

    print("\n=== PANEL SUMMARY ===")
    print(f"Original source rows: {len(df):,}")
    print(f"Rows in weekly panel: {len(panel):,}")
    print(f"Original rows retained: {panel['WAS_RECORDED'].sum():,}")
    print(
        "Rows assigned zero under the working assumption: "
        f"{(~panel['WAS_RECORDED']).sum():,}"
    )
    print(f"Distinct products: {panel['PRODUCT_CODE'].nunique():,}")
    print(
        "Missing prices in zero-filled rows: "
        f"{panel.loc[~panel['WAS_RECORDED'], 'AVG_UNIT_PRICE'].isna().sum():,}"
    )

    print("\n=== WEEKLY PANEL COVERAGE ===")
    print(coverage.to_string(index=False))

    print("\n=== PRODUCTS FIRST OBSERVED BY WEEK ===")
    print(new_products.to_string(index=False))

    print(f"\nPanel saved to: {panel_path}")
    print(f"Coverage report saved to: {coverage_path}")
    print(f"First-seen report saved to: {new_products_path}")


if __name__ == "__main__":
    main()
