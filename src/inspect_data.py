
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}


def find_dataset():
    files = [
        path
        for path in RAW_DATA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        raise FileNotFoundError(
            f"No supported dataset found in {RAW_DATA_DIR}"
        )

    if len(files) > 1:
        names = "\n".join(path.name for path in files)
        raise ValueError(
            "Multiple datasets found. Keep only the intended dataset "
            "in data/raw for this audit, or specify the file explicitly:\n"
            f"{names}"
        )

    return files[0]


def load_dataset(path):
    extension = path.suffix.lower()

    if extension == ".csv":
        return pd.read_csv(path, low_memory=False)
    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if extension == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file format: {extension}")


def main():
    dataset_path = find_dataset()
    df = load_dataset(dataset_path)

    print("\n=== DATASET ===")
    print(f"File: {dataset_path.name}")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")

    print("\n=== FIRST FIVE ROWS ===")
    print(df.head().to_string(index=False))

    print("\n=== COLUMN NAMES ===")
    print(df.columns.tolist())

    print("\n=== DATA TYPES ===")
    print(df.dtypes.to_string())

    print("\n=== MISSING VALUES ===")
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
    })
    print(missing.sort_values("missing_count", ascending=False).to_string())

    print("\n=== DUPLICATE ROWS ===")
    print(f"Exact duplicate rows: {df.duplicated().sum():,}")

    print("\n=== NUMERIC SUMMARY ===")
    print(df.describe().T.to_string())

    print("\n=== UNIQUE VALUES IN KEY COLUMNS ===")
    for column in [
        "STORE_CODE",
        "PRODUCT_CODE",
        "CATEGORY_CODE",
        "SALES_WEEK",
    ]:
        if column in df.columns:
            print(f"{column}: {df[column].nunique(dropna=True):,} unique values")

    if "ACTUAL" in df.columns:
        print("\n=== TARGET VARIABLE: ACTUAL ===")
        print(df["ACTUAL"].describe().to_string())
        print(f"Zero sales rows: {(df['ACTUAL'] == 0).sum():,}")
        print(f"Negative sales rows: {(df['ACTUAL'] < 0).sum():,}")


if __name__ == "__main__":
    main()
