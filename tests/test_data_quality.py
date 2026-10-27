import pandas as pd
import pytest

from src.audit_quality import validate_sales_data


@pytest.fixture
def valid_sales_data():
    return pd.DataFrame({
        "STORE_CODE": [89888, 89888],
        "PRODUCT_CODE": [101, 101],
        "CATEGORY_CODE": [10, 10],
        "SALES_WEEK": ["2024-06-03", "2024-06-10"],
        "ACTUAL": [12, 15],
        "AVG_UNIT_PRICE": [0, 20.5],
    })


def test_valid_data_passes(valid_sales_data):
    assert validate_sales_data(valid_sales_data) is None


def test_missing_column_is_rejected(valid_sales_data):
    data = valid_sales_data.drop(columns=["ACTUAL"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_sales_data(data)


def test_invalid_date_is_rejected(valid_sales_data):
    valid_sales_data.loc[0, "SALES_WEEK"] = "not-a-date"

    with pytest.raises(ValueError, match="invalid dates"):
        validate_sales_data(valid_sales_data)


def test_duplicate_store_product_week_is_rejected(valid_sales_data):
    valid_sales_data.loc[1, "SALES_WEEK"] = "2024-06-03"

    with pytest.raises(ValueError, match="Duplicate"):
        validate_sales_data(valid_sales_data)


def test_non_numeric_sales_is_rejected(valid_sales_data):
    data = valid_sales_data.copy()
    data["ACTUAL"] = data["ACTUAL"].astype("string")
    data.loc[0, "ACTUAL"] = "invalid"

    with pytest.raises(ValueError, match="ACTUAL"):
        validate_sales_data(data)

def test_zero_price_is_allowed(valid_sales_data):
    assert validate_sales_data(valid_sales_data) is None