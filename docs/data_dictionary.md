# Data Dictionary

## Source columns

| Field | Meaning | Notes from audit |
|---|---|---|
| `STORE_CODE` | Store identifier | Only 89888 is present. |
| `PRODUCT_CODE` | Product identifier | Used to group product history; numeric magnitude is not meaningful. |
| `CATEGORY_CODE` | Category identifier | Two products have more than one observed category code. |
| `SALES_WEEK` | Start date of sales week | 18 consecutive Mondays from 2024-06-03 through 2024-09-30. |
| `ACTUAL` | Quantity sold | Target. Source values were positive whole numbers but stored as floats. |
| `AVG_UNIT_PRICE` | Average unit price | 113 source rows have zero prices; extreme prices were not automatically removed. |

## Panel and audit columns

| Field | Meaning |
|---|---|
| `FIRST_SEEN_WEEK` | First week a product appeared in the source; not necessarily its true launch date. |
| `WAS_RECORDED` | `True` if source contained the row; `False` if the panel process created it. |
| `SPLIT` | Chronological split: train, validation, or test. |
| `FEATURE_ELIGIBLE` | A prior one-week lag exists; this is an eligibility flag, not a predictive feature. |
| `VALIDATION_EVALUABLE` / `TEST_EVALUABLE` | Split-specific eligibility according to the original panel-building script. |

## Engineered features

| Feature | Meaning |
|---|---|
| `LAG_1`, `LAG_2`, `LAG_4`, `LAG_8` | Product's target quantity at an earlier weekly lag. |
| `ROLLING_MEAN_4`, `ROLLING_MEAN_8` | Average of preceding target values, excluding the current week. |
| `ROLLING_STD_4` | Standard deviation of preceding target values; requires two prior values. |
| `PRICE_LAST_KNOWN` | Most recent previously recorded product price. |
| `PRICE_LOG1P_LAST_KNOWN` | Log(1 + last-known price). |
| `PRICE_RECORDED_PREVIOUS_WEEK` | Whether a price record existed in the immediately preceding week. |
| `CATEGORY_LAST_KNOWN` | Most recent prior category code for a product. |
| `WEEK_INDEX` | Position of a date in the 18-week observed calendar, zero-based. |
| `MONTH` | Calendar month extracted from `SALES_WEEK`. |
| `WEEKS_SINCE_FIRST_SEEN` | Weeks elapsed since first observed source record. |

## Semantic warning

Unobserved product-week rows created during panel expansion are assigned zero `ACTUAL` under a working assumption, not a verified fact. The source-row flag must not be used to predict the same week's target: it exposes whether a record existed for that week.
