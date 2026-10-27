# SKU Sales Forecasting with Machine Learning

A reproducible assessment project for weekly product sales in a single store. It compares simple history-based baselines with Linear Regression, Random Forest, and XGBoost, and records the assumptions and limitations that affect evaluation.

> **Status:** assessment/portfolio prototype, not production-ready. The reported MAPE target was not met. Treatment of absent product-week rows as zero is a working assumption that requires business confirmation.

## Objective

Forecast weekly recorded sales quantity (`ACTUAL`) for each product at `STORE_CODE = 89888`. The supplied CSV has six fields:

| Field | Meaning |
|---|---|
| `STORE_CODE` | Store identifier |
| `PRODUCT_CODE` | Product identifier |
| `CATEGORY_CODE` | Product category identifier |
| `SALES_WEEK` | Start date of the sales week |
| `ACTUAL` | Quantity sold; target variable |
| `AVG_UNIT_PRICE` | Average selling price per unit |

Only one store appears in the source, so cross-store generalization cannot be evaluated.

## Data and assumptions

The initial audit of `Sales_Data.csv` found 436,927 source records, 57,143 product codes, 1,421 categories, and 18 consecutive weekly dates from June 3 to September 30, 2024. There were no missing values, exact duplicate rows, or repeated store-product-week keys in the source. Source `ACTUAL` values were positive whole numbers, although stored as floats. Median quantity was 3, mean approximately 14.03, and maximum 3,828. There were 113 zero-price records, two products with multiple observed category codes, and some extreme prices; values were not automatically deleted or capped.

The source does not contain explicit zero-sales rows. The analytical panel expands each product from its first observed week onward and assigns zero `ACTUAL` when a product-week has no source record. `WAS_RECORDED` indicates whether a row existed in the source.

**This is an assumption, not a fact confirmed by the data.** An absent row could mean zero recorded sales, an out-of-stock product, assortment changes, discontinuation, or a data issue. This is a forecast of recorded sales under the panel assumption—not a verified forecast of unconstrained demand. See [`docs/decisions_and_limitations.md`](docs/decisions_and_limitations.md).

## Methodology

### Chronological split

| Split | Weeks | Dates |
|---|---:|---|
| Train | 12 | 2024-06-03 to 2024-08-19 |
| Validation | 3 | 2024-08-26 to 2024-09-09 |
| Final test | 3 | 2024-09-16 to 2024-09-30 |

After model selection, the final Random Forest was refitted using eligible train and validation rows before evaluation on the test weeks. The evaluation is **rolling one-week-ahead**: actuals from a completed week may inform the next week's features. This is not a three-week forecast issued once with no intervening actuals.

### Features

Features included product-level previous sales lags (`LAG_1`, `LAG_2`, `LAG_4`, `LAG_8`), rolling summaries shifted to exclude current-week `ACTUAL`, last-known price, previous-week price availability, calendar features, and weeks since first observed. Product codes were used to group historical data, not as a continuous numeric measure.

### Models compared

- Last-week naive forecast
- Previous-four-week moving average
- Linear Regression
- Random Forest
- XGBoost

The original Random Forest settings were retained after a small chronological tuning comparison failed to improve mean fold metrics. Hyperparameter search was limited by the short history.

## Final test results

Test dates: September 16, 23, and 30, 2024. Metrics cover eligible product-week rows with earlier product history; first-observed product-weeks are excluded.

| Model | MAE (units/product-week) | RMSE | MAPE on positive actuals | WAPE across eligible rows | R^2 |
|---|---:|---:|---:|---:|---:|
| Random Forest | **2.5972** | **13.3287** | 65.53% | **41.37%** | **0.8789** |
| Last-week naive | 2.7726 | 14.0882 | 83.63% | 44.16% | 0.8647 |
| Four-week mean | 2.6787 | 13.5502 | 70.43% | 42.67% | 0.8749 |

Random Forest reduced MAE by about 3.0% relative to the four-week mean baseline across all eligible test rows. R^2 exceeded the client's 0.75 benchmark, but MAPE was substantially higher than the requested 15%. The project **does not meet every stated benchmark**.

### Sensitivity to zero-filled rows

The ranking changes when scoring only rows originally present in the CSV:

| Source rows only | Random Forest | Four-week mean |
|---|---:|---:|
| MAE | 4.9784 | **4.9340** |
| RMSE | 19.7995 | **19.4390** |
| MAPE on positive actuals | **65.53%** | 70.43% |
| WAPE | 33.86% | **33.56%** |
| R^2 | 0.8816 | **0.8859** |

Across all eligible rows, 94,800 of 165,464 final-test rows (about 57.3%) were zero-filled. On original-source rows alone, the four-week mean had slightly better MAE, RMSE, WAPE, and R^2; Random Forest had lower MAPE. The four-week mean also had a closer aggregate total across test weeks. Always state the population and zero-fill assumption when reporting metrics.

MAPE is calculated only for positive actuals because conventional MAPE is undefined at zero. MAE is a quantity error per product-week, not a percentage of store-wide weekly sales.

## Outputs

The scripts create charts and reports under `reports/`, such as:

- `reports/figures/actual_distribution.png`
- `reports/figures/weekly_recorded_sales.png`
- `reports/figures/weekly_product_coverage.png`
- `reports/figures/positive_price_distribution.png`
- `reports/figures/validation_forecasts.png`
- `reports/figures/final_test_forecast.png`

Generated outputs are ignored by Git by default because some are client-derived and row-level prediction reports may be sensitive. Only add an aggregate chart to Git after confirming permission to share it.

## Repository structure

```text
.
├── data/raw/                  # Local input; not committed
├── data/processed/            # Generated data/features; not committed
├── docs/                      # Methodology, data dictionary, model card, MLOps plan
├── models/                    # Serialized models; not committed by default
├── notebooks/                 # Exploratory notebooks
├── reports/figures/           # Generated visuals; not committed by default
├── src/                       # Analysis, feature engineering, modeling scripts
├── tests/                     # Automated tests (to be expanded)
├── .github/workflows/         # CI checks
├── .gitignore
├── .python-version
├── README.md
├── requirements.txt           # Existing environment's package snapshot
└── requirements-ci.txt        # Cross-platform dependencies for CI checks
```

## Reproduce locally

Use Python 3.11. Put the authorized input file at `data/raw/Sales_Data.csv`; do not commit it.

From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then run the scripts as needed, in dependency order. For the full analysis, typical commands are:

```powershell
python src\inspect_data.py
python src\audit_quality.py
python src\anomalies_investigation.py
python src\create_eda_plots.py
python src\create_time_splits.py
python src\build_weekly_panel.py
python src\baseline_evaluation.py
python src\features_building.py
python src\initial_model_training.py
python src\RF_model_tuning.py
python src\plot_validation_forecasts.py
python src\evaluation_final_test.py
python src\evaluation_of_zero_fill_sensitivity.py
```

Because the client dataset is deliberately not in Git, a fresh clone cannot run the data-dependent scripts until the same authorized input is supplied locally. `pip freeze` records the installed environment; refresh `requirements.txt` only intentionally after confirming the environment is the project's `.venv`.

## Limitations and next steps

- Only 18 weeks from one store; yearly seasonality and cross-store behavior cannot be established.
- Missing product-week semantics are unconfirmed.
- Inventory availability, stockouts, promotions, costs, and assortment are unavailable.
- The final test spans only three weeks.
- MAPE target was not achieved.
- This is a portfolio project, not yet a production deployment. The current Docker workflow reproduces predictions for the historical test period. Future improvements include forecasting newly arriving weeks, automated scheduling, monitoring, and deployment.


## Dockerized Batch Prediction and Forward Forecasting

The project supports two Docker-based workflows using the saved Random Forest model and prepared historical feature dataset.

### Prerequisites

- Docker Desktop installed and running.
- The processed feature dataset at `data/processed/sales_model_features.csv`.
- For batch mode, a trained model at `models/random_forest_final.joblib`.

The dataset and models remain outside the Docker image and are mounted into the container at runtime.

### Historical Batch Prediction

Run from the project root in Windows PowerShell:

```powershell
.\scripts\run_batch_docker.ps1
```

This loads the saved final-test model and reproduces predictions for the historical test period.

Output: `reports/batch_predictions_test.csv`

### Forward Forecasting

Run:

```powershell
.\scripts\run_batch_docker.ps1 -Mode forward
```

This trains a separate Random Forest using eligible historical data and generates one-week-ahead forecasts for products recorded in the latest historical week that have sufficient sales history.

Outputs:

- `reports/next_week_forecast.csv`
- `models/random_forest_forward.joblib`

### Validation

The Docker batch predictions were compared against the original final-test predictions for all 165,464 rows. The maximum absolute prediction difference was approximately `1e-12`.

The forward-forecast workflow was also verified inside Docker, generating 23,511 product forecasts.

### Scope and Limitations

The source data ends on September 30, 2024. The current forward forecast is therefore a historical demonstration for October 7, 2024, not a forecast for the present date.

Inventory availability, stockouts, promotions, costs, and assortment data are unavailable. Products without recorded sales in the latest historical week are excluded from the forward-forecast cohort. This is a portfolio project, not yet a production deployment.
