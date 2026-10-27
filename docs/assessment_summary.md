# Weekly Sales Forecasting with Machine Learning

## Objective

The objective was to forecast weekly sales quantity (`ACTUAL`) for individual products in store 89888 using historical sales data. Linear Regression, Random Forest, and XGBoost were evaluated against last-week and four-week moving-average baselines.

## Dataset and preparation

The source dataset contained 436,927 records, 57,143 distinct products, 1,421 categories, and 18 consecutive weekly dates from June 3 to September 30, 2024. Only one store was represented.

Initial checks found no missing values, exact duplicate rows, or repeated store-product-week combinations. Observed `ACTUAL` values were positive whole numbers, although stored as floating-point values. The target was highly skewed: median sales quantity was 3, compared with a mean of 14.03 and a maximum of 3,828. The price field contained 113 zero-price records, while some unusually high prices required additional verification. These values were retained rather than removed arbitrarily.

Because the source had no rows with zero sales, a weekly product panel was constructed from each product's first observed sales week onward. Missing product-week records in this panel were assigned `ACTUAL = 0` under a documented working assumption. A provenance flag distinguished original records from added rows. This assumption could not be verified against inventory or assortment data, so results are interpreted as forecasts of recorded sales rather than unconstrained customer demand.

## Feature engineering and modeling

Leakage-aware features included previous-week, two-week, four-week, and eight-week sales lags; rolling means and standard deviation calculated from prior weeks; the last known price and its log transformation; calendar information; and weeks since a product was first observed. Current-week target information was excluded from predictive features.

The first 12 weeks were used for training, the next three for validation, and the final three for testing. Initial comparisons covered Linear Regression, Random Forest, and XGBoost. Random Forest settings were compared through two chronological validation folds. The original configuration was retained because the alternative configurations did not improve average validation performance. The selected model was refitted on training and validation data before final test evaluation.

## Final test results

| Model | MAE | RMSE | MAPE on positive sales | WAPE across all eligible rows | R^2 |
|---|---:|---:|---:|---:|---:|
| Random Forest | 2.5972 | 13.3287 | 65.53% | 41.37% | 0.8789 |
| Last-week naive | 2.7726 | 14.0882 | 83.63% | 44.16% | 0.8647 |
| Four-week mean | 2.6787 | 13.5502 | 70.43% | 42.67% | 0.8749 |

Random Forest performed best on MAE, RMSE, WAPE, and R^2 across all eligible product-week test rows. Its MAE was approximately 3.0% lower than that of the four-week mean baseline. R^2 exceeded the client's benchmark of 0.75. However, MAPE remained substantially above the requested 15%, so the project did not meet every stated performance target.

Sensitivity analysis revealed that the model ranking depended on the zero-fill assumption. On original-source test rows only, the four-week mean had slightly lower MAE (4.9340 versus 4.9784) and WAPE (33.56% versus 33.86%) than Random Forest. Random Forest produced lower MAPE on positive sales. In aggregate, the four-week mean forecast was also closer to total recorded sales across the three test weeks, although aggregate accuracy is different from product-week accuracy.

## Visualization and limitations

The forecast comparison chart is saved as `reports/figures/final_test_forecast.png`. It compares actual quantity with Random Forest and the four-week mean baseline over the final test weeks.

The dataset covers only four months and one store, so it cannot establish yearly seasonal patterns or generalization to other stores. No inventory, stockout, promotion, or assortment data was provided. Consequently, zero-filled rows might represent no recorded sales, unavailability, or other missing observations. The zero-fill rule is a modeling assumption rather than a confirmed business fact. Products' first observed weeks are excluded from lag-based evaluation because no earlier sales history is available.

MAPE is calculated only on rows with positive actuals because it is undefined for zero actuals. Its high value also reflects the large number of low-quantity sales observations. MAE is reported in units per product-week, while WAPE provides a scale-normalized measure across the full evaluated panel.

## Conclusion

Random Forest is the leading model by product-week error metrics under the all-row, zero-filled evaluation. The four-week moving average remains a credible alternative, particularly for aggregate weekly totals and evaluation on originally recorded sales rows. The results should therefore be presented with the zero-fill assumption and evaluation scope clearly stated. Additional inventory and product-availability information, a longer sales history, and more robust out-of-time evaluation would be required before making stronger claims about operational forecasting accuracy.