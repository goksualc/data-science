# HW4 Summary — Best Models + Next-Quarter Predictions

Selection criterion: highest training `R²` (coefficient of determination), where:

`R² = 1 - SS_res / SS_tot`.

R² reference: [Wikipedia – Coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination)

Holdout rule: train on the previous 8 quarters and predict the held-out latest quarter (the “next quarter” for the homework).

---

## 1) Best model per (symbol, metric) — by training R²

| symbol   | metric    | best_model   |   r2_best_train | tag_used                                            |
|:---------|:----------|:-------------|----------------:|:----------------------------------------------------|
| AAPL     | dividends | logarithmic  |          0.0248 | CommonStockDividendsPerShareDeclared                |
| AAPL     | earnings  | logarithmic  |          0.0025 | NetIncomeLoss                                       |
| AAPL     | revenue   | logarithmic  |          0.0188 | RevenueFromContractWithCustomerExcludingAssessedTax |
| GE       | dividends | linear       |          0.6697 | CommonStockDividendsPerShareDeclared                |
| GE       | earnings  | logarithmic  |          0.2318 | NetIncomeLoss                                       |
| GE       | revenue   | logarithmic  |          0.1615 | Revenues                                            |
| GOOG     | dividends | linear       |          0.8203 | PaymentsOfDividends                                 |
| GOOG     | earnings  | exponential  |          0.4646 | NetIncomeLoss                                       |
| GOOG     | revenue   | linear       |          0.5401 | Revenues                                            |
| IBM      | dividends | linear       |          0.0205 | PaymentsOfDividendsCommonStock                      |
| IBM      | earnings  | linear       |          0.0569 | NetIncomeLoss                                       |
| IBM      | revenue   | linear       |          0.0296 | Revenues                                            |
| META     | dividends | logarithmic  |          0.6615 | PaymentsOfDividends                                 |
| META     | earnings  | linear       |          0.4098 | NetIncomeLoss                                       |
| META     | revenue   | linear       |          0.2345 | RevenueFromContractWithCustomerExcludingAssessedTax |
| MSFT     | dividends | logarithmic  |          0.1297 | CommonStockDividendsPerShareDeclared                |
| MSFT     | earnings  | logarithmic  |          0.1937 | NetIncomeLoss                                       |
| MSFT     | revenue   | logarithmic  |          0.1803 | RevenueFromContractWithCustomerExcludingAssessedTax |
| PG       | dividends | logarithmic  |          0.0888 | PaymentsOfDividends                                 |
| PG       | earnings  | logarithmic  |          0.0802 | NetIncomeLoss                                       |
| PG       | revenue   | logarithmic  |          0.0544 | Revenues                                            |

---

## 2) Held-out “next quarter” prediction vs actual (and error)

| symbol   | metric    | test_end   | best_model   |   r2_best_train | y_test_actual   | y_pred_next_quarter   | error_abs   | error_rel   | tag_used                                            |
|:---------|:----------|:------------|:-------------|----------------:|:----------------|:------------------------|:-------------|:-------------|:----------------------------------------------------|
| AAPL     | dividends | 2026-03-28 | logarithmic  |          0.0248 | 0.52            | 0.4465                 | 0.07349     | -14.13%     | CommonStockDividendsPerShareDeclared                |
| AAPL     | earnings  | 2026-03-28 | logarithmic  |          0.0025 | 7.168e+10       | 5.735e+10              | 1.433e+10   | -19.99%     | NetIncomeLoss                                       |
| AAPL     | revenue   | 2026-03-28 | logarithmic  |          0.0188 | 2.549e+11       | 2.015e+11              | 5.342e+10   | -20.95%     | RevenueFromContractWithCustomerExcludingAssessedTax |
| GE       | dividends | 2026-03-31 | linear       |          0.6697 | 0.47            | 1.046                   | 0.5757      | 122.49%     | CommonStockDividendsPerShareDeclared                |
| GE       | earnings  | 2026-03-31 | logarithmic  |          0.2318 | 1.904e+09       | 3.115e+09              | 1.211e+09   | 63.61%      | NetIncomeLoss                                       |
| GE       | revenue   | 2026-03-31 | logarithmic  |          0.1615 | 1.239e+10       | 1.966e+10              | 7.273e+09   | 58.69%      | Revenues                                            |
| GOOG     | dividends | 2026-03-31 | linear       |          0.8203 | 2.542e+09       | 7.462e+09              | 4.92e+09    | 193.55%     | PaymentsOfDividends                                 |
| GOOG     | earnings  | 2026-03-31 | exponential  |          0.4646 | 6.258e+10       | 8.263e+10              | 2.005e+10   | 32.04%      | NetIncomeLoss                                       |
| GOOG     | revenue   | 2026-03-31 | linear       |          0.5401 | 1.099e+11       | 2.687e+11              | 1.589e+11   | 144.55%     | Revenues                                            |
| IBM      | dividends | 2026-03-31 | linear       |          0.0205 | 1.576e+09       | 3.594e+09              | 2.018e+09   | 128.02%     | PaymentsOfDividendsCommonStock                      |
| IBM      | earnings  | 2026-03-31 | linear       |          0.0569 | 1.216e+09       | 3.588e+09              | 2.372e+09   | 195.07%     | NetIncomeLoss                                       |
| IBM      | revenue   | 2026-03-31 | linear       |          0.0296 | 1.592e+10       | 3.639e+10              | 2.047e+10   | 128.62%     | Revenues                                            |
| META     | dividends | 2026-03-31 | logarithmic  |          0.6615 | 1.346e+09       | 3.513e+09              | 2.167e+09   | 160.97%     | PaymentsOfDividends                                 |
| META     | earnings  | 2026-03-31 | linear       |          0.4098 | 2.677e+10       | 3.920e+10              | 1.243e+10   | 46.43%      | NetIncomeLoss                                       |
| META     | revenue   | 2026-03-31 | linear       |          0.2345 | 5.631e+10       | 1.140e+11              | 5.773e+10   | 102.52%     | RevenueFromContractWithCustomerExcludingAssessedTax |
| MSFT     | dividends | 2026-03-31 | logarithmic  |          0.1297 | 2.73            | 1.82                    | 0.9104      | -33.35%     | CommonStockDividendsPerShareDeclared                |
| MSFT     | earnings  | 2026-03-31 | logarithmic  |          0.1937 | 9.798e+10       | 5.808e+10              | 3.990e+10   | -40.72%     | NetIncomeLoss                                       |
| MSFT     | revenue   | 2026-03-31 | logarithmic  |          0.1803 | 2.418e+11       | 1.538e+11              | 8.806e+10   | -36.41%     | RevenueFromContractWithCustomerExcludingAssessedTax |
| PG       | dividends | 2026-03-31 | logarithmic  |          0.0888 | 7.623e+09       | 5.228e+09              | 2.395e+09   | -31.42%     | PaymentsOfDividends                                 |
| PG       | earnings  | 2026-03-31 | logarithmic  |          0.0802 | 1.300e+10       | 9.008e+09              | 3.994e+09   | -30.71%     | NetIncomeLoss                                       |
| PG       | revenue   | 2026-03-31 | logarithmic  |          0.0544 | 6.583e+10       | 4.558e+10              | 2.025e+10   | -30.76%     | Revenues                                            |

