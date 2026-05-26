# CISC 7700X HW#4 — Quarterly Financial Modeling (SEC XBRL)

This homework downloads quarterly financial statement values from the SEC public XBRL endpoint (“companyfacts”), fits four simple curve families to each metric:

- Linear: `y = a + b x`
- Logarithmic: `y = a + b log(x)`
- Exponential: `y = b * exp(a*x)`  (only if `y > 0`)
- Power: `y = b * x^a` (only if `y > 0`)

For each `(symbol, metric)` pair, it chooses the “best” model using the **coefficient of determination** (R²):

`R² = 1 - SS_res / SS_tot`  (see Wikipedia’s “Coefficient of determination”).  
Source: [Wikipedia – Coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination).

It then predicts the **next quarter** by holding out the most recent quarterly value as the “target”, fitting models on the previous quarters, and comparing the prediction to the held-out “latest quarter”.

We follow the homework hint for how to compute error / validate predictions (see hint file).  
Source: [hw4.hint.txt](https://www.theparticle.com/cs/bc/dsci/hw4.hint.txt).

## Run

From the repo root:

```bash
python3 hw4/hw4_models.py
```

Optional:

```bash
python3 hw4/hw4_models.py --n-train 8 --min-points 9 --symbols IBM,MSFT,AAPL,GOOG,META,PG,GE
```

## Outputs

The script writes:

- `hw4/outputs/hw4_quarterly_model_results.csv`
- `hw4/outputs/hw4_quarterly_model_results.json`
- `hw4/outputs/hw4_summary.md` (human-readable best-model + prediction-error tables)

It also caches SEC API responses under:

- `hw4/.cache/`

