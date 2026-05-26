# CISC 7700X HW#6 — Classification model on `hw6.data.csv`

Dataset:
- `hw6/hw6.data.csv`
- **Last column is the label**
- Other columns are numeric features

## Model

Baseline classifier:
- **StandardScaler + multinomial Logistic Regression** (`lbfgs`)

## Train/test split

- Random split: **70% train / 30% test**
- Stratified by label
- Default seed: `42`

## Run

From repo root:

```bash
python3 hw6/hw6_train_eval.py --data hw6/hw6.data.csv --test-size 0.30 --seed 42
```

Outputs:
- prints **test accuracy** and **confusion matrix**
- writes `hw6/outputs/hw6_results.json`

