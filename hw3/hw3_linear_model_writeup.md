# HW3 Linear Model (Cost-Sensitive Evaluation)

## Dataset
Used `hw3/hw3.data1.csv` (repo copy of `hw3.data1.csv.gz`).

CSV header format:
- Features: `column1 ... column20`
- Label: `label` in `{ -1, +1 }`

Let the positive class be `+1`.

## Given linear model
Score:

`y = 24*column1 + -15*column2 + -38*column3 + -7*column4 + -41*column5 + 35*column6 + 0*column7 + -2*column8 + 19*column9 + 33*column10 + -3*column11 + 7*column12 + 3*column13 + -47*column14 + 26*column15 + 10*column16 + 40*column17 + -1*column18 + 3*column19 + 0*column20 - 6`

Decision rule:
- Predict `+1` if `y > 0`
- Otherwise predict `-1`

### Confusion matrix + accuracy (N = 10,000)

Confusion matrix (rows = true, cols = predicted):

|          | Pred -1 | Pred +1 |
|----------|----------|----------|
| True -1  | 5259     | 484      |
| True +1  | 707      | 3550     |

Counts:
- TP = 3550
- TN = 5259
- FP = 484
- FN = 707

Accuracy:
- `(TP + TN) / N = (3550 + 5259) / 10000 = 0.8809`

## Economic gain (cost-sensitive)
Costs:
- False negative (FN): $1000
- False positive (FP): $100
- Correct prediction: $0

Total cost:
`cost = 1000*FN + 100*FP = 1000*707 + 100*484 = 755,400`

Economic gain:
- If we define `economic_gain = -cost` (since correct answers cost $0),
- Total economic gain = `-755,400`
- Expected economic gain per instance = `-755,400 / 10,000 = -75.54`

## How to tweak the model to increase economic gain
Because the cost of a false negative is 10x the cost of a false positive, we should:
- Reduce `FN` even if it increases `FP`

For a linear model, a simple way is to shift the decision threshold:
- Predict `+1` if `y > t` (instead of `t = 0`)

This is equivalent to adjusting the intercept/bias by `-t`.

## Cost-maximizing tweak (threshold scan)
I scanned candidate thresholds `t` over the score distribution (midpoints between observed scores), and chose the threshold that minimizes:
- `1000*FN + 100*FP`

Best threshold found:
- `t = -1004.5`

Equivalent bias adjustment (keeping the provided coefficients):
- Original bias = `-6`
- New bias = `-6 - t = -6 - (-1004.5) = 998.5`

### Confusion matrix + accuracy for the tuned model

Decision rule:
- Predict `+1` if `y > -1004.5`
- Else predict `-1`

Confusion matrix (rows = true, cols = predicted):

|          | Pred -1 | Pred +1 |
|----------|----------|----------|
| True -1  | 4624     | 1119     |
| True +1  | 0        | 4257     |

Counts:
- TP = 4257
- TN = 4624
- FP = 1119
- FN = 0

Accuracy:
- `(TP + TN) / N = (4257 + 4624) / 10000 = 0.8881`

Economic gain:
- Total cost = `1000*0 + 100*1119 = 111,900`
- Total economic gain = `-111,900`
- Expected economic gain per instance = `-111,900 / 10,000 = -11.19`

