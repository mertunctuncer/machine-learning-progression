# Day 02 - Multivariable Linear Regression

An extension of the Day 1 implementation to multivariable linear regression on the full Boston Housing dataset.
All 13 features are used to predict median home value (MEDV). No ML frameworks are used for the model itself only NumPy, Pandas, and sklearn's `train_test_split`.
Structure of the code was also improved to make it more manageable.

## Dataset

The [Boston Housing dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html). The task is to predict median home value (MEDV) from all 13 features.

The dataset is split 80/20 into train and test sets. 
Two training configurations were compared: with and without IQR-based outlier removal (k=3.0) applied to all continuous features and the target in the training set only. 
The variance of each feature is checked after filtering and any zero-variance features are dropped before training.

## Implementation

All components are implemented from scratch using only NumPy and Pandas.

**Normalizers**
- `MinMaxNormalizer` scales features to [0, 1]
- `ZScoreNormalizer` standardizes to zero mean and unit variance

**Loss functions**
- `MSELoss` mean squared error
- `MAELoss` mean absolute error
- `RMSELoss` root mean squared error

**Evaluation metrics**
- MSE, MAE, RMSE, R², MAPE, MaxError

## Training

Gradient descent with convergence detected by relative change falling below tol=1e-6 for patience=20 consecutive epochs, hard cap at 50,000 epochs. 
A learning rate sweep over [1e-1, 1e-2, 1e-3, 1e-4, 1e-5] is run per (normalization, loss) combination before the real training run.
Only runs where a converging lr was found are included in the results.
Divergence (nan/inf loss) is detected beforehand.

## Results

Six combinations were attempted. 
MinMax+MAE failed to converge under any tested learning rate and was skipped. 
Z-score+MAE technically converged but took far longer than other runs and produced a noticeably worse R² — MAE's sign-based gradient oscillates near the optimum and does not work well with plain gradient descent.
The best model was Z-score + MSE (R² = 0.6683).

### Best model: Z-score + MSE without outlier removal (R² = 0.6683)

### Feature weights
| Feature  | Weight  |
|----------|---------|
| LSTAT    | -0.3872 |
| RM       | +0.3379 |
| DIS      | -0.3304 |
| RAD      | +0.2355 |
| PTRATIO  | -0.2184 |
| NOX      | -0.2167 |
| TAX      | -0.1828 |
| B        | +0.1212 |
| CRIM     | -0.1071 |
| CHAS     | +0.0774 |
| ZN       | +0.0736 |
| INDUS    | +0.0278 |
| AGE      | -0.0191 |

LSTAT (lower status population %) and RM (average rooms) are the strongest predictors, which matches domain intuition. AGE and INDUS contribute almost nothing.

![graphs](multivariable_analysis.png)

## Usage

From the project directory:

```bash
uv sync
```

Then run the notebook.