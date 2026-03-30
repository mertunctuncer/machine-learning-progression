# Machine Learning Progression | from scratch

Implementing one ML algorithm or technique from scratch every day. 
The rule is that before using a library implementation of something, it has to be built manually at least once. 
Once that is done, the library version is fair game going forward.

## Implementations

| Day | Topic                                                                  | Description                                                                                                                                                                                                                        |
|-----|------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 01  | [Linear Regression](01_single_variable_linear_regression/)             | Univariate linear regression trained with gradient descent on the Boston Housing dataset. Compares normalization strategies (MinMax, Z-score) and loss functions (MSE, MAE, RMSE) and how they affect convergence and fit quality. |
| 02  | [Multivariable Linear Regression](02_multivariable_linear_regression/) | Extends Day 1 to all 13 Boston Housing features. Adds R², MAPE, and MaxError metrics. Compares outlier removal strategies and finds that aggressive filtering hurts performance on this small dataset.                             |

## Structure

Each day gets its own directory with a README, data, and implementation. Directories are numbered by day.

## Running

Projects use [uv](https://github.com/astral-sh/uv) for dependency management. From any day's directory:

```bash
uv sync
```

Then run the notebook. Check each day's README for setup and anything else specific to that project.