# Machine Learning Progression | from scratch

Implementing one ML algorithm or technique from scratch every day. The rule is that before using a library implementation of something, It has to be built manually at least once. Once thats dibe, the library version is fair game going forward.

## Implementations

| Day | Topic | Description |
|-----|-------|-------------|
| 01 | [Linear Regression](01-linear-regression-predict-ho/) | Univariate linear regression trained with gradient descent on the Boston Housing dataset. Compares normalization strategies (MinMax, Z-score) and loss functions (MSE, MAE, RMSE) and how they affect convergence and fit quality. |

## Structure

Each day gets its own directory with a README, data, and implementation. Directories are numbered by day.

## Running

Projects use [uv](https://github.com/astral-sh/uv) for dependency management. From any day's directory:

```bash
uv sync
```
then run the notebook

Check each day's README for setup and anything else specific to that project.