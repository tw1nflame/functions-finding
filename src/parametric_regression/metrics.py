"""Regression metrics shared by fitting and selection."""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ._validation import validate_xy


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Return MAE, RMSE and R² for equal finite 1-D arrays.

    R² is NaN for one observation. For constant targets sklearn's finite
    convention is used: 1 for an exact prediction and 0 otherwise.
    """
    actual, predicted = validate_xy(y_true, y_pred)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return {
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted, force_finite=True))
            if actual.size >= 2
            else float("nan"),
        }
