"""Общие метрики регрессии для подбора параметров и выбора моделей."""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ._validation import validate_xy


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Вычислить MAE, RMSE и R² для конечных одномерных массивов одинаковой длины.

    При одном наблюдении R² равен NaN. Для постоянной целевой величины используется
    соглашение sklearn: 1 при точном прогнозе, 0 в остальных случаях.
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
