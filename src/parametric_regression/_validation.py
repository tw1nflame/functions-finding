"""Shared validation; missing observations are never silently dropped."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def as_vector(values: ArrayLike, name: str, *, allow_empty: bool = False) -> FloatArray:
    """Return a finite real 1-D float array or raise ValueError."""
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a one-dimensional array of real numbers.")
    array = np.asarray(array, dtype=np.float64)
    if not allow_empty and array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values (no NaN or infinity).")
    return array


def validate_xy(x: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Return validated x and y of equal length."""
    x_array, y_array = as_vector(x, "x"), as_vector(y, "y")
    if x_array.shape != y_array.shape:
        raise ValueError("x and y must have the same length.")
    return x_array, y_array
