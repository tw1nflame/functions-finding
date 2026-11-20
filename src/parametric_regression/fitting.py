"""Shared linear/nonlinear parameter fitting with deterministic multi-start."""

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

from ._validation import FloatArray, validate_xy
from .families import BaseFunctionFamily
from .metrics import regression_metrics
from .results import FitResult


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def fit_family(
    family: BaseFunctionFamily,
    x: ArrayLike,
    y: ArrayLike,
    *,
    max_nfev: int = 2000,
    max_starts: int = 12,
) -> FitResult:
    """Fit a family to x/y and return coefficients, training metrics and diagnostics.

    Invalid shared inputs/settings raise ValueError. Family-specific failures
    (domain, insufficient data, overflow, convergence) return success=False.
    max_nfev limits each nonlinear start; max_starts caps deterministic guesses.
    Only supplied data are used for initialization and optimization.
    """
    x, y = validate_xy(x, y)
    _positive_integer(max_nfev, "max_nfev")
    _positive_integer(max_starts, "max_starts")
    if not isinstance(family, BaseFunctionFamily):
        raise TypeError("family must implement BaseFunctionFamily.")
    result = FitResult(family=family, success=False)
    try:
        family.validate_domain(x)
        if len(x) < family.n_parameters or np.unique(x).size < family.n_parameters:
            raise ValueError(f"At least {family.n_parameters} distinct x values are required.")
        if np.ptp(x) == 0:
            raise ValueError("x must vary for parameter fitting.")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
                design = family.design_matrix(x)
                if design is not None:
                    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
                    if rank < family.n_parameters:
                        raise ValueError("Linear design matrix is rank deficient.")
                    params = family.linear_parameters(coefficients, x)
                    result.message = "Linear least squares converged."
                    result.nfev = 1
                    result.jacobian_rank, result.jacobian_condition = _jacobian_diagnostics(design)
                else:
                    params, message, count, diagnostics, rank, condition = _nonlinear_fit(
                        family, x, y, max_nfev, max_starts
                    )
                    result.message, result.nfev = message, count
                    result.warnings = diagnostics
                    result.jacobian_rank, result.jacobian_condition = rank, condition
                prediction = family.evaluate(x, params)
                result.warnings += family.identifiability_warnings(x, params)
                result.train_metrics = regression_metrics(y, prediction)
                # Validate formula generation before declaring a usable result.
                family.formula(params)
                family.formula(params, latex=True)
            result.warnings += tuple(dict.fromkeys(str(w.message) for w in caught))
        result.params = np.array(params, dtype=float, copy=True)
        result.params.setflags(write=False)
        result.success = True
    except Exception as error:
        # Extension families can fail independently; selection must retain that failure.
        result.message = f"{type(error).__name__}: {error}"
    return result


def _nonlinear_fit(
    family: BaseFunctionFamily,
    x: FloatArray,
    y: FloatArray,
    max_nfev: int,
    max_starts: int,
) -> tuple[FloatArray, str, int, tuple[str, ...], int, float]:
    best = None
    failures: list[str] = []
    count = 0
    scale = max(float(np.std(y)), float(np.max(np.abs(y))) * 1e-8, 1e-12)

    def residual(params: FloatArray) -> FloatArray:
        return (family.evaluate(x, params) - y) / scale

    guesses = family.initial_guesses(x, y)[:max_starts]
    for index, guess in enumerate(guesses):
        try:
            family.validate_parameters(guess)
            attempt = least_squares(
                residual,
                guess,
                bounds=family.parameter_bounds(),
                method="trf",
                x_scale="jac",
                max_nfev=max_nfev,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            count += attempt.nfev
            if not attempt.success or not np.isfinite(attempt.cost):
                failures.append(f"start {index + 1}: {attempt.message}")
                continue
            if best is None or attempt.cost < best.cost:
                best = attempt
        except Exception as error:
            failures.append(f"start {index + 1}: {type(error).__name__}: {error}")
    if best is None:
        raise RuntimeError("No converged start. " + ("; ".join(failures) or "No feasible guesses."))
    rank, condition = _jacobian_diagnostics(best.jac)
    if rank < family.n_parameters:
        failures.append("Jacobian is rank deficient; individual parameters may be unidentifiable.")
    elif condition > 1e8:
        failures.append(
            "Jacobian is ill-conditioned; individual parameters may be weakly identifiable."
        )
    return best.x, str(best.message), count, tuple(failures), rank, condition


def _jacobian_diagnostics(matrix: FloatArray) -> tuple[int, float]:
    """Assess local identifiability after column normalization (units independent)."""
    norms = np.linalg.norm(matrix, axis=0)
    normalized = matrix / np.where(norms > 0, norms, 1.0)
    return int(np.linalg.matrix_rank(normalized)), float(np.linalg.cond(normalized))
