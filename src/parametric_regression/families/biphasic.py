"""Ordered two-rate survival, without interchangeable mixture components."""

import numpy as np
from scipy.special import expit

from .._validation import FloatArray
from ._observed import _PositiveInitialFamily


class BiphasicExponentialFamily(_PositiveInitialFamily):
    """y = y0*(sigmoid(q)*exp(-(k_slow+delta_k)*t) + sigmoid(-q)*exp(-k_slow*t)).

    k_slow > 0 and delta_k > 0 impose k_fast > k_slow. q is unrestricted.
    Components describe two time scales, not necessarily two biological populations.
    """

    name = "biphasic_exponential"
    parameter_names = ("k_slow", "delta_k", "q")
    positive_parameters = ("k_slow", "delta_k")
    expression_template = (
        "@y0@ * ((1/(1+exp(-(@q@))))*exp(-((@k_slow@)+(@delta_k@))*@t@)"
        " + (1/(1+exp(@q@)))*exp(-(@k_slow@)*@t@))"
    )
    latex_expression_template = (
        r"@y0@\left[\frac{\exp(-((@k_slow@)+(@delta_k@))@t@)}{1+\exp(-(@q@))}"
        r"+\frac{\exp(-(@k_slow@)@t@)}{1+\exp(@q@)}\right]"
    )

    def parameter_bounds(self) -> tuple[FloatArray, FloatArray]:
        """Order rates using a positive increment; leave the logit unconstrained."""
        return np.array([np.finfo(float).tiny, np.finfo(float).tiny, -np.inf]), np.full(3, np.inf)

    def _evaluate(self, x: FloatArray, params: FloatArray) -> FloatArray:
        slow, increment, q = params
        weight, complement = expit(q), expit(-q)
        log_weight = np.log(weight) if weight > 0 else -np.logaddexp(0.0, -q)
        log_complement = np.log(complement) if complement > 0 else -np.logaddexp(0.0, q)
        with np.errstate(over="ignore"):
            slow_reduction = slow * x
            fast_reduction = increment * x
            mixture = np.logaddexp(log_weight - fast_reduction, log_complement)
        result = np.exp(np.log(self.initial_value) - slow_reduction + mixture)
        result[x == 0] = self.initial_value
        return result

    def initial_guess(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Return the first ordered two-rate guess."""
        return self.initial_guesses(x, y)[0]

    def identifiability_warnings(self, x: FloatArray, params: FloatArray) -> tuple[str, ...]:
        """Flag negligible mixture weights or rates unresolved over the observed interval."""
        slow, increment, q = params
        weight = expit(q)
        warnings = []
        if weight < 1e-4 or weight > 1 - 1e-4:
            warnings.append(
                "Biphasic components are weakly identifiable: a mixture weight is negligible."
            )
        span = float(np.ptp(x)) if len(x) else 0.0
        if increment < 1e-3 * slow or (span > 0 and increment < 1e-3 / span):
            warnings.append(
                "Biphasic rates are practically indistinguishable over the observed interval."
            )
        return tuple(warnings)

    def initial_guesses(self, x: FloatArray, y: FloatArray) -> list[FloatArray]:
        """Estimate the late slope and try nine weight/rate-separation combinations."""
        self.validate_domain(x)
        span = float(np.ptp(x))
        valid = (y > 0) & (x >= np.median(x))
        slow = 0.5 / span
        if np.count_nonzero(valid) >= 2 and np.unique(x[valid]).size >= 2:
            slope = np.linalg.lstsq(
                np.column_stack([np.ones(np.count_nonzero(valid)), x[valid]]),
                np.log(y[valid]),
                rcond=None,
            )[0][1]
            slow = max(float(-slope), 0.01 / span)
        return [
            np.array([slow, factor / span, q])
            for factor in (3.0, 10.0, 30.0)
            for q in (0.0, -1.5, 1.5)
        ]
