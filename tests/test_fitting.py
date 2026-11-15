import numpy as np
import pytest

from parametric_regression import (
    BaseFunctionFamily,
    ExponentialFamily,
    GompertzFamily,
    LogisticFamily,
    PolynomialFamily,
    RationalSaturationFamily,
    fit_family,
    regression_metrics,
)

CASES = [
    (PolynomialFamily(1), [2, 3], (-2, 3)),
    (PolynomialFamily(2), [2, -3, 0.5], (-2, 3)),
    (PolynomialFamily(3), [2, -3, 0.5, 0.25], (-2, 3)),
    (ExponentialFamily(), [2, 5, -0.3], (0, 20)),
    (ExponentialFamily(), [-2, 1.5, 0.4], (0, 8)),
    (LogisticFamily(), [2, 5, 1.2, 4], (-2, 10)),
    (LogisticFamily(), [2, 5, -1.2, 4], (-2, 10)),
    (GompertzFamily(), [2, 5, 0.8, 4], (-2, 12)),
    (GompertzFamily(), [8, -5, 0.8, 4], (-2, 12)),
    (GompertzFamily(), [2, 5, -0.8, 4], (-4, 12)),
    (RationalSaturationFamily(), [2, 5, 3], (0, 20)),
]


@pytest.mark.parametrize("family,params,domain", CASES)
def test_recovers_identifiable_noiseless_parameters(family, params, domain):
    x = np.linspace(*domain, 120)
    y = family.evaluate(x, params)
    fitted = fit_family(family, x, y)
    assert fitted.success, fitted.message
    np.testing.assert_allclose(fitted.params, params, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(fitted.predict(x), y, rtol=1e-6, atol=1e-6)
    assert fitted.metrics["rmse"] < 1e-6
    assert set(fitted.coefficients) == set(family.parameter_names)
    assert fitted.formula and fitted.latex_formula


@pytest.mark.parametrize("family,params,domain", CASES)
def test_noisy_curves_and_unsorted_x(family, params, domain):
    rng = np.random.default_rng(42)
    x = rng.permutation(np.linspace(*domain, 100))
    clean = family.evaluate(x, params)
    fitted = fit_family(family, x, clean + rng.normal(0, 0.03, len(x)))
    assert fitted.success, fitted.message
    grid = np.linspace(*domain, 250)
    assert regression_metrics(family.evaluate(grid, params), fitted.predict(grid))["rmse"] < 0.04


def test_optimizer_budget_failure_is_diagnostic():
    x = np.linspace(0, 20, 100)
    result = fit_family(ExponentialFamily(), x, 2 + 5 * np.exp(-0.3 * x), max_nfev=1)
    assert not result.success
    assert "No converged start" in result.message
    with pytest.raises(RuntimeError):
        result.predict([1, 2])


@pytest.mark.parametrize(
    "x,y",
    [([], []), ([1, 2], [1]), ([1, np.nan], [1, 2]), ([1, 2], [np.inf, 2]), ([[1], [2]], [1, 2])],
)
def test_bad_observations_raise(x, y):
    with pytest.raises(ValueError):
        fit_family(PolynomialFamily(), x, y)


def test_insufficient_distinct_points_and_domain_failures():
    for family, x, y in (
        (PolynomialFamily(3), [0, 1, 2], [0, 1, 4]),
        (PolynomialFamily(2), [1, 1, 1, 2], [1, 2, 3, 4]),
        (RationalSaturationFamily(), [-1, 0, 1, 2], [1, 2, 3, 4]),
    ):
        result = fit_family(family, x, y)
        assert not result.success
        assert result.message


def test_centered_polynomial_fit():
    x = 1000 + np.linspace(-10, 10, 100)
    y = 3 + 2 * (x - 1000) + 0.5 * (x - 1000) ** 2
    result = fit_family(PolynomialFamily(2), x, y)
    assert result.success
    np.testing.assert_allclose(result.predict(x), y, atol=1e-8)


def test_metrics_against_hand_computed_values_and_constant_target():
    metrics = regression_metrics([1, 2, 3], [2, 2, 2])
    assert metrics == pytest.approx({"mae": 2 / 3, "rmse": np.sqrt(2 / 3), "r2": 0})
    assert regression_metrics([2, 2], [2, 2])["r2"] == 1
    assert regression_metrics([2, 2], [3, 3])["r2"] == 0
    assert np.isnan(regression_metrics([1], [2])["r2"])


class SineFamily(BaseFunctionFamily):
    """A new family with no changes to fitting or selection."""

    name = "sine"
    parameter_names = ("amplitude", "offset")

    def _evaluate(self, x, params):
        return params[0] * np.sin(x) + params[1]

    def initial_guess(self, x, y):
        return np.array([1.0, np.mean(y)])

    def formula(self, params, *, latex=False):
        return f"y = {params[0]}*sin(x) + {params[1]}"


def test_extension_family_works_without_registry_changes():
    from parametric_regression import ModelSelector

    x = np.linspace(0, 12, 100)
    result = ModelSelector([PolynomialFamily(), SineFamily()]).fit(x, 3 * np.sin(x) + 2)
    assert result.best_family == "sine"
    np.testing.assert_allclose(result.best_result.params, [3, 2], atol=1e-7)
