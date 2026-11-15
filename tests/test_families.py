import numpy as np
import pytest

from parametric_regression import (
    ExponentialFamily,
    GompertzFamily,
    LogisticFamily,
    PolynomialFamily,
    RationalSaturationFamily,
)


@pytest.mark.parametrize(
    "family,params,expected",
    [
        (PolynomialFamily(1), [2, 3], [2, 5, 8]),
        (PolynomialFamily(2), [2, 3, 4], [2, 9, 24]),
        (PolynomialFamily(3), [1, 2, 3, 4], [1, 10, 49]),
        (ExponentialFamily(), [2, 5, -0.3], 2 + 5 * np.exp(-0.3 * np.arange(3))),
        (LogisticFamily(), [2, 5, 1, 1], 2 + 5 / (1 + np.exp(-np.array([-1, 0, 1])))),
        (GompertzFamily(), [2, 5, 1, 1], 2 + 5 * np.exp(-np.exp(-np.array([-1, 0, 1])))),
        (RationalSaturationFamily(), [2, 5, 1], [2, 4.5, 2 + 10 / 3]),
    ],
)
def test_known_values_and_output_shape(family, params, expected):
    prediction = family.evaluate([0, 1, 2], params)
    np.testing.assert_allclose(prediction, expected)
    assert prediction.shape == (3,)
    assert np.isfinite(prediction).all()
    assert family.evaluate([], params).shape == (0,)
    assert "y =" in family.formula(params)
    assert "y =" in family.formula(params, latex=True)
    for invalid in (params[:-1], [np.nan] * len(params), [np.inf] * len(params)):
        with pytest.raises(ValueError):
            family.evaluate([1, 2], invalid)


@pytest.mark.parametrize("degree", [0, 4, -1, 1.5, True])
def test_invalid_polynomial_degree(degree):
    with pytest.raises(ValueError):
        PolynomialFamily(degree)


@pytest.mark.parametrize(
    "x,params", [([-1, 0], [1, 2, 3]), ([0, 1], [1, 2, 0]), ([0, 1], [1, 2, -1])]
)
def test_rational_domain_and_parameters(x, params):
    with pytest.raises(ValueError):
        RationalSaturationFamily().evaluate(x, params)


def test_extreme_saturation_is_finite():
    x = np.array([-1e6, 0, 1e6])
    for family in (LogisticFamily(), GompertzFamily()):
        prediction = family.evaluate(x, [2, 5, 100, 0])
        assert np.isfinite(prediction).all()
        np.testing.assert_allclose(prediction[[0, 2]], [2, 7])
    np.testing.assert_allclose(RationalSaturationFamily().evaluate([1e308], [0, 2, 1e308]), [1])


def test_exponential_avoids_intermediate_overflow_but_rejects_true_overflow():
    family = ExponentialFamily()
    np.testing.assert_allclose(family.evaluate([1], [2, 1e-300, 700]), [2 + 1e-300 * np.exp(700)])
    assert np.isfinite(family.evaluate([1], [0, 1e-300, 800])).all()
    np.testing.assert_array_equal(family.evaluate([1e100], [2, 0, 1e100]), [2])
    with pytest.raises(FloatingPointError):
        family.evaluate([1], [0, 1, 800])


@pytest.mark.parametrize("values", [1, [[1], [2]], [np.nan], [np.inf], [1j], ["1"], [True]])
def test_invalid_input_vectors(values):
    with pytest.raises(ValueError):
        PolynomialFamily().evaluate(values, [1, 2])


def test_logistic_amplitude_bound_and_latex_scientific_notation():
    with pytest.raises(ValueError):
        LogisticFamily().evaluate([1], [0, -1, 1, 1])
    assert r"\times 10^{-20}" in PolynomialFamily().formula([1e-20, 2], latex=True)
