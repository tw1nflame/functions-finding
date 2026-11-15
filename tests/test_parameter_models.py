import numpy as np
import pytest

from parametric_regression import (
    ConstantParameterModel,
    LinearParameterModel,
    PolynomialParameterModel,
)


@pytest.mark.parametrize(
    "model,expected_complexity",
    [
        (ConstantParameterModel(), 1),
        (LinearParameterModel(), 3),
        (PolynomialParameterModel(), 6),
        (PolynomialParameterModel(interactions=False), 5),
    ],
)
def test_parameter_models_recover_known_response(model, expected_complexity):
    rng = np.random.default_rng(5)
    X = rng.uniform(-2, 2, (40, 2))
    y = np.full(40, 3.0)
    if model.degree >= 1:
        y += 2 * X[:, 0] - X[:, 1]
    if model.degree == 2:
        y += 0.4 * X[:, 0] ** 2 + 0.7 * X[:, 1] ** 2
        if model.interactions:
            y += 0.8 * X[:, 0] * X[:, 1]
    assert model.fit(X, y, feature_names=["T", "pH"]) is model
    assert model.complexity(2) == expected_complexity
    np.testing.assert_allclose(model.predict(X), y, atol=1e-10)
    assert model.formula(latex=True)
    T, pH = X[:, 0], X[:, 1]
    predicted_from_formula = eval(model.formula(), {"__builtins__": {}}, {"T": T, "pH": pH})
    np.testing.assert_allclose(predicted_from_formula, y, atol=1e-9)


def test_rank_and_sample_count_guard():
    with pytest.raises(ValueError, match="rank deficient"):
        LinearParameterModel().fit([[1, 1], [2, 2], [3, 3]], [2, 3, 4])
    with pytest.raises(ValueError, match="more coefficients"):
        PolynomialParameterModel().fit([[1], [2]], [2, 3])


def test_weights_and_fixed_constant():
    X = np.arange(3)[:, None]
    fitted = ConstantParameterModel().fit(X, [1, 1, 5], sample_weight=[0.5, 0.5, 1])
    np.testing.assert_allclose(fitted.predict([[100]]), [3])
    fixed = ConstantParameterModel(7).fit(X, [1, 2, 3])
    assert fixed.complexity(1) == 0
    np.testing.assert_allclose(fixed.predict([[10]]), [7])


def test_failed_refit_clears_old_parameter_model():
    model = LinearParameterModel().fit([[1], [2], [3]], [1, 2, 3])
    with pytest.raises(ValueError):
        model.fit([[1], [1]], [1, 2])
    with pytest.raises(RuntimeError):
        model.predict([[2]])


@pytest.mark.parametrize(
    "X,y",
    [
        ([1, 2], [1, 2]),
        ([[np.nan]], [1]),
        ([[1]], [np.inf]),
        ([[1]], [1, 2]),
        ([["1"]], [1]),
        ([[1j]], [1]),
    ],
)
def test_invalid_coefficient_data(X, y):
    with pytest.raises(ValueError):
        LinearParameterModel().fit(X, y)


def test_no_unfitted_prediction_and_invalid_options():
    with pytest.raises(RuntimeError):
        LinearParameterModel().predict([[1]])
    with pytest.raises(ValueError):
        PolynomialParameterModel(degree=3)
    with pytest.raises(ValueError):
        PolynomialParameterModel(interactions="yes")
