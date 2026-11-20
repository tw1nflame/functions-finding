"""Specialized condition laws: scales, domains, weights and shared fitting."""

import numpy as np
import pytest

from parametric_regression import (
    ArrheniusParameterModel,
    HillParameterModel,
    MonodParameterModel,
    RatkowskyParameterModel,
)


def case(kind):
    if kind == "arrhenius":
        model = ArrheniusParameterModel(298.15)
        x = np.linspace(285, 320, 15)
        y = np.log(0.2) - 40000 / model.R * (1 / x - 1 / 298.15)
    elif kind == "ratkowsky":
        model = RatkowskyParameterModel(temperature_max=40)
        x = np.linspace(10, 35, 15)
        y = 2 * np.log(0.03) + 2 * np.log(x + 4)
    elif kind == "hill":
        model = HillParameterModel()
        x = np.r_[0, np.geomspace(0.01, 30, 20)]
        y = 1 - 2 * x**1.7 / (2**1.7 + x**1.7)
    else:
        model = MonodParameterModel()
        x = np.r_[0, np.geomspace(0.01, 30, 20)]
        y = 0.7 * x / (2 + x)
    return model, x, y


@pytest.mark.parametrize("kind", ["arrhenius", "ratkowsky", "hill", "monod"])
def test_specialized_recovery_and_formulas(kind):
    model, x, y = case(kind)
    assert model.fit(x[:, None], y, feature_names=["condition"]) is model
    np.testing.assert_allclose(model.predict(x[:, None]), y, atol=1e-7)
    assert model.formula() and model.formula(latex=True)
    assert "condition" in model.formula()
    assert "@" not in model.formula()
    with pytest.raises(ValueError, match="one"):
        model.predict(np.ones((3, 2)))
    if kind == "arrhenius":
        np.testing.assert_allclose([model.k_ref_, model.E_a_], [0.2, 40000], rtol=1e-7)
    if kind == "ratkowsky":
        np.testing.assert_allclose([model.b_, model.T_min_], [0.03, -4], atol=1e-7)
    if kind == "hill":
        np.testing.assert_allclose(list(model.parameters_.values()), [1, -2, 2, 1.7], atol=1e-7)
    if kind == "monod":
        np.testing.assert_allclose([model.mu_max_, model.K_s_], [0.7, 2], atol=1e-7)
        np.testing.assert_array_equal(model.predict([[0]]), [0])


@pytest.mark.parametrize("kind", ["arrhenius", "ratkowsky", "hill", "monod"])
def test_weighted_fits_match_balanced_observations(kind):
    model, x, y = case(kind)
    y = y + 0.002 * np.sin(np.arange(len(y)))
    if kind == "monod":
        y[0] = 0
    model.fit(x[:, None], y)
    counts = np.arange(len(x)) % 3 + 1
    duplicate = np.repeat(np.arange(len(x)), counts)
    weighted, _, _ = case(kind)
    weighted.fit(x[duplicate, None], y[duplicate], sample_weight=1 / counts[duplicate])
    np.testing.assert_allclose(weighted.predict(x[:, None]), model.predict(x[:, None]), atol=1e-6)


@pytest.mark.parametrize("kind", ["arrhenius", "ratkowsky", "hill", "monod"])
def test_parameter_model_validation_and_failed_refit(kind):
    model, x, y = case(kind)
    with pytest.raises(RuntimeError):
        model.predict([[1]])
    with pytest.raises(RuntimeError):
        model.formula()
    with pytest.raises(ValueError):
        model.fit(np.column_stack([x, x]), y)
    with pytest.raises(ValueError):
        model.fit(x[:, None], y, sample_weight=np.zeros(len(y)))
    with pytest.raises(ValueError):
        model.fit(x[:, None], y, feature_names=["a", "b"])
    model.fit(x[:, None], y)
    with pytest.raises(ValueError, match="distinct"):
        model.fit(np.ones((20, 1)), np.ones(20))
    with pytest.raises(RuntimeError):
        model.predict([[1]])


def test_arrhenius_kelvin_and_nonnegative_activation_energy():
    for ref in (0, -1, np.inf, np.nan):
        with pytest.raises(ValueError):
            ArrheniusParameterModel(ref)
    model, x, y = case("arrhenius")
    with pytest.raises(ValueError, match="Kelvin"):
        model.fit([[-1], [1]], [0, 1])
    model.fit(x[:, None], y)
    with pytest.raises(ValueError, match="Kelvin"):
        model.predict([[0]])
    model.fit(x[:, None], -y)
    assert model.E_a_ == 0


def test_ratkowsky_declared_domain_and_no_automatic_upper_extrapolation():
    model, x, y = case("ratkowsky")
    model.fit(x[:, None], y)
    with pytest.raises(ValueError, match="T_min"):
        model.predict([[model.T_min_]])
    with pytest.raises(ValueError, match="upper"):
        model.predict([[41]])
    model = RatkowskyParameterModel().fit(x[:, None], y)
    with pytest.raises(ValueError, match="upper"):
        model.predict([[36]])
    with pytest.raises(ValueError):
        RatkowskyParameterModel(temperature_max=20).fit(x[:, None], y)


@pytest.mark.parametrize("kind", ["hill", "monod"])
def test_concentration_domains_and_saturation_degeneracy(kind):
    model, x, y = case(kind)
    model.fit(x[:, None], y)
    with pytest.raises(ValueError):
        model.predict([[-1]])
    with pytest.raises(ValueError):
        model.fit(x[:, None], np.zeros_like(y))
    if kind == "monod":
        with pytest.raises(ValueError, match="nonnegative"):
            model.fit(x[:, None], -np.ones_like(y))


def test_hill_stable_sigmoid_without_powers_overflow():
    model, x, y = case("hill")
    model.fit(x[:, None], y)
    predicted = model.predict([[0], [1e308]])
    np.testing.assert_allclose(predicted, [1, -1], atol=1e-7)
