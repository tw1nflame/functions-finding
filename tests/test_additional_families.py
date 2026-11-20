"""Domain, stability, recovery and identifiability of opt-in temporal families."""

import numpy as np
import pytest
from scipy.special import expit

from parametric_regression import (
    BiphasicExponentialFamily,
    DelayedExponentialFamily,
    ExponentialRelaxationFamily,
    LogarithmicFamily,
    ModelSelector,
    PowerLawFamily,
    WeibullSurvivalFamily,
    fit_family,
)

CASES = [
    (PowerLawFamily(), [2, -0.4, 1.3], np.linspace(0, 10, 60)),
    (LogarithmicFamily(), [2, 3, 1.5], np.linspace(0, 30, 60)),
    (DelayedExponentialFamily(2), [3, 0.3, 2], np.linspace(0, 20, 80)),
    (DelayedExponentialFamily(2), [-1.5, 0.3, 2], np.linspace(0, 20, 80)),
    (WeibullSurvivalFamily(100), [15, 1.4], np.linspace(0, 35, 70)),
    (BiphasicExponentialFamily(100), [0.06, 0.4, 0.3], np.linspace(0, 60, 90)),
]


@pytest.mark.parametrize("family,params,time", CASES)
def test_recover_well_identified_temporal_parameters(family, params, time):
    target = family.evaluate(time, params)
    fitted = fit_family(family, time, target)
    assert fitted.success, fitted.message
    np.testing.assert_allclose(fitted.params, params, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(fitted.predict(time), target, atol=2e-6)
    assert fitted.jacobian_rank == family.n_parameters
    assert fitted.formula and fitted.latex_formula
    assert "@" not in fitted.formula + fitted.latex_formula
    np.testing.assert_array_equal(family.evaluate([], params), [])


def test_values_and_initial_conditions():
    t = np.array([0.0, 1.0, 2.0, 5.0])
    np.testing.assert_allclose(PowerLawFamily().evaluate(t, [2, -3, 0.5]), 2 - 3 * np.sqrt(t))
    np.testing.assert_allclose(LogarithmicFamily().evaluate(t, [2, -3, 4]), 2 - 3 * np.log1p(t / 4))
    np.testing.assert_allclose(
        WeibullSurvivalFamily(100).evaluate([0, 15, 30], [15, 1]), [100, 10, 1]
    )
    q = 0.7
    expected = 100 * (expit(q) * np.exp(-0.5 * t) + expit(-q) * np.exp(-0.1 * t))
    np.testing.assert_allclose(BiphasicExponentialFamily(100).evaluate(t, [0.1, 0.4, q]), expected)
    f = DelayedExponentialFamily(2)
    np.testing.assert_array_equal(f.evaluate([0, 1, 2], [3, 0.4, 2]), [2, 2, 2])
    np.testing.assert_allclose(
        f.evaluate(t, [3, 0.4, 0]), ExponentialRelaxationFamily(2).evaluate(t, [3, 0.4])
    )


@pytest.mark.parametrize("family,params,time", CASES)
def test_temporal_domains_and_parameter_constraints(family, params, time):
    with pytest.raises(ValueError):
        family.evaluate([-1, 0], params)
    with pytest.raises(ValueError):
        family.evaluate(time, params[:-1])
    for name in family.positive_parameters:
        bad = np.array(params, dtype=float)
        bad[family.parameter_names.index(name)] = 0
        with pytest.raises(ValueError):
            family.evaluate(time, bad)
    assert family.expression(dict.fromkeys(family.parameter_names, "z"))
    assert family.expression(dict.fromkeys(family.parameter_names, "z"), latex=True)


@pytest.mark.parametrize("family", [WeibullSurvivalFamily, BiphasicExponentialFamily])
def test_positive_known_initial_value(family):
    for value in (0, -1, np.inf, np.nan, True, "100", [100]):
        with pytest.raises(ValueError):
            family(value)
    template = family()
    bound = template.with_initial_value(17)
    assert type(bound) is family and bound.initial_value == 17
    assert template.initial_value is None
    with pytest.raises(ValueError, match="initial_value"):
        template.validate_domain(np.array([0.0]))


def test_extreme_values_avoid_spurious_intermediate_overflow():
    np.testing.assert_allclose(
        PowerLawFamily().evaluate([1e200], [0, 1e-300, 2]), [1e100], rtol=1e-12
    )
    np.testing.assert_array_equal(PowerLawFamily().evaluate([0, 1e308], [2, 0, 1e308]), [2, 2])
    with pytest.raises(FloatingPointError):
        PowerLawFamily().evaluate([1e308], [0, 1, 2])
    got = LogarithmicFamily().evaluate([0, 1e308], [2, 1, 3e-308])
    np.testing.assert_allclose(got, [2, 2 + np.log(1e308) - np.log(3e-308)])
    np.testing.assert_array_equal(
        WeibullSurvivalFamily(100).evaluate([0, 1e308], [1e-300, 1e308]), [100, 0]
    )
    for q in (-1000.0, 1000.0):
        y = BiphasicExponentialFamily(100).evaluate([0, 1e308], [1e308, 1e308, q])
        np.testing.assert_array_equal(y, [100, 0])
    for family, params in [
        (ExponentialRelaxationFamily(2), [3, 1e308]),
        (DelayedExponentialFamily(2), [3, 1e308, 0]),
    ]:
        np.testing.assert_array_equal(family.evaluate([0, 1e308], params), [2, 5])


def test_delayed_zero_lag_and_degenerate_components_are_diagnosed():
    family = DelayedExponentialFamily(2)
    with pytest.raises(ValueError):
        family.evaluate([0, 1], [3, 0.2, -1])
    assert isinstance(family.with_initial_value(3), DelayedExponentialFamily)
    t = np.linspace(0, 20, 60)
    result = fit_family(family, t, family.evaluate(t, [3, 0.2, 0]))
    assert result.success
    assert result.params[2] < 1e-5
    for f in (PowerLawFamily(), LogarithmicFamily(), DelayedExponentialFamily(2)):
        result = fit_family(f, t, np.full_like(t, 2))
        assert result.success
        assert result.jacobian_rank < f.n_parameters or result.jacobian_condition > 1e8
        assert any("Jacobian" in warning for warning in result.warnings)


def test_selector_opt_in_and_failure_isolation():
    assert len(ModelSelector().families) == 7
    t = np.linspace(0, 10, 60)
    target = PowerLawFamily().evaluate(t, [1, 2, 0.6])
    result = ModelSelector(families=[WeibullSurvivalFamily(), PowerLawFamily()]).fit(
        t, target, split="time"
    )
    assert result.best_family == "power_law"
    assert not result.candidates[0].success
    assert result.metrics["rmse"] < 1e-6


def test_biphasic_practical_degeneracy_has_an_explicit_warning():
    family = BiphasicExponentialFamily(100)
    time = np.linspace(0, 30, 70)
    warnings = family.identifiability_warnings(time, np.array([0.1, 1e-10, 0.0]))
    assert any("indistinguishable" in warning for warning in warnings)
    fitted = fit_family(family, time, family.evaluate(time, [0.1, 1e-10, 0]))
    assert fitted.success
    assert any("Biphasic" in warning or "Jacobian" in warning for warning in fitted.warnings)


@pytest.mark.parametrize("family,params,time", CASES)
def test_new_families_recover_curves_with_noise(family, params, time):
    target = family.evaluate(time, params)
    noise = np.random.default_rng(42).normal(0, np.ptp(target) * 0.001, len(time))
    fitted = fit_family(family, time, target + noise)
    assert fitted.success, fitted.message
    assert np.max(np.abs(fitted.predict(time) - target)) < np.ptp(target) * 0.01
