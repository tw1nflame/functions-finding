import numpy as np
import pandas as pd
import pytest

from parametric_regression import (
    ExponentialFamily,
    ModelSelector,
    PolynomialFamily,
    RationalSaturationFamily,
    fit_family,
    regression_metrics,
)


@pytest.mark.parametrize("split", ["random", "time"])
def test_selects_exponential_and_evaluates_holdout(split):
    x = np.linspace(0, 20, 100)
    y = 2 + 5 * np.exp(-0.3 * x)
    result = ModelSelector().fit(x, y, split=split)
    assert result.best_family == "exponential"
    assert result.criterion == "validation_rmse"
    assert result.split_protocol.method == split
    assert len(result.candidates) == 7
    indices = result.split_protocol.validation_indices
    assert result.metrics == regression_metrics(y[indices], result.predict(x[indices]))
    assert result.metrics["rmse"] < 1e-7
    assert result.formula and result.latex_formula and result.coefficients
    assert result.to_dataframe()["selected"].sum() == 1
    assert all(candidate.split_protocol is result.split_protocol for candidate in result.candidates)


def test_no_validation_leakage_in_parameters_or_initialization():
    x = np.linspace(0, 20, 80)
    y = 2 + 5 * np.exp(-0.3 * x)
    selector = ModelSelector([ExponentialFamily(), PolynomialFamily(2)])
    original = selector.fit(x, y, split="time")
    corrupted = y.copy()
    corrupted[original.split_protocol.validation_indices] += 10000
    changed = selector.fit(x, corrupted, split="time")
    train = original.split_protocol.train_indices
    for first, second in zip(original.candidates, changed.candidates, strict=True):
        direct = fit_family(first.family, x[train], y[train])
        np.testing.assert_array_equal(first.params, second.params)
        np.testing.assert_array_equal(first.params, direct.params)
        assert second.validation_metrics["rmse"] > 9000


@pytest.mark.parametrize("split", ["random", "time", "group"])
def test_reproducible_partition_and_group_isolation(split):
    x = np.tile(np.linspace(0, 20, 20), 5)
    groups = np.repeat(["a", "b", "c", "d", "e"], 20)
    selector = ModelSelector([PolynomialFamily()], random_state=14)
    kwargs = {"groups": groups} if split == "group" else {}
    one = selector.fit(x, x + 2, split=split, **kwargs)
    two = selector.fit(x, x + 2, split=split, **kwargs)
    train, validation = one.split_protocol.train_indices, one.split_protocol.validation_indices
    np.testing.assert_array_equal(train, two.split_protocol.train_indices)
    np.testing.assert_array_equal(validation, two.split_protocol.validation_indices)
    assert not set(train) & set(validation)
    assert set(train) | set(validation) == set(range(len(x)))
    if split == "group":
        assert not set(groups[train]) & set(groups[validation])
    if split == "time":
        assert x[train].max() < x[validation].min()


def test_failed_candidates_and_all_failed_are_retained():
    x = np.linspace(-10, -1, 30)
    result = ModelSelector([RationalSaturationFamily(), PolynomialFamily()]).fit(x, 2 * x)
    assert result.best_family == "polynomial_1"
    assert not result.candidates[0].success
    assert "x >= 0" in result.candidates[0].message
    failed = ModelSelector([RationalSaturationFamily()]).fit(x, x)
    assert not failed.success
    assert failed.best_family is None and failed.formula is None and failed.metrics == {}
    assert len(failed.candidates) == 1
    with pytest.raises(RuntimeError, match="No successful candidate"):
        failed.predict([0, 1])


def test_constant_target_prefers_linear():
    x = np.linspace(0, 10, 50)
    result = ModelSelector().fit(x, np.full_like(x, 3))
    assert result.best_family == "polynomial_1"
    assert result.metrics["rmse"] < 1e-12
    np.testing.assert_allclose(result.predict([0, 100]), [3, 3], atol=1e-12)


def test_simplicity_threshold_is_relative_to_global_minimum():
    x = np.linspace(-1, 1, 100)
    rng = np.random.default_rng(9)
    y = 2 * x + 0.2 * x**2 + rng.normal(0, 0.1, len(x))
    families = [PolynomialFamily(3), PolynomialFamily(2), PolynomialFamily(1)]
    strict = ModelSelector(families, relative_tolerance=0, absolute_tolerance=0).fit(x, y)
    rmses = [candidate.metrics["rmse"] for candidate in strict.candidates]
    assert strict.best_index == int(np.argmin(rmses))
    relaxed = ModelSelector(families, relative_tolerance=100).fit(x, y)
    assert relaxed.best_family == "polynomial_1"


def test_small_samples_and_candidate_specific_minimum():
    with pytest.raises(ValueError, match="two observations"):
        ModelSelector().fit([0, 1, 2], [0, 1, 2])
    x = np.arange(4, dtype=float)
    result = ModelSelector([PolynomialFamily(3), PolynomialFamily(1)], validation_size=0.5).fit(
        x, x
    )
    assert not result.candidates[0].success
    assert result.best_family == "polynomial_1"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"families": []},
        {"validation_size": 0},
        {"validation_size": 1},
        {"relative_tolerance": -1},
        {"absolute_tolerance": np.nan},
        {"max_nfev": 0},
        {"max_starts": 1.5},
        {"random_state": -1},
    ],
)
def test_invalid_selector_settings(kwargs):
    with pytest.raises(ValueError):
        ModelSelector(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"split": "unknown"},
        {"split": "group"},
        {"split": "group", "groups": [1] * 20},
        {"split": "group", "groups": [1, 2]},
        {"split": "group", "groups": [1] * 10 + [None] * 10},
        {"split": "random", "groups": [1] * 10 + [2] * 10},
    ],
)
def test_invalid_split_settings(kwargs):
    with pytest.raises(ValueError):
        ModelSelector([PolynomialFamily()]).fit(np.arange(20), np.arange(20), **kwargs)


def test_dataframe_arbitrary_column_names_and_positional_indices():
    df = pd.DataFrame(
        {
            "temperature": np.arange(40),
            "response": np.arange(40) * 3,
            "batch": np.repeat(["a", "b", "c", "d"], 10),
        },
        index=np.arange(100, 140),
    )
    selector = ModelSelector([PolynomialFamily()])
    result = selector.fit_dataframe(
        df, x_column="temperature", y_column="response", group_column="batch", split="group"
    )
    assert result.metrics["rmse"] < 1e-12
    assert max(result.split_protocol.train_indices) < len(df)
    with pytest.raises(ValueError, match="Missing DataFrame"):
        selector.fit_dataframe(df, x_column="missing", y_column="response")
    df["temperature"] = pd.array(df["temperature"], dtype="Float64")
    df.loc[100, "temperature"] = pd.NA
    with pytest.raises(ValueError, match="finite"):
        selector.fit_dataframe(df, x_column="temperature", y_column="response")


def test_validation_prediction_failure_does_not_stop_comparison():
    # The custom family's domain ends before the chronological holdout.
    class RestrictedPolynomial(PolynomialFamily):
        def validate_domain(self, values):
            if np.any(values > 5):
                raise ValueError("Outside calibrated domain")

    x = np.linspace(0, 6, 30)
    result = ModelSelector([RestrictedPolynomial(), PolynomialFamily()]).fit(x, x, split="time")
    assert not result.candidates[0].success
    assert "Validation failed" in result.candidates[0].message
    assert result.best_index == 1
