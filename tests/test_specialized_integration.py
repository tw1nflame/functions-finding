"""Holdout trajectories for new families and explicit scientific target scales."""

import numpy as np
import pandas as pd
import pytest

from parametric_regression import (
    ArrheniusParameterModel,
    BiphasicExponentialFamily,
    ConstantParameterModel,
    DelayedExponentialFamily,
    ExponentialRelaxationFamily,
    HillParameterModel,
    LinearParameterModel,
    LogarithmicFamily,
    ModelSelector,
    MonodParameterModel,
    MultiExperimentRegressor,
    PowerLawFamily,
    RatkowskyParameterModel,
    WeibullSurvivalFamily,
)


def fit_table(
    family, model, condition_values, parameter_function, *, fixed=None, auto=False, transforms=None
):
    frames = []
    bound = (
        family.with_initial_value(
            100 if isinstance(family, (WeibullSurvivalFamily, BiphasicExponentialFamily)) else 2
        )
        if family.requires_initial_value
        else family
    )
    for i, value in enumerate(condition_values):
        partition = (
            "train"
            if i < len(condition_values) - 4
            else ("validation" if i < len(condition_values) - 2 else "test")
        )
        t = np.linspace(0, 40, 65)
        frames.append(
            pd.DataFrame(
                {
                    "experiment": i,
                    "time": t + 100 if family.requires_initial_value else t,
                    "z": value,
                    "y": bound.evaluate(t, parameter_function(value)),
                    "partition": partition,
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    kwargs = {"parameter_candidates": model} if auto else {"parameter_models": model}
    result = MultiExperimentRegressor(
        family, fixed_parameters=fixed, parameter_transforms=transforms, **kwargs
    ).fit(
        frame,
        time_column="time",
        target_column="y",
        group_column="experiment",
        condition_columns=["z"],
        split_column="partition",
    )
    return result, frame


@pytest.mark.parametrize("kind", ["power", "logarithmic", "delayed", "weibull", "biphasic"])
def test_new_temporal_families_in_two_stage_regression(kind):
    if kind == "power":
        family = PowerLawFamily()
        models = {"c": LinearParameterModel(), "a": LinearParameterModel()}
        fixed = {"b": 1.3}

        def parameters(z):
            return [1 + 0.02 * z, 0.5 + 0.03 * z, 1.3]
    elif kind == "logarithmic":
        family = LogarithmicFamily()
        models = {"c": LinearParameterModel(), "a": LinearParameterModel()}
        fixed = {"tau": 2.0}

        def parameters(z):
            return [1 + 0.02 * z, 0.5 + 0.03 * z, 2.0]
    elif kind == "delayed":
        family = DelayedExponentialFamily()
        models = {"A": LinearParameterModel()}
        fixed = {"k": 0.3, "lag": 2.0}

        def parameters(z):
            return [0.5 + 0.03 * z, 0.3, 2.0]
    elif kind == "weibull":
        family = WeibullSurvivalFamily()
        models = {"delta": LinearParameterModel()}
        fixed = {"p": 1.4}

        def parameters(z):
            return [np.exp(2 + 0.03 * z), 1.4]
    else:
        family = BiphasicExponentialFamily()
        models = {"q": LinearParameterModel()}
        fixed = {"k_slow": 0.06, "delta_k": 0.4}

        def parameters(z):
            return [0.06, 0.4, -0.5 + 0.1 * z]

    result, _ = fit_table(family, models, np.arange(10.0), parameters, fixed=fixed)
    assert result.success, (result.message, result.parameter_candidates)
    assert result.metrics["test"]["complete"]
    assert result.metrics["test"]["observation"]["rmse"] < 1e-5
    assert result.formula and result.latex_formula
    assert not result.experiment_parameters.loc[
        result.experiment_parameters.partition.ne("train"), "fit_success"
    ].any()
    if family.requires_initial_value:
        assert result.experiment_parameters.time_origin.eq(100).all()
        assert result.metrics["test"]["n_scored_observations"] == 128


@pytest.mark.parametrize("kind", ["arrhenius", "ratkowsky", "monod"])
@pytest.mark.parametrize("auto", [False, True])
def test_positive_rate_models_undo_log_exactly_once(kind, auto):
    if kind == "arrhenius":
        model = ArrheniusParameterModel()
        conditions = np.linspace(285, 320, 14)

        def rate(z):
            return 0.2 * np.exp(-40000 / model.R * (1 / z - 1 / 298.15))
    elif kind == "ratkowsky":
        model = RatkowskyParameterModel(temperature_max=40)
        conditions = np.linspace(5, 35, 14)

        def rate(z):
            return 0.03**2 * (z + 4) ** 2
    else:
        model = MonodParameterModel()
        conditions = np.array([0.05, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10, 30, 0.2, 4, 0.7, 7])

        def rate(z):
            return 0.5 * z / (2 + z)

    models = [ConstantParameterModel(), LinearParameterModel(), model] if auto else {"k": model}
    result, _ = fit_table(
        ExponentialRelaxationFamily(),
        models,
        conditions,
        lambda z: [0.7, rate(z)],
        fixed={"A": 0.7},
        auto=auto,
    )
    assert result.success, (result.message, result.parameter_candidates)
    assert result.metrics["test"]["observation"]["rmse"] < 1e-5
    relationship = result.parameter_models["k"]
    assert relationship.model.name == kind
    assert relationship.transform == ("identity" if kind == "monod" else "log")
    predicted = result.predict_parameters({"z": float(conditions[2])})["k"].iloc[0]
    np.testing.assert_allclose(predicted, rate(conditions[2]), rtol=1e-5)
    if kind == "monod":
        with pytest.raises(ValueError):
            result.predict_parameters({"z": 0.0}, warn_extrapolation=False)


@pytest.mark.parametrize("auto", [False, True])
def test_relaxation_hill_and_candidate_selection(auto):
    hill = HillParameterModel()
    models = [ConstantParameterModel(), LinearParameterModel(), hill] if auto else {"A": hill}
    conditions = np.array([0, 0.03, 0.1, 0.3, 1, 3, 10, 30, 0.2, 2, 0.6, 5])

    def amplitude(z):
        return 0.2 + 1.2 * z**1.5 / (2**1.5 + z**1.5)

    result, _ = fit_table(
        ExponentialRelaxationFamily(),
        models,
        conditions,
        lambda z: [amplitude(z), 0.2],
        fixed={"k": 0.2},
        auto=auto,
    )
    assert result.success, (result.message, result.parameter_candidates)
    assert result.parameter_models["A"].model.name == "hill"
    assert result.parameter_models["A"].transform == "identity"
    assert result.metrics["test"]["observation"]["rmse"] < 1e-5


@pytest.mark.parametrize(
    "model,transform", [(ArrheniusParameterModel(), "identity"), (MonodParameterModel(), "log")]
)
def test_explicit_incompatible_transform_is_a_diagnostic(model, transform):
    conditions = np.linspace(290, 320, 12)
    result, _ = fit_table(
        ExponentialRelaxationFamily(),
        {"k": model},
        conditions,
        lambda z: [0.7, 0.1 + 0.001 * (z - 290)],
        fixed={"A": 0.7},
        transforms={"k": transform},
    )
    assert not result.success
    assert result.parameter_candidates.message.str.contains("requires").any()
    assert result.parameter_candidates.message.str.contains("explicitly").any()


def test_simpler_single_rate_candidate_is_preferred():
    t = np.linspace(0, 30, 70)
    target = 100 * np.exp(-0.2 * t)
    result = ModelSelector(
        families=[ExponentialRelaxationFamily(100), BiphasicExponentialFamily(100)]
    ).fit(t, target, split="time")
    assert result.best_family == "exponential_relaxation"


def test_biphasic_weak_fitted_parameters_are_excluded_from_stage_two():
    result, _ = fit_table(
        BiphasicExponentialFamily(),
        {name: ConstantParameterModel() for name in ("k_slow", "delta_k", "q")},
        np.arange(10.0),
        lambda z: [0.1, 1e-10, 0.0],
    )
    assert not result.success
    train = result.experiment_parameters.query("partition == 'train'")
    assert train.fit_success.all()
    assert not train.used_for_parameter_models.any()
    assert train.message.str.contains("Excluded").all()


def test_known_fixed_biphasic_parameters_do_not_require_identification():
    result, _ = fit_table(
        BiphasicExponentialFamily(),
        {},
        np.arange(10.0),
        lambda z: [0.1, 1e-10, 0.0],
        fixed={"k_slow": 0.1, "delta_k": 1e-10, "q": 0.0},
    )
    assert result.success, result.message


@pytest.mark.parametrize("transform", ["log", "identity"])
def test_hill_for_rate_has_the_documented_physical_scale(transform):
    conditions = np.array([0, 0.03, 0.1, 0.3, 1, 3, 10, 30, 0.2, 2, 0.6, 5])

    def rate(z):
        value = -2 + 0.7 * z**1.5 / (2**1.5 + z**1.5)
        return np.exp(value) if transform == "log" else value + 2.1

    result, _ = fit_table(
        ExponentialRelaxationFamily(),
        {"k": HillParameterModel()},
        conditions,
        lambda z: [0.7, rate(z)],
        fixed={"A": 0.7},
        transforms={"k": transform},
    )
    assert result.success, (result.message, result.parameter_candidates)
    assert result.parameter_models["k"].transform == transform
    assert result.metrics["test"]["observation"]["rmse"] < 1e-5
