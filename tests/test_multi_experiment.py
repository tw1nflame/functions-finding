import numpy as np
import pandas as pd
import pytest

from parametric_regression import (
    BaseParameterModel,
    ConstantParameterModel,
    ExponentialFamily,
    ExponentialRelaxationFamily,
    ExtrapolationWarning,
    GompertzFamily,
    LinearParameterModel,
    LogisticFamily,
    MultiExperimentRegressor,
    PolynomialFamily,
    PolynomialParameterModel,
    RationalSaturationFamily,
    fit_family,
)


def synthetic_data(*, two_factors=False, noise=0.0, constant_rate=False, trajectories=1):
    rng = np.random.default_rng(11)
    frames = []
    for i in range(18):
        temperature = 15 + 20 * i / 17
        ph = float(rng.uniform(6, 8)) if two_factors else 7.0
        amplitude = 0.2 + 0.015 * temperature
        rate = 0.12 if constant_rate else np.exp(-3.2 + 0.04 * temperature - 0.1 * (ph - 7))
        partition = "train" if i < 12 else ("validation" if i < 15 else "test")
        for trajectory in range(trajectories):
            time = np.linspace(0, 50, 35)
            initial = 1 + 0.03 * i + 0.05 * trajectory
            y = initial + amplitude * (-np.expm1(-rate * time))
            error = rng.normal(0, noise, len(time))
            error[0] = 0
            frames.append(
                pd.DataFrame(
                    {
                        "batch": f"E{i}",
                        "organism": f"E{i}-T{trajectory}",
                        "elapsed": time,
                        "response": y + error,
                        "temperature": temperature,
                        "pH": ph,
                        "partition": partition,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def fit_data(model, df, *, two_factors=False, split=True):
    return model.fit(
        df,
        time_column="elapsed",
        target_column="response",
        group_column="batch",
        trajectory_column="organism",
        condition_columns=["temperature", "pH"] if two_factors else ["temperature"],
        split_column="partition" if split else None,
    )


@pytest.mark.parametrize("two_factors", [False, True])
def test_manual_recovers_dependencies_and_new_conditions(two_factors):
    df = synthetic_data(two_factors=two_factors)
    model = MultiExperimentRegressor(
        ExponentialRelaxationFamily(),
        parameter_models={"A": LinearParameterModel(), "k": LinearParameterModel()},
    )
    result = fit_data(model, df, two_factors=two_factors)
    assert result.success, (result.message, result.parameter_candidates)
    assert result.metrics["validation"]["experiment"]["rmse"] < 1e-6
    assert result.metrics["test"]["experiment"]["rmse"] < 1e-6
    conditions = {"temperature": 24.0, **({"pH": 7.0} if two_factors else {})}
    time = np.linspace(0, 40, 70)
    expected = 1.25 + (0.2 + 0.015 * 24) * (-np.expm1(-np.exp(-3.2 + 0.04 * 24) * time))
    np.testing.assert_allclose(
        result.predict(time, conditions, initial_value=1.25), expected, atol=1e-6
    )
    np.testing.assert_allclose(
        model.predict(time, conditions, initial_value=1.25), expected, atol=1e-6
    )
    assert result.parameter_models["k"].transform == "log"
    assert "exp(" in result.parameter_formulas["k"]
    assert "y0" in result.formula
    assert result.latex_formula and result.parameter_latex_formulas
    training = result.experiment_parameters.query("partition == 'train'")
    np.testing.assert_allclose(
        training["param:A"], 0.2 + 0.015 * training["condition:temperature"], atol=1e-6
    )


def test_auto_prefers_constant_and_scores_final_curve():
    result = fit_data(
        MultiExperimentRegressor(ExponentialRelaxationFamily()), synthetic_data(constant_rate=True)
    )
    assert result.success, result.message
    assert result.parameter_models["k"].model.name == "constant"
    assert result.parameter_models["A"].model.name == "linear"
    selected = result.candidate_scores.query("selected").iloc[0]
    assert selected.validation_score == result.evaluations["validation"].experiment["rmse"]
    assert len(result.candidate_scores) == 9
    assert result.metrics["test"]["experiment"]["rmse"] < 1e-6


def test_relaxation_known_initial_state_and_coefficients():
    family = ExponentialRelaxationFamily(1.25)
    t = np.linspace(0, 40, 50)
    np.testing.assert_allclose(family.evaluate(t, [0.5, 0.2]), 1.25 + 0.5 * (1 - np.exp(-0.2 * t)))
    fitted = fit_family(family, t, family.evaluate(t, [0.5, 0.2]))
    assert fitted.success
    np.testing.assert_allclose(fitted.params, [0.5, 0.2], atol=1e-7)
    for x, params in (([-1], [0.5, 0.2]), ([1], [0.5, 0]), ([1], [0.5, -1])):
        with pytest.raises(ValueError):
            family.evaluate(x, params)
    with pytest.raises(ValueError, match="initial_value"):
        ExponentialRelaxationFamily().evaluate([0, 1], [0.5, 0.2])


def test_existing_time_family_without_initial_state():
    frames = []
    for i in range(10):
        t = np.linspace(-2, 3, 20)
        frames.append(
            pd.DataFrame({"id": i, "t": t, "z": i, "y": (1 + i * 0.1) + (2 + i * 0.2) * t})
        )
    df = pd.concat(frames, ignore_index=True)
    result = MultiExperimentRegressor(
        PolynomialFamily(),
        parameter_models={"a0": LinearParameterModel(), "a1": LinearParameterModel()},
    ).fit(df, time_column="t", target_column="y", group_column="id", condition_columns=["z"])
    assert result.success, result.message
    np.testing.assert_allclose(result.predict([0, 1], {"z": 4}), [1.4, 4.2], atol=1e-8)
    assert all(r.transform == "identity" for r in result.parameter_models.values())


def test_noisy_data_remains_predictive():
    result = fit_data(
        MultiExperimentRegressor(
            ExponentialRelaxationFamily(),
            parameter_models={"A": LinearParameterModel(), "k": LinearParameterModel()},
        ),
        synthetic_data(two_factors=True, noise=0.004),
        two_factors=True,
    )
    assert result.success, result.message
    assert result.metrics["test"]["experiment"]["rmse"] < 0.015


def test_test_targets_do_not_change_selection_or_any_fitted_model():
    df = synthetic_data()
    original = fit_data(MultiExperimentRegressor(ExponentialRelaxationFamily()), df)
    changed_df = df.copy()
    mask = (changed_df["partition"] == "test") & (changed_df["elapsed"] > 0)
    changed_df.loc[mask, "response"] += 100
    changed = fit_data(MultiExperimentRegressor(ExponentialRelaxationFamily()), changed_df)
    assert original.success and changed.success
    assert original.formula == changed.formula
    pd.testing.assert_frame_equal(original.candidate_scores, changed.candidate_scores)
    assert changed.metrics["test"]["observation"]["rmse"] > 99
    assert original.metrics["validation"] == changed.metrics["validation"]


def test_no_holdout_fitting_and_shared_train_coefficients(monkeypatch):
    from parametric_regression.multi_experiment import model as module

    original_fit = module.fit_trajectory
    fitted_ids = []

    def traced_fit(family, trajectory, fixed, **kwargs):
        assert trajectory.partition == "train"
        fitted_ids.append(trajectory.identifier)
        return original_fit(family, trajectory, fixed, **kwargs)

    monkeypatch.setattr(module, "fit_trajectory", traced_fit)
    df = synthetic_data(trajectories=2)
    result = fit_data(MultiExperimentRegressor(ExponentialRelaxationFamily()), df)
    altered = df.copy()
    altered.loc[(altered.partition == "validation") & (altered.elapsed > 0), "response"] += 2
    other = fit_data(MultiExperimentRegressor(ExponentialRelaxationFamily()), altered)
    assert result.success and other.success
    assert len(fitted_ids) == 2 * 12 * 2
    for identifier in result.train_fits:
        np.testing.assert_array_equal(
            result.train_fits[identifier].params, other.train_fits[identifier].params
        )
    holdout = result.experiment_parameters.query("partition != 'train'")
    assert holdout[["param:A", "param:k"]].isna().all().all()
    assert not holdout.used_for_parameter_models.any()


def test_group_splits_keep_related_trajectories_together_and_repeat():
    df = synthetic_data(trajectories=3)
    model = MultiExperimentRegressor(ExponentialRelaxationFamily(), random_state=7)
    first = fit_data(model, df, split=False)
    second = fit_data(model, df, split=False)
    np.testing.assert_array_equal(
        first.split_protocol.row_partitions, second.split_protocol.row_partitions
    )
    for positions in df.groupby("batch").indices.values():
        assert len(set(first.split_protocol.row_partitions[positions])) == 1
    assert set(first.split_protocol.experiments.partition) == {"train", "validation", "test"}
    assert (
        len(first.train_fits) == 3 * (first.split_protocol.experiments.partition == "train").sum()
    )


def test_preassigned_split_leakage_is_rejected():
    df = synthetic_data(trajectories=2)
    df.loc[df.organism == "E0-T1", "partition"] = "validation"
    with pytest.raises(ValueError, match="crosses split"):
        fit_data(MultiExperimentRegressor(ExponentialRelaxationFamily()), df)


def test_auto_uses_curve_error_not_parameter_training_error():
    frames = []
    for i, partition in enumerate(["train", "train", "validation", "test"]):
        t = np.arange(1, 5, dtype=float)
        slope = 1 if partition == "train" else 2
        frames.append(pd.DataFrame({"id": i, "t": t, "y": slope * t, "z": i, "part": partition}))
    result = MultiExperimentRegressor(
        PolynomialFamily(),
        fixed_parameters={"a0": 0},
        parameter_candidates=[ConstantParameterModel(0.99), ConstantParameterModel(1.1)],
    ).fit(
        pd.concat(frames),
        time_column="t",
        target_column="y",
        group_column="id",
        condition_columns=["z"],
        split_column="part",
    )
    assert result.success, result.message
    # 0.99 is closer to train coefficient estimates, but 1.1 predicts validation y better.
    assert result.parameter_models["a1"].model.value == 1.1


def test_multiple_trajectories_do_not_inflate_independent_sample_count():
    df = synthetic_data(trajectories=5)
    df = df[df.batch.isin(["E0", "E1", "E12", "E15"])].reset_index(drop=True)
    result = fit_data(MultiExperimentRegressor(ExponentialRelaxationFamily()), df)
    assert result.success, result.message
    skipped = result.parameter_candidates.query("model == 'quadratic'")
    assert len(skipped) == 2 and not skipped.success.any()
    assert skipped.message.str.contains("independent training experiments").all()
    assert len(result.candidate_scores) == 4


def test_duplicate_trajectories_do_not_overweight_coefficient_fit():
    frames = []
    for i, (n_trajectories, intercept, partition) in enumerate(
        [(5, 1, "train"), (1, 9, "train"), (1, 5, "validation"), (1, 5, "test")]
    ):
        for j in range(n_trajectories):
            t = np.arange(10, dtype=float)
            frames.append(
                pd.DataFrame(
                    {
                        "id": i,
                        "trajectory": f"{i}-{j}",
                        "t": t,
                        "y": intercept + 2 * t,
                        "z": i,
                        "part": partition,
                    }
                )
            )
    result = MultiExperimentRegressor(
        PolynomialFamily(),
        parameter_models={"a0": ConstantParameterModel(), "a1": ConstantParameterModel()},
    ).fit(
        pd.concat(frames),
        time_column="t",
        target_column="y",
        group_column="id",
        trajectory_column="trajectory",
        condition_columns=["z"],
        split_column="part",
    )
    assert result.success
    np.testing.assert_allclose(result.parameter_models["a0"].predict([[0]]), [5], atol=1e-8)


def test_observation_and_experiment_aggregation_have_different_weights():
    from parametric_regression.multi_experiment.validation import Trajectory, evaluate_curves

    items = []
    for i, (n, error) in enumerate([(100, 1), (2, 3)]):
        t = np.arange(n, dtype=float)
        items.append(
            Trajectory(
                i,
                i,
                "validation",
                np.arange(n),
                t,
                np.zeros(n),
                np.array([error]),
                scored=np.ones(n, dtype=bool),
            )
        )
    report = evaluate_curves(items, lambda item: np.full(len(item.time), item.conditions[0]))
    assert report.complete
    assert report.experiment["rmse"] == pytest.approx(2)
    assert report.observation["rmse"] == pytest.approx(np.sqrt(118 / 102))


def test_time_origin_initial_state_and_baseline_exclusion():
    df = synthetic_data()
    initial = df.groupby("organism", sort=False).first().response.to_dict()
    df["elapsed"] += 100
    model = MultiExperimentRegressor(ExponentialRelaxationFamily())
    result = model.fit(
        df,
        time_column="elapsed",
        target_column="response",
        group_column="batch",
        trajectory_column="organism",
        condition_columns=["temperature"],
        split_column="partition",
        initial_values=initial,
    )
    assert result.success
    assert (result.experiment_parameters.time_origin == 100).all()
    report = result.evaluations["validation"]
    assert (~report.predictions.scored).sum() == 3
    assert report.summary()["n_scored_observations"] == 3 * 34
    np.testing.assert_allclose(result.predict([0, 1], {"temperature": 24}, initial_value=3)[0], 3)


def test_unidentifiable_coefficients_are_excluded_and_known_fix_can_rescue():
    df = synthetic_data()
    flat = df.copy()
    flat["response"] = flat.groupby("organism").response.transform("first")
    rejected = fit_data(MultiExperimentRegressor(ExponentialRelaxationFamily()), flat)
    assert not rejected.success
    assert rejected.warnings
    assert not rejected.experiment_parameters.used_for_parameter_models.any()
    # Known k removes the unidentifiable coefficient; A=0 is then well-defined.
    rescued = fit_data(
        MultiExperimentRegressor(
            ExponentialRelaxationFamily(),
            fixed_parameters={"k": 0.1},
            parameter_models={"A": ConstantParameterModel()},
        ),
        flat,
    )
    assert rescued.success, rescued.message
    assert rescued.metrics["test"]["experiment"]["rmse"] < 1e-7
    np.testing.assert_allclose(rescued.predict_parameters({"temperature": 24})["k"], [0.1])


@pytest.mark.parametrize("issue", ["missing_target", "few_times", "flat_time", "missing_condition"])
def test_failed_train_trajectories_are_reported_without_silent_dropping(issue):
    df = synthetic_data()
    selected = df.organism == "E0-T0"
    if issue == "missing_target":
        df.loc[df.index[selected][1], "response"] = np.nan
    elif issue == "missing_condition":
        df.loc[df.index[selected][1], "temperature"] = np.nan
    elif issue == "flat_time":
        df.loc[selected, "elapsed"] = 0
    else:
        df = df[~selected | (df.elapsed <= df.elapsed.unique()[1])].copy()
    result = fit_data(MultiExperimentRegressor(ExponentialRelaxationFamily()), df)
    assert result.success, result.message
    row = result.experiment_parameters.query("trajectory_id == 'E0-T0'").iloc[0]
    assert not row.used_for_parameter_models
    assert row.message and result.warnings
    assert len(result.experiment_parameters) == 18


def test_bad_validation_not_scored_as_successful_subset_and_bad_test_reported():
    df = synthetic_data()
    bad_validation = df.copy()
    bad_validation.loc[
        bad_validation.index[bad_validation.partition == "validation"][1], "response"
    ] = np.nan
    result = fit_data(MultiExperimentRegressor(ExponentialRelaxationFamily()), bad_validation)
    assert not result.success
    assert not result.candidate_scores.success.any()
    bad_test = df.copy()
    bad_test.loc[bad_test.index[bad_test.partition == "test"][1], "response"] = np.nan
    result = fit_data(MultiExperimentRegressor(ExponentialRelaxationFamily()), bad_test)
    assert result.success
    assert not result.metrics["test"]["complete"]
    assert len(result.evaluations["test"].failures) == 1


def test_optimization_failure_is_retained():
    result = fit_data(
        MultiExperimentRegressor(ExponentialRelaxationFamily(), max_nfev=1), synthetic_data()
    )
    assert not result.success
    rows = result.experiment_parameters.query("partition == 'train'")
    assert rows.message.str.contains("No converged start").all()
    with pytest.raises(RuntimeError):
        result.predict([1], {"temperature": 25}, initial_value=1)


def test_manual_insufficient_experiments_and_search_limit():
    df = synthetic_data(two_factors=True)
    df = df[df.batch.isin(["E0", "E1", "E2", "E12", "E15"])].copy()
    result = fit_data(
        MultiExperimentRegressor(
            ExponentialRelaxationFamily(),
            parameter_models={"A": PolynomialParameterModel(), "k": LinearParameterModel()},
        ),
        df,
        two_factors=True,
    )
    assert not result.success
    assert "A" in result.message
    result = fit_data(
        MultiExperimentRegressor(ExponentialRelaxationFamily(), max_combinations=1),
        synthetic_data(),
    )
    assert not result.success and "max_combinations" in result.message


def test_prediction_conditions_rowwise_initial_values_and_extrapolation():
    result = fit_data(MultiExperimentRegressor(ExponentialRelaxationFamily()), synthetic_data())
    params = result.predict_parameters({"temperature": 24})
    assert params.k.iloc[0] > 0
    expected = [
        result.predict([t], {"temperature": temperature}, initial_value=y0)[0]
        for t, temperature, y0 in [(0, 20, 1), (5, 24, 2), (10, 26, 3)]
    ]
    actual = result.predict(
        [0, 5, 10], pd.DataFrame({"temperature": [20, 24, 26]}), initial_value=[1, 2, 3]
    )
    np.testing.assert_allclose(actual, expected)
    assert result.prediction_scenarios({"temperature": 15}) == ["seen_conditions"]
    assert result.prediction_scenarios({"temperature": 24}) == ["interpolation"]
    with pytest.warns(ExtrapolationWarning):
        result.predict([0, 1], {"temperature": 100}, initial_value=1)
    assert result.prediction_scenarios({"temperature": 100}) == ["extrapolation"]
    for conditions in ({"unknown": 20}, {"temperature": np.nan}, {"temperature": "20"}):
        with pytest.raises(ValueError):
            result.predict([0, 1], conditions, initial_value=1)
    with pytest.raises(ValueError, match="initial_value"):
        result.predict([0, 1], {"temperature": 24})


def test_composed_formula_agrees_with_prediction_and_optional_parameter_truth():
    result = fit_data(MultiExperimentRegressor(ExponentialRelaxationFamily()), synthetic_data())
    time = np.linspace(0, 30, 50)
    rhs = result.formula.split(" = ", 1)[1]
    from_formula = eval(
        rhs, {"__builtins__": {}}, {"exp": np.exp, "t": time, "temperature": 24, "y0": 1.2}
    )
    np.testing.assert_allclose(
        from_formula, result.predict(time, {"temperature": 24}, initial_value=1.2), atol=1e-9
    )
    conditions = pd.DataFrame({"temperature": [20, 24, 26]})
    truth = pd.DataFrame(
        {
            "A": 0.2 + 0.015 * conditions.temperature,
            "k": np.exp(-3.2 + 0.04 * conditions.temperature),
        }
    )
    scores = result.score_parameters(conditions, truth)
    assert scores["A"]["rmse"] < 1e-6 and scores["k"]["rmse"] < 1e-6


def test_custom_parameter_model_without_changes_to_time_fitting():
    from sklearn.linear_model import LinearRegression

    class CustomLinear(BaseParameterModel):
        name = "custom_linear"

        def complexity(self, n_features):
            return n_features + 1

        def fit(self, X, y, *, feature_names=None, sample_weight=None):
            self.regression = LinearRegression().fit(X, y, sample_weight=sample_weight)
            self.names = feature_names
            return self

        def predict(self, X):
            return self.regression.predict(X)

        def formula(self, *, latex=False):
            return (
                str(self.regression.intercept_)
                + " + "
                + " + ".join(
                    f"({c})*{name}"
                    for c, name in zip(self.regression.coef_, self.names, strict=True)
                )
            )

    template = CustomLinear()
    result = fit_data(
        MultiExperimentRegressor(
            ExponentialRelaxationFamily(),
            parameter_models={"A": template, "k": template},
        ),
        synthetic_data(),
    )
    assert result.success
    assert not hasattr(template, "regression")
    assert result.metrics["test"]["experiment"]["rmse"] < 1e-6


@pytest.mark.parametrize(
    "issue", ["varying_condition", "missing_id", "bad_time", "bad_label", "crossed_trajectory"]
)
def test_invalid_experiment_schema_is_rejected(issue):
    df = synthetic_data()
    if issue == "varying_condition":
        df.loc[1, "temperature"] += 1
    elif issue == "missing_id":
        df.loc[1, "batch"] = None
    elif issue == "bad_time":
        df["elapsed"] = df.elapsed.astype(str)
    elif issue == "bad_label":
        df.loc[df.batch == "E0", "partition"] = "other"
    else:
        df.loc[df.batch == "E1", "organism"] = "E0-T0"
    with pytest.raises(ValueError):
        fit_data(MultiExperimentRegressor(ExponentialRelaxationFamily()), df)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"parameter_models": {"bad": ConstantParameterModel()}},
        {"parameter_selection": "manual"},
        {"parameter_selection": "unknown"},
        {"parameter_transforms": {"k": "square"}},
        {"fixed_parameters": {"k": 0}},
        {"test_size": 0.9},
        {"aggregation": "trajectory"},
        {"max_combinations": 0},
    ],
)
def test_invalid_multi_settings(kwargs):
    with pytest.raises(ValueError):
        MultiExperimentRegressor(ExponentialRelaxationFamily(), **kwargs)


def test_arbitrary_condition_names_do_not_collide_with_time_symbol():
    df = synthetic_data().rename(columns={"temperature": "t"})
    result = MultiExperimentRegressor(ExponentialRelaxationFamily()).fit(
        df,
        time_column="elapsed",
        target_column="response",
        group_column="batch",
        condition_columns=["t"],
        split_column="partition",
    )
    assert result.success
    time = np.array([0, 10, 20])
    assert "y(t_1, t, y0)" in result.formula
    prediction = eval(
        result.formula.split(" = ", 1)[1],
        {"__builtins__": {}},
        {"exp": np.exp, "t_1": time, "t": 24, "y0": 1},
    )
    np.testing.assert_allclose(
        prediction, result.predict(time, {"t": 24}, initial_value=1), atol=1e-9
    )


def test_optional_no_test_partition_and_observation_selection():
    result = fit_data(
        MultiExperimentRegressor(
            ExponentialRelaxationFamily(), test_size=0, aggregation="observation"
        ),
        synthetic_data(),
        split=False,
    )
    assert result.success
    assert "test" not in result.evaluations
    selected = result.candidate_scores.query("selected").iloc[0]
    assert selected.validation_score == result.evaluations["validation"].observation["rmse"]


def test_all_fixed_parameters_and_bound_model_reuse():
    df = synthetic_data(constant_rate=True)
    df["response"] = 1 + 0.5 * (-np.expm1(-0.12 * df.elapsed))
    result = fit_data(
        MultiExperimentRegressor(
            ExponentialRelaxationFamily(),
            parameter_selection="manual",
            fixed_parameters={"A": 0.5, "k": 0.12},
        ),
        df,
    )
    assert result.success
    assert result.candidate_scores.n_coefficients.iloc[0] == 0
    assert result.metrics["test"]["experiment"]["rmse"] < 1e-12


def test_quadratic_interaction_coefficient_dependency_is_recovered():
    df = synthetic_data(two_factors=True)
    amplitude = 0.2 + 0.015 * df.temperature + 0.004 * df.temperature * (df.pH - 7)
    y0 = df.groupby("organism").response.transform("first")
    df["response"] = y0 + amplitude * (-np.expm1(-0.12 * df.elapsed))
    result = fit_data(MultiExperimentRegressor(ExponentialRelaxationFamily()), df, two_factors=True)
    assert result.success
    assert result.parameter_models["A"].model.name == "quadratic"
    assert result.parameter_models["k"].model.name == "constant"
    assert result.metrics["test"]["experiment"]["rmse"] < 1e-6


@pytest.mark.parametrize(
    "family,base_params",
    [
        (ExponentialFamily(), [2, 5, -0.3]),
        (LogisticFamily(), [2, 5, 0.8, 5]),
        (GompertzFamily(), [2, 5, 0.8, 5]),
        (RationalSaturationFamily(), [2, 5, 3]),
    ],
)
def test_existing_nonlinear_families_work_in_both_stages(family, base_params):
    frames = []
    for i, z in enumerate(np.linspace(-1, 1, 10)):
        t = np.linspace(0, 20, 40)
        params = np.asarray(base_params) + 0.01 * z
        for j, name in enumerate(family.parameter_names):
            if name in family.positive_parameters:
                params[j] = base_params[j] * np.exp(0.01 * z)
        part = "train" if i < 6 else ("validation" if i < 8 else "test")
        frames.append(
            pd.DataFrame(
                {"id": i, "t": t, "y": family.evaluate(t, params), "z": z, "partition": part}
            )
        )
    result = MultiExperimentRegressor(
        family, parameter_models={name: LinearParameterModel() for name in family.parameter_names}
    ).fit(
        pd.concat(frames, ignore_index=True),
        time_column="t",
        target_column="y",
        group_column="id",
        condition_columns=["z"],
        split_column="partition",
    )
    assert result.success, (result.message, result.experiment_parameters)
    assert result.metrics["test"]["experiment"]["rmse"] < 1e-5
    assert result.formula and result.latex_formula


@pytest.mark.parametrize("identifier_type", ["categorical", "tuple"])
def test_categorical_and_composite_identifiers(identifier_type):
    df = synthetic_data()
    for column in ("batch", "organism"):
        if identifier_type == "categorical":
            df[column] = pd.Categorical(df[column], categories=[*df[column].unique(), "unused"])
        else:
            df[column] = df[column].map(lambda value: ("lab", value))
    result = fit_data(MultiExperimentRegressor(ExponentialRelaxationFamily()), df)
    assert result.success, result.message
    assert len(result.experiment_parameters) == 18
    assert result.metrics["test"]["experiment"]["rmse"] < 1e-6
