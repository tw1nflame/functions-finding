import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from test_multi_experiment import fit_data, synthetic_data

from parametric_regression import ExponentialRelaxationFamily, MultiExperimentRegressor
from parametric_regression.multi_experiment.visualization import (
    plot_experiments,
    plot_new_trajectory,
    plot_parameter_dependency,
    plot_prediction_comparison,
)


def test_all_four_multi_experiment_figures(tmp_path):
    result = fit_data(
        MultiExperimentRegressor(ExponentialRelaxationFamily()),
        synthetic_data(two_factors=True),
        two_factors=True,
    )
    assert result.success
    t = np.linspace(0, 30, 50)
    truth = 1 + (0.2 + 0.015 * 24) * (-np.expm1(-np.exp(-3.2 + 0.04 * 24) * t))
    figures = [
        plot_experiments(result, max_trajectories=3),
        plot_experiments(result, partition="test", max_trajectories=2),
        plot_parameter_dependency(result, "k", "temperature", reference_conditions={"pH": 7}),
        plot_new_trajectory(result, t, {"temperature": 24, "pH": 7}, initial_value=1, truth=truth),
        plot_prediction_comparison(result, partition="test"),
    ]
    for i, figure in enumerate(figures):
        assert isinstance(figure, Figure)
        path = tmp_path / f"figure-{i}.png"
        figure.savefig(path)
        assert path.stat().st_size > 1000
        plt.close(figure)
