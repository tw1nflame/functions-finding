import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from parametric_regression import ModelSelector, PolynomialFamily, fit_family
from parametric_regression.visualization import plot_fit


def test_figure_split_comparison_and_save(tmp_path):
    x = np.linspace(0, 10, 40)
    y = 2 + 3 * x
    result = ModelSelector([PolynomialFamily(1), PolynomialFamily(2)]).fit(x, y)
    figure = plot_fit(result, x, y, top_n=2, xlabel="Time", ylabel="Response")
    assert isinstance(figure, Figure)
    assert len(figure.axes[0].lines) == 2
    assert len(figure.axes[0].collections) == 2
    figure.savefig(tmp_path / "comparison.png")
    assert (tmp_path / "comparison.png").stat().st_size > 1000
    plt.close(figure)
    with pytest.raises(ValueError, match="original selection rows"):
        plot_fit(result, x[:-1], y[:-1])


def test_standalone_fit_plot():
    x = np.linspace(0, 10, 40)
    result = fit_family(PolynomialFamily(), x, x)
    figure = plot_fit(result, x, x)
    assert len(figure.axes[0].collections) == 1
    plt.close(figure)
