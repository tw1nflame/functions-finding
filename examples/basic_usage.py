"""Reproducible synthetic experiment; writes a figure and candidate table."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from parametric_regression import ModelSelector
from parametric_regression.visualization import plot_fit


def main() -> None:
    """Fit synthetic decay and save artifacts without opening GUI windows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo.png"))
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    x = np.linspace(0, 20, 120)
    y = 2 + 5 * np.exp(-0.3 * x) + rng.normal(0, 0.03, x.size)
    result = ModelSelector(random_state=42).fit(x, y, split="time")
    if not result.success:
        raise RuntimeError(result.to_dataframe().to_string(index=False))
    print("Selected family:", result.best_family)
    print(result.formula)
    print("Coefficients:", result.coefficients)
    print("Validation metrics:", result.metrics)
    print("Predictions at 21, 22, 23:", result.predict([21, 22, 23]))
    print(result.to_dataframe().drop(columns=["warnings", "message"]).to_string(index=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_fit(result, x, y, top_n=3, xlabel="Time", ylabel="Response")
    figure.savefig(args.output, dpi=150)
    plt.close(figure)
    result.to_dataframe().to_csv(args.output.with_suffix(".csv"), index=False)
    print("Figure:", args.output.resolve())


if __name__ == "__main__":
    main()
