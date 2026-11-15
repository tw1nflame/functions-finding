"""Headless synthetic two-stage regression demo with four saved analysis figures."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parametric_regression import ExponentialRelaxationFamily, MultiExperimentRegressor
from parametric_regression.multi_experiment.visualization import (
    plot_experiments,
    plot_new_trajectory,
    plot_parameter_dependency,
    plot_prediction_comparison,
)


def make_synthetic_data(random_state: int = 42) -> pd.DataFrame:
    """Generate 30 independent experiments with known A(T) and log k(T,pH)."""
    rng = np.random.default_rng(random_state)
    frames = []
    for experiment in range(30):
        temperature = float(rng.uniform(18, 34))
        ph = float(rng.uniform(6, 8))
        time = np.linspace(0, 50, 50)
        initial = 0.9 + 0.01 * experiment
        amplitude = 0.2 + 0.015 * temperature
        rate = np.exp(-3.2 + 0.04 * temperature - 0.15 * (ph - 7))
        response = initial + amplitude * (-np.expm1(-rate * time))
        noise = rng.normal(0, 0.002, len(time))
        noise[0] = 0  # Initial state is observed accurately in this demonstration.
        frames.append(
            pd.DataFrame(
                {
                    "experiment_id": f"E{experiment:02d}",
                    "time": time,
                    "temperature": temperature,
                    "pH": ph,
                    "response": response + noise,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    """Fit Auto, print the analytical dependency and write figures/tables/metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/multi_experiment"))
    args = parser.parse_args()
    df = make_synthetic_data()
    result = MultiExperimentRegressor(
        ExponentialRelaxationFamily(),
        parameter_selection="auto",
        random_state=42,
    ).fit(
        df,
        time_column="time",
        target_column="response",
        group_column="experiment_id",
        condition_columns=["temperature", "pH"],
    )
    if not result.success:
        raise RuntimeError(
            result.message + "\n" + result.parameter_candidates.to_string(index=False)
        )
    print(result.message)
    print("Parameter models:", {name: r.model.name for name, r in result.parameter_models.items()})
    for name, formula in result.parameter_formulas.items():
        print(f"{name} = {formula}")
    print(result.formula)
    print("Validation:", result.metrics["validation"]["experiment"])
    print("Test:", result.metrics["test"]["experiment"])
    for message in result.warnings:
        print("Diagnostic:", message)
    time = np.linspace(0, 50, 100)
    conditions = {"temperature": 25.0, "pH": 7.0}
    truth = 1.1 + (0.2 + 0.015 * 25) * (-np.expm1(-np.exp(-3.2 + 0.04 * 25) * time))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "trajectories": plot_experiments(result, max_trajectories=4),
        "parameter_k": plot_parameter_dependency(
            result, "k", "temperature", reference_conditions={"pH": 7}
        ),
        "new_trajectory": plot_new_trajectory(
            result, time, conditions, initial_value=1.1, truth=truth
        ),
        "heldout_predictions": plot_prediction_comparison(result, partition="test"),
    }
    for name, figure in figures.items():
        figure.savefig(args.output_dir / f"{name}.png", dpi=150)
        plt.close(figure)
    result.experiment_parameters.to_csv(args.output_dir / "experiment_parameters.csv", index=False)
    result.candidate_scores.to_csv(args.output_dir / "candidate_scores.csv", index=False)
    result.split_protocol.experiments.to_csv(args.output_dir / "split.csv", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2), encoding="utf-8"
    )
    print("Artifacts:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
