"""Acquisition, workbook audit and experiment orchestration; models use the library API.

The small OOXML reader reads stored values, never evaluates Excel formulas. It avoids
an additional Excel-engine dependency and deliberately validates this workbook's schema.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import urllib.request
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from parametric_regression import (
    ConstantParameterModel,
    LinearParameterModel,
    ModelSelector,
    MultiExperimentRegressor,
    regression_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "real_bacterial_growth"
FILENAME = "Figure_2_multi_species_three_concentrations_linked (2).xlsx"
URL = (
    "https://zenodo.org/records/22798201/files/"
    "Figure_2_multi_species_three_concentrations_linked%20%282%29.xlsx?download=1"
)
SHA256 = "26450f3e8e0b834e384bbec308c537da65bc7afb61930f641036acc6932338a6"
MD5 = "d1f9ac3ceadb1278e0e513c2ea32121a"
SEED = 42
FIT_OPTIONS = {"max_nfev": 2000, "max_starts": 12}
CONCENTRATIONS = {
    "No Treatment": 0.0,
    "C8-HSL 0.05 mM": 0.05,
    "C8-HSL 0.2 mM": 0.2,
    "C8-HSL 1.0 mM": 1.0,
}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf8")


def acquire():
    """Use the pinned local original, or download it; never substitute observations."""
    path = ROOT / "data" / "external" / FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        try:
            with urllib.request.urlopen(URL, timeout=90) as response:
                content = response.read()
        except OSError as error:
            raise RuntimeError(f"Download failed. Save {URL} manually as {path}") from error
        if hashlib.sha256(content).hexdigest() != SHA256:
            raise ValueError("Downloaded file differs from audited v5 workbook; re-audit required.")
        path.write_bytes(content)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError("Local workbook checksum mismatch; original data must not be edited.")
    assert hashlib.md5(content).hexdigest() == MD5
    save_json(
        path.parent / "bacterial_growth_provenance.json",
        {
            "url": URL,
            "doi": "10.5281/zenodo.22798201",
            "version": "v5",
            "published": "2026-09-16",
            "retrieved": "2026-11-08",
            "sha256": SHA256,
            "zenodo_md5": MD5,
            "bytes": len(content),
            "license": "CC-BY-4.0",
            "authors": ["Anjali Ghosh", "Schnieder Fernandes"],
        },
    )
    return path


def read_workbook(path):
    """Return sheet -> rectangular rows, including cached formula results."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    sheets = {}
    with ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = [
                "".join(node.itertext())
                for node in ET.fromstring(archive.read("xl/sharedStrings.xml"))
            ]
        links = {
            node.attrib["Id"]: node.attrib["Target"]
            for node in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        }
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        for sheet in workbook.findall("s:sheets/s:sheet", ns):
            target = links[sheet.attrib[f"{{{rel_ns}}}id"]]
            target = target.lstrip("/") if target.startswith("/") else "xl/" + target
            xml = ET.fromstring(archive.read(target))
            rows = []
            for row in xml.findall("s:sheetData/s:row", ns):
                values = []
                for cell in row.findall("s:c", ns):
                    letters = re.match(r"[A-Z]+", cell.attrib["r"])[0]
                    column = 0
                    for letter in letters:
                        column = column * 26 + ord(letter) - ord("A") + 1
                    values.extend([None] * (column - len(values)))
                    kind = cell.attrib.get("t")
                    node = cell.find("s:v", ns)
                    value = None if node is None else node.text
                    if kind == "inlineStr":
                        value = "".join(n.text or "" for n in cell.findall(".//s:t", ns))
                    elif kind == "s" and value is not None:
                        value = shared[int(value)]
                    elif kind == "b" and value is not None:
                        value = value == "1"
                    elif kind not in ("str", "e") and value is not None:
                        value = float(value)
                    values[column - 1] = value
                rows.append(values)
            width = max(map(len, rows))
            sheets[sheet.attrib["name"]] = [r + [None] * (width - len(r)) for r in rows]
    return sheets


def prepare_data():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sheets = read_workbook(acquire())
    audit = pd.DataFrame(
        [
            {
                "sheet": name,
                "rows_including_header": len(rows),
                "columns": len(rows[0]),
                "first_row": str(rows[0]),
            }
            for name, rows in sheets.items()
        ]
    )
    audit.to_csv(OUTPUT / "workbook_sheets.csv", index=False)
    save_json(OUTPUT / "analysis_guide.json", sheets["Analysis Guide"])
    rows = sheets["Raw Data"]
    assert rows[0] == ["Strain", "Condition", "Time (hrs)", "Replicate", "OD570"]
    data = pd.DataFrame(rows[1:], columns=rows[0]).rename(
        columns={
            "Strain": "strain",
            "Condition": "source_condition",
            "Time (hrs)": "time_h",
            "Replicate": "replicate_id",
            "OD570": "od570",
        }
    )
    assert set(data.source_condition) == set(CONCENTRATIONS)
    data["concentration_mM"] = data.source_condition.map(CONCENTRATIONS)
    data["curve_id"] = data.strain + "|" + data.source_condition + "|" + data.replicate_id
    # Source only identifies replicate labels, not dates or cross-condition experiment IDs.
    data["replicate_block"] = data.strain + "|" + data.replicate_id
    data["experiment_id"] = pd.NA
    data["is_control"] = data.source_condition.eq("No Treatment")
    data["source_sheet"] = "Raw Data"
    data["source_excel_row"] = np.arange(2, len(data) + 2)
    data["background_correction"] = "not specified in this workbook"
    assert not data.drop(columns="experiment_id").isna().any().any()
    assert np.isfinite(data[["time_h", "od570", "concentration_mM"]]).all().all()
    assert not data.duplicated(["curve_id", "time_h"]).any()
    quality = data.groupby("strain").agg(
        curves=("curve_id", "nunique"),
        rows=("od570", "size"),
        min_time=("time_h", "min"),
        max_time=("time_h", "max"),
        minimum_od=("od570", "min"),
        negative_od=("od570", lambda x: (x < 0).sum()),
    )
    complete = data.groupby("curve_id").time_h.apply(lambda x: set(x) == set(range(15)))
    assert complete.all()
    # Prespecified preference: explicitly named Salmonella strain, complete design.
    assert quality.loc["ST14028", "curves"] == 12
    selected = data.loc[data.strain.eq("ST14028")].copy().reset_index(drop=True)
    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    data.to_csv(processed / "bacterial_growth_all.csv", index=False)
    selected.to_csv(processed / "bacterial_growth.csv", index=False)
    quality.to_csv(OUTPUT / "strain_quality.csv")
    save_json(
        OUTPUT / "environment.json",
        {
            "python": platform.python_version(),
            "random_state": SEED,
            "fit_options": FIT_OPTIONS,
            "packages": {
                p: version(p)
                for p in [
                    "numpy",
                    "scipy",
                    "pandas",
                    "scikit-learn",
                    "matplotlib",
                    "parametric-regression",
                ]
            },
        },
    )
    return selected, audit, quality


def save_figure(fig, name):
    fig.savefig(OUTPUT / f"{name}.png", dpi=170, bbox_inches="tight")
    return fig


def plot_eda(data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for (condition, frame), ax in zip(data.groupby("source_condition"), axes.flat, strict=True):
        for replicate, curve in frame.groupby("replicate_id"):
            ax.plot(curve.time_h, curve.od570, ".-", alpha=0.65, label=replicate)
        stats = frame.groupby("time_h").od570.agg(["mean", "std"])
        ax.plot(stats.index, stats["mean"], "k-", lw=2, label="Mean")
        ax.fill_between(
            stats.index,
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            color="grey",
            alpha=0.2,
            label="SD (n=3)",
        )
        ax.set(title=condition, xlabel="Time (h)", ylabel="Deposited OD570")
        ax.legend(fontsize=8)
    fig.suptitle("ST14028: individual biological replicate labels and descriptive mean ± SD")
    fig.tight_layout()
    save_figure(fig, "eda_curves")
    endpoint = data.loc[data.time_h.eq(data.time_h.max())]
    fig2, ax = plt.subplots(figsize=(8, 4))
    for index, (condition, frame) in enumerate(endpoint.groupby("source_condition")):
        ax.scatter(np.full(len(frame), index), frame.od570, label=condition)
    ax.set_xticks(range(4), sorted(endpoint.source_condition.unique()), rotation=15)
    ax.set(ylabel="OD570 at 14 h", title="All three endpoints per condition (no significance test)")
    fig2.tight_layout()
    save_figure(fig2, "eda_endpoints")
    return endpoint.groupby("source_condition").od570.agg(["mean", "std", "min", "max"])


def individual_fits(data):
    results, records, predictions = {}, [], []
    selector = ModelSelector(validation_size=0.2, random_state=SEED, **FIT_OPTIONS)
    for identifier, curve in data.groupby("curve_id", sort=True):
        curve = curve.sort_values("time_h")
        result = selector.fit(curve.time_h, curve.od570, split="time")
        results[identifier] = result
        table = result.to_dataframe()
        table["curve_id"] = identifier
        table["concentration_mM"] = curve.concentration_mM.iloc[0]
        table["replicate_id"] = curve.replicate_id.iloc[0]
        for index, fit in enumerate(result.candidates):
            table.loc[index, "coefficients"] = json.dumps(fit.coefficients)
            table.loc[index, "formula"] = fit.formula
            table.loc[index, "jacobian_condition"] = fit.jacobian_condition
            if fit.success:
                pred = curve[["curve_id", "time_h", "od570"]].copy()
                pred["family"] = fit.family_name
                pred["predicted"] = fit.predict(curve.time_h)
                pred["partition"] = np.where(curve.time_h <= 11, "train", "validation")
                predictions.append(pred)
        records.append(table)
    table = pd.concat(records, ignore_index=True)
    table.to_csv(OUTPUT / "individual_candidates.csv", index=False)
    table.loc[table.selected].to_csv(OUTPUT / "individual_winners.csv", index=False)
    pd.concat(predictions).to_csv(OUTPUT / "individual_predictions.csv", index=False)
    return results, table


def choose_family(results, training_ids):
    """Prespecified aggregate chronological validation rule, restricted to fit-train IDs."""
    rows = []
    for index, family in enumerate(ModelSelector().families):
        fits = [results[key].candidates[index] for key in sorted(training_ids)]
        complete = all(f.success for f in fits)
        stable = complete and all(
            f.jacobian_rank == f.n_parameters and f.jacobian_condition < 1e8 for f in fits
        )
        rows.append(
            {
                "family": family.name,
                "index": index,
                "n_parameters": family.n_parameters,
                "complete": complete,
                "stable": stable,
                "mean_validation_rmse": np.mean([f.validation_metrics["rmse"] for f in fits])
                if stable
                else np.inf,
            }
        )
    table = pd.DataFrame(rows)
    best = table.mean_validation_rmse.min()
    if not np.isfinite(best):
        raise RuntimeError("No complete stable common family on training curves.")
    eligible = table.loc[table.mean_validation_rmse <= best * 1.01 + 1e-12]
    winner = eligible.sort_values(["n_parameters", "mean_validation_rmse", "index"]).iloc[0]
    return ModelSelector().families[int(winner["index"])], table


def make_split(data, heldout_replicate=None):
    """Keep all trajectories of a replicate label together in the inner split."""
    frame = data.loc[~data.is_control].copy()
    if heldout_replicate is None:
        frame["partition"] = np.where(
            frame.concentration_mM.eq(0.2),
            "test",
            np.where(frame.replicate_id.eq("Rep3"), "validation", "train"),
        )
        # New-condition task deliberately shares replicate labels across outer partitions.
        # These are analysis groups, NOT asserted independent biological experiments.
        frame["analysis_group"] = frame.partition + "|" + frame.replicate_block
    else:
        remaining = sorted(set(frame.replicate_id) - {heldout_replicate})
        frame["partition"] = np.where(
            frame.replicate_id.eq(heldout_replicate),
            "test",
            np.where(frame.replicate_id.eq(remaining[-1]), "validation", "train"),
        )
        frame["analysis_group"] = frame.replicate_block
    return frame.reset_index(drop=True)


def run_comparison(frame, results, name):
    train = frame.loc[frame.partition.eq("train")]
    validation = frame.loc[frame.partition.eq("validation")]
    test = frame.loc[frame.partition.eq("test")]
    assert set(train.curve_id).isdisjoint(test.curve_id)
    assert set(validation.curve_id).isdisjoint(test.curve_id)
    if name == "new_concentration":
        assert not frame.loc[frame.partition.ne("test"), "concentration_mM"].eq(0.2).any()
    family, family_scores = choose_family(results, set(train.curve_id))
    family_scores.to_csv(OUTPUT / f"{name}_family_selection.csv", index=False)
    frame[
        [
            "curve_id",
            "replicate_id",
            "replicate_block",
            "concentration_mM",
            "time_h",
            "partition",
            "analysis_group",
        ]
    ].to_csv(OUTPUT / f"{name}_split.csv", index=False)
    model = MultiExperimentRegressor(
        family,
        parameter_candidates=[ConstantParameterModel(), LinearParameterModel()],
        random_state=SEED,
        max_combinations=81,
        **FIT_OPTIONS,
    ).fit(
        frame,
        time_column="time_h",
        target_column="od570",
        group_column="analysis_group",
        trajectory_column="curve_id",
        condition_columns=["concentration_mM"],
        split_column="partition",
    )
    model.experiment_parameters.to_csv(OUTPUT / f"{name}_parameters.csv", index=False)
    model.candidate_scores.to_csv(OUTPUT / f"{name}_parameter_selection.csv", index=False)
    save_json(
        OUTPUT / f"{name}_model.json",
        {
            "success": model.success,
            "message": model.message,
            "warnings": model.warnings,
            "family": family.name,
            "formula": model.formula,
            "parameter_formulas": model.parameter_formulas,
            "metrics": model.metrics,
            "family_selection_curve_ids": sorted(set(train.curve_id)),
        },
    )
    # Same fit-train and inner-validation observations for direct regression.
    # C has only two fit levels in the main task. Remove C^2/C^3 terms whose
    # effects cannot be identified; retain all total-degree<=d terms with C exponent<=1.
    features = ["time_h", "concentration_mM"]
    direct_candidates, direct_scores = {}, []
    for degree in (1, 2, 3):
        pipeline = make_pipeline(StandardScaler(), PolynomialFeatures(degree, include_bias=False))
        design = pipeline.fit_transform(train[features])
        powers = pipeline[-1].powers_
        mask = powers[:, 1] <= 1
        regressor = LinearRegression().fit(design[:, mask], train.od570)
        prediction = regressor.predict(pipeline.transform(validation[features])[:, mask])
        direct_candidates[degree] = (pipeline, mask, regressor)
        direct_scores.append(
            {
                "degree": degree,
                "terms": int(mask.sum()),
                **regression_metrics(validation.od570, prediction),
            }
        )
    scores = pd.DataFrame(direct_scores)
    minimum = scores.rmse.min()
    degree = int(scores.loc[scores.rmse <= minimum * 1.01 + 1e-12].degree.min())
    pipeline, mask, baseline = direct_candidates[degree]
    scores["selected"] = scores.degree.eq(degree)
    scores.to_csv(OUTPUT / f"{name}_direct_selection.csv", index=False)
    predicted = {"Direct Polynomial": baseline.predict(pipeline.transform(test[features])[:, mask])}
    if model.success:
        predicted["MultiExperimentRegressor"] = model.predict(
            test.time_h, test[["concentration_mM"]]
        )
    # Nearest concentration, mean of fit-train replicate observations at each time.
    means = train.groupby(["concentration_mM", "time_h"]).od570.mean()
    levels = np.sort(train.concentration_mM.unique())
    predicted["Nearest concentration"] = np.array(
        [
            means.loc[levels[np.argmin(np.abs(levels - c))], t]
            for c, t in zip(test.concentration_mM, test.time_h, strict=True)
        ]
    )
    metrics, by_rep, predictions = [], [], []
    for approach, prediction in predicted.items():
        metrics.append({"approach": approach, **regression_metrics(test.od570, prediction)})
        out = test.copy()
        out["approach"], out["predicted"] = approach, prediction
        predictions.append(out)
        for replicate, subset in out.groupby("replicate_id"):
            by_rep.append(
                {
                    "approach": approach,
                    "replicate_id": replicate,
                    **regression_metrics(subset.od570, subset.predicted),
                }
            )
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUTPUT / f"{name}_metrics.csv", index=False)
    pd.DataFrame(by_rep).to_csv(OUTPUT / f"{name}_metrics_by_replicate.csv", index=False)
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(OUTPUT / f"{name}_predictions.csv", index=False)
    return model, metrics, predictions, family_scores


def plot_individual(data, results, family_name):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    curves = data.loc[data.replicate_id.eq("Rep1") & ~data.is_control]
    for column, (_, curve) in enumerate(curves.groupby("concentration_mM")):
        result = results[curve.curve_id.iloc[0]]
        fixed = next(f for f in result.candidates if f.family_name == family_name)
        ax, residual = axes[:, column]
        ax.scatter(curve.time_h, curve.od570, s=22, color="black", label="Observed")
        for label, fit in [("Selected", result.best_result), ("Common family", fixed)]:
            if fit.success:
                prediction = fit.predict(curve.time_h)
                ax.plot(curve.time_h, prediction, label=f"{label}: {fit.family_name}")
                residual.plot(curve.time_h, curve.od570 - prediction, ".-", label=label)
        ax.set(title=f"C={curve.concentration_mM.iloc[0]:g} mM; Rep1", ylabel="OD570")
        residual.set(xlabel="Time (h)", ylabel="Observed − predicted")
        for axis in (ax, residual):
            axis.axvspan(11.5, 14, alpha=0.12, color="orange", label="Validation 12–14 h")
            axis.legend(fontsize=7)
        residual.axhline(0, color="grey", lw=1)
    fig.tight_layout()
    return save_figure(fig, "individual_fits_residuals")


def plot_holdout(frame, predictions):
    fig, ax = plt.subplots(figsize=(10, 5))
    for (c, replicate), curve in frame.loc[frame.partition.ne("test")].groupby(
        ["concentration_mM", "replicate_id"]
    ):
        ax.plot(
            curve.time_h,
            curve.od570,
            color="grey",
            alpha=0.3,
            label=f"Train/validation {c:g} mM" if replicate == "Rep1" else None,
        )
    test = frame.loc[frame.partition.eq("test")]
    for replicate, curve in test.groupby("replicate_id"):
        ax.scatter(curve.time_h, curve.od570, s=25, label=f"Test 0.2 mM {replicate}")
    for approach, curve in predictions.loc[predictions.replicate_id.eq("Rep1")].groupby("approach"):
        ax.plot(curve.time_h, curve.predicted, lw=2, label=approach)
    ax.set(xlabel="Time (h)", ylabel="OD570", title="Completely held-out concentration: 0.2 mM")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, "new_concentration_predictions")
