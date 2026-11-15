"""Guard real-data extraction and the concentration/replicate holdout boundaries."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bacterial_growth_support", ROOT / "examples" / "bacterial_growth_support.py"
)
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)


@pytest.fixture
def source_data():
    sheets = support.read_workbook(ROOT / "data" / "external" / support.FILENAME)
    raw = sheets["Raw Data"]
    frame = pd.DataFrame(raw[1:], columns=raw[0])
    frame = frame.loc[frame.Strain.eq("ST14028")].rename(
        columns={
            "Condition": "source_condition",
            "Replicate": "replicate_id",
            "Time (hrs)": "time_h",
        }
    )
    frame["concentration_mM"] = frame.source_condition.map(support.CONCENTRATIONS)
    frame["is_control"] = frame.source_condition.eq("No Treatment")
    frame["curve_id"] = frame.source_condition + "|" + frame.replicate_id
    frame["replicate_block"] = "ST14028|" + frame.replicate_id
    return frame


def test_workbook_pinned_values_and_schema(source_data):
    path = ROOT / "data" / "external" / support.FILENAME
    assert support.hashlib.sha256(path.read_bytes()).hexdigest() == support.SHA256
    sheets = support.read_workbook(path)
    assert len(sheets) == 8
    assert sheets["Raw Data"][0] == ["Strain", "Condition", "Time (hrs)", "Replicate", "OD570"]
    assert sheets["Raw Data"][1] == ["ECOLI", "No Treatment", 0, "Rep1", 0]
    assert len(sheets["Raw Data"]) == 1261
    assert len(source_data) == 180
    assert (
        source_data.groupby("curve_id").time_h.apply(lambda t: sorted(t) == list(range(15))).all()
    )


def test_concentration_holdout_excludes_every_test_curve(source_data):
    frame = support.make_split(source_data)
    assert not frame.is_control.any()
    assert frame.groupby("partition").size().to_dict() == {
        "test": 45,
        "train": 60,
        "validation": 30,
    }
    assert frame.loc[frame.partition.eq("test"), "concentration_mM"].eq(0.2).all()
    assert not frame.loc[frame.partition.ne("test"), "concentration_mM"].eq(0.2).any()
    assert set(frame.loc[frame.partition.eq("validation"), "replicate_id"]) == {"Rep3"}
    assert frame.groupby("curve_id").partition.nunique().eq(1).all()
    assert frame.groupby("analysis_group").partition.nunique().eq(1).all()


@pytest.mark.parametrize("replicate", ["Rep1", "Rep2", "Rep3"])
def test_replicate_holdout_excludes_whole_label_across_conditions(source_data, replicate):
    frame = support.make_split(source_data, replicate)
    assert frame.groupby("replicate_block").partition.nunique().eq(1).all()
    assert frame.loc[frame.partition.eq("test"), "replicate_id"].eq(replicate).all()
    assert set(frame.loc[frame.partition.eq("test"), "concentration_mM"]) == {0.05, 0.2, 1.0}
    assert frame.groupby("partition").size().eq(45).all()


def test_common_family_rule_never_reads_excluded_curves():
    class Forbidden:
        @property
        def candidates(self):
            raise AssertionError("Family selection accessed a held-out curve")

    fits = [
        SimpleNamespace(
            success=True,
            jacobian_rank=family.n_parameters,
            n_parameters=family.n_parameters,
            jacobian_condition=2.0,
            validation_metrics={"rmse": 0.1 + index},
        )
        for index, family in enumerate(support.ModelSelector().families)
    ]
    results = {
        "train": SimpleNamespace(candidates=fits),
        "test": Forbidden(),
        "validation": Forbidden(),
    }
    family, scores = support.choose_family(results, {"train"})
    assert family.name == "polynomial_1"
    assert len(scores) == 7
