"""Interpretable parametric regression and reproducible holdout selection."""

from .families import (
    BaseFunctionFamily,
    ExponentialFamily,
    ExponentialRelaxationFamily,
    GompertzFamily,
    LogisticFamily,
    PolynomialFamily,
    RationalSaturationFamily,
)
from .fitting import fit_family
from .metrics import regression_metrics
from .multi_experiment import (
    ExtrapolationWarning,
    MultiExperimentRegressor,
    MultiExperimentResult,
)
from .parameter_models import (
    BaseParameterModel,
    ConstantParameterModel,
    LinearParameterModel,
    PolynomialParameterModel,
)
from .results import FitResult, SelectionResult, SplitInfo
from .selection import ModelSelector

__version__ = "0.2.0"
__all__ = [
    "BaseFunctionFamily",
    "ExponentialFamily",
    "ExponentialRelaxationFamily",
    "BaseParameterModel",
    "ConstantParameterModel",
    "LinearParameterModel",
    "PolynomialParameterModel",
    "MultiExperimentRegressor",
    "MultiExperimentResult",
    "ExtrapolationWarning",
    "FitResult",
    "GompertzFamily",
    "LogisticFamily",
    "ModelSelector",
    "PolynomialFamily",
    "RationalSaturationFamily",
    "SelectionResult",
    "SplitInfo",
    "fit_family",
    "regression_metrics",
]
