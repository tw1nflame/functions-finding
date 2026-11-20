"""Interpretable parametric regression and reproducible holdout selection."""

from .families import (
    BaseFunctionFamily,
    BiphasicExponentialFamily,
    DelayedExponentialFamily,
    ExponentialFamily,
    ExponentialRelaxationFamily,
    GompertzFamily,
    LogarithmicFamily,
    LogisticFamily,
    PolynomialFamily,
    PowerLawFamily,
    RationalSaturationFamily,
    WeibullSurvivalFamily,
)
from .fitting import fit_family
from .metrics import regression_metrics
from .multi_experiment import (
    ExtrapolationWarning,
    MultiExperimentRegressor,
    MultiExperimentResult,
)
from .parameter_models import (
    ArrheniusParameterModel,
    BaseParameterModel,
    ConstantParameterModel,
    HillParameterModel,
    LinearParameterModel,
    MonodParameterModel,
    PolynomialParameterModel,
    RatkowskyParameterModel,
)
from .results import FitResult, SelectionResult, SplitInfo
from .selection import ModelSelector

__version__ = "0.2.0"
__all__ = [
    "BaseFunctionFamily",
    "PowerLawFamily",
    "LogarithmicFamily",
    "DelayedExponentialFamily",
    "WeibullSurvivalFamily",
    "BiphasicExponentialFamily",
    "ArrheniusParameterModel",
    "RatkowskyParameterModel",
    "HillParameterModel",
    "MonodParameterModel",
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
