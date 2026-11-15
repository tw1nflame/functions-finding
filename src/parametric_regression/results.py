"""Results retain all candidates, diagnostics and the exact evaluation protocol."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray
from .families import BaseFunctionFamily

SplitMethod = Literal["random", "time", "group"]


@dataclass(frozen=True)
class SplitInfo:
    """Original row positions used for training/validation, plus split settings."""

    method: SplitMethod
    train_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]
    validation_size: float
    random_state: int | None

    def __post_init__(self) -> None:
        for name in ("train_indices", "validation_indices"):
            indices = np.array(getattr(self, name), dtype=np.int64, copy=True)
            indices.setflags(write=False)
            object.__setattr__(self, name, indices)


@dataclass
class FitResult:
    """One fitted family or a failed attempt with its diagnostic message.

    Coefficients always belong to the training fit. ``metrics`` refers to
    validation metrics when evaluated by ModelSelector, training metrics otherwise.
    """

    family: BaseFunctionFamily
    success: bool
    params: FloatArray | None = None
    train_metrics: dict[str, float] = field(default_factory=dict)
    validation_metrics: dict[str, float] = field(default_factory=dict)
    message: str = ""
    warnings: tuple[str, ...] = ()
    split_protocol: SplitInfo | None = None
    nfev: int = 0
    jacobian_rank: int | None = None
    jacobian_condition: float | None = None

    @property
    def family_name(self) -> str:
        """Stable family identifier."""
        return self.family.name

    @property
    def n_parameters(self) -> int:
        """Number of free parameters used for the simplicity preference."""
        return self.family.n_parameters

    @property
    def coefficients(self) -> dict[str, float]:
        """Named coefficients; empty if no solution was obtained."""
        if self.params is None:
            return {}
        return dict(zip(self.family.parameter_names, map(float, self.params), strict=True))

    @property
    def formula(self) -> str | None:
        """Plain formula in the original input coordinate."""
        return None if self.params is None else self.family.formula(self.params)

    @property
    def latex_formula(self) -> str | None:
        """LaTeX formula without math delimiters."""
        return None if self.params is None else self.family.formula(self.params, latex=True)

    @property
    def metrics(self) -> dict[str, float]:
        """Validation metrics if available, otherwise training metrics."""
        return dict(self.validation_metrics or self.train_metrics)

    def predict(self, x: ArrayLike) -> FloatArray:
        """Predict a finite 1-D x vector; raise RuntimeError for a failed fit."""
        if not self.success or self.params is None:
            raise RuntimeError(f"Cannot predict with {self.family_name}: {self.message}")
        return self.family.evaluate(x, self.params)


@dataclass
class SelectionResult:
    """Selection outcome, including failed candidates and an optional winner."""

    candidates: tuple[FitResult, ...]
    best_index: int | None
    split_protocol: SplitInfo
    relative_tolerance: float
    absolute_tolerance: float
    criterion: str = "validation_rmse"

    @property
    def success(self) -> bool:
        """Whether at least one model fitted and predicted validation data."""
        return self.best_index is not None

    @property
    def best_result(self) -> FitResult:
        """Winning training fit; raise with all diagnostics if every candidate failed."""
        if self.best_index is None:
            details = "; ".join(f"{r.family_name}: {r.message}" for r in self.candidates)
            raise RuntimeError(f"No successful candidate. {details}")
        return self.candidates[self.best_index]

    @property
    def best_family(self) -> str | None:
        """Winning family name, or None when no candidate succeeded."""
        return self.best_result.family_name if self.success else None

    @property
    def formula(self) -> str | None:
        """Winning numeric formula, or None."""
        return self.best_result.formula if self.success else None

    @property
    def latex_formula(self) -> str | None:
        """Winning LaTeX formula, or None."""
        return self.best_result.latex_formula if self.success else None

    @property
    def coefficients(self) -> dict[str, float]:
        """Named coefficients of the winner."""
        return self.best_result.coefficients if self.success else {}

    @property
    def metrics(self) -> dict[str, float]:
        """Winning validation metrics; empty when every candidate failed."""
        return self.best_result.metrics if self.success else {}

    def predict(self, x: ArrayLike) -> FloatArray:
        """Predict using the selected training fit, without implicit refitting."""
        return self.best_result.predict(x)

    def to_dataframe(self) -> pd.DataFrame:
        """Return all candidates in input order, with separate train/validation metrics."""
        return pd.DataFrame(
            [
                {
                    "family": result.family_name,
                    "success": result.success,
                    "selected": index == self.best_index,
                    "n_parameters": result.n_parameters,
                    **{f"train_{key}": value for key, value in result.train_metrics.items()},
                    **{
                        f"validation_{key}": value
                        for key, value in result.validation_metrics.items()
                    },
                    "message": result.message,
                    "warnings": "; ".join(result.warnings),
                }
                for index, result in enumerate(self.candidates)
            ]
        )
