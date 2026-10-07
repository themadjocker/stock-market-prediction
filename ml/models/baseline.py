from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.models.base import BaseModel
from ml.models.target import TargetSpec


class NaiveBaselineModel(BaseModel):
    """
    Deterministic naive baseline for financial time series forecasting.

    Strategies:
    - "zero": Predicts constant 0.0 (the standard statistical baseline for daily log returns).
    - "mean": Predicts the historical training mean return.
    - "majority": Predicts majority direction class (for binary targets).
    """

    def __init__(self, strategy: str = "zero"):
        if strategy not in {"zero", "mean", "majority"}:
            raise ValueError(f"Unknown baseline strategy: {strategy}. Must be 'zero', 'mean', or 'majority'.")
        self.strategy = strategy
        self._fitted_value: float = 0.0
        self._is_fitted: bool = False
        self._target_spec: TargetSpec | None = None

    @property
    def name(self) -> str:
        return f"NaiveBaseline({self.strategy})"

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | None = None,
        metadata_train: pd.DataFrame | None = None,
        metadata_val: pd.DataFrame | None = None,
    ) -> NaiveBaselineModel:
        spec = self._validate_and_set_target(y_train)
        if self.strategy == "majority" and spec.task_type != "classification":
            raise ValueError("The majority baseline is only valid for classification targets.")
        if self.strategy in {"zero", "mean"} and spec.task_type != "regression":
            raise ValueError(f"The {self.strategy} baseline is only valid for regression targets.")

        y_vals = y_train.iloc[:, 0].to_numpy(dtype=float)
        valid = y_vals[np.isfinite(y_vals)]

        if len(valid) == 0:
            raise ValueError("y_train contains no finite target values.")

        if self.strategy == "zero":
            self._fitted_value = 0.0
            self._pos_ratio = np.nan
        elif self.strategy == "mean":
            self._fitted_value = float(np.mean(valid))
            self._pos_ratio = np.nan
        elif self.strategy == "majority":
            # Majority binary class: 1.0 if >=50% are positive, else 0.0
            pos_ratio = np.mean(valid > 0.0)
            self._fitted_value = 1.0 if pos_ratio >= 0.5 else 0.0
            self._pos_ratio = float(pos_ratio)

        self._is_fitted = True
        return self

    def predict(
        self,
        X: pd.DataFrame,
        metadata: pd.DataFrame | None = None,
        history_X: pd.DataFrame | None = None,
        history_metadata: pd.DataFrame | None = None,
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        n_samples = len(X)
        return np.full(n_samples, self._fitted_value, dtype=float)

    def predict_proba(
        self,
        X: pd.DataFrame,
        metadata: pd.DataFrame | None = None,
        history_X: pd.DataFrame | None = None,
        history_metadata: pd.DataFrame | None = None,
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        if self.target_spec is None or self.target_spec.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks.")
        n_samples = len(X)
        p1 = self._pos_ratio
        p0 = 1.0 - p1
        preds = np.empty((n_samples, 2), dtype=float)
        preds[:, 0] = p0
        preds[:, 1] = p1
        return preds

    def get_params(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "fitted_value": self._fitted_value}
