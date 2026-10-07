from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from ml.models.metrics import calculate_classification_metrics, calculate_metrics
from ml.models.target import TargetSpec, validate_target_frame


class BaseModel(ABC):
    """
    Standard interface for all forecasting models.

    All models (Baseline, XGBoost, LSTM, Transformer) conform to this contract,
    ensuring uniform training, inference, and metric evaluation.
    """

    def _validate_and_set_target(self, y_train: pd.DataFrame, target_spec: TargetSpec | None = None) -> TargetSpec:
        spec = validate_target_frame(y_train, target_spec)
        self._target_spec = spec
        return spec

    @property
    def target_spec(self) -> TargetSpec | None:
        return getattr(self, "_target_spec", None)

    @property
    def requires_selection_holdout(self) -> bool:
        """Whether fit() uses a chronological internal holdout for model selection."""
        return False

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier of the model architecture."""
        ...

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | None = None,
        metadata_train: pd.DataFrame | None = None,
        metadata_val: pd.DataFrame | None = None,
    ) -> BaseModel:
        """
        Fit model parameters on training partitions.

        For selection-based models, X_val/y_val represent an internal chronological
        selection holdout used only for early stopping/checkpoint selection. The outer
        validation fold must not be passed here by the trainer.
        """
        ...

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
        metadata: pd.DataFrame | None = None,
        history_X: pd.DataFrame | None = None,
        history_metadata: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """
        Generate out-of-sample predictions for features X.

        Returns 1D numpy array of shape (N,) matching the target format.
        """
        ...

    @abstractmethod
    def predict_proba(
        self,
        X: pd.DataFrame,
        metadata: pd.DataFrame | None = None,
        history_X: pd.DataFrame | None = None,
        history_metadata: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """
        Generate out-of-sample class probabilities for features X.

        Returns 2D numpy array of shape (N, 2) where column 0 is P(class=0)
        and column 1 is P(class=1).
        """
        ...

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> dict[str, float]:
        """
        Evaluate model predictions against ground truth targets y.
        """
        spec = validate_target_frame(y, self.target_spec)
        y_true = y.iloc[:, 0].to_numpy(dtype=float)
        if spec.task_type == "classification":
            y_pred_proba = self.predict_proba(X)
            return calculate_classification_metrics(y_true, y_pred_proba)

        y_pred = self.predict(X)
        return calculate_metrics(y_true, y_pred)

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return dictionary of model configuration hyperparameters."""
        ...
