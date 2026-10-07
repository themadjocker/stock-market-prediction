from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from ml.models.base import BaseModel
from ml.models.target import TargetSpec


class XGBoostModel(BaseModel):
    """
    Deterministic Gradient Boosted Decision Tree model for stock return forecasting.

    Accepts 2D tabular feature matrices. Supports validation early stopping without
    contaminating final test data.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        early_stopping_rounds: int | None = 10,
        n_jobs: int = 1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = n_jobs

        self._model: xgb.XGBRegressor | xgb.XGBClassifier | None = None
        self._target_spec: TargetSpec | None = None
        self._is_fitted: bool = False
        self._selection_best_iteration: int | None = None

    @property
    def requires_selection_holdout(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return f"XGBoost(depth={self.max_depth}, lr={self.learning_rate})"

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | None = None,
        metadata_train: pd.DataFrame | None = None,
        metadata_val: pd.DataFrame | None = None,
    ) -> XGBoostModel:
        if X_train.empty or y_train.empty:
            raise ValueError("Training data cannot be empty.")

        spec = self._validate_and_set_target(y_train)
        X_tr = X_train.to_numpy(dtype=float)
        y_tr = y_train.iloc[:, 0].to_numpy(dtype=float)

        model_params: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "objective": "binary:logistic" if spec.task_type == "classification" else "reg:squarederror",
            "eval_metric": "logloss" if spec.task_type == "classification" else "rmse",
        }

        eval_set = None
        if (
            self.early_stopping_rounds is not None
            and X_val is not None
            and y_val is not None
            and not X_val.empty
            and not y_val.empty
        ):
            X_v = X_val.to_numpy(dtype=float)
            self._validate_and_set_target(y_val, spec)
            y_v = y_val.iloc[:, 0].to_numpy(dtype=float)
            eval_set = [(X_v, y_v)]
            model_params["early_stopping_rounds"] = self.early_stopping_rounds

        model_cls = xgb.XGBClassifier if spec.task_type == "classification" else xgb.XGBRegressor
        self._model = model_cls(**model_params)
        self._model.fit(
            X_tr,
            y_tr,
            eval_set=eval_set,
            verbose=False,
        )

        if eval_set is not None and self._model is not None:
            self._selection_best_iteration = int(getattr(self._model, "best_iteration", self.n_estimators - 1))
        else:
            self._selection_best_iteration = None

        self._is_fitted = True
        return self

    def predict(
        self,
        X: pd.DataFrame,
        metadata: pd.DataFrame | None = None,
        history_X: pd.DataFrame | None = None,
        history_metadata: pd.DataFrame | None = None,
    ) -> np.ndarray:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before predict.")
        if X.empty:
            return np.array([], dtype=float)

        kwargs = {}
        if self._selection_best_iteration is not None:
            kwargs["iteration_range"] = (0, self._selection_best_iteration + 1)

        preds = self._model.predict(X.to_numpy(dtype=float), **kwargs)
        return preds.astype(float)

    def predict_proba(
        self,
        X: pd.DataFrame,
        metadata: pd.DataFrame | None = None,
        history_X: pd.DataFrame | None = None,
        history_metadata: pd.DataFrame | None = None,
    ) -> np.ndarray:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before predict_proba.")
        if X.empty:
            return np.empty((0, 2), dtype=float)
        if self.target_spec is None or self.target_spec.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks.")

        if not isinstance(self._model, xgb.XGBClassifier):
            raise RuntimeError("predict_proba requires a classification XGBoost model.")

        kwargs = {}
        if self._selection_best_iteration is not None:
            kwargs["iteration_range"] = (0, self._selection_best_iteration + 1)

        return self._model.predict_proba(X.to_numpy(dtype=float), **kwargs).astype(float)

    def get_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "early_stopping_rounds": self.early_stopping_rounds,
            "selection_best_iteration": self._selection_best_iteration,
        }
