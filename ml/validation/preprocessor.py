from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


class TrainOnlyPreprocessor:
    """
    Leakage-safe feature scaler with strict train-only fitting invariant.

    Guarantees that scaling statistics (mean, std, median, IQR) are estimated
    exclusively from training observations. Re-fitting on validation or test partitions
    is explicitly rejected at runtime.
    """

    def __init__(self, scaler_type: str = "standard"):
        if scaler_type not in {"standard", "robust"}:
            raise ValueError(f"Unsupported scaler_type: {scaler_type}. Must be 'standard' or 'robust'.")
        self.scaler_type = scaler_type
        self._scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
        self._is_fitted: bool = False
        self._feature_columns: tuple[str, ...] | None = None
        self._fitted_params: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def feature_columns(self) -> tuple[str, ...] | None:
        return self._feature_columns

    @property
    def fitted_params(self) -> dict[str, Any]:
        if not self._is_fitted:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return dict(self._fitted_params)

    def fit(self, X: pd.DataFrame) -> TrainOnlyPreprocessor:
        """
        Fit scaler parameters exclusively on training features.

        Calling fit a second time raises RuntimeError to prevent accidental re-fitting.
        """
        if self._is_fitted:
            raise RuntimeError(
                "Preprocessor is already fitted. Re-fitting is strictly prohibited to prevent data leakage."
            )

        if X.empty:
            raise ValueError("Cannot fit preprocessor on an empty DataFrame.")

        self._feature_columns = tuple(X.columns)
        self._scaler.fit(X.to_numpy(dtype=float))
        self._is_fitted = True

        if isinstance(self._scaler, StandardScaler):
            mean = self._scaler.mean_
            scale = self._scaler.scale_
            var = self._scaler.var_
            if mean is None or scale is None or var is None:
                raise RuntimeError("StandardScaler did not expose fitted statistics after fit.")
            self._fitted_params = {
                "mean": mean.copy(),
                "scale": scale.copy(),
                "var": var.copy(),
            }
        else:
            center = self._scaler.center_
            scale = self._scaler.scale_
            if center is None or scale is None:
                raise RuntimeError("RobustScaler did not expose fitted statistics after fit.")
            self._fitted_params = {
                "center": center.copy(),
                "scale": scale.copy(),
            }

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform candidate features using the pre-fitted training statistics.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before calling transform.")

        if tuple(X.columns) != self._feature_columns:
            raise ValueError(f"Feature columns mismatch. Expected {self._feature_columns}, got {tuple(X.columns)}")

        if X.empty:
            return pd.DataFrame(columns=self._feature_columns, index=X.index)

        scaled_values = self._scaler.transform(X.to_numpy(dtype=float))
        return pd.DataFrame(scaled_values, columns=self._feature_columns, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and return scaled training features."""
        return self.fit(X).transform(X)
