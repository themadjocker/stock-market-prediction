from __future__ import annotations

import numpy as np
import pandas as pd

from ml.models.xgboost_model import XGBoostModel


def make_synthetic_data(n_samples: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "f1": np.random.randn(n_samples),
            "f2": np.random.randn(n_samples),
        }
    )
    # y = 0.5 * f1 - 0.2 * f2 + noise
    y = pd.DataFrame({"target_return_1d": 0.5 * X["f1"] - 0.2 * X["f2"] + 0.01 * np.random.randn(n_samples)})
    return X, y


def test_xgboost_model_fit_predict_deterministic() -> None:
    X, y = make_synthetic_data(80)
    X_train, y_train = X.iloc[:60], y.iloc[:60]
    X_val, y_val = X.iloc[60:], y.iloc[60:]

    model1 = XGBoostModel(n_estimators=20, max_depth=2, random_state=42)
    model1.fit(X_train, y_train, X_val, y_val)
    preds1 = model1.predict(X_val)

    model2 = XGBoostModel(n_estimators=20, max_depth=2, random_state=42)
    model2.fit(X_train, y_train, X_val, y_val)
    preds2 = model2.predict(X_val)

    assert len(preds1) == 20
    np.testing.assert_array_equal(preds1, preds2)


def test_xgboost_model_evaluate() -> None:
    X, y = make_synthetic_data(60)
    model = XGBoostModel(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X.iloc[:40], y.iloc[:40])

    metrics = model.evaluate(X.iloc[40:], y.iloc[40:])
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "directional_accuracy" in metrics
    assert metrics["rmse"] > 0.0
