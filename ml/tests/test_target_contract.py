from __future__ import annotations

import pandas as pd
import pytest

from ml.features.targets import TARGET_HORIZONS, build_targets
from ml.models.baseline import NaiveBaselineModel
from ml.models.target import (
    ACTIVE_TARGET_HORIZONS,
    TargetSpec,
    target_spec_for,
    validate_target_frame,
)
from ml.models.xgboost_model import XGBoostModel


def test_active_target_horizons_are_one_and_five() -> None:
    assert TARGET_HORIZONS == (1, 5)
    assert ACTIVE_TARGET_HORIZONS == (1, 5)


def test_active_target_specs_are_explicit() -> None:
    assert target_spec_for("target_return_1d") == TargetSpec("target_return_1d", "regression", 1)
    assert target_spec_for("target_return_5d") == TargetSpec("target_return_5d", "regression", 5)
    assert target_spec_for("target_up_1d") == TargetSpec("target_up_1d", "classification", 1)
    assert target_spec_for("target_up_5d") == TargetSpec("target_up_5d", "classification", 5)


def test_unknown_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="inactive horizon"):
        target_spec_for("target_return_20d")


def test_training_requires_exactly_one_target() -> None:
    y = pd.DataFrame({"target_return_1d": [0.1, 0.2], "target_return_5d": [0.3, 0.4]})
    with pytest.raises(ValueError, match="exactly one target column"):
        validate_target_frame(y)


def test_classification_target_requires_binary_values() -> None:
    y = pd.DataFrame({"target_up_1d": [0.0, 1.0, 0.5]})
    with pytest.raises(ValueError, match="0/1"):
        validate_target_frame(y)


def test_regression_target_is_resolved_correctly() -> None:
    y = pd.DataFrame({"target_return_1d": [0.1, -0.2]})
    spec = validate_target_frame(y)
    assert spec.task_type == "regression"
    assert spec.horizon == 1


def test_classification_target_is_resolved_correctly() -> None:
    y = pd.DataFrame({"target_up_5d": [0.0, 1.0]})
    spec = validate_target_frame(y)
    assert spec.task_type == "classification"
    assert spec.horizon == 5


def test_baseline_strategy_matches_target_task() -> None:
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    y_reg = pd.DataFrame({"target_return_1d": [0.1, -0.1, 0.2]})
    y_cls = pd.DataFrame({"target_up_1d": [1.0, 1.0, 0.0]})

    NaiveBaselineModel("mean").fit(X, y_reg)
    NaiveBaselineModel("majority").fit(X, y_cls)

    with pytest.raises(ValueError, match="only valid for classification"):
        NaiveBaselineModel("majority").fit(X, y_reg)
    with pytest.raises(ValueError, match="only valid for regression"):
        NaiveBaselineModel("mean").fit(X, y_cls)


def test_xgboost_supports_explicit_classification_target() -> None:
    X = pd.DataFrame({"f1": [-2.0, -1.0, 1.0, 2.0, -1.5, 1.5]})
    y = pd.DataFrame({"target_up_1d": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]})
    model = XGBoostModel(n_estimators=5, max_depth=2, random_state=42, early_stopping_rounds=None)
    model.fit(X, y)
    preds = model.predict(X)
    assert set(preds.tolist()).issubset({0.0, 1.0})
    assert model.target_spec is not None
    assert model.target_spec.task_type == "classification"


def test_build_targets_default_exposes_only_active_horizons() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 6,
            "session_date": pd.date_range("2026-01-01", periods=6, freq="D"),
            "adj_close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        }
    )
    result = build_targets(frame)
    assert "target_return_1d" in result.columns
    assert "target_return_5d" in result.columns
    assert "target_up_1d" in result.columns
    assert "target_up_5d" in result.columns
    assert "target_return_20d" not in result.columns
    assert "target_up_20d" not in result.columns
