from __future__ import annotations

import numpy as np
import pytest

from ml.models.metrics import calculate_metrics


def test_calculate_metrics_perfect_predictions() -> None:
    y_true = np.array([0.01, 0.02, -0.01, 0.03])
    y_pred = np.array([0.01, 0.02, -0.01, 0.03])

    metrics = calculate_metrics(y_true, y_pred)
    assert np.isclose(metrics["rmse"], 0.0)
    assert np.isclose(metrics["mae"], 0.0)
    assert np.isclose(metrics["r2"], 1.0)
    assert np.isclose(metrics["directional_accuracy"], 1.0)
    assert np.isclose(metrics["pearson_ic"], 1.0)


def test_calculate_metrics_imperfect_predictions() -> None:
    y_true = np.array([0.02, -0.02, 0.04, -0.01])
    y_pred = np.array([0.01, 0.01, 0.03, -0.02])

    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["rmse"] > 0.0
    assert metrics["mae"] > 0.0
    # Signs: (+, +), (-, +)[wrong], (+, +), (-, -) -> 3/4 = 0.75
    assert np.isclose(metrics["directional_accuracy"], 0.75)


def test_calculate_metrics_empty_and_nan() -> None:
    metrics = calculate_metrics([], [])
    assert np.isnan(metrics["rmse"])

    metrics_nan = calculate_metrics([np.nan, np.nan], [np.nan, np.nan])
    assert np.isnan(metrics_nan["rmse"])


def test_calculate_metrics_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        calculate_metrics([1.0, 2.0], [1.0])


def test_metrics_constant_target_has_defined_r2_and_ic() -> None:
    metrics = calculate_metrics([1.0, 1.0, 1.0], [1.0, 2.0, 0.0])
    assert metrics["r2"] == 0.0
    assert metrics["pearson_ic"] == 0.0


def test_metrics_rejects_non_finite_classification_values() -> None:
    from ml.models.metrics import calculate_classification_metrics

    with pytest.raises(ValueError, match="finite"):
        calculate_classification_metrics([0.0, 1.0], [[1.0, 0.0], [np.nan, np.nan]])


def test_classification_metrics_are_task_aware() -> None:
    from ml.models.metrics import calculate_classification_metrics

    # [0.1, 0.9, 0.4, 0.8] as probability of class 1
    probs = [[0.9, 0.1], [0.1, 0.9], [0.6, 0.4], [0.2, 0.8]]
    metrics = calculate_classification_metrics([0.0, 1.0, 1.0, 0.0], probs)
    assert np.isclose(metrics["accuracy"], 0.5)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "logloss" in metrics
