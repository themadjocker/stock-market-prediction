from __future__ import annotations

import numpy as np


def calculate_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> dict[str, float]:
    """
    Pure calculation of standard regression and directional financial metrics.

    Metrics:
    - rmse: Root Mean Squared Error
    - mae: Mean Absolute Error
    - r2: Coefficient of Determination
    - directional_accuracy: Directional Hit Rate (sign match)
    - pearson_ic: Pearson correlation coefficient between predicted and actual returns
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()

    if len(y_t) != len(y_p):
        raise ValueError(f"Shape mismatch: y_true has length {len(y_t)}, y_pred has length {len(y_p)}")

    if len(y_t) == 0:
        return {
            "rmse": np.nan,
            "relative_rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "directional_accuracy": np.nan,
            "pearson_ic": np.nan,
        }

    # Mask NaNs if any exist
    valid_mask = np.isfinite(y_t) & np.isfinite(y_p)
    if not valid_mask.any():
        return {
            "rmse": np.nan,
            "relative_rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "directional_accuracy": np.nan,
            "pearson_ic": np.nan,
        }

    y_t = y_t[valid_mask]
    y_p = y_p[valid_mask]

    diff = y_t - y_p
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))

    # Relative RMSE (zero-baseline)
    zero_rmse = float(np.sqrt(np.mean(y_t**2)))
    relative_rmse = rmse / zero_rmse if zero_rmse > 1e-12 else np.nan

    # R2
    ss_res = np.sum(diff**2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    if ss_tot > 1e-12:
        r2 = float(1.0 - (ss_res / ss_tot))
    else:
        r2 = 0.0

    # Directional accuracy (Hit Rate)
    # Check sign alignment: both > 0 or both < 0 or both == 0
    sign_true = np.sign(y_t)
    sign_pred = np.sign(y_p)
    dir_acc = float(np.mean(sign_true == sign_pred))

    # Pearson IC (correlation)
    std_t = np.std(y_t)
    std_p = np.std(y_p)
    if std_t > 1e-12 and std_p > 1e-12:
        cov = np.mean((y_t - np.mean(y_t)) * (y_p - np.mean(y_p)))
        pearson_ic = float(cov / (std_t * std_p))
    else:
        pearson_ic = 0.0

    return {
        "rmse": rmse,
        "relative_rmse": relative_rmse,
        "mae": mae,
        "r2": r2,
        "directional_accuracy": dir_acc,
        "pearson_ic": pearson_ic,
    }


def calculate_classification_metrics(
    y_true: np.ndarray | list[float],
    y_pred_proba: np.ndarray | list[list[float]],
) -> dict[str, float]:
    """Calculate binary classification metrics for explicit 0/1 targets using probabilities."""
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p_proba = np.asarray(y_pred_proba, dtype=np.float64)

    if len(y_t) == 0:
        return {
            "logloss": np.nan,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
        }

    if y_p_proba.ndim != 2 or y_p_proba.shape[1] != 2:
        raise ValueError(f"y_pred_proba must be 2D array of shape (N, 2), got {y_p_proba.shape}")

    if len(y_t) != len(y_p_proba):
        raise ValueError(f"Shape mismatch: y_true has length {len(y_t)}, y_pred_proba has length {len(y_p_proba)}")

    if not np.isfinite(y_t).all() or not np.isfinite(y_p_proba).all():
        raise ValueError("Classification metrics require finite values.")

    if not set(np.unique(y_t).tolist()).issubset({0.0, 1.0}):
        raise ValueError("y_true must contain only 0/1 values for classification metrics.")

    p1 = np.clip(y_p_proba[:, 1], 1e-15, 1.0 - 1e-15)

    # Calculate LogLoss
    logloss = float(-np.mean(y_t * np.log(p1) + (1 - y_t) * np.log(1 - p1)))

    pred = (y_p_proba[:, 1] >= 0.5).astype(int)
    true = y_t.astype(int)

    tp = int(np.sum((true == 1) & (pred == 1)))
    fp = int(np.sum((true == 0) & (pred == 1)))
    fn = int(np.sum((true == 1) & (pred == 0)))

    accuracy = float(np.mean(true == pred))
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    return {
        "logloss": logloss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
