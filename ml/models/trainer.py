from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.models.base import BaseModel
from ml.models.metrics import calculate_classification_metrics, calculate_metrics
from ml.models.target import validate_target_frame
from ml.validation.model_selection import make_inner_selection_split
from ml.validation.split import ValidatedFoldBundle


@dataclass(frozen=True)
class ValidationResult:
    """
    Immutable out-of-sample evaluation result for a single validation fold.
    """

    model_name: str
    fold_id: int
    manifest_hash: str
    metrics: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    timestamps: pd.Series
    selection_start: pd.Timestamp | None = None
    selection_fit_count: int | None = None
    selection_count: int | None = None


def train_and_validate(
    model: BaseModel,
    bundle: ValidatedFoldBundle,
) -> ValidationResult:
    """
    Train a model using an internal chronological selection holdout and evaluate
    out-of-sample on the untouched outer validation fold.

    Enforces strict physical separation:
    - Selection-based models receive only an inner holdout for early stopping/checkpoint selection.
    - The outer validation fold is never passed to model.fit().
    - Outer validation predictions are evaluated only after model selection is complete.
    - Test partitions are structurally excluded from this workflow.
    """
    if not isinstance(bundle, ValidatedFoldBundle):
        raise TypeError(f"Expected ValidatedFoldBundle, got {type(bundle)}")

    target_spec = validate_target_frame(bundle.y_train, model.target_spec)
    validate_target_frame(bundle.y_val, target_spec)
    selection_start = None
    selection_fit_count = None
    selection_count = None

    if model.requires_selection_holdout:
        inner = make_inner_selection_split(
            bundle.metadata_train,
            target_horizon=target_spec.horizon,
        )
        X_fit = bundle.X_train.iloc[inner.fit_indices].reset_index(drop=True)
        y_fit = bundle.y_train.iloc[inner.fit_indices].reset_index(drop=True)
        metadata_fit = bundle.metadata_train.iloc[inner.fit_indices].reset_index(drop=True)
        X_selection = bundle.X_train.iloc[inner.selection_indices].reset_index(drop=True)
        y_selection = bundle.y_train.iloc[inner.selection_indices].reset_index(drop=True)
        metadata_selection = bundle.metadata_train.iloc[inner.selection_indices].reset_index(drop=True)
        selection_start = inner.selection_start
        selection_fit_count = len(inner.fit_indices)
        selection_count = len(inner.selection_indices)

        # Outer validation is deliberately NOT passed to fit(). It is reserved for final
        # out-of-sample evaluation of this fold.
        model.fit(
            X_train=X_fit,
            y_train=y_fit,
            X_val=X_selection,
            y_val=y_selection,
            metadata_train=metadata_fit,
            metadata_val=metadata_selection,
        )
    else:
        # Baselines do not use a selection holdout and can consume the complete outer-train partition.
        model.fit(
            X_train=bundle.X_train,
            y_train=bundle.y_train,
            metadata_train=bundle.metadata_train,
        )

    # Predict out-of-sample on validation features
    y_pred = model.predict(
        bundle.X_val,
        metadata=bundle.metadata_val,
        history_X=bundle.X_train,
        history_metadata=bundle.metadata_train,
    )
    y_true = bundle.y_val.iloc[:, 0].to_numpy(dtype=float)

    if target_spec.task_type == "classification":
        metrics = calculate_classification_metrics(y_true, y_pred)
    else:
        metrics = calculate_metrics(y_true, y_pred)

    timestamps = pd.to_datetime(bundle.metadata_val["session_date"])

    return ValidationResult(
        model_name=model.name,
        fold_id=bundle.fold_id,
        manifest_hash=bundle.manifest_hash,
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        timestamps=timestamps,
        selection_start=selection_start,
        selection_fit_count=selection_fit_count,
        selection_count=selection_count,
    )
