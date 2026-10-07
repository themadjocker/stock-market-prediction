from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.models.base import BaseModel
from ml.models.trainer import ValidationResult, train_and_validate
from ml.validation.split import ValidatedFoldBundle


@dataclass(frozen=True)
class ModelExperimentResult:
    """Immutable P5 result for one model evaluated across validation folds."""

    model_name: str
    fold_results: tuple[ValidationResult, ...]
    aggregate_metrics: dict[str, float]


def _aggregate_fold_metrics(results: Iterable[ValidationResult]) -> dict[str, float]:
    rows = [r.metrics for r in results]
    if not rows:
        return {}

    keys = sorted({key for row in rows for key in row})
    aggregate: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        aggregate[key] = float(np.mean(finite)) if finite.size else float("nan")
    return aggregate


def run_model_across_folds(
    model_factory: Callable[[], BaseModel],
    folds: Iterable[ValidatedFoldBundle],
) -> ModelExperimentResult:
    """Train a fresh model per fold and aggregate only out-of-sample fold metrics.

    ``model_factory`` must return a new BaseModel for every invocation. Reusing a
    fitted model between temporal folds is deliberately prohibited because it would
    carry learned state across validation boundaries.
    """
    fold_results: list[ValidationResult] = []
    model_names: set[str] = set()

    for bundle in folds:
        model = model_factory()
        if not isinstance(model, BaseModel):
            raise TypeError("model_factory must return a BaseModel instance")
        result = train_and_validate(model, bundle)
        fold_results.append(result)
        model_names.add(result.model_name)

    if not fold_results:
        raise ValueError("At least one validation fold is required.")
    if len(model_names) != 1:
        raise RuntimeError("Model factory returned inconsistent model names across folds.")

    return ModelExperimentResult(
        model_name=fold_results[0].model_name,
        fold_results=tuple(fold_results),
        aggregate_metrics=_aggregate_fold_metrics(fold_results),
    )


def compare_experiments(
    experiments: Iterable[ModelExperimentResult],
    *,
    primary_metric: str = "rmse",
) -> pd.DataFrame:
    """Build a deterministic model-comparison table from validation results only."""
    rows = []
    for experiment in experiments:
        row = {"model": experiment.model_name, **experiment.aggregate_metrics}
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["model", primary_metric])

    frame = pd.DataFrame(rows)
    if primary_metric not in frame.columns:
        raise ValueError(
            f"Primary metric '{primary_metric}' is unavailable. "
            f"Available metrics: {sorted(set(frame.columns) - {'model'})}"
        )
    return frame.sort_values(
        by=[primary_metric, "model"],
        ascending=[True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
