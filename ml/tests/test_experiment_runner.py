from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features.dataset import build_model_dataset
from ml.models.baseline import NaiveBaselineModel
from ml.models.experiment import compare_experiments, run_model_across_folds
from ml.models.registry import available_models, create_model
from ml.validation.split import walk_forward_split


def make_dataset(n_days: int = 90):
    dates = pd.date_range("2026-01-01", periods=n_days, freq="B")
    rows = []
    for ticker in ["AAA", "BBB"]:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "ticker": ticker,
                    "session_date": date,
                    "f1": float(i),
                    "f2": float(i % 7),
                    "target_return_1d": float(np.sin(i / 5.0) * 0.01),
                }
            )
    return build_model_dataset(
        pd.DataFrame(rows),
        feature_columns=["f1", "f2"],
        target_columns=["target_return_1d"],
    )


def test_registry_exposes_stable_model_names() -> None:
    names = available_models()
    assert {
        "baseline_zero",
        "baseline_mean",
        "baseline_majority",
        "xgboost",
        "lstm",
        "transformer",
    }.issubset(names)
    assert isinstance(create_model("baseline_zero"), NaiveBaselineModel)
    assert create_model("xgboost", n_estimators=2).name.startswith("XGBoost")
    with pytest.raises(ValueError):
        create_model("does_not_exist")


def test_experiment_runner_uses_fresh_model_per_fold_and_aggregates_metrics() -> None:
    dataset = make_dataset()
    folds = [
        bundle
        for bundle, _ in walk_forward_split(
            dataset,
            min_train_sessions=40,
            val_sessions=10,
            step_sessions=10,
            target_horizon=1,
        )[:2]
    ]

    created: list[NaiveBaselineModel] = []

    def factory() -> NaiveBaselineModel:
        model = NaiveBaselineModel(strategy="zero")
        created.append(model)
        return model

    result = run_model_across_folds(factory, folds)
    assert len(result.fold_results) == 2
    assert len(created) == 2
    assert created[0] is not created[1]
    assert np.isfinite(result.aggregate_metrics["rmse"])


def test_compare_experiments_sorts_by_primary_metric() -> None:
    dataset = make_dataset()
    folds = [
        bundle
        for bundle, _ in walk_forward_split(
            dataset,
            min_train_sessions=40,
            val_sessions=10,
            step_sessions=10,
            target_horizon=1,
        )[:1]
    ]

    zero = run_model_across_folds(lambda: NaiveBaselineModel(strategy="zero"), folds)
    mean = run_model_across_folds(lambda: NaiveBaselineModel(strategy="mean"), folds)
    frame = compare_experiments([zero, mean], primary_metric="rmse")
    assert "rmse" in frame.columns
    assert frame.iloc[0]["rmse"] <= frame.iloc[1]["rmse"]
