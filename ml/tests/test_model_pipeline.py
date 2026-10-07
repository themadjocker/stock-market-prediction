from __future__ import annotations

import numpy as np
import pandas as pd

from ml.features.dataset import ModelDataset, build_model_dataset
from ml.models.baseline import NaiveBaselineModel
from ml.models.lstm_model import LSTMModel
from ml.models.trainer import ValidationResult, train_and_validate
from ml.models.transformer_model import TransformerModel
from ml.models.xgboost_model import XGBoostModel
from ml.validation.split import chronological_split


def make_pipeline_dataset(n_days: int = 80) -> ModelDataset:
    dates = pd.date_range("2026-01-01", periods=n_days, freq="B")
    rows = []
    for ticker in ["AAA", "BBB"]:
        for i, d in enumerate(dates):
            rows.append(
                {
                    "ticker": ticker,
                    "session_date": d,
                    "close": 100.0 + i * 0.5,
                    "volume": 1000.0 + i * 10,
                    "target_return_5d": 0.01 * (1 if i % 2 == 0 else -1),
                }
            )
    df = pd.DataFrame(rows)
    return build_model_dataset(
        df,
        feature_columns=["close", "volume"],
        target_columns=["target_return_5d"],
    )


def test_end_to_end_pipeline_all_models() -> None:
    dataset = make_pipeline_dataset(n_days=80)

    # 1. P4 Validation split
    fold_bundle, _, _ = chronological_split(
        dataset,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        target_horizon=5,
    )

    models = [
        NaiveBaselineModel(strategy="zero"),
        NaiveBaselineModel(strategy="mean"),
        XGBoostModel(n_estimators=10, max_depth=2, random_state=42),
        LSTMModel(lookback=4, hidden_dim=8, epochs=2, batch_size=16, seed=42),
        TransformerModel(lookback=4, d_model=16, nhead=2, epochs=2, batch_size=16, seed=42),
    ]

    for model in models:
        result = train_and_validate(model, fold_bundle)
        assert isinstance(result, ValidationResult)
        assert result.manifest_hash == fold_bundle.manifest_hash
        assert len(result.y_pred) == len(fold_bundle.y_val)
        assert "rmse" in result.metrics
        assert "directional_accuracy" in result.metrics
        assert np.isfinite(result.metrics["rmse"])


def test_trainer_does_not_pass_outer_validation_to_selection_models() -> None:
    dataset = make_pipeline_dataset(n_days=80)
    fold_bundle, _, _ = chronological_split(
        dataset,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        target_horizon=5,
    )

    class SpyXGB(XGBoostModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.captured: dict[str, object] = {}

        def fit(self, *args, **kwargs):
            self.captured = {
                "x_train": kwargs.get("X_train"),
                "y_train": kwargs.get("y_train"),
                "x_val": kwargs.get("X_val"),
                "y_val": kwargs.get("y_val"),
                "metadata_val": kwargs.get("metadata_val"),
            }
            return super().fit(*args, **kwargs)

    model = SpyXGB(n_estimators=5, max_depth=2, random_state=42)
    result = train_and_validate(model, fold_bundle)

    captured = model.captured
    captured_x_val = captured["x_val"]
    captured_x_train = captured["x_train"]
    assert isinstance(captured_x_val, pd.DataFrame)
    assert isinstance(captured_x_train, pd.DataFrame)
    assert len(captured_x_val) < len(fold_bundle.X_val) or not captured_x_val.equals(fold_bundle.X_val)
    assert len(captured_x_val) == result.selection_count
    assert len(captured_x_train) == result.selection_fit_count
    assert result.selection_start is not None
    assert pd.Timestamp(result.selection_start) < pd.Timestamp(fold_bundle.metadata_val["session_date"].min())


def test_inner_selection_split_is_chronological_and_purges_target_overlap() -> None:
    from ml.validation.model_selection import make_inner_selection_split

    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    metadata = pd.DataFrame({"ticker": ["AAA"] * len(dates), "session_date": dates})
    split = make_inner_selection_split(metadata, target_horizon=5, selection_fraction=0.2, min_selection_sessions=5)

    assert split.fit_indices.max() < split.selection_indices.min()
    assert split.selection_indices.tolist() == list(range(24, 30))
    # Rows 19-23 have outcomes on/after the selection boundary and must be purged.
    assert split.fit_indices.max() == 18
    assert split.purged_fit_count == 5
