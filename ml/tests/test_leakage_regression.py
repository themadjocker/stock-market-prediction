from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features.dataset import ModelDataset, build_model_dataset
from ml.validation.preprocessor import TrainOnlyPreprocessor
from ml.validation.purging import purge_train_horizon
from ml.validation.split import (
    chronological_split,
)


def make_test_dataset(n_days: int = 100) -> ModelDataset:
    dates = pd.date_range("2026-01-01", periods=n_days, freq="B")
    df = pd.DataFrame(
        {
            "ticker": ["AAA"] * n_days,
            "session_date": dates,
            "close": np.linspace(100.0, 200.0, n_days),
            "target_return_5d": np.linspace(0.01, 0.05, n_days),
        }
    )
    return build_model_dataset(
        df,
        feature_columns=["close"],
        target_columns=["target_return_5d"],
    )


def test_adversarial_case_a_target_outcome_in_validation_purged() -> None:
    """
    Case A: Sample session date < val_start, but session_date + h >= val_start.
    Must be PURGED from training.
    """
    dates = pd.date_range("2026-01-01", periods=20, freq="B")
    metadata = pd.DataFrame({"ticker": ["AAA"] * 20, "session_date": dates})

    val_start = dates[10]
    horizon = 3  # sessions 7, 8, 9 have outcome dates 10, 11, 12 >= val_start

    train_indices = np.arange(10)  # sessions 0..9
    safe_indices, purged_count = purge_train_horizon(
        metadata,
        train_indices,
        val_start_date=val_start,
        horizon=horizon,
    )

    assert purged_count == 3
    # Sessions 7, 8, 9 must not be in safe_indices
    assert set(safe_indices) == {0, 1, 2, 3, 4, 5, 6}


def test_adversarial_case_b_val_target_outcome_in_test_purged() -> None:
    """
    Case B: Validation sample session date < test_start, but session_date + h >= test_start.
    Must be PURGED from validation.
    """
    dataset = make_test_dataset(n_days=60)
    bundle, test_bundle, split = chronological_split(
        dataset,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        target_horizon=5,
    )

    assert test_bundle is not None
    test_start = pd.to_datetime(test_bundle.metadata_test["session_date"]).min()

    # Calculate outcome dates for all remaining validation samples
    # For every surviving validation sample, outcome date must be strictly < test_start
    dates = pd.to_datetime(dataset.metadata["session_date"]).to_numpy()
    val_indices = split.val_indices
    for idx in val_indices:
        outcome_idx = idx + 5
        if outcome_idx < len(dates):
            outcome_date = dates[outcome_idx]
            assert outcome_date < test_start


def test_adversarial_case_c_val_data_cannot_affect_train_preprocessing() -> None:
    """
    Case C: Tampering with validation values must NOT alter training scaler statistics.
    """
    train_df = pd.DataFrame({"close": [100.0, 110.0, 120.0]})
    val_df_poisoned = pd.DataFrame({"close": [100000.0, 999999.0]})

    p1 = TrainOnlyPreprocessor()
    p1.fit(train_df)
    mean_clean = p1.fitted_params["mean"].copy()

    # Even if transformed on poisoned validation data
    p1.transform(val_df_poisoned)
    mean_after = p1.fitted_params["mean"].copy()

    np.testing.assert_array_equal(mean_clean, mean_after)


def test_adversarial_case_d_refitting_on_val_raises_runtime_error() -> None:
    """
    Case D: Attempting to fit on validation data directly triggers RuntimeError.
    """
    train_df = pd.DataFrame({"close": [100.0, 110.0]})
    val_df = pd.DataFrame({"close": [120.0, 130.0]})

    preprocessor = TrainOnlyPreprocessor()
    preprocessor.fit(train_df)

    with pytest.raises(RuntimeError, match="Re-fitting is strictly prohibited"):
        preprocessor.fit(val_df)


def test_adversarial_case_e_split_manifest_changes_when_boundary_moves() -> None:
    """
    Case E: Moving split boundary changes the manifest hash.
    """
    dataset = make_test_dataset(n_days=80)
    bundle1, _, _ = chronological_split(dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    bundle2, _, _ = chronological_split(dataset, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20)

    assert bundle1.manifest_hash != bundle2.manifest_hash


def test_adversarial_case_f_manifest_is_strictly_deterministic() -> None:
    """
    Case F: Running fold generation twice over identical data produces identical hash.
    """
    dataset = make_test_dataset(n_days=80)
    bundle1, _, _ = chronological_split(dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    bundle2, _, _ = chronological_split(dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    assert bundle1.manifest_hash == bundle2.manifest_hash


def test_adversarial_case_g_test_data_quarantined_from_validated_bundle() -> None:
    """
    Case G: ValidatedFoldBundle physically does not contain test attributes.
    """
    dataset = make_test_dataset(n_days=80)
    bundle, test_bundle, _ = chronological_split(dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    assert not hasattr(bundle, "X_test")
    assert not hasattr(bundle, "y_test")
    assert hasattr(test_bundle, "X_test")
    assert hasattr(test_bundle, "y_test")
