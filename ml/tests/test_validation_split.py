from __future__ import annotations

import pandas as pd

from ml.features.dataset import ModelDataset, build_model_dataset
from ml.validation.split import (
    QuarantinedTestBundle,
    ValidatedFoldBundle,
    chronological_split,
    walk_forward_split,
)


def make_panel_dataset(tickers: list[str] | None = None, n_days: int = 100) -> ModelDataset:
    if tickers is None:
        tickers = ["AAA", "BBB"]
    dates = pd.date_range("2026-01-01", periods=n_days, freq="B")
    rows = []
    for ticker in tickers:
        base_price = 100.0 if ticker == "AAA" else 50.0
        for i, d in enumerate(dates):
            rows.append(
                {
                    "ticker": ticker,
                    "session_date": d,
                    "close": base_price + i * 0.5,
                    "volume": 1000.0 + i * 10,
                    "target_return_1d": 0.005,
                    "target_return_5d": 0.02,
                }
            )
    df = pd.DataFrame(rows)
    return build_model_dataset(
        df,
        feature_columns=["close", "volume"],
        target_columns=["target_return_1d", "target_return_5d"],
    )


def test_chronological_split_global_panel_alignment() -> None:
    dataset = make_panel_dataset(["AAA", "BBB"], n_days=100)
    bundle, test_bundle, split = chronological_split(
        dataset,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        target_horizon=5,
    )

    assert isinstance(bundle, ValidatedFoldBundle)
    assert isinstance(test_bundle, QuarantinedTestBundle)

    # Invariant: Strict global chronology
    train_dates = pd.to_datetime(bundle.metadata_train["session_date"])
    val_dates = pd.to_datetime(bundle.metadata_val["session_date"])
    test_dates = pd.to_datetime(test_bundle.metadata_test["session_date"])

    assert train_dates.max() < val_dates.min()
    assert val_dates.max() < test_dates.min()

    # Panel alignment: No date exists in both train and val
    assert set(train_dates).isdisjoint(set(val_dates))
    assert set(val_dates).isdisjoint(set(test_dates))

    # All tickers present in train and val
    assert set(bundle.metadata_train["ticker"].unique()) == {"AAA", "BBB"}
    assert set(bundle.metadata_val["ticker"].unique()) == {"AAA", "BBB"}


def test_walk_forward_split_expanding() -> None:
    dataset = make_panel_dataset(["AAA"], n_days=120)
    folds = walk_forward_split(
        dataset,
        min_train_sessions=60,
        val_sessions=20,
        step_sessions=20,
        target_horizon=5,
        expanding=True,
    )

    assert len(folds) >= 2
    prev_train_len = 0
    for fold_id, (bundle, split) in enumerate(folds):
        assert bundle.fold_id == fold_id
        train_dates = pd.to_datetime(bundle.metadata_train["session_date"])
        val_dates = pd.to_datetime(bundle.metadata_val["session_date"])

        # Strict chronology per fold
        assert train_dates.max() < val_dates.min()

        # Expanding window property: train size strictly increases
        assert len(bundle.X_train) > prev_train_len
        prev_train_len = len(bundle.X_train)


def test_walk_forward_embargo_is_enforced_in_next_fold_train() -> None:
    dataset = make_panel_dataset(["AAA"], n_days=140)
    folds = walk_forward_split(
        dataset,
        min_train_sessions=60,
        val_sessions=10,
        step_sessions=15,
        target_horizon=1,
        embargo_sessions=2,
        expanding=True,
    )

    assert len(folds) >= 2

    # Fold 0: validation ends at session 69.
    # Fold 1 training extends through session 74 because step_sessions=15,
    # so sessions 70 and 71 are the post-validation gap. The 2-session
    # embargo must remove those dates from Fold 1 training.
    _, fold0 = folds[0]
    fold1_bundle, fold1 = folds[1]

    fold0_val_end = fold0.val_dates[1]
    fold1_train_dates = pd.to_datetime(fold1_bundle.metadata_train["session_date"])

    expected_embargo_dates = pd.date_range(
        start=fold0_val_end + pd.offsets.BDay(1),
        periods=2,
        freq="B",
    )

    assert all(date not in set(fold1_train_dates) for date in expected_embargo_dates)
    assert fold1.manifest.split_config["train_embargoed_count"] == 2


def test_walk_forward_embargo_zero_preserves_training_membership() -> None:
    dataset = make_panel_dataset(["AAA"], n_days=140)

    folds_without_embargo = walk_forward_split(
        dataset,
        min_train_sessions=60,
        val_sessions=10,
        step_sessions=15,
        target_horizon=1,
        embargo_sessions=0,
        expanding=True,
    )
    folds_with_zero = walk_forward_split(
        dataset,
        min_train_sessions=60,
        val_sessions=10,
        step_sessions=15,
        target_horizon=1,
        embargo_sessions=0,
        expanding=True,
    )

    assert len(folds_without_embargo) == len(folds_with_zero)
    for (bundle_a, split_a), (bundle_b, split_b) in zip(folds_without_embargo, folds_with_zero):
        assert split_a.train_indices.tolist() == split_b.train_indices.tolist()
        assert split_a.val_indices.tolist() == split_b.val_indices.tolist()
        assert split_a.manifest.manifest_hash == split_b.manifest.manifest_hash
