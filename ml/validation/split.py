from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.features.dataset import ModelDataset
from ml.validation.manifest import SplitManifest, generate_split_manifest
from ml.validation.preprocessor import TrainOnlyPreprocessor
from ml.validation.purging import (
    apply_embargo,
    protect_validation_horizon,
    purge_train_horizon,
)


@dataclass(frozen=True)
class TemporalSplit:
    """Represents the raw partitioned row indices and dates for a temporal fold."""

    fold_id: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray | None
    train_dates: tuple[pd.Timestamp, pd.Timestamp]
    val_dates: tuple[pd.Timestamp, pd.Timestamp]
    test_dates: tuple[pd.Timestamp, pd.Timestamp] | None
    manifest: SplitManifest


@dataclass(frozen=True)
class ValidatedFoldBundle:
    """
    Immutable data bundle passed from P4 Validation to P5 Modeling.

    Contains exclusively training and validation partitions. Test data is strictly
    quarantined and cannot be accessed through this object.
    """

    fold_id: int
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_val: pd.DataFrame
    y_val: pd.DataFrame
    metadata_train: pd.DataFrame
    metadata_val: pd.DataFrame
    manifest_hash: str
    X_train_raw: pd.DataFrame | None = None
    X_val_raw: pd.DataFrame | None = None
    X_train_context_raw: pd.DataFrame | None = None
    metadata_train_context: pd.DataFrame | None = None


@dataclass(frozen=True)
class QuarantinedTestBundle:
    """
    Quarantined test partition held out strictly for future out-of-sample evaluation (P7).

    Must never be passed to or consumed by P5 model training or validation loops.
    """

    X_test: pd.DataFrame
    y_test: pd.DataFrame
    metadata_test: pd.DataFrame
    manifest_hash: str


def chronological_split(
    dataset: ModelDataset,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    target_horizon: int = 5,
    embargo_sessions: int = 0,
    scaler_type: str = "standard",
) -> tuple[ValidatedFoldBundle, QuarantinedTestBundle | None, TemporalSplit]:
    """
    Partition ModelDataset into strictly chronological, leakage-safe partitions.

    Global panel alignment: Splitting is performed across the unique sorted calendar
    session dates of the entire stock panel to ensure Date D belongs to exactly one partition.

    Target-horizon-aware purging: Purges training samples whose label extends into validation,
    and purges validation samples whose label extends into test (Fix 1).
    """
    if train_ratio <= 0.0 or val_ratio <= 0.0 or test_ratio < 0.0:
        raise ValueError("train_ratio and val_ratio must be positive; test_ratio must be non-negative.")

    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.4f}")

    if target_horizon <= 0:
        raise ValueError(f"target_horizon must be a positive integer, got {target_horizon}")

    if dataset.n_rows == 0:
        raise ValueError("Cannot split an empty ModelDataset.")

    dates = pd.to_datetime(dataset.metadata["session_date"])
    unique_dates = np.sort(dates.unique())
    n_unique_dates = len(unique_dates)

    if n_unique_dates < 3:
        raise ValueError(f"Dataset has only {n_unique_dates} unique trading dates; need at least 3 for splitting.")

    # Compute date split indices
    n_train_dates = int(np.floor(n_unique_dates * train_ratio))
    n_val_dates = int(np.floor(n_unique_dates * val_ratio))
    if test_ratio > 0.0:
        n_val_dates = max(1, n_val_dates)
        n_train_dates = max(1, n_train_dates)
        train_date_slice = unique_dates[:n_train_dates]
        val_date_slice = unique_dates[n_train_dates : n_train_dates + n_val_dates]
        test_date_slice = unique_dates[n_train_dates + n_val_dates :]
        if len(test_date_slice) == 0:
            # Adjust if rounding ate the test window
            test_date_slice = unique_dates[-1:]
            val_date_slice = unique_dates[n_train_dates:-1]
    else:
        train_date_slice = unique_dates[:n_train_dates]
        val_date_slice = unique_dates[n_train_dates:]
        test_date_slice = np.array([], dtype="datetime64[ns]")

    val_start_date = pd.Timestamp(val_date_slice[0])
    test_start_date = pd.Timestamp(test_date_slice[0]) if len(test_date_slice) > 0 else None

    # Map dates to raw row indices
    raw_train_mask = dates.isin(train_date_slice).to_numpy()
    raw_val_mask = dates.isin(val_date_slice).to_numpy()
    raw_test_mask = (
        dates.isin(test_date_slice).to_numpy() if len(test_date_slice) > 0 else np.zeros(len(dates), dtype=bool)
    )

    raw_train_indices = np.flatnonzero(raw_train_mask)
    raw_val_indices = np.flatnonzero(raw_val_mask)
    raw_test_indices = np.flatnonzero(raw_test_mask) if len(test_date_slice) > 0 else None

    n_train_raw = len(raw_train_indices)
    n_val_raw = len(raw_val_indices)

    # 1. Purge train horizon
    train_indices, _ = purge_train_horizon(
        dataset.metadata,
        raw_train_indices,
        val_start_date=val_start_date,
        horizon=target_horizon,
    )

    if len(train_indices) == 0:
        raise ValueError(
            f"All {n_train_raw} training rows were purged by horizon {target_horizon} "
            f"before validation start {val_start_date.strftime('%Y-%m-%d')}."
        )

    # 2. Protect validation horizon against test overlap (Fix 1)
    val_indices, _ = protect_validation_horizon(
        dataset.metadata,
        raw_val_indices,
        test_start_date=test_start_date,
        horizon=target_horizon,
    )

    # 3. Apply embargo post-validation if requested
    if embargo_sessions > 0 and test_start_date is not None and raw_test_indices is not None:
        embargoed_test_indices, _ = apply_embargo(
            dataset.metadata,
            raw_test_indices,
            embargo_start_date=test_start_date,
            embargo_sessions=embargo_sessions,
        )
        test_indices = embargoed_test_indices
    else:
        test_indices = raw_test_indices

    train_dates = (
        pd.Timestamp(train_date_slice[0]),
        pd.Timestamp(train_date_slice[-1]),
    )
    val_dates = (pd.Timestamp(val_date_slice[0]), pd.Timestamp(val_date_slice[-1]))
    test_dates = (
        (pd.Timestamp(test_date_slice[0]), pd.Timestamp(test_date_slice[-1])) if len(test_date_slice) > 0 else None
    )

    split_config = {
        "strategy": "chronological",
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "target_horizon": target_horizon,
        "embargo_sessions": embargo_sessions,
        "scaler_type": scaler_type,
    }

    manifest = generate_split_manifest(
        fold_id=0,
        split_config=split_config,
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        train_indices_raw_count=n_train_raw,
        train_indices_final=train_indices,
        val_indices_raw_count=n_val_raw,
        val_indices_final=val_indices,
        test_indices=test_indices,
    )

    temporal_split = TemporalSplit(
        fold_id=0,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        manifest=manifest,
    )

    # Fit preprocessor strictly on train features
    preprocessor = TrainOnlyPreprocessor(scaler_type=scaler_type)
    X_train_scaled = preprocessor.fit_transform(dataset.X.iloc[train_indices])
    X_val_scaled = preprocessor.transform(dataset.X.iloc[val_indices])

    fold_bundle = ValidatedFoldBundle(
        fold_id=0,
        X_train=X_train_scaled.reset_index(drop=True),
        y_train=dataset.y.iloc[train_indices].reset_index(drop=True),
        X_val=X_val_scaled.reset_index(drop=True),
        y_val=dataset.y.iloc[val_indices].reset_index(drop=True),
        metadata_train=dataset.metadata.iloc[train_indices].reset_index(drop=True),
        metadata_val=dataset.metadata.iloc[val_indices].reset_index(drop=True),
        manifest_hash=manifest.manifest_hash,
        X_train_raw=dataset.X.iloc[train_indices].reset_index(drop=True),
        X_val_raw=dataset.X.iloc[val_indices].reset_index(drop=True),
        X_train_context_raw=dataset.X.iloc[raw_train_indices].reset_index(drop=True),
        metadata_train_context=dataset.metadata.iloc[raw_train_indices].reset_index(drop=True),
    )

    quarantined_test: QuarantinedTestBundle | None = None
    if test_indices is not None and len(test_indices) > 0:
        X_test_scaled = preprocessor.transform(dataset.X.iloc[test_indices])
        quarantined_test = QuarantinedTestBundle(
            X_test=X_test_scaled.reset_index(drop=True),
            y_test=dataset.y.iloc[test_indices].reset_index(drop=True),
            metadata_test=dataset.metadata.iloc[test_indices].reset_index(drop=True),
            manifest_hash=manifest.manifest_hash,
        )

    return fold_bundle, quarantined_test, temporal_split


def walk_forward_split(
    dataset: ModelDataset,
    *,
    min_train_sessions: int = 60,
    val_sessions: int = 20,
    step_sessions: int | None = None,
    target_horizon: int = 5,
    embargo_sessions: int = 0,
    expanding: bool = True,
    scaler_type: str = "standard",
) -> list[tuple[ValidatedFoldBundle, TemporalSplit]]:
    """
    Generate sequential, leakage-safe walk-forward folds (expanding or rolling).

    Every fold independently enforces target-horizon purging and train-only preprocessing.
    """
    if min_train_sessions <= 0 or val_sessions <= 0:
        raise ValueError("min_train_sessions and val_sessions must be positive integers.")

    if target_horizon <= 0:
        raise ValueError("target_horizon must be a positive integer.")

    step = val_sessions if step_sessions is None else step_sessions
    if step <= 0:
        raise ValueError("step_sessions must be a positive integer.")
    if step < val_sessions:
        raise ValueError(
            "step_sessions must be greater than or equal to val_sessions "
            "so walk-forward validation windows do not overlap."
        )

    dates = pd.to_datetime(dataset.metadata["session_date"])
    unique_dates = np.sort(dates.unique())
    n_unique_dates = len(unique_dates)

    required_sessions = min_train_sessions + target_horizon + val_sessions
    if n_unique_dates < required_sessions:
        raise ValueError(
            f"Insufficient unique session dates ({n_unique_dates}) for walk-forward validation. "
            f"Requires at least {required_sessions} sessions."
        )

    folds: list[tuple[ValidatedFoldBundle, TemporalSplit]] = []
    current_train_end_idx = min_train_sessions
    fold_id = 0
    previous_val_end_idx: int | None = None

    while current_train_end_idx + val_sessions <= n_unique_dates:
        if expanding:
            train_date_slice = unique_dates[:current_train_end_idx]
        else:
            start_idx = max(0, current_train_end_idx - min_train_sessions)
            train_date_slice = unique_dates[start_idx:current_train_end_idx]

        val_date_slice = unique_dates[current_train_end_idx : current_train_end_idx + val_sessions]
        val_start_date = pd.Timestamp(val_date_slice[0])

        # If a subsequent fold exists, the next fold start date is next_val_start
        next_val_start_date = (
            pd.Timestamp(unique_dates[current_train_end_idx + val_sessions])
            if current_train_end_idx + val_sessions < n_unique_dates
            else None
        )

        raw_train_mask = dates.isin(train_date_slice).to_numpy()
        raw_val_mask = dates.isin(val_date_slice).to_numpy()

        raw_train_indices = np.flatnonzero(raw_train_mask)
        raw_val_indices = np.flatnonzero(raw_val_mask)

        # 1. Apply the temporal embargo from the previous validation window.
        # The embargo protects the observations immediately following the previous
        # validation period from becoming training data in this fold.
        if embargo_sessions > 0 and previous_val_end_idx is not None:
            embargo_start_idx = previous_val_end_idx + 1
            if embargo_start_idx < n_unique_dates:
                embargo_start_date = pd.Timestamp(unique_dates[embargo_start_idx])
                embargo_candidate_indices, embargoed_count = apply_embargo(
                    dataset.metadata,
                    raw_train_indices,
                    embargo_start_date=embargo_start_date,
                    embargo_sessions=embargo_sessions,
                )
            else:
                embargo_candidate_indices = raw_train_indices
                embargoed_count = 0
        else:
            embargo_candidate_indices = raw_train_indices
            embargoed_count = 0

        # 2. Purge train horizon independently from the embargo.
        train_indices, horizon_purged_count = purge_train_horizon(
            dataset.metadata,
            embargo_candidate_indices,
            val_start_date=val_start_date,
            horizon=target_horizon,
        )

        if len(train_indices) == 0:
            raise ValueError(f"Fold {fold_id}: all training rows were purged by horizon {target_horizon}.")

        # 2. Protect validation horizon if subsequent fold follows (Fix 1 & 2)
        val_indices, _ = protect_validation_horizon(
            dataset.metadata,
            raw_val_indices,
            test_start_date=next_val_start_date,
            horizon=target_horizon,
        )

        train_dates = (
            pd.Timestamp(train_date_slice[0]),
            pd.Timestamp(train_date_slice[-1]),
        )
        val_dates = (pd.Timestamp(val_date_slice[0]), pd.Timestamp(val_date_slice[-1]))

        split_config = {
            "strategy": "walk_forward",
            "expanding": expanding,
            "min_train_sessions": min_train_sessions,
            "val_sessions": val_sessions,
            "step_sessions": step,
            "target_horizon": target_horizon,
            "embargo_sessions": embargo_sessions,
            "scaler_type": scaler_type,
        }

        manifest = generate_split_manifest(
            fold_id=fold_id,
            split_config={
                **split_config,
                "train_horizon_purged_count": horizon_purged_count,
                "train_embargoed_count": embargoed_count,
            },
            train_dates=train_dates,
            val_dates=val_dates,
            test_dates=None,
            train_indices_raw_count=len(raw_train_indices),
            train_indices_final=train_indices,
            val_indices_raw_count=len(raw_val_indices),
            val_indices_final=val_indices,
            test_indices=None,
        )

        temporal_split = TemporalSplit(
            fold_id=fold_id,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=None,
            train_dates=train_dates,
            val_dates=val_dates,
            test_dates=None,
            manifest=manifest,
        )

        # Fit preprocessor strictly on this fold's train data
        preprocessor = TrainOnlyPreprocessor(scaler_type=scaler_type)
        X_train_scaled = preprocessor.fit_transform(dataset.X.iloc[train_indices])
        X_val_scaled = preprocessor.transform(dataset.X.iloc[val_indices])

        fold_bundle = ValidatedFoldBundle(
            fold_id=fold_id,
            X_train=X_train_scaled.reset_index(drop=True),
            y_train=dataset.y.iloc[train_indices].reset_index(drop=True),
            X_val=X_val_scaled.reset_index(drop=True),
            y_val=dataset.y.iloc[val_indices].reset_index(drop=True),
            metadata_train=dataset.metadata.iloc[train_indices].reset_index(drop=True),
            metadata_val=dataset.metadata.iloc[val_indices].reset_index(drop=True),
            manifest_hash=manifest.manifest_hash,
            X_train_raw=dataset.X.iloc[train_indices].reset_index(drop=True),
            X_val_raw=dataset.X.iloc[val_indices].reset_index(drop=True),
            X_train_context_raw=dataset.X.iloc[raw_train_indices].reset_index(drop=True),
            metadata_train_context=dataset.metadata.iloc[raw_train_indices].reset_index(drop=True),
        )

        folds.append((fold_bundle, temporal_split))
        previous_val_end_idx = current_train_end_idx + val_sessions - 1
        fold_id += 1
        current_train_end_idx += step

    return folds
