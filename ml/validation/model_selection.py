from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.validation.purging import purge_train_horizon


@dataclass(frozen=True)
class InnerSelectionSplit:
    """Chronological inner split used only for model-selection controls such as early stopping."""

    fit_indices: np.ndarray
    selection_indices: np.ndarray
    selection_start: pd.Timestamp
    raw_fit_count: int
    purged_fit_count: int


def make_inner_selection_split(
    metadata: pd.DataFrame,
    *,
    target_horizon: int,
    selection_fraction: float = 0.20,
    min_selection_sessions: int = 5,
) -> InnerSelectionSplit:
    """Create a chronological fit/selection split inside an already validated outer-train partition.

    The selection window is the latest contiguous global session-date block. Training rows whose
    forward label resolves on/after the selection boundary are purged, so the inner selection set
    is not used to train the model through overlapping targets.
    """
    if metadata.empty:
        raise ValueError("Cannot create an inner selection split from empty metadata.")
    if target_horizon <= 0:
        raise ValueError("target_horizon must be positive.")
    if not 0.0 < selection_fraction < 0.5:
        raise ValueError("selection_fraction must be between 0 and 0.5.")
    if min_selection_sessions <= 0:
        raise ValueError("min_selection_sessions must be positive.")

    dates = pd.to_datetime(metadata["session_date"])
    unique_dates = np.sort(dates.unique())
    n_dates = len(unique_dates)
    selection_sessions = max(min_selection_sessions, int(np.ceil(n_dates * selection_fraction)))
    if n_dates <= selection_sessions + 2:
        raise ValueError(
            f"Insufficient sessions ({n_dates}) for an inner selection holdout of {selection_sessions} sessions."
        )

    selection_start_idx = n_dates - selection_sessions
    selection_start = pd.Timestamp(unique_dates[selection_start_idx])

    raw_fit_indices = np.flatnonzero(dates < selection_start)
    selection_indices = np.flatnonzero(dates >= selection_start)
    fit_indices, _ = purge_train_horizon(
        metadata,
        raw_fit_indices,
        val_start_date=selection_start,
        horizon=target_horizon,
    )

    if len(fit_indices) == 0:
        raise ValueError("All inner training rows were purged before the selection boundary.")
    if len(selection_indices) == 0:
        raise ValueError("Inner selection split is empty.")

    return InnerSelectionSplit(
        fit_indices=fit_indices,
        selection_indices=selection_indices,
        selection_start=selection_start,
        raw_fit_count=len(raw_fit_indices),
        purged_fit_count=len(raw_fit_indices) - len(fit_indices),
    )


def make_p6_inner_folds(
    metadata: pd.DataFrame,
    *,
    target_horizon: int,
    min_selection_sessions: int = 5,
) -> list[InnerSelectionSplit]:
    """Create the 2 chronological inner folds for P6 Optuna tuning.

    Fold 1: 60% Inner Fit, 20% Inner Selection.
    Fold 2: 80% Inner Fit, 20% Inner Selection.
    """
    if metadata.empty:
        raise ValueError("Cannot create inner folds from empty metadata.")

    dates = pd.to_datetime(metadata["session_date"])
    unique_dates = np.sort(dates.unique())
    n_dates = len(unique_dates)

    # 20% selection
    selection_sessions = max(min_selection_sessions, int(np.ceil(n_dates * 0.20)))

    folds = []
    # Fold 1: ends at 80% mark, selection is 60%-80%
    # Fold 2: ends at 100% mark, selection is 80%-100%

    # Fold 1 geometry:
    f1_end_idx = int(np.floor(n_dates * 0.80))
    if f1_end_idx <= selection_sessions + 2:
        raise ValueError("Insufficient sessions for Fold 1 geometry.")

    f1_selection_start_idx = f1_end_idx - selection_sessions
    f1_selection_start = pd.Timestamp(unique_dates[f1_selection_start_idx])
    f1_end_date = pd.Timestamp(unique_dates[f1_end_idx])

    f1_raw_fit = np.flatnonzero(dates < f1_selection_start)
    f1_sel = np.flatnonzero((dates >= f1_selection_start) & (dates < f1_end_date))

    f1_fit, _ = purge_train_horizon(
        metadata,
        f1_raw_fit,
        val_start_date=f1_selection_start,
        horizon=target_horizon,
    )

    folds.append(
        InnerSelectionSplit(
            fit_indices=f1_fit,
            selection_indices=f1_sel,
            selection_start=f1_selection_start,
            raw_fit_count=len(f1_raw_fit),
            purged_fit_count=len(f1_raw_fit) - len(f1_fit),
        )
    )

    # Fold 2 geometry: (matches Protocol A)
    f2_selection_start_idx = n_dates - selection_sessions
    f2_selection_start = pd.Timestamp(unique_dates[f2_selection_start_idx])

    f2_raw_fit = np.flatnonzero(dates < f2_selection_start)
    f2_sel = np.flatnonzero(dates >= f2_selection_start)

    f2_fit, _ = purge_train_horizon(
        metadata,
        f2_raw_fit,
        val_start_date=f2_selection_start,
        horizon=target_horizon,
    )

    folds.append(
        InnerSelectionSplit(
            fit_indices=f2_fit,
            selection_indices=f2_sel,
            selection_start=f2_selection_start,
            raw_fit_count=len(f2_raw_fit),
            purged_fit_count=len(f2_raw_fit) - len(f2_fit),
        )
    )

    return folds
