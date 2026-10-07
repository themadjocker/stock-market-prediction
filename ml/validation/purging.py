from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_label_outcome_dates(
    metadata: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    """
    Calculate the actual session date on which the forward horizon h resolves.

    Uses actual ticker-specific trading sessions rather than calendar day offsets.
    If t + h exceeds the available sessions for that ticker, the outcome date is NaT.
    """
    if horizon <= 0:
        raise ValueError(f"Target horizon must be a positive integer, got {horizon}")

    if not {"ticker", "session_date"}.issubset(metadata.columns):
        raise ValueError("metadata must contain 'ticker' and 'session_date'")

    dates = pd.to_datetime(metadata["session_date"])
    outcome_dates = pd.Series(index=metadata.index, dtype="datetime64[ns]")

    frame = pd.DataFrame(
        {
            "ticker": metadata["ticker"],
            "session_date": dates,
            "orig_idx": metadata.index,
        }
    )

    for _, ticker_group in frame.groupby("ticker", sort=False):
        sorted_group = ticker_group.sort_values("session_date")
        # Shift -horizon to get the date h sessions into the future
        future_dates = sorted_group["session_date"].shift(-horizon)
        outcome_dates.loc[sorted_group["orig_idx"]] = future_dates.to_numpy()

    return outcome_dates


def purge_train_horizon(
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    val_start_date: pd.Timestamp,
    horizon: int,
) -> tuple[np.ndarray, int]:
    """
    Purge training observations whose forward label outcome falls on or after val_start_date.

    Any training sample with outcome_date >= val_start_date or unresolved within
    the pre-validation history is purged to prevent forward label leakage into validation.
    """
    if len(train_indices) == 0:
        return np.array([], dtype=int), 0

    train_meta = metadata.iloc[train_indices]
    outcome_dates = calculate_label_outcome_dates(train_meta, horizon)

    val_start = pd.to_datetime(val_start_date)

    # A sample is safe ONLY if its outcome date is strictly before val_start_date
    safe_mask = outcome_dates.notna() & (outcome_dates < val_start)
    safe_indices = train_indices[safe_mask.to_numpy()]
    purged_count = int(len(train_indices) - len(safe_indices))

    return safe_indices, purged_count


def protect_validation_horizon(
    metadata: pd.DataFrame,
    val_indices: np.ndarray,
    test_start_date: pd.Timestamp | None,
    horizon: int,
) -> tuple[np.ndarray, int]:
    """
    Protect the validation-to-test boundary.

    Purges validation samples whose label outcome falls on or after test_start_date.
    Ensures validation ground truth does not depend on future observations occurring
    inside the final test period.
    """
    if test_start_date is None or len(val_indices) == 0:
        return val_indices, 0

    val_meta = metadata.iloc[val_indices]
    outcome_dates = calculate_label_outcome_dates(val_meta, horizon)

    test_start = pd.to_datetime(test_start_date)

    safe_mask = outcome_dates.notna() & (outcome_dates < test_start)
    safe_indices = val_indices[safe_mask.to_numpy()]
    purged_count = int(len(val_indices) - len(safe_indices))

    return safe_indices, purged_count


def apply_embargo(
    metadata: pd.DataFrame,
    candidate_indices: np.ndarray,
    embargo_start_date: pd.Timestamp,
    embargo_sessions: int,
) -> tuple[np.ndarray, int]:
    """
    Exclude trading sessions immediately following embargo_start_date.

    The embargo spans exactly embargo_sessions unique calendar market dates starting at
    or immediately following embargo_start_date.
    """
    if embargo_sessions <= 0 or len(candidate_indices) == 0:
        return candidate_indices, 0

    start_date = pd.to_datetime(embargo_start_date)
    dates = pd.to_datetime(metadata["session_date"])

    # Global unique trading dates
    unique_dates = np.sort(dates.unique())
    after_dates = unique_dates[unique_dates >= start_date]

    if len(after_dates) == 0:
        return candidate_indices, 0

    embargo_window = set(after_dates[:embargo_sessions])

    candidate_dates = dates.iloc[candidate_indices]
    keep_mask = ~candidate_dates.isin(embargo_window).to_numpy()

    embargoed_indices = candidate_indices[keep_mask]
    purged_count = int(len(candidate_indices) - len(embargoed_indices))

    return embargoed_indices, purged_count
