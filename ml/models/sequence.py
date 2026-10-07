from __future__ import annotations

import numpy as np
import pandas as pd


def construct_lookback_sequences(
    X: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    lookback: int = 10,
    endpoint_indices: np.ndarray | list[int] | None = None,
    max_lookback: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct backward-only [B, L, F] sequences without crossing ticker boundaries.

    ``endpoint_indices`` optionally restricts which rows are prediction endpoints.
    This is used at validation time with ``train + validation`` history so validation
    predictions can use legitimate pre-validation context without seeing future rows.

    Returned indices always refer to the row positions in ``X``.
    """
    if lookback <= 0:
        raise ValueError(f"lookback must be a positive integer, got {lookback}")
    if metadata is not None and len(X) != len(metadata):
        raise ValueError("X and metadata must contain the same number of rows.")
    if X.empty:
        return np.empty((0, lookback, X.shape[1]), dtype=float), np.array([], dtype=int)

    n_rows, n_features = X.shape
    X_vals = X.to_numpy(dtype=float)

    if endpoint_indices is None:
        requested = np.arange(n_rows, dtype=int)
    else:
        requested = np.asarray(endpoint_indices, dtype=int)
        if requested.ndim != 1:
            raise ValueError("endpoint_indices must be one-dimensional.")
        if np.any(requested < 0) or np.any(requested >= n_rows):
            raise IndexError("endpoint_indices contains an out-of-range row index.")
        requested_set = set(requested.tolist())

    effective_max_lookback = max_lookback if max_lookback is not None else lookback
    if lookback > effective_max_lookback:
        raise ValueError(f"lookback ({lookback}) cannot be greater than max_lookback ({effective_max_lookback})")

    if metadata is None or "ticker" not in metadata.columns:
        if n_rows < effective_max_lookback:
            return np.empty((0, lookback, n_features), dtype=float), np.array([], dtype=int)
        sequences: list[np.ndarray] = []
        valid: list[int] = []
        for end_idx in requested:
            # Sequence must end at end_idx and have length `lookback`.
            # But it can only be constructed if the data has enough history for `effective_max_lookback`.
            req_start_idx = int(end_idx) - effective_max_lookback + 1
            if req_start_idx < 0:
                continue
            start_idx = int(end_idx) - lookback + 1
            sequences.append(X_vals[start_idx : int(end_idx) + 1])
            valid.append(int(end_idx))
        if not sequences:
            return np.empty((0, lookback, n_features), dtype=float), np.array([], dtype=int)
        return np.asarray(sequences, dtype=float), np.asarray(valid, dtype=int)

    metadata = metadata.reset_index(drop=True)
    if "session_date" in metadata.columns:
        session_dates = pd.to_datetime(metadata["session_date"])
        if session_dates.isna().any():
            raise ValueError("metadata.session_date contains invalid/missing timestamps.")
    else:
        session_dates = None

    requested_set = set(requested.tolist())
    sequences_list: list[np.ndarray] = []
    indices_list: list[int] = []

    frame = pd.DataFrame(
        {
            "ticker": metadata["ticker"].astype(str).to_numpy(),
            "orig_idx": np.arange(n_rows),
        }
    )

    for ticker, group in frame.groupby("ticker", sort=False):
        group_indices = group["orig_idx"].to_numpy(dtype=int)

        if session_dates is not None:
            group_dates = session_dates.iloc[group_indices]
            if not group_dates.is_monotonic_increasing:
                raise ValueError(f"Metadata for ticker '{ticker}' must be sorted chronologically.")

        for position in range(effective_max_lookback - 1, len(group_indices)):
            end_idx = int(group_indices[position])
            if end_idx not in requested_set:
                continue
            window_indices = group_indices[position - lookback + 1 : position + 1]
            if session_dates is not None:
                endpoint_date = session_dates.iloc[end_idx]
                if session_dates.iloc[window_indices].gt(endpoint_date).any():
                    raise ValueError("Sequence contains an observation after its prediction endpoint.")
            sequences_list.append(X_vals[window_indices])
            indices_list.append(end_idx)

    if not sequences_list:
        return np.empty((0, lookback, n_features), dtype=float), np.array([], dtype=int)

    order = np.argsort(np.asarray(indices_list, dtype=int))
    sequence_array = np.asarray(sequences_list, dtype=float)[order]
    valid_indices = np.asarray(indices_list, dtype=int)[order]
    return sequence_array, valid_indices
