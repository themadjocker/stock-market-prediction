from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.models.sequence import construct_lookback_sequences


def test_construct_lookback_sequences_single_series() -> None:
    X = pd.DataFrame({"f1": np.arange(10, dtype=float), "f2": np.arange(10, 20, dtype=float)})
    lookback = 4

    seqs, valid_idx = construct_lookback_sequences(X, lookback=lookback)

    # 10 rows with lookback 4 -> 10 - 4 + 1 = 7 sequences
    assert seqs.shape == (7, 4, 2)
    assert len(valid_idx) == 7
    # First sequence ends at index 3
    assert valid_idx[0] == 3
    np.testing.assert_array_equal(seqs[0, :, 0], [0.0, 1.0, 2.0, 3.0])
    # Last sequence ends at index 9
    assert valid_idx[-1] == 9
    np.testing.assert_array_equal(seqs[-1, :, 0], [6.0, 7.0, 8.0, 9.0])


def test_construct_lookback_sequences_never_crosses_ticker_boundaries() -> None:
    dates = pd.date_range("2026-01-01", periods=6, freq="B")
    metadata = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "session_date": list(dates[:3]) + list(dates[:3]),
        }
    )
    X = pd.DataFrame({"f1": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0]})
    lookback = 2

    seqs, valid_idx = construct_lookback_sequences(X, metadata=metadata, lookback=lookback)

    # For AAA: 3 rows, lookback 2 -> 2 seqs (indices 1, 2)
    # For BBB: 3 rows, lookback 2 -> 2 seqs (indices 4, 5)
    # Total = 4 sequences
    assert seqs.shape == (4, 2, 1)
    assert list(valid_idx) == [1, 2, 4, 5]

    # First sequence for BBB (orig_idx 4) MUST NOT contain any AAA values!
    bbb_seq_0 = seqs[2, :, 0]
    np.testing.assert_array_equal(bbb_seq_0, [20.0, 21.0])
    assert 12.0 not in bbb_seq_0


def test_lookback_invariant_max_date_is_prediction_date() -> None:
    dates = pd.date_range("2026-01-01", periods=10, freq="B")
    metadata = pd.DataFrame({"ticker": ["AAA"] * 10, "session_date": dates})
    X = pd.DataFrame({"f1": np.arange(10, dtype=float)})
    lookback = 3

    seqs, valid_idx = construct_lookback_sequences(X, metadata=metadata, lookback=lookback)

    for i, orig_idx in enumerate(valid_idx):
        prediction_date = dates[orig_idx]
        seq_values = seqs[i, :, 0]
        # Invariant: the final observation in the sequence is row orig_idx
        assert seq_values[-1] == X.iloc[orig_idx]["f1"]
        # No value in the sequence corresponds to any date > prediction_date
        assert orig_idx >= lookback - 1


def test_validation_endpoints_can_use_pre_validation_history_without_future_rows() -> None:
    dates = pd.date_range("2026-01-01", periods=8, freq="B")
    metadata = pd.DataFrame({"ticker": ["AAA"] * 8, "session_date": dates})
    X = pd.DataFrame({"f1": np.arange(8, dtype=float)})

    # D1..D5 are training history; D6..D8 are validation endpoints.
    endpoints = np.arange(5, 8, dtype=int)
    seqs, valid_idx = construct_lookback_sequences(
        X,
        metadata=metadata,
        lookback=3,
        endpoint_indices=endpoints,
    )

    assert list(valid_idx) == [5, 6, 7]
    np.testing.assert_array_equal(seqs[0, :, 0], [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(seqs[1, :, 0], [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(seqs[2, :, 0], [5.0, 6.0, 7.0])


def test_endpoint_indices_never_allow_future_rows() -> None:
    dates = pd.date_range("2026-01-01", periods=6, freq="B")
    metadata = pd.DataFrame({"ticker": ["AAA"] * 6, "session_date": dates})
    X = pd.DataFrame({"f1": np.arange(6, dtype=float)})

    seqs, valid_idx = construct_lookback_sequences(
        X,
        metadata=metadata,
        lookback=3,
        endpoint_indices=np.array([3, 4]),
    )

    assert list(valid_idx) == [3, 4]
    assert all(seq[-1, 0] == X.iloc[idx, 0] for seq, idx in zip(seqs, valid_idx))
    assert not any(4.0 in seq[:, 0] for seq in [seqs[0]])


def test_metadata_length_mismatch_is_rejected() -> None:
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    metadata = pd.DataFrame({"ticker": ["AAA", "AAA"]})
    with pytest.raises(ValueError, match="same number of rows"):
        construct_lookback_sequences(X, metadata=metadata, lookback=2)
