from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features.dataset import (
    ModelDataset,
    build_model_dataset,
    validate_information_cutoff,
)


def make_data(rows: int = 8) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "ticker": ["AAA"] * rows,
            "session_date": dates,
            "close": np.arange(100.0, 100.0 + rows),
            "sma_20": np.arange(101.0, 101.0 + rows),
            "rsi_14": np.linspace(40.0, 55.0, rows),
            "market_return_1d": np.linspace(-0.01, 0.01, rows),
            "sector_return_1d": np.linspace(-0.02, 0.02, rows),
            "target_return_1d": np.linspace(-0.01, 0.03, rows),
            "target_up_1d": [0, 0, 1, 1, 1, 0, 1, 1],
        }
    )


FEATURES = ("close", "sma_20", "rsi_14", "market_return_1d", "sector_return_1d")
TARGETS = ("target_return_1d", "target_up_1d")


def test_builds_expected_model_dataset() -> None:
    result = build_model_dataset(make_data(), feature_columns=FEATURES, target_columns=TARGETS)
    assert isinstance(result, ModelDataset)
    assert result.feature_columns == FEATURES
    assert result.target_columns == TARGETS
    assert result.n_rows == 8
    assert list(result.metadata.columns) == ["ticker", "session_date"]


def test_preserves_metadata_row_alignment() -> None:
    data = make_data()
    result = build_model_dataset(data, feature_columns=FEATURES, target_columns=TARGETS)
    pd.testing.assert_series_equal(result.metadata["session_date"], data["session_date"], check_names=False)
    pd.testing.assert_series_equal(result.X["close"], data["close"], check_names=False)


def test_missing_feature_rows_are_dropped() -> None:
    data = make_data()
    data.loc[2, "rsi_14"] = np.nan
    result = build_model_dataset(data, feature_columns=FEATURES, target_columns=TARGETS)
    assert len(result.X) == 7
    assert pd.Timestamp("2026-01-05") not in set(result.metadata["session_date"])


def test_missing_target_rows_are_dropped() -> None:
    data = make_data()
    data.loc[5, "target_return_1d"] = np.nan
    result = build_model_dataset(data, feature_columns=FEATURES, target_columns=TARGETS)
    assert len(result.X) == 7


def test_can_keep_incomplete_rows_for_diagnostics() -> None:
    data = make_data()
    data.loc[2, "rsi_14"] = np.nan
    result = build_model_dataset(
        data,
        feature_columns=FEATURES,
        target_columns=TARGETS,
        drop_incomplete=False,
    )
    assert len(result.X) == 8
    assert result.X.loc[2, "rsi_14"] != result.X.loc[2, "rsi_14"]


def test_feature_target_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        build_model_dataset(
            make_data(),
            feature_columns=FEATURES + ("target_return_1d",),
            target_columns=TARGETS,
        )


def test_target_like_feature_is_rejected() -> None:
    with pytest.raises(ValueError, match="Target-like"):
        build_model_dataset(
            make_data(),
            feature_columns=("close", "target_up_1d"),
            target_columns=("target_return_1d",),
        )


def test_missing_columns_are_rejected() -> None:
    data = make_data().drop(columns=["rsi_14"])
    with pytest.raises(ValueError, match="Missing required columns"):
        build_model_dataset(data, feature_columns=FEATURES, target_columns=TARGETS)


def test_duplicate_ticker_date_is_rejected() -> None:
    data = pd.concat([make_data(), make_data().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        build_model_dataset(data, feature_columns=FEATURES, target_columns=TARGETS)


def test_unsorted_rows_per_ticker_are_rejected() -> None:
    data = make_data().iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="chronological"):
        build_model_dataset(data, feature_columns=FEATURES, target_columns=TARGETS)


def test_non_numeric_feature_is_rejected() -> None:
    data = make_data()
    data["rsi_14"] = "bad"
    with pytest.raises(ValueError, match="must be numeric"):
        build_model_dataset(data, feature_columns=FEATURES, target_columns=TARGETS)


def test_infinite_feature_is_rejected() -> None:
    data = make_data()
    data.loc[0, "sma_20"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        build_model_dataset(data, feature_columns=FEATURES, target_columns=TARGETS)


def test_information_at_cutoff_is_allowed() -> None:
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    data = pd.DataFrame({"as_of_timestamp": dates, "information_timestamp": dates})
    validate_information_cutoff(data)


def test_future_information_is_rejected() -> None:
    data = pd.DataFrame(
        {
            "as_of_timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "information_timestamp": pd.to_datetime(["2026-01-01", "2026-01-03"]),
        }
    )
    with pytest.raises(ValueError, match="Future information detected"):
        validate_information_cutoff(data)
