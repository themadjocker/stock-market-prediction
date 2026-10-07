from __future__ import annotations

import math

import pandas as pd
import pytest

from ml.features.returns import build_return_features


def test_return_features_have_correct_values() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 4,
            "session_date": pd.date_range(
                "2026-01-01",
                periods=4,
                freq="D",
            ),
            "open": [100.0, 110.0, 100.0, 125.0],
            "high": [112.0, 115.0, 125.0, 130.0],
            "low": [98.0, 95.0, 95.0, 120.0],
            "close": [110.0, 100.0, 125.0, 128.0],
        }
    )

    result = build_return_features(frame)

    assert math.isclose(
        float(result["log_return_1d"].to_numpy(dtype=float)[1]),
        math.log(100.0 / 110.0),
    )

    assert math.isclose(
        float(result["log_return_2d"].to_numpy(dtype=float)[2]),
        math.log(125.0 / 110.0),
    )

    assert math.isclose(
        float(result["intraday_return"].to_numpy(dtype=float)[1]),
        math.log(100.0 / 110.0),
    )

    assert math.isclose(
        float(result["overnight_gap"].to_numpy(dtype=float)[1]),
        math.log(110.0 / 110.0),
    )

    assert math.isclose(
        float(result["high_low_range"].to_numpy(dtype=float)[0]),
        math.log(112.0 / 98.0),
    )


def test_return_features_do_not_cross_ticker_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "BBB"],
            "session_date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-01",
                    "2026-01-02",
                ]
            ),
            "open": [100.0, 110.0, 1000.0, 1200.0],
            "high": [110.0, 115.0, 1210.0, 1250.0],
            "low": [95.0, 95.0, 950.0, 1150.0],
            "close": [110.0, 100.0, 1200.0, 1150.0],
        }
    )

    result = build_return_features(frame)

    # First observation of each ticker cannot have a lagged return
    # or overnight gap supplied by another ticker.
    assert pd.isna(result.loc[0, "log_return_1d"])
    assert pd.isna(result.loc[0, "overnight_gap"])

    assert pd.isna(result.loc[2, "log_return_1d"])
    assert pd.isna(result.loc[2, "overnight_gap"])


def test_return_warmup_is_exact_per_ticker() -> None:
    rows_per_ticker = 25

    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "session_date": pd.date_range(
                        "2026-01-01",
                        periods=rows_per_ticker,
                        freq="D",
                    ),
                    "open": [100.0 + i for i in range(rows_per_ticker)],
                    "high": [102.0 + i for i in range(rows_per_ticker)],
                    "low": [98.0 + i for i in range(rows_per_ticker)],
                    "close": [101.0 + i for i in range(rows_per_ticker)],
                }
            )
            for ticker in ("AAA", "BBB")
        ],
        ignore_index=True,
    )

    result = build_return_features(frame)

    for horizon in (1, 2, 5, 10, 20):
        column = f"log_return_{horizon}d"

        for _, ticker_result in result.groupby("ticker", sort=False):
            values = ticker_result[column].reset_index(drop=True)

            assert values.iloc[:horizon].isna().all()
            assert values.iloc[horizon:].notna().all()


def test_future_values_do_not_change_past_features() -> None:
    base = pd.DataFrame(
        {
            "ticker": ["AAA"] * 5,
            "session_date": pd.date_range(
                "2026-01-01",
                periods=5,
                freq="D",
            ),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [102.0, 103.0, 104.0, 105.0, 106.0],
            "low": [98.0, 99.0, 100.0, 101.0, 102.0],
            "close": [101.0, 102.0, 103.0, 104.0, 105.0],
        }
    )

    changed = base.copy()
    changed.loc[4, "open"] = 5000.0
    changed.loc[4, "high"] = 6000.0
    changed.loc[4, "low"] = 4000.0
    changed.loc[4, "close"] = 5500.0

    base_features = build_return_features(base)
    changed_features = build_return_features(changed)

    feature_columns = [
        "log_return_1d",
        "log_return_2d",
        "log_return_5d",
        "log_return_10d",
        "log_return_20d",
        "intraday_return",
        "overnight_gap",
        "high_low_range",
    ]

    for column in feature_columns:
        pd.testing.assert_series_equal(
            base_features.loc[:3, column],
            changed_features.loc[:3, column],
            check_names=False,
        )


def test_unsorted_data_is_rejected() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "session_date": pd.to_datetime(["2026-01-02", "2026-01-01"]),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [98.0, 99.0],
            "close": [101.0, 102.0],
        }
    )

    with pytest.raises(ValueError, match="chronological"):
        build_return_features(frame)
