from __future__ import annotations

import math

import pandas as pd
import pytest

from ml.features.volatility import build_volatility_volume_features


def test_volatility_and_volume_features_have_expected_values() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 6,
            "session_date": pd.date_range(
                "2026-01-01",
                periods=6,
                freq="D",
            ),
            "close": [
                100.0,
                110.0,
                99.0,
                108.0,
                120.0,
                114.0,
            ],
            "volume": [
                1000.0,
                1100.0,
                900.0,
                1200.0,
                1500.0,
                1300.0,
            ],
        }
    )

    result = build_volatility_volume_features(
        frame,
        volatility_windows=(3,),
        volume_windows=(3,),
    )

    daily_returns = [
        math.log(110.0 / 100.0),
        math.log(99.0 / 110.0),
        math.log(108.0 / 99.0),
    ]

    expected_volatility = pd.Series(daily_returns).std()

    assert math.isclose(
        float(result["volatility_3d"].to_numpy(dtype=float)[3]),
        expected_volatility,
    )

    expected_volume_mean = (1100.0 + 900.0 + 1200.0) / 3.0

    assert math.isclose(
        float(result["volume_sma_3d"].to_numpy(dtype=float)[3]),
        expected_volume_mean,
    )

    expected_volume_zscore = (1200.0 - expected_volume_mean) / pd.Series([1100.0, 900.0, 1200.0]).std()

    assert math.isclose(
        float(result["volume_zscore_3d"].to_numpy(dtype=float)[3]),
        expected_volume_zscore,
    )


def test_volatility_does_not_cross_ticker_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "ticker": [
                "AAA",
                "AAA",
                "AAA",
                "AAA",
                "BBB",
                "BBB",
                "BBB",
                "BBB",
            ],
            "session_date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                ]
            ),
            "close": [
                100.0,
                101.0,
                102.0,
                103.0,
                1000.0,
                1100.0,
                1200.0,
                1300.0,
            ],
            "volume": [
                1000.0,
                1100.0,
                1200.0,
                1300.0,
                10000.0,
                11000.0,
                12000.0,
                13000.0,
            ],
        }
    )

    result = build_volatility_volume_features(
        frame,
        volatility_windows=(3,),
        volume_windows=(3,),
    )

    assert pd.isna(result.loc[0, "volatility_3d"])
    assert pd.isna(result.loc[4, "volatility_3d"])

    assert pd.isna(result.loc[0, "volume_change_1d"])
    assert pd.isna(result.loc[4, "volume_change_1d"])


def test_volatility_warmup_is_exact_per_ticker() -> None:
    rows_per_ticker = 6

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
                    "close": [float(100 + index) for index in range(rows_per_ticker)],
                    "volume": [float(1000 + 100 * index) for index in range(rows_per_ticker)],
                }
            )
            for ticker in ("AAA", "BBB")
        ],
        ignore_index=True,
    )

    result = build_volatility_volume_features(
        frame,
        volatility_windows=(3,),
        volume_windows=(3,),
    )

    for _, ticker_result in result.groupby("ticker", sort=False):
        volatility = ticker_result["volatility_3d"].reset_index(drop=True)

        assert volatility.iloc[:3].isna().all()
        assert volatility.iloc[3:].notna().all()


def test_future_values_do_not_change_past_features() -> None:
    base = pd.DataFrame(
        {
            "ticker": ["AAA"] * 6,
            "session_date": pd.date_range(
                "2026-01-01",
                periods=6,
                freq="D",
            ),
            "close": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0],
            "volume": [
                1000.0,
                1100.0,
                1200.0,
                1150.0,
                1250.0,
                1300.0,
            ],
        }
    )

    changed = base.copy()

    changed.loc[5, "close"] = 5000.0
    changed.loc[5, "volume"] = 50000.0

    base_features = build_volatility_volume_features(
        base,
        volatility_windows=(3,),
        volume_windows=(3,),
    )

    changed_features = build_volatility_volume_features(
        changed,
        volatility_windows=(3,),
        volume_windows=(3,),
    )

    feature_columns = [
        "volatility_3d",
        "volume_change_1d",
        "volume_sma_3d",
        "volume_zscore_3d",
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
            "close": [100.0, 101.0],
            "volume": [1000.0, 1100.0],
        }
    )

    with pytest.raises(ValueError, match="chronological"):
        build_volatility_volume_features(frame)


def test_invalid_volume_is_rejected() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "session_date": pd.to_datetime(["2026-01-01"]),
            "close": [100.0],
            "volume": [0.0],
        }
    )

    with pytest.raises(ValueError, match="positive"):
        build_volatility_volume_features(frame)
