from __future__ import annotations

import math

import pandas as pd
import pytest

from ml.features.targets import build_targets


def test_targets_are_forward_looking_with_correct_values() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 6,
            "session_date": pd.date_range("2026-01-01", periods=6, freq="D"),
            "adj_close": [100.0, 110.0, 99.0, 100.0, 120.0, 108.0],
        }
    )

    result = build_targets(frame)

    assert math.isclose(
        float(result["target_return_1d"].to_numpy(dtype=float)[0]),
        math.log(110.0 / 100.0),
    )

    assert math.isclose(
        float(result["target_return_5d"].to_numpy(dtype=float)[0]),
        math.log(108.0 / 100.0),
    )

    assert result.loc[0, "target_up_1d"] == 1.0
    assert result.loc[1, "target_up_1d"] == 0.0


def test_targets_do_not_cross_ticker_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "session_date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-05",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-05",
                ]
            ),
            "adj_close": [100.0, 110.0, 121.0, 1000.0, 2000.0, 4000.0],
        }
    )

    result = build_targets(frame)

    # AAA has only three observations, so a 5-session target
    # must remain unavailable rather than reaching into BBB.
    assert result.loc[:2, "target_return_5d"].isna().all()

    assert math.isclose(
        float(result["target_return_1d"].to_numpy(dtype=float)[3]),
        math.log(2000.0 / 1000.0),
    )


def test_unsorted_ticker_data_is_rejected() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "session_date": pd.to_datetime(["2026-01-02", "2026-01-01"]),
            "adj_close": [110.0, 100.0],
        }
    )

    with pytest.raises(ValueError, match="chronological"):
        build_targets(frame)


def test_duplicate_sessions_are_rejected() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "session_date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "adj_close": [100.0, 101.0],
        }
    )

    with pytest.raises(ValueError, match="Duplicate"):
        build_targets(frame)


def test_missing_required_columns_are_rejected() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "session_date": pd.to_datetime(["2026-01-01"]),
        }
    )

    with pytest.raises(ValueError, match="adj_close"):
        build_targets(frame)


def test_target_null_boundaries_are_exact_per_ticker() -> None:
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
                    "adj_close": [float(100 + index) for index in range(rows_per_ticker)],
                }
            )
            for ticker in ("AAA", "BBB")
        ],
        ignore_index=True,
    )

    result = build_targets(frame)

    for horizon in (1, 5):
        return_column = f"target_return_{horizon}d"
        up_column = f"target_up_{horizon}d"

        for ticker, ticker_result in result.groupby("ticker", sort=False):
            return_target = ticker_result[return_column].reset_index(drop=True)
            up_target = ticker_result[up_column].reset_index(drop=True)

            # Exactly the final `horizon` observations lack future data.
            assert int(return_target.isna().sum()) == horizon
            assert int(up_target.isna().sum()) == horizon

            # The observation immediately before the missing tail still has
            # a valid target.
            assert pd.notna(return_target.iloc[-horizon - 1])
            assert pd.notna(up_target.iloc[-horizon - 1])

            # Every observation in the terminal unavailable region is NaN.
            assert return_target.iloc[-horizon:].isna().all()
            assert up_target.iloc[-horizon:].isna().all()

            # Make the assertion failure identify the ticker clearly.
            assert ticker in {"AAA", "BBB"}
