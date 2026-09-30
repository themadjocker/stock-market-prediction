from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features.technical import (
    TECHNICAL_FEATURE_COLUMNS,
    build_technical_features,
)


def make_prices(
    ticker: str,
    periods: int = 260,
    multiplier: float = 1.0,
) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=periods, freq="B")
    base = (
        100.0
        + np.arange(periods, dtype=float) * 0.15
        + 2.0 * np.sin(np.arange(periods, dtype=float) / 7.0)
    ) * multiplier

    return pd.DataFrame(
        {
            "ticker": ticker,
            "session_date": dates,
            "open": base * 0.998,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base,
        }
    )


def test_builds_expected_technical_columns() -> None:
    data = make_prices("AAA")

    result = build_technical_features(data)

    assert list(result.columns[-len(TECHNICAL_FEATURE_COLUMNS) :]) == list(
        TECHNICAL_FEATURE_COLUMNS
    )
    assert len(result) == len(data)


def test_moving_average_warmup_is_expected() -> None:
    data = make_prices("AAA")
    result = build_technical_features(data)

    assert result["sma_10"].iloc[:9].isna().all()
    assert result["sma_10"].iloc[9:].notna().all()

    assert result["sma_200"].iloc[:199].isna().all()
    assert result["sma_200"].iloc[199:].notna().all()


def test_core_indicators_become_defined_after_warmup() -> None:
    data = make_prices("AAA")
    result = build_technical_features(data)

    assert result["rsi_14"].iloc[14:].notna().any()
    assert result["macd_12_26"].iloc[25:].notna().any()
    assert result["adx_14"].iloc[30:].notna().any()
    assert result["bb_percent_b_20_2"].iloc[19:].notna().any()
    assert result["trend_slope_50"].iloc[49:].notna().any()


def test_di_spread_matches_plus_di_minus_minus_di() -> None:
    data = make_prices("AAA")
    result = build_technical_features(data)

    expected = result["plus_di_14"] - result["minus_di_14"]

    pd.testing.assert_series_equal(
        result["di_spread_14"],
        expected,
        check_names=False,
    )


def test_features_are_isolated_per_ticker() -> None:
    aaa = make_prices("AAA")
    bbb = make_prices("BBB", multiplier=10.0)

    combined = pd.concat([aaa, bbb], ignore_index=True)
    combined_result = build_technical_features(combined)

    aaa_result = build_technical_features(aaa)

    combined_aaa = combined_result.loc[
        combined_result["ticker"].eq("AAA"),
        list(TECHNICAL_FEATURE_COLUMNS),
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        combined_aaa,
        aaa_result[list(TECHNICAL_FEATURE_COLUMNS)].reset_index(drop=True),
        check_dtype=True,
    )


def test_future_rows_cannot_change_earlier_features() -> None:
    data = make_prices("AAA")
    mutated = data.copy()

    cutoff = 180
    mutated.loc[cutoff:, "close"] *= 4.0
    mutated.loc[cutoff:, "high"] *= 4.0
    mutated.loc[cutoff:, "low"] *= 4.0
    mutated.loc[cutoff:, "open"] *= 4.0

    baseline = build_technical_features(data)
    changed = build_technical_features(mutated)

    pd.testing.assert_frame_equal(
        baseline.loc[: cutoff - 1, list(TECHNICAL_FEATURE_COLUMNS)].reset_index(
            drop=True
        ),
        changed.loc[: cutoff - 1, list(TECHNICAL_FEATURE_COLUMNS)].reset_index(
            drop=True
        ),
        check_dtype=True,
    )


@pytest.mark.parametrize(
    "bad_column",
    ["open", "high", "low", "close"],
)
def test_rejects_missing_values_in_price_columns(bad_column: str) -> None:
    data = make_prices("AAA")
    data.loc[0, bad_column] = np.nan

    with pytest.raises(ValueError, match=f"{bad_column} contains missing values"):
        build_technical_features(data)


def test_rejects_non_chronological_ticker_rows() -> None:
    data = make_prices("AAA")
    data.loc[[0, 1], "session_date"] = data.loc[[1, 0], "session_date"].to_numpy()

    with pytest.raises(ValueError, match="must be chronological"):
        build_technical_features(data)


def test_no_unexpected_infinities_are_returned() -> None:
    data = make_prices("AAA")
    result = build_technical_features(data)

    technical_values = result[list(TECHNICAL_FEATURE_COLUMNS)].to_numpy(
        dtype=float
    )

    assert not np.isinf(technical_values).any()
