from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features.context import (
    ALL_CONTEXT_COLUMNS,
    MARKET_CONTEXT_COLUMNS,
    SECTOR_CONTEXT_COLUMNS,
    build_context_features,
    build_market_context_features,
    build_sector_context_features,
)


def make_prices(
    ticker: str,
    sector: str,
    periods: int = 80,
    multiplier: float = 1.0,
) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=periods, freq="B")
    base = (
        100.0
        + np.arange(periods, dtype=float) * 0.25
        + 2.0 * np.sin(np.arange(periods, dtype=float) / 5.0)
    ) * multiplier

    return pd.DataFrame(
        {
            "ticker": ticker,
            "session_date": dates,
            "sector": sector,
            "close": base,
        }
    )


def make_market(periods: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=periods, freq="B")
    close = 1000.0 + np.arange(periods, dtype=float) * 1.5

    return pd.DataFrame(
        {
            "session_date": dates,
            "close": close,
        }
    )


def test_market_context_adds_expected_columns() -> None:
    stocks = make_prices("AAA", "IT")
    result = build_market_context_features(
        stocks,
        market_index=make_market(),
    )

    assert set(MARKET_CONTEXT_COLUMNS).issubset(result.columns)
    assert "market_regime_trend" in result.columns
    assert len(result) == len(stocks)


def test_market_features_are_point_in_time() -> None:
    stocks = make_prices("AAA", "IT")
    market = make_market()

    baseline = build_market_context_features(
        stocks,
        market_index=market,
    )

    changed_market = market.copy()
    changed_market.loc[60:, "close"] *= 5.0

    changed = build_market_context_features(
        stocks,
        market_index=changed_market,
    )

    check_cols = [
        "market_return_1d",
        "market_return_5d",
        "market_return_20d",
        "market_volatility_20d",
        "market_sma_20",
        "market_sma_50",
    ]

    pd.testing.assert_frame_equal(
        baseline.loc[:59, check_cols].reset_index(drop=True),
        changed.loc[:59, check_cols].reset_index(drop=True),
        check_dtype=True,
    )


def test_market_relative_return_is_stock_minus_market() -> None:
    stocks = make_prices("AAA", "IT")
    result = build_market_context_features(
        stocks,
        market_index=make_market(),
    )

    expected = (
        result["stock_return_5d_context"] - result["market_return_5d"]
    )

    pd.testing.assert_series_equal(
        result["stock_vs_market_return_5d"],
        expected,
        check_names=False,
    )


def test_sector_context_isolated_by_sector() -> None:
    aaa = make_prices("AAA", "IT", multiplier=1.0)
    bbb = make_prices("BBB", "IT", multiplier=1.1)
    ccc = make_prices("CCC", "BANK", multiplier=3.0)

    data = pd.concat([aaa, bbb, ccc], ignore_index=True)
    result = build_sector_context_features(data)

    aaa_only = build_sector_context_features(aaa)

    result_aaa = result.loc[
        result["ticker"].eq("AAA"),
        ["session_date", *SECTOR_CONTEXT_COLUMNS],
    ].reset_index(drop=True)

    expected_aaa = aaa_only[
        ["session_date", *SECTOR_CONTEXT_COLUMNS]
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        result_aaa,
        expected_aaa,
        check_dtype=True,
    )


def test_sector_breadth_has_valid_probability_range() -> None:
    aaa = make_prices("AAA", "IT")
    bbb = make_prices("BBB", "IT")
    data = pd.concat([aaa, bbb], ignore_index=True)

    result = build_sector_context_features(data)

    values = result["sector_breadth_1d"].dropna()

    assert not values.empty
    assert values.between(0.0, 1.0).all()


def test_combined_context_adds_optional_cross_assets() -> None:
    stocks = pd.concat(
        [
            make_prices("AAA", "IT"),
            make_prices("BBB", "BANK", multiplier=2.0),
        ],
        ignore_index=True,
    )
    dates = make_market()["session_date"]

    vix = pd.DataFrame(
        {"session_date": dates, "close": np.linspace(12, 20, len(dates))}
    )
    usd_inr = pd.DataFrame(
        {"session_date": dates, "close": np.linspace(83, 86, len(dates))}
    )
    crude = pd.DataFrame(
        {"session_date": dates, "close": np.linspace(70, 80, len(dates))}
    )

    result = build_context_features(
        stocks,
        market_index=make_market(),
        india_vix=vix,
        usd_inr=usd_inr,
        crude=crude,
    )

    assert set(ALL_CONTEXT_COLUMNS).issubset(result.columns)
    assert result["india_vix"].notna().all()

    first_session = dates.iloc[0]
    later_rows = result["session_date"].gt(first_session)

    assert result.loc[later_rows, "usd_inr_return_1d"].notna().all()
    assert result.loc[later_rows, "crude_return_1d"].notna().all()
    assert result.loc[
        result["session_date"].eq(first_session), "usd_inr_return_1d"
    ].isna().all()
    assert result.loc[
        result["session_date"].eq(first_session), "crude_return_1d"
    ].isna().all()


def test_missing_cross_assets_are_kept_as_nan() -> None:
    stocks = make_prices("AAA", "IT")
    result = build_context_features(
        stocks,
        market_index=make_market(),
    )

    assert result["india_vix"].isna().all()
    assert result["usd_inr_return_1d"].isna().all()
    assert result["crude_return_1d"].isna().all()


def test_unsorted_market_index_is_rejected() -> None:
    market = make_market()
    market = market.iloc[::-1].reset_index(drop=True)

    with pytest.raises(ValueError, match="chronological"):
        build_market_context_features(
            make_prices("AAA", "IT"),
            market_index=market,
        )


def test_duplicate_market_dates_are_rejected() -> None:
    market = make_market()
    market.loc[1, "session_date"] = market.loc[0, "session_date"]

    with pytest.raises(ValueError, match="duplicate"):
        build_market_context_features(
            make_prices("AAA", "IT"),
            market_index=market,
        )


def test_stock_rows_do_not_cross_ticker_boundaries() -> None:
    aaa = make_prices("AAA", "IT")
    bbb = make_prices("BBB", "BANK", multiplier=10.0)
    combined = pd.concat([aaa, bbb], ignore_index=True)

    result = build_market_context_features(
        combined,
        market_index=make_market(),
    )
    aaa_only = build_market_context_features(
        aaa,
        market_index=make_market(),
    )

    cols = [
        "stock_return_1d_context",
        "stock_return_5d_context",
        "stock_return_20d_context",
    ]

    pd.testing.assert_frame_equal(
        result.loc[result["ticker"].eq("AAA"), cols].reset_index(drop=True),
        aaa_only[cols].reset_index(drop=True),
        check_dtype=True,
    )


def test_no_unexpected_infinities_are_returned() -> None:
    stocks = pd.concat(
        [
            make_prices("AAA", "IT"),
            make_prices("BBB", "BANK"),
        ],
        ignore_index=True,
    )

    result = build_context_features(
        stocks,
        market_index=make_market(),
    )

    numeric = result[list(ALL_CONTEXT_COLUMNS)].to_numpy(dtype=float)

    assert not np.isinf(numeric).any()


def test_expected_warmup_for_market_features() -> None:
    stocks = make_prices("AAA", "IT")
    result = build_market_context_features(
        stocks,
        market_index=make_market(),
    )

    assert result["market_return_5d"].iloc[:5].isna().all()
    assert result["market_sma_20"].iloc[:19].isna().all()
    assert result["market_sma_50"].iloc[:49].isna().all()
