from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_CONTEXT_COLUMNS = (
    "market_return_1d",
    "market_return_5d",
    "market_return_20d",
    "market_volatility_20d",
    "market_sma_20",
    "market_sma_50",
    "market_close_to_sma_20",
    "market_close_to_sma_50",
    "stock_vs_market_return_1d",
    "stock_vs_market_return_5d",
    "stock_vs_market_return_20d",
    "market_breadth_1d",
    "market_breadth_5d",
)

SECTOR_CONTEXT_COLUMNS = (
    "sector_return_1d",
    "sector_return_5d",
    "sector_return_20d",
    "sector_volatility_20d",
    "stock_vs_sector_return_1d",
    "stock_vs_sector_return_5d",
    "stock_vs_sector_return_20d",
    "sector_breadth_1d",
)

REGIME_COLUMNS = (
    "market_regime_trend",
    "market_regime_volatility",
)

CROSS_ASSET_COLUMNS = (
    "india_vix",
    "india_vix_change_1d",
    "usd_inr_return_1d",
    "crude_return_1d",
)

ALL_CONTEXT_COLUMNS = MARKET_CONTEXT_COLUMNS + SECTOR_CONTEXT_COLUMNS + REGIME_COLUMNS + CROSS_ASSET_COLUMNS

_STOCK_REQUIRED_COLUMNS = {
    "ticker",
    "session_date",
    "close",
    "sector",
}

_INDEX_REQUIRED_COLUMNS = {
    "session_date",
    "close",
}


def _validate_index_frame(data: pd.DataFrame, name: str) -> None:
    missing = _INDEX_REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(sorted(missing))}")

    if data.empty:
        return

    if data["session_date"].duplicated().any():
        raise ValueError(f"{name} contains duplicate session_date rows.")

    if not data["session_date"].is_monotonic_increasing:
        raise ValueError(f"{name} session_date must be chronological.")

    values = pd.to_numeric(data["close"], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{name} close contains missing or non-numeric values.")

    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} close contains non-finite values.")

    if (values <= 0).any():
        raise ValueError(f"{name} close must contain only positive values.")


def _validate_stock_frame(data: pd.DataFrame) -> None:
    missing = _STOCK_REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Stock data is missing required columns: {', '.join(sorted(missing))}")

    if data.empty:
        return

    if data[["ticker", "session_date"]].duplicated().any():
        raise ValueError("Duplicate ticker/session_date rows detected.")

    for ticker, ticker_frame in data.groupby("ticker", sort=False):
        if not ticker_frame["session_date"].is_monotonic_increasing:
            raise ValueError(f"session_date must be chronological for ticker {ticker}.")

        values = pd.to_numeric(ticker_frame["close"], errors="coerce")
        if values.isna().any():
            raise ValueError(f"close contains missing or non-numeric values for ticker {ticker}.")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"close contains non-finite values for ticker {ticker}.")
        if (values <= 0).any():
            raise ValueError(f"close must contain only positive values for ticker {ticker}.")


def _rolling_return(series: pd.Series, horizon: int) -> pd.Series:
    return pd.Series(
        np.log(series / series.shift(horizon)),
        index=series.index,
        dtype="float64",
    )


def _rolling_volatility(series: pd.Series, window: int) -> pd.Series:
    returns = _rolling_return(series, 1)
    return returns.rolling(window=window, min_periods=window).std()


def _attach_index_features(
    stock_frame: pd.DataFrame,
    index_frame: pd.DataFrame,
    *,
    prefix: str,
) -> pd.DataFrame:
    """Attach point-in-time index features using exact date alignment."""
    result = stock_frame.copy()

    index = index_frame[["session_date", "close"]].copy()
    index["return_1d"] = _rolling_return(index["close"], 1)
    index["return_5d"] = _rolling_return(index["close"], 5)
    index["return_20d"] = _rolling_return(index["close"], 20)
    index["volatility_20d"] = _rolling_volatility(index["close"], 20)
    index["sma_20"] = index["close"].rolling(window=20, min_periods=20).mean()
    index["sma_50"] = index["close"].rolling(window=50, min_periods=50).mean()
    index["close_to_sma_20"] = index["close"] / index["sma_20"] - 1.0
    index["close_to_sma_50"] = index["close"] / index["sma_50"] - 1.0

    index = index.rename(
        columns={
            column: f"{prefix}_{column}"
            for column in (
                "return_1d",
                "return_5d",
                "return_20d",
                "volatility_20d",
                "sma_20",
                "sma_50",
                "close_to_sma_20",
                "close_to_sma_50",
            )
        }
    )

    return result.merge(
        index[
            [
                "session_date",
                f"{prefix}_return_1d",
                f"{prefix}_return_5d",
                f"{prefix}_return_20d",
                f"{prefix}_volatility_20d",
                f"{prefix}_sma_20",
                f"{prefix}_sma_50",
                f"{prefix}_close_to_sma_20",
                f"{prefix}_close_to_sma_50",
            ]
        ],
        on="session_date",
        how="left",
        validate="many_to_one",
    )


def _sector_aggregate(
    stock_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create equal-weight sector context from the stock universe.

    This deliberately avoids future information: each sector statistic is
    calculated only from same-day stock data and prior observations.
    """
    frame = stock_data.copy()
    frame["return_1d"] = frame.groupby("ticker", sort=False)["close"].transform(
        lambda values: _rolling_return(values, 1)
    )
    frame["return_5d"] = frame.groupby("ticker", sort=False)["close"].transform(
        lambda values: _rolling_return(values, 5)
    )
    frame["return_20d"] = frame.groupby("ticker", sort=False)["close"].transform(
        lambda values: _rolling_return(values, 20)
    )

    sector_daily = (
        frame.groupby(["sector", "session_date"], sort=False)
        .agg(
            sector_return_1d=("return_1d", "mean"),
            sector_return_5d=("return_5d", "mean"),
            sector_return_20d=("return_20d", "mean"),
            sector_breadth_1d=("return_1d", lambda values: values.gt(0).mean()),
        )
        .reset_index()
    )

    sector_daily["sector_volatility_20d"] = sector_daily.groupby("sector", sort=False)["sector_return_1d"].transform(
        lambda values: values.rolling(window=20, min_periods=20).std()
    )

    return sector_daily


def build_market_context_features(
    stock_data: pd.DataFrame,
    *,
    market_index: pd.DataFrame,
) -> pd.DataFrame:
    _validate_stock_frame(stock_data)
    _validate_index_frame(market_index, "market_index")

    frame = stock_data.copy()

    if frame.empty:
        for column in MARKET_CONTEXT_COLUMNS + REGIME_COLUMNS:
            frame[column] = pd.Series(index=frame.index, dtype="float64")
        return frame

    frame = _attach_index_features(frame, market_index, prefix="market")

    stock_returns = frame.groupby("ticker", sort=False)["close"]

    frame["stock_return_1d_context"] = stock_returns.transform(lambda values: _rolling_return(values, 1))
    frame["stock_return_5d_context"] = stock_returns.transform(lambda values: _rolling_return(values, 5))
    frame["stock_return_20d_context"] = stock_returns.transform(lambda values: _rolling_return(values, 20))

    frame["stock_vs_market_return_1d"] = frame["stock_return_1d_context"] - frame["market_return_1d"]
    frame["stock_vs_market_return_5d"] = frame["stock_return_5d_context"] - frame["market_return_5d"]
    frame["stock_vs_market_return_20d"] = frame["stock_return_20d_context"] - frame["market_return_20d"]

    market_positive = (
        frame[["session_date", "ticker", "stock_return_1d_context"]]
        .drop_duplicates(["session_date", "ticker"])
        .groupby("session_date", sort=False)["stock_return_1d_context"]
        .mean()
        .rename("market_breadth_1d")
    )
    breadth_map_1d = (
        frame[["session_date", "stock_return_1d_context"]]
        .groupby("session_date", sort=False)["stock_return_1d_context"]
        .apply(lambda values: values.gt(0).mean())
        .rename("market_breadth_1d")
    )
    breadth_map_5d = (
        frame[["session_date", "stock_return_5d_context"]]
        .groupby("session_date", sort=False)["stock_return_5d_context"]
        .apply(lambda values: values.gt(0).mean())
        .rename("market_breadth_5d")
    )

    # Avoid accidental duplicate columns from intermediate groupings.
    _ = market_positive

    frame = frame.drop(
        columns=["market_breadth_1d", "market_breadth_5d"],
        errors="ignore",
    )
    frame = frame.join(breadth_map_1d, on="session_date")
    frame = frame.join(breadth_map_5d, on="session_date")

    frame["market_regime_trend"] = np.select(
        [
            frame["market_close_to_sma_50"] > 0.02,
            frame["market_close_to_sma_50"] < -0.02,
        ],
        [1.0, -1.0],
        default=0.0,
    )

    rolling_vol = frame["market_volatility_20d"]
    volatility_median = rolling_vol.expanding(min_periods=60).median()
    frame["market_regime_volatility"] = np.select(
        [
            rolling_vol > volatility_median * 1.5,
            rolling_vol < volatility_median * 0.75,
        ],
        [1.0, -1.0],
        default=0.0,
    )

    return frame


def build_sector_context_features(
    stock_data: pd.DataFrame,
) -> pd.DataFrame:
    _validate_stock_frame(stock_data)

    frame = stock_data.copy()

    if frame.empty:
        for column in SECTOR_CONTEXT_COLUMNS:
            frame[column] = pd.Series(index=frame.index, dtype="float64")
        return frame

    sector_daily = _sector_aggregate(frame)

    frame["stock_return_1d_context"] = frame.groupby("ticker", sort=False)["close"].transform(
        lambda values: _rolling_return(values, 1)
    )
    frame["stock_return_5d_context"] = frame.groupby("ticker", sort=False)["close"].transform(
        lambda values: _rolling_return(values, 5)
    )
    frame["stock_return_20d_context"] = frame.groupby("ticker", sort=False)["close"].transform(
        lambda values: _rolling_return(values, 20)
    )

    frame = frame.merge(
        sector_daily,
        on=["sector", "session_date"],
        how="left",
        validate="many_to_one",
    )

    frame["stock_vs_sector_return_1d"] = frame["stock_return_1d_context"] - frame["sector_return_1d"]
    frame["stock_vs_sector_return_5d"] = frame["stock_return_5d_context"] - frame["sector_return_5d"]
    frame["stock_vs_sector_return_20d"] = frame["stock_return_20d_context"] - frame["sector_return_20d"]

    return frame.drop(
        columns=[
            "stock_return_1d_context",
            "stock_return_5d_context",
            "stock_return_20d_context",
        ]
    )


def add_cross_asset_context(
    stock_data: pd.DataFrame,
    *,
    india_vix: pd.DataFrame | None = None,
    usd_inr: pd.DataFrame | None = None,
    crude: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = stock_data.copy()

    optional = {
        "india_vix": india_vix,
        "usd_inr": usd_inr,
        "crude": crude,
    }

    for name, data in optional.items():
        if data is None:
            continue

        _validate_index_frame(data, name)

        values = data[["session_date", "close"]].copy()

        if name == "india_vix":
            values["india_vix"] = values["close"]
            values["india_vix_change_1d"] = values["india_vix"].pct_change()
            values = values[["session_date", "india_vix", "india_vix_change_1d"]]
        else:
            values[f"{name}_return_1d"] = _rolling_return(values["close"], 1)
            values = values[["session_date", f"{name}_return_1d"]]

        frame = frame.merge(
            values,
            on="session_date",
            how="left",
            validate="many_to_one",
        )

    for column in CROSS_ASSET_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan

    return frame


def build_context_features(
    stock_data: pd.DataFrame,
    *,
    market_index: pd.DataFrame,
    india_vix: pd.DataFrame | None = None,
    usd_inr: pd.DataFrame | None = None,
    crude: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the complete P1.2.5 market/context feature layer."""
    frame = build_market_context_features(
        stock_data,
        market_index=market_index,
    )

    sector = build_sector_context_features(frame)
    sector_only = sector[
        [
            "ticker",
            "session_date",
            *SECTOR_CONTEXT_COLUMNS,
        ]
    ]

    frame = frame.merge(
        sector_only,
        on=["ticker", "session_date"],
        how="left",
        validate="one_to_one",
    )

    frame = add_cross_asset_context(
        frame,
        india_vix=india_vix,
        usd_inr=usd_inr,
        crude=crude,
    )

    return frame
