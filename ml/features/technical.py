from __future__ import annotations

import numpy as np
import pandas as pd
import talib


TECHNICAL_FEATURE_COLUMNS = (
    "sma_10",
    "sma_20",
    "sma_50",
    "sma_200",
    "ema_10",
    "ema_20",
    "ema_50",
    "close_to_sma_10",
    "close_to_sma_20",
    "close_to_sma_50",
    "close_to_sma_200",
    "rsi_14",
    "macd_12_26",
    "macd_signal_9",
    "macd_hist_12_26_9",
    "stoch_k_14",
    "stoch_d_14_3",
    "adx_14",
    "plus_di_14",
    "minus_di_14",
    "di_spread_14",
    "bb_percent_b_20_2",
    "bb_bandwidth_20_2",
    "trend_slope_20",
    "trend_slope_50",
)

_REQUIRED_COLUMNS = {
    "ticker",
    "session_date",
    "open",
    "high",
    "low",
    "close",
}


def _validate_input(data: pd.DataFrame) -> None:
    missing_columns = _REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    if data.empty:
        return

    if data[["ticker", "session_date"]].duplicated().any():
        raise ValueError("Duplicate ticker/session_date rows detected.")

    numeric_columns = ["open", "high", "low", "close"]
    for column in numeric_columns:
        values = data[column]
        if values.isna().any():
            raise ValueError(f"{column} contains missing values.")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} contains non-finite values.")
        if (values <= 0).any():
            raise ValueError(f"{column} must contain only positive values.")

    if (data["high"] < data["low"]).any():
        raise ValueError("high must be greater than or equal to low.")

    if (data["high"] < data[["open", "close"]].max(axis=1)).any():
        raise ValueError("high must be greater than or equal to open and close.")

    if (data["low"] > data[["open", "close"]].min(axis=1)).any():
        raise ValueError("low must be less than or equal to open and close.")

    for ticker, ticker_frame in data.groupby("ticker", sort=False):
        if not ticker_frame["session_date"].is_monotonic_increasing:
            raise ValueError(
                f"session_date must be chronological for ticker {ticker}."
            )


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return np.where(np.isfinite(result), result, np.nan)


def build_technical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build technical-analysis features independently for each ticker.

    The function preserves input row order and index. All indicators use
    information from the current row and prior rows only; no future rows are
    explicitly used to calculate an earlier observation.

    TA-Lib provides the underlying indicator implementations.
    """
    _validate_input(data)

    frame = data.copy()

    if frame.empty:
        for column in TECHNICAL_FEATURE_COLUMNS:
            frame[column] = pd.Series(index=frame.index, dtype="float64")
        return frame

    original_index = frame.index
    frame = frame.reset_index(drop=True)

    for column in TECHNICAL_FEATURE_COLUMNS:
        frame[column] = np.nan

    for _, ticker_frame in frame.groupby("ticker", sort=False):
        positions = ticker_frame.index.to_numpy()

        close = ticker_frame["close"].to_numpy(dtype=float)
        high = ticker_frame["high"].to_numpy(dtype=float)
        low = ticker_frame["low"].to_numpy(dtype=float)

        sma_10 = talib.SMA(close, timeperiod=10)
        sma_20 = talib.SMA(close, timeperiod=20)
        sma_50 = talib.SMA(close, timeperiod=50)
        sma_200 = talib.SMA(close, timeperiod=200)

        ema_10 = talib.EMA(close, timeperiod=10)
        ema_20 = talib.EMA(close, timeperiod=20)
        ema_50 = talib.EMA(close, timeperiod=50)

        close_to_sma_10 = _safe_divide(close, sma_10) - 1.0
        close_to_sma_20 = _safe_divide(close, sma_20) - 1.0
        close_to_sma_50 = _safe_divide(close, sma_50) - 1.0
        close_to_sma_200 = _safe_divide(close, sma_200) - 1.0

        rsi_14 = talib.RSI(close, timeperiod=14)

        macd, macd_signal, macd_hist = talib.MACD(
            close,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9,
        )

        stoch_k, stoch_d = talib.STOCH(
            high,
            low,
            close,
            fastk_period=14,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0,
        )

        adx_14 = talib.ADX(high, low, close, timeperiod=14)
        plus_di_14 = talib.PLUS_DI(high, low, close, timeperiod=14)
        minus_di_14 = talib.MINUS_DI(high, low, close, timeperiod=14)
        di_spread_14 = plus_di_14 - minus_di_14

        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close,
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0,
        )
        bb_width_denominator = bb_middle
        bb_percent_b_denominator = bb_upper - bb_lower
        bb_percent_b = _safe_divide(
            close - bb_lower,
            bb_percent_b_denominator,
        )
        bb_bandwidth = _safe_divide(
            bb_upper - bb_lower,
            bb_width_denominator,
        )

        trend_slope_20 = talib.LINEARREG_SLOPE(close, timeperiod=20)
        trend_slope_50 = talib.LINEARREG_SLOPE(close, timeperiod=50)

        feature_values = {
            "sma_10": sma_10,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "ema_10": ema_10,
            "ema_20": ema_20,
            "ema_50": ema_50,
            "close_to_sma_10": close_to_sma_10,
            "close_to_sma_20": close_to_sma_20,
            "close_to_sma_50": close_to_sma_50,
            "close_to_sma_200": close_to_sma_200,
            "rsi_14": rsi_14,
            "macd_12_26": macd,
            "macd_signal_9": macd_signal,
            "macd_hist_12_26_9": macd_hist,
            "stoch_k_14": stoch_k,
            "stoch_d_14_3": stoch_d,
            "adx_14": adx_14,
            "plus_di_14": plus_di_14,
            "minus_di_14": minus_di_14,
            "di_spread_14": di_spread_14,
            "bb_percent_b_20_2": bb_percent_b,
            "bb_bandwidth_20_2": bb_bandwidth,
            "trend_slope_20": trend_slope_20,
            "trend_slope_50": trend_slope_50,
        }

        for column, values in feature_values.items():
            frame.loc[positions, column] = values

    frame[list(TECHNICAL_FEATURE_COLUMNS)] = (
        frame[list(TECHNICAL_FEATURE_COLUMNS)]
        .replace([np.inf, -np.inf], np.nan)
        .astype("float64")
    )

    frame.index = original_index
    return frame
