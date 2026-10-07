from __future__ import annotations

import numpy as np
import pandas as pd

VOLATILITY_WINDOWS = (5, 20, 60)
VOLUME_WINDOWS = (5, 20)

_REQUIRED_COLUMNS = {
    "ticker",
    "session_date",
    "close",
    "volume",
}


def build_volatility_volume_features(
    data: pd.DataFrame,
    *,
    volatility_windows: tuple[int, ...] = VOLATILITY_WINDOWS,
    volume_windows: tuple[int, ...] = VOLUME_WINDOWS,
) -> pd.DataFrame:
    """
    Build leakage-safe volatility and volume features.

    All rolling calculations are performed independently per ticker.
    Features use only information available at the current row.
    """

    missing_columns = _REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    if not volatility_windows:
        raise ValueError("At least one volatility window is required.")

    if not volume_windows:
        raise ValueError("At least one volume window is required.")

    for windows, name in (
        (volatility_windows, "volatility"),
        (volume_windows, "volume"),
    ):
        if any(not isinstance(window, int) or isinstance(window, bool) or window <= 0 for window in windows):
            raise ValueError(f"{name} windows must be positive integers.")

        if len(set(windows)) != len(windows):
            raise ValueError(f"{name} windows must be unique.")

    frame = data.copy()

    if frame.empty:
        for window in volatility_windows:
            frame[f"volatility_{window}d"] = pd.Series(
                index=frame.index,
                dtype="float64",
            )

        frame["volume_change_1d"] = pd.Series(
            index=frame.index,
            dtype="float64",
        )

        for window in volume_windows:
            frame[f"volume_sma_{window}d"] = pd.Series(
                index=frame.index,
                dtype="float64",
            )

            frame[f"volume_zscore_{window}d"] = pd.Series(
                index=frame.index,
                dtype="float64",
            )

        return frame

    if frame[["ticker", "session_date"]].duplicated().any():
        raise ValueError("Duplicate ticker/session_date rows detected.")

    for column in ("close", "volume"):
        if frame[column].isna().any():
            raise ValueError(f"{column} contains missing values.")

        if not np.isfinite(frame[column]).all():
            raise ValueError(f"{column} contains non-finite values.")

    if (frame["close"] <= 0).any():
        raise ValueError("close must contain only positive values.")

    if (frame["volume"] <= 0).any():
        raise ValueError("volume must contain only positive values.")

    grouped = frame.groupby("ticker", sort=False)

    for ticker, ticker_frame in grouped:
        if not ticker_frame["session_date"].is_monotonic_increasing:
            raise ValueError(f"session_date must be chronological for ticker {ticker}.")

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------

    previous_close = grouped["close"].shift(1)

    daily_log_return = pd.Series(
        np.log(frame["close"] / previous_close),
        index=frame.index,
        dtype="float64",
    )

    return_groups = daily_log_return.groupby(
        frame["ticker"],
        sort=False,
    )

    for window in volatility_windows:
        frame[f"volatility_{window}d"] = return_groups.transform(
            lambda series, window=window: series.rolling(
                window=window,
                min_periods=window,
            ).std()
        )

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------

    previous_volume = grouped["volume"].shift(1)

    frame["volume_change_1d"] = np.log(frame["volume"] / previous_volume)

    volume_groups = frame["volume"].groupby(
        frame["ticker"],
        sort=False,
    )

    for window in volume_windows:
        volume_sma = volume_groups.transform(
            lambda series, window=window: series.rolling(
                window=window,
                min_periods=window,
            ).mean()
        )

        volume_std = volume_groups.transform(
            lambda series, window=window: series.rolling(
                window=window,
                min_periods=window,
            ).std()
        )

        frame[f"volume_sma_{window}d"] = volume_sma

        frame[f"volume_zscore_{window}d"] = (frame["volume"] - volume_sma) / volume_std

    return frame
