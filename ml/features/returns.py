from __future__ import annotations

import numpy as np
import pandas as pd

RETURN_HORIZONS = (1, 2, 5, 10, 20)

_REQUIRED_COLUMNS = {
    "ticker",
    "session_date",
    "open",
    "high",
    "low",
    "close",
}


def build_return_features(
    data: pd.DataFrame,
    horizons: tuple[int, ...] = RETURN_HORIZONS,
) -> pd.DataFrame:
    """
    Build price/return features using information available at each row.

    All lagged calculations are performed independently per ticker.
    """
    missing_columns = _REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    if not horizons:
        raise ValueError("At least one return horizon is required.")

    if any(
        not isinstance(horizon, int)
        or isinstance(horizon, bool)
        or horizon <= 0
        for horizon in horizons
    ):
        raise ValueError("Return horizons must be positive integers.")

    if len(set(horizons)) != len(horizons):
        raise ValueError("Return horizons must be unique.")

    frame = data.copy()

    if frame.empty:
        return frame

    if frame[["ticker", "session_date"]].duplicated().any():
        raise ValueError("Duplicate ticker/session_date rows detected.")

    numeric_columns = ["open", "high", "low", "close"]

    for column in numeric_columns:
        if frame[column].isna().any():
            raise ValueError(f"{column} contains missing values.")

        if not np.isfinite(frame[column]).all():
            raise ValueError(f"{column} contains non-finite values.")

        if (frame[column] <= 0).any():
            raise ValueError(f"{column} must contain only positive values.")

    if (frame["high"] < frame["low"]).any():
        raise ValueError("high must be greater than or equal to low.")

    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        raise ValueError("high must be greater than or equal to open and close.")

    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
        raise ValueError("low must be less than or equal to open and close.")

    grouped = frame.groupby("ticker", sort=False)

    for ticker, ticker_frame in grouped:
        if not ticker_frame["session_date"].is_monotonic_increasing:
            raise ValueError(
                f"session_date must be chronological for ticker {ticker}."
            )

    for horizon in horizons:
        previous_close = grouped["close"].shift(horizon)
        frame[f"log_return_{horizon}d"] = np.log(
            frame["close"] / previous_close
        )

    previous_close = grouped["close"].shift(1)

    frame["intraday_return"] = np.log(frame["close"] / frame["open"])
    frame["overnight_gap"] = np.log(frame["open"] / previous_close)
    frame["high_low_range"] = np.log(frame["high"] / frame["low"])

    return frame
