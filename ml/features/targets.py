from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_HORIZONS = (1, 5)

_REQUIRED_COLUMNS = {
    "ticker",
    "session_date",
    "adj_close",
}


def build_targets(
    data: pd.DataFrame,
    horizons: tuple[int, ...] = TARGET_HORIZONS,
) -> pd.DataFrame:
    missing_columns = _REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    if not horizons:
        raise ValueError("At least one target horizon is required.")

    if any(not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0 for horizon in horizons):
        raise ValueError("Target horizons must be positive integers.")

    if len(set(horizons)) != len(horizons):
        raise ValueError("Target horizons must be unique.")

    frame = data.copy()

    if frame.empty:
        for horizon in horizons:
            frame[f"target_return_{horizon}d"] = pd.Series(
                index=frame.index,
                dtype="float64",
            )
            frame[f"target_up_{horizon}d"] = pd.Series(
                index=frame.index,
                dtype="float64",
            )
        return frame

    if frame[["ticker", "session_date"]].duplicated().any():
        raise ValueError("Duplicate ticker/session_date rows detected.")

    if frame["adj_close"].isna().any():
        raise ValueError("adj_close contains missing values.")

    if not np.isfinite(frame["adj_close"]).all():
        raise ValueError("adj_close contains non-finite values.")

    if (frame["adj_close"] <= 0).any():
        raise ValueError("adj_close must contain only positive values.")

    for ticker, ticker_frame in frame.groupby("ticker", sort=False):
        if not ticker_frame["session_date"].is_monotonic_increasing:
            raise ValueError(f"session_date must be chronological for ticker {ticker}.")

    for horizon in horizons:
        future_close = frame.groupby("ticker", sort=False)["adj_close"].shift(-horizon)

        target_return = pd.Series(
            np.log(future_close / frame["adj_close"]),
            index=frame.index,
            dtype="float64",
        )
        target_up = target_return.gt(0).astype("float64")
        target_up[target_return.isna()] = np.nan

        frame[f"target_return_{horizon}d"] = target_return
        frame[f"target_up_{horizon}d"] = target_up

    return frame
