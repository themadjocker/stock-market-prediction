"""Market-data ingestion and normalization.

The rest of the ML pipeline should consume the canonical schema produced
by this module rather than depending directly on yfinance's dataframe shape.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

CANONICAL_COLUMNS: tuple[str, ...] = (
    "session_date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)

COLUMN_MAP: dict[str, str] = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def _flatten_ticker_columns(
    frame: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Normalize yfinance's MultiIndex output for a single ticker."""

    if not isinstance(frame.columns, pd.MultiIndex):
        return frame

    ticker_level: int | None = None

    for level in range(frame.columns.nlevels):
        values = frame.columns.get_level_values(level)
        if ticker in values:
            ticker_level = level
            break

    if ticker_level is None:
        raise ValueError(f"Could not find ticker {ticker!r} in yfinance MultiIndex columns.")

    result = frame.xs(
        ticker,
        axis=1,
        level=ticker_level,
        drop_level=True,
    )

    if isinstance(result, pd.Series):
        raise TypeError(f"Expected DataFrame after flattening ticker {ticker!r}, but received a Series.")

    return result


def normalize_ohlcv(
    frame: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Convert raw yfinance output into the project canonical schema."""

    ticker = ticker.upper().strip()

    if frame.empty:
        raise ValueError(f"No market data returned for ticker {ticker!r}.")

    frame = _flatten_ticker_columns(frame.copy(), ticker)

    missing_columns = set(COLUMN_MAP) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing expected market-data columns for {ticker}: {sorted(missing_columns)}")

    frame = frame.rename(columns=COLUMN_MAP)

    index = pd.DatetimeIndex(pd.to_datetime(frame.index))

    if index.tz is not None:
        index = index.tz_localize(None)

    frame.index = index.normalize()
    frame.index.name = "session_date"

    frame = frame.loc[:, list(COLUMN_MAP.values())].copy()

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.reset_index()
    frame.insert(1, "ticker", ticker)

    result = frame.loc[:, list(CANONICAL_COLUMNS)]

    if isinstance(result, pd.Series):
        raise TypeError("Expected a DataFrame after selecting canonical columns.")

    return result


def fetch_ohlcv(
    ticker: str,
    *,
    period: str = "5y",
) -> pd.DataFrame:
    """Download and normalize daily OHLCV data for one ticker."""

    ticker = ticker.upper().strip()

    raw = yf.download(
        ticker,
        period=period,
        auto_adjust=False,
        progress=False,
    )

    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f"Expected yfinance to return a DataFrame for {ticker!r}, got {type(raw).__name__}.")

    return normalize_ohlcv(raw, ticker)


def save_csv(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    """Persist canonical market data as CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize daily OHLCV data.")
    parser.add_argument(
        "--ticker",
        required=True,
        help="Ticker symbol, for example AAPL.",
    )
    parser.add_argument(
        "--period",
        default="5y",
        help="yfinance period, for example 5d, 1y, or 5y.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )

    args = parser.parse_args()

    data = fetch_ohlcv(args.ticker, period=args.period)

    if args.output is not None:
        save_csv(data, args.output)

    print(f"TICKER: {args.ticker.upper()}")
    print(f"ROWS: {len(data)}")
    print(f"COLUMNS: {', '.join(data.columns)}")
    print(data.head().to_string(index=False))


if __name__ == "__main__":
    main()
