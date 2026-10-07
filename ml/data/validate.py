"""Validation for canonical market data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.fetch import CANONICAL_COLUMNS, NUMERIC_COLUMNS


def validate_market_data(frame: pd.DataFrame) -> None:
    """Raise ValueError when canonical market data violates invariants."""

    errors: list[str] = []

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    missing_columns = [column for column in CANONICAL_COLUMNS if column not in frame.columns]

    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")

    if errors:
        raise ValueError("\n".join(errors))

    data = frame.loc[:, list(CANONICAL_COLUMNS)].copy()

    # ------------------------------------------------------------------
    # Dates
    # ------------------------------------------------------------------
    parsed_dates = pd.to_datetime(data["session_date"], errors="coerce")

    if parsed_dates.isna().any():
        errors.append(f"Invalid session_date values: {int(parsed_dates.isna().sum())}")

    # ------------------------------------------------------------------
    # Tickers
    # ------------------------------------------------------------------
    if data["ticker"].isna().any():
        errors.append(f"Missing ticker values: {int(data['ticker'].isna().sum())}")

    ticker_values = data["ticker"].astype("string").str.strip()

    if (ticker_values == "").any():
        errors.append("Blank ticker values detected.")

    # ------------------------------------------------------------------
    # Numeric columns
    # ------------------------------------------------------------------
    for column in NUMERIC_COLUMNS:
        numeric = pd.to_numeric(data[column], errors="coerce")

        if numeric.isna().any():
            errors.append(f"{column} contains {int(numeric.isna().sum())} non-numeric or missing values.")
            continue

        if not np.isfinite(numeric.to_numpy()).all():
            errors.append(f"{column} contains non-finite values.")

    # ------------------------------------------------------------------
    # Price sanity
    # ------------------------------------------------------------------
    for column in ("open", "high", "low", "close", "adj_close"):
        numeric = pd.to_numeric(data[column], errors="coerce")

        if (numeric <= 0).any():
            errors.append(f"{column} contains non-positive values.")

    volume = pd.to_numeric(data["volume"], errors="coerce")

    if (volume < 0).any():
        errors.append("volume contains negative values.")

    # ------------------------------------------------------------------
    # OHLC relationships
    # ------------------------------------------------------------------
    open_price = pd.to_numeric(data["open"], errors="coerce")
    high_price = pd.to_numeric(data["high"], errors="coerce")
    low_price = pd.to_numeric(data["low"], errors="coerce")
    close_price = pd.to_numeric(data["close"], errors="coerce")

    invalid_ohlc = (
        (low_price > high_price)
        | (open_price < low_price)
        | (open_price > high_price)
        | (close_price < low_price)
        | (close_price > high_price)
    )

    if invalid_ohlc.any():
        errors.append(f"Invalid OHLC relationships: {int(invalid_ohlc.sum())} rows.")

    # ------------------------------------------------------------------
    # Duplicate sessions
    # ------------------------------------------------------------------
    duplicate_mask = data.duplicated(
        subset=["ticker", "session_date"],
        keep=False,
    )

    if duplicate_mask.any():
        errors.append(f"Duplicate ticker/session_date rows: {int(duplicate_mask.sum())}.")

    # ------------------------------------------------------------------
    # Chronological ordering per ticker
    # ------------------------------------------------------------------
    temp = data.assign(_parsed_date=parsed_dates)

    for ticker, group in temp.groupby("ticker", sort=False):
        if not group["_parsed_date"].is_monotonic_increasing:
            errors.append(f"session_date is not monotonically increasing for {ticker}.")

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------
    if errors:
        raise ValueError("Market-data validation failed:\n- " + "\n- ".join(errors))


def validate_csv(input_path: Path) -> pd.DataFrame:
    """Read and validate a canonical market-data CSV."""

    frame = pd.read_csv(input_path)
    validate_market_data(frame)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate canonical market-data CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to canonical market-data CSV.",
    )

    args = parser.parse_args()

    data = validate_csv(args.input)

    print("VALIDATION: PASS")
    print(f"ROWS: {len(data)}")
    print(f"COLUMNS: {', '.join(data.columns)}")
    print(f"DATE RANGE: {data['session_date'].min()} -> {data['session_date'].max()}")
    print(f"TICKERS: {data['ticker'].nunique()}")


if __name__ == "__main__":
    main()
