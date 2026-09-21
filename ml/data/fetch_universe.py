"""Download and validate the development stock universe."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.data.fetch import fetch_ohlcv, save_csv
from ml.data.universe import DEV_UNIVERSE
from ml.data.validate import validate_market_data


def fetch_universe(
    tickers: tuple[str, ...],
    *,
    period: str,
    output_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    """Fetch, validate, and save one CSV per ticker."""

    output_dir.mkdir(parents=True, exist_ok=True)

    successful: list[str] = []
    failed: dict[str, str] = {}

    for index, ticker in enumerate(tickers, start=1):
        print(f"\n[{index}/{len(tickers)}] {ticker}")

        try:
            data = fetch_ohlcv(ticker, period=period)

            validate_market_data(data)

            output_path = output_dir / f"{ticker}.csv"
            save_csv(data, output_path)

            successful.append(ticker)

            print(f"  ROWS: {len(data)}")
            print(
                f"  DATE RANGE: "
                f"{data['session_date'].min()} -> "
                f"{data['session_date'].max()}"
            )
            print(f"  SAVED: {output_path}")

        except Exception as exc:
            failed[ticker] = str(exc)
            print(f"  FAILED: {exc}")

    return successful, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and validate the development stock universe."
    )
    parser.add_argument(
        "--period",
        default="5y",
        help="yfinance period, for example 1y or 5y.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for per-ticker CSV files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N development tickers.",
    )

    args = parser.parse_args()

    tickers = DEV_UNIVERSE[: args.limit] if args.limit else DEV_UNIVERSE

    print(f"UNIVERSE SIZE: {len(tickers)}")
    print(f"PERIOD: {args.period}")
    print(f"OUTPUT: {args.output_dir}")

    successful, failed = fetch_universe(
        tickers,
        period=args.period,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("UNIVERSE DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"SUCCESSFUL: {len(successful)}")
    print(f"FAILED: {len(failed)}")

    if failed:
        print("\nFAILURES:")
        for ticker, error in failed.items():
            print(f"- {ticker}: {error}")

        raise SystemExit(1)

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()
