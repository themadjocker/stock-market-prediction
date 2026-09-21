"""Audit the downloaded market-data corpus."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.data.validate import validate_market_data


def audit_file(path: Path) -> dict[str, object]:
    """Validate and summarize one ticker file."""

    frame = pd.read_csv(path)
    validate_market_data(frame)

    dates = pd.to_datetime(frame["session_date"])

    return {
        "ticker": path.stem,
        "rows": len(frame),
        "first_date": dates.min(),
        "last_date": dates.max(),
        "unique_dates": dates.nunique(),
        "duplicates": int(frame.duplicated(
            subset=["ticker", "session_date"]
        ).sum()),
        "missing_values": int(frame.isna().sum().sum()),
        "min_close": float(frame["close"].min()),
        "max_close": float(frame["close"].max()),
        "min_volume": float(frame["volume"].min()),
        "max_volume": float(frame["volume"].max()),
    }


def audit_corpus(raw_dir: Path) -> pd.DataFrame:
    """Validate and summarize the entire raw-data corpus."""

    files = sorted(raw_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    records = [audit_file(path) for path in files]

    return (
        pd.DataFrame(records)
        .sort_values("ticker")
        .reset_index(drop=True)
    )


def main() -> None:
    raw_dir = Path("data/raw")

    report = audit_corpus(raw_dir)

    print("=" * 80)
    print("RAW DATA CORPUS AUDIT")
    print("=" * 80)

    print(f"TICKERS: {len(report)}")
    print(f"TOTAL ROWS: {int(report['rows'].sum())}")

    print("\nROW COUNTS:")
    print(report["rows"].value_counts().sort_index().to_string())

    print("\nDATE COVERAGE:")
    print(
        report[
            ["ticker", "rows", "first_date", "last_date"]
        ].to_string(index=False)
    )

    print("\nDUPLICATES:")
    print(
        report[
            ["ticker", "duplicates", "unique_dates"]
        ].to_string(index=False)
    )

    print("\nMISSING VALUES:")
    print(
        report[
            ["ticker", "missing_values"]
        ].to_string(index=False)
    )

    print("\nPRICE / VOLUME RANGES:")
    print(
        report[
            [
                "ticker",
                "min_close",
                "max_close",
                "min_volume",
                "max_volume",
            ]
        ].to_string(index=False)
    )

    print("\nSUMMARY CHECKS:")
    print(
        f"UNIQUE ROW COUNTS: {report['rows'].nunique()}"
    )
    print(
        f"UNIQUE START DATES: {report['first_date'].nunique()}"
    )
    print(
        f"UNIQUE END DATES: {report['last_date'].nunique()}"
    )
    print(
        f"TOTAL DUPLICATES: {int(report['duplicates'].sum())}"
    )
    print(
        f"TOTAL MISSING VALUES: {int(report['missing_values'].sum())}"
    )

    print("\nSTATUS: AUDIT COMPLETE")


if __name__ == "__main__":
    main()
