"""Development stock universe for the forecasting pipeline.

This is a fixed, diversified development set used to validate the
data and ML pipeline. It is not the final research universe.
"""

DEV_UNIVERSE: tuple[str, ...] = (
    # Technology
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "AVGO",
    "AMD",
    "CRM",
    "ORCL",
    "ADBE",
    "CSCO",
    "IBM",
    "QCOM",
    "AMAT",
    "MU",
    "TXN",
    # Financials
    "JPM",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "V",
    "MA",
    "BRK-B",
    # Healthcare
    "JNJ",
    "UNH",
    "LLY",
    "MRK",
    "ABBV",
    "PFE",
    "TMO",
    # Energy
    "XOM",
    "CVX",
    "COP",
    # Industrials
    "CAT",
    "DE",
    "GE",
    "HON",
    "BA",
    # Consumer
    "WMT",
    "COST",
    "HD",
    "MCD",
    "KO",
    "PEP",
    "PG",
    # Communication / Media
    "DIS",
    "NFLX",
    # Utilities
    "NEE",
)


def get_dev_universe() -> tuple[str, ...]:
    """Return the fixed development universe."""
    return DEV_UNIVERSE
