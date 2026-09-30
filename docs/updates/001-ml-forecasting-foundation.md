# Update 001 · ML forecasting foundation

This update records the complete ML foundation delivered across **P1.1** and **P1.2**. It is the canonical progress record for the forecasting foundation and should be extended in-place as the project advances through the remaining P1 phases.

---

## P1.1 · Data foundation ✅

### Repository and ML structure

The project was reorganized into dedicated ML, backend, frontend, data, and documentation areas so the forecasting pipeline can evolve independently from the application layer.

Key foundation pieces established:

- `ml/data/` for ingestion, validation, auditing, and universe handling.
- `ml/features/` for feature and target construction.
- `ml/tests/` for isolated ML tests.
- `data/raw/` and `data/processed/` placeholders for the future data corpus.
- `ml/pyproject.toml` and `ml/uv.lock` for reproducible Python dependency management.

### Canonical market-data contract

The ingestion layer was built around a project-owned canonical schema instead of exposing the rest of the pipeline directly to the provider dataframe shape.

Canonical fields:

- `session_date`
- `ticker`
- `open`
- `high`
- `low`
- `close`
- `adj_close`
- `volume`

The current ingestion adapter uses `yfinance` for development data and normalizes provider output into this stable schema.

### Data validation

The market-data validator checks structural and numerical invariants before downstream feature generation:

- required columns and schema presence
- valid session dates
- non-empty ticker values
- numeric and finite OHLCV values
- positive price values
- non-negative volume
- valid OHLC relationships
- duplicate `ticker/session_date` keys
- chronological ordering within each ticker

### Corpus auditing

A corpus-audit layer was added to validate stored CSV files and summarize coverage, row counts, dates, duplicates, missing values, and price/volume ranges.

### Development universe

A fixed diversified development universe was introduced so the pipeline could be tested across multiple stocks and sectors before the final research universe is wired in.

> **Market-scope note:** the currently tracked development universe is still the earlier US development set. The planned Indian-market pivot, including Nifty 50 and historical constituent handling, is a later integration step and is not represented here as already completed.

### P1.1 checkpoint

**Status: COMPLETE ✅**

---

# P1.2 · Feature + dataset foundation ✅

P1.2 was completed across six sub-phases, creating the feature-engineering and model-dataset contract required before model training.

### P1.2.1 · Forecasting targets ✅

Forward-looking targets were added at multiple trading-session horizons:

- 1 session
- 5 sessions
- 20 sessions

For each horizon the pipeline constructs:

- forward log-return target
- direction / up target

Target construction is isolated per ticker and explicitly validates chronological input, duplicate keys, positive finite prices, target-horizon validity, and missing future observations.

The final observations of each ticker naturally carry unavailable future targets and remain available for later dataset/warm-up handling.

### P1.2.2 · Returns / price features ✅

Lagged and price-structure features were added using information available at each observation:

- lagged log returns at 1 / 2 / 5 / 10 / 20 sessions
- intraday return
- overnight gap
- high-low range

The implementation validates ticker/session uniqueness, chronological ordering, finite positive prices, and OHLC consistency before calculating the features.

### P1.2.3 · Volatility + volume ✅

The volatility and volume layer adds:

- rolling 5-session volatility
- rolling 20-session volatility
- ATR(14)
- normalized ATR percentage
- 20-session volume average
- volume ratio versus the 20-session average
- 20-session volume z-score
- dollar volume
- rolling dollar-volume average

Warm-up rows are left as missing until enough history exists, and calculations are isolated per ticker.

### P1.2.4 · Technical features ✅

The technical layer adds a broader set of model inputs:

- SMA 10 / 20 / 50 / 200
- EMA 10 / 20 / 50
- close-to-SMA ratios
- RSI(14)
- MACD, signal, and histogram
- stochastic %K / %D
- ADX(14)
- +DI(14) / -DI(14)
- directional spread
- Bollinger Bands
- Bollinger %B and bandwidth
- rolling 20 / 50-session price-trend slopes

Technical indicators are treated as **features**, not as hard-coded trading rules.

The implementation uses the current TA-Lib package and keeps calculations isolated by ticker so one security cannot contaminate another security's indicator history.

### P1.2.5 · Market / context features ✅

Context features extend the stock-level view to market and sector conditions:

**Market context**

- benchmark 1 / 5 / 20-session returns
- benchmark rolling 20-session volatility
- benchmark SMA 20 / 50
- benchmark price-to-SMA ratios
- simple market trend/regime state
- stock-vs-market relative performance at 1 / 5 / 20 sessions

**Sector context**

- sector 1 / 5 / 20-session returns
- sector rolling volatility

The interfaces are intentionally provider-agnostic so the same feature contract can later consume the planned Indian market, Nifty 50, sector, and cross-asset feeds.

### P1.2.6 · Model dataset builder ✅

A model-ready dataset contract was added to make feature/target assembly explicit and defensive.

It provides:

- explicit feature-column schema
- explicit target-column schema
- feature/target overlap protection
- protection against target-prefixed feature leakage
- required-column validation
- duplicate `ticker/session_date` rejection
- chronological-order enforcement
- numeric-type validation
- infinite-value rejection
- configurable incomplete-row handling
- preservation of ticker/date metadata
- optional preservation of prediction-cutoff timestamps

A dedicated information-cutoff validator also rejects data whose `information_timestamp` occurs after the prediction `as_of_timestamp`, providing an explicit guard against future-information leakage.

### P1.2 validation gate 🧪

Focused dataset-builder suite:

```text
uv run --project .\ml pytest .\ml\tests\test_dataset.py -q
14 passed
```

Full ML test suite:

```text
uv run --project .\ml pytest .\ml\tests -q
55 passed
```

Current test coverage is distributed across:

- `test_returns.py`
- `test_targets.py`
- `test_volatility.py`
- `test_technical.py`
- `test_context.py`
- `test_dataset.py`

### P1.2 checkpoint

**Status: COMPLETE ✅**

The project now has a validated feature-engineering and model-dataset contract layer before model selection, training, and financial evaluation begin.

---

## Engineering principles established by Update 001

This foundation deliberately keeps the following boundaries explicit:

1. **Targets are future-looking; features must be information-available at prediction time.**
2. **Per-ticker calculations never bleed across securities.**
3. **Chronological ordering is a hard invariant, not a convenience.**
4. **Feature engineering and dataset assembly do not perform model training, scaling, or splitting prematurely.**
5. **The Indian-market pivot is an upcoming data-integration task, not something to fake by relabeling the current development universe.**
6. **The next major gate is leakage-safe validation before model comparisons are trusted.**

---

## Next gate · P1.3 · Leakage-Safe Validation 🚦

The next phase will establish the evaluation boundary for the entire forecasting system:

- chronological train / validation / test splits
- walk-forward validation
- multi-horizon label purging
- embargo / gap handling where required
- train-only fitting of learned transforms such as scalers
- deterministic split manifests
- explicit tests proving that future information cannot cross the evaluation boundary
