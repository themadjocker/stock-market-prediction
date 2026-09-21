# Stock Market Prediction
https://delightful-conkies-82be02.netlify.app/

## Package manager

This project uses **pnpm**. Do not commit `package-lock.json`; the canonical lockfile is `pnpm-lock.yaml`.

### Setup

```bash
corepack enable
corepack prepare pnpm@11.27.1 --activate
pnpm install
```

If Corepack is unavailable on your machine, install pnpm 11 directly and then run `pnpm install`.

### Development

```bash
pnpm dev
```

### Build

```bash
pnpm build
```

### Lint

```bash
pnpm lint
```

### Preview

```bash
pnpm preview
```

## Python environments

Backend environment:

```bash
uv sync --directory backend
```

ML environment:

```bash
uv sync --project ml
```

## Project structure

```text
frontend/   React + Vite frontend
backend/    FastAPI backend
ml/         Data, features, models, training, evaluation
data/       Local datasets and data markers
docs/       Engineering notes and updates
```

---

# 📜 Project Update Log

> A chronological record of the engineering milestones, decisions, validations, and major changes made throughout the project.

---

<details open>
<summary><h2>🚀 Update 001 · ML Data Foundation</h2></summary>

**📅 Date:** September 21, 2026
**🟢 Status:** Complete

### 🎯 What changed

The first production-oriented market-data foundation for the forecasting pipeline was established.

The project moved from a frontend-focused application toward a multi-stock machine-learning pipeline with reproducible market-data ingestion and validation.

### 🧩 Implemented

| Area           | Implementation                                     |
| -------------- | -------------------------------------------------- |
| 🌐 Universe    | Fixed 50-stock diversified US development universe |
| 📡 Data source | `yfinance`                                         |
| 🕐 Frequency   | Daily market sessions                              |
| 📦 Storage     | Per-ticker CSV files under `data/raw/`             |
| 🧱 Schema      | Canonical OHLCV + adjusted-close schema            |
| 🔍 Validation  | Automated structural and market-data sanity checks |
| 📊 Audit       | Corpus-level coverage and integrity audit          |
| 🗓️ Alignment  | Exact trading-session alignment across all tickers  |

### 📈 Verified Dataset

| Metric                     |                      Result |
| -------------------------- | --------------------------: |
| **Development tickers**    |                      **50** |
| **Total rows**             |                  **62,750** |
| **Rows per ticker**        |                   **1,255** |
| **Historical range**       | **2021-09-20 → 2026-09-18** |
| **Duplicate sessions**     |                       **0** |
| **Missing values**         |                       **0** |
| **Identical session sets** |                     **Yes** |

### 🔎 Corpus Verification

```text
TICKERS: 50
TOTAL ROWS: 62750

UNIQUE ROW COUNTS: 1
UNIQUE START DATES: 1
UNIQUE END DATES: 1

TOTAL DUPLICATES: 0
TOTAL MISSING VALUES: 0

STATUS: AUDIT COMPLETE
```

### 🗓️ Trading-Session Alignment

```text
TICKERS: 50
UNIQUE DATE SETS: 1
COMMON DATES: 1255
UNION DATES: 1255

ALL DATE SETS IDENTICAL: True
```

### 🧠 Engineering Notes

The 50-stock universe is currently a **development universe**, not the final research universe.

The ingestion layer converts the provider-specific dataframe format into a stable internal schema:

```text
session_date
ticker
open
high
low
close
adj_close
volume
```

The raw market dataset is generated locally and is intended to remain outside version control.

The development corpus can be regenerated with:

```powershell
uv run --project .\ml python -m ml.data.fetch_universe --period 5y
```

### ✅ Milestone Result

```text
DATA INGESTION        ✅
NORMALIZATION         ✅
VALIDATION            ✅
50-STOCK CORPUS       ✅
COVERAGE AUDIT        ✅
SESSION ALIGNMENT     ✅

P1 DATA FOUNDATION    🟢 COMPLETE
P1 MODELING           🔴 NOT STARTED
```

### 🔜 Next Engineering Phase

**Target Definition + Leakage-Safe Feature Engineering**

The next phase will establish:

* 🎯 next-day forecasting targets
* 🧮 feature construction
* ⏱️ information cutoffs
* 🛡️ leakage controls
* 📅 time-aware validation
* 📊 training-example generation

> **Milestone:** The data foundation is ready. The predictive modeling pipeline begins next.

</details>

---
