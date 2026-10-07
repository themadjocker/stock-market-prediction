from __future__ import annotations

import numpy as np
import pandas as pd

from ml.validation.purging import (
    apply_embargo,
    calculate_label_outcome_dates,
    protect_validation_horizon,
    purge_train_horizon,
)


def make_metadata(tickers: list[str], days: int = 30) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        dates = pd.date_range("2026-01-01", periods=days, freq="B")
        for d in dates:
            rows.append({"ticker": ticker, "session_date": d})
    return pd.DataFrame(rows)


def test_calculate_label_outcome_dates_exact() -> None:
    meta = make_metadata(["AAA"], days=10)
    outcomes = calculate_label_outcome_dates(meta, horizon=3)

    # First session is 2026-01-01 (Thu). Outcome +3 sessions is 2026-01-06 (Tue)
    dates = pd.date_range("2026-01-01", periods=10, freq="B")
    assert outcomes.iloc[0] == dates[3]
    assert outcomes.iloc[6] == dates[9]
    # Tail 3 observations cannot resolve
    assert outcomes.iloc[7:].isna().all()


def test_purge_train_horizon_drops_overlapping_samples() -> None:
    meta = make_metadata(["AAA", "BBB"], days=20)
    dates = pd.date_range("2026-01-01", periods=20, freq="B")

    # Let train be sessions 0..11, validation starts at session 12 (dates[12])
    train_mask = meta["session_date"].isin(dates[:12]).to_numpy()
    train_indices = np.flatnonzero(train_mask)

    val_start = dates[12]
    horizon = 5  # 5 sessions

    safe_indices, purged_count = purge_train_horizon(
        meta,
        train_indices,
        val_start_date=val_start,
        horizon=horizon,
    )

    safe_dates = meta.iloc[safe_indices]["session_date"]
    # The latest safe training date must be dates[6], because 6 + 5 = 11 < 12!
    # Any date >= dates[7] has outcome >= dates[12], which is >= val_start!
    assert safe_dates.max() == dates[6]
    # 5 sessions per ticker (sessions 7, 8, 9, 10, 11) must be purged -> 10 total
    assert purged_count == 10


def test_protect_validation_horizon_drops_test_overlapping_samples() -> None:
    meta = make_metadata(["AAA"], days=20)
    dates = pd.date_range("2026-01-01", periods=20, freq="B")

    val_indices = np.arange(10, 16)  # sessions 10..15
    test_start = dates[16]
    horizon = 2

    # sessions 10..13 have outcomes: 10+2=12, 11+2=13, 12+2=14, 13+2=15 < 16
    # session 14+2=16 >= test_start -> purged!
    # session 15+2=17 >= test_start -> purged!
    safe_indices, purged_count = protect_validation_horizon(
        meta,
        val_indices,
        test_start_date=test_start,
        horizon=horizon,
    )

    assert len(safe_indices) == 4
    assert purged_count == 2
    assert 14 not in safe_indices
    assert 15 not in safe_indices


def test_apply_embargo_excludes_specified_sessions() -> None:
    meta = make_metadata(["AAA"], days=15)
    dates = pd.date_range("2026-01-01", periods=15, freq="B")

    candidate_indices = np.arange(10, 15)  # dates[10..14]
    embargo_start = dates[10]

    # 2 sessions embargo -> dates[10] and dates[11] must be dropped
    embargoed_indices, purged_count = apply_embargo(
        meta,
        candidate_indices,
        embargo_start_date=embargo_start,
        embargo_sessions=2,
    )

    assert purged_count == 2
    assert len(embargoed_indices) == 3
    remaining_dates = meta.iloc[embargoed_indices]["session_date"].tolist()
    assert dates[10] not in remaining_dates
    assert dates[11] not in remaining_dates
    assert dates[12] in remaining_dates
