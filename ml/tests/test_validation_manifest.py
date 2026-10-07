from __future__ import annotations

import numpy as np
import pandas as pd

from ml.validation.manifest import generate_split_manifest


def test_deterministic_manifest_hashes() -> None:
    train_dates = (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-20"))
    val_dates = (pd.Timestamp("2026-01-21"), pd.Timestamp("2026-01-31"))
    train_idx = np.array([0, 1, 2, 3, 4, 5])
    val_idx = np.array([6, 7, 8])

    manifest1 = generate_split_manifest(
        fold_id=0,
        split_config={"h": 5},
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=None,
        train_indices_raw_count=8,
        train_indices_final=train_idx,
        val_indices_raw_count=3,
        val_indices_final=val_idx,
        test_indices=None,
    )

    manifest2 = generate_split_manifest(
        fold_id=0,
        split_config={"h": 5},
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=None,
        train_indices_raw_count=8,
        train_indices_final=train_idx,
        val_indices_raw_count=3,
        val_indices_final=val_idx,
        test_indices=None,
    )

    # Identical inputs must yield identical hash
    assert manifest1.manifest_hash == manifest2.manifest_hash
    assert manifest1.train_index_hash == manifest2.train_index_hash

    # Changed index must produce changed hash
    manifest3 = generate_split_manifest(
        fold_id=0,
        split_config={"h": 5},
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=None,
        train_indices_raw_count=8,
        train_indices_final=np.array([0, 1, 2, 3, 4]),  # different
        val_indices_raw_count=3,
        val_indices_final=val_idx,
        test_indices=None,
    )

    assert manifest1.manifest_hash != manifest3.manifest_hash
