from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitManifest:
    """
    Deterministic audit record capturing the full provenance of a validation fold.
    """

    fold_id: int
    split_config: dict[str, Any]
    train_date_bounds: tuple[str, str]
    val_date_bounds: tuple[str, str]
    test_date_bounds: tuple[str, str] | None
    n_train_raw: int
    n_train_purged: int
    n_train_final: int
    n_val_raw: int
    n_val_purged: int
    n_val_final: int
    n_test: int
    train_index_hash: str
    val_index_hash: str
    manifest_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _compute_index_hash(indices: np.ndarray) -> str:
    """Compute deterministic SHA-256 digest of sorted row index array."""
    sorted_bytes = np.sort(indices).astype(np.int64).tobytes()
    return hashlib.sha256(sorted_bytes).hexdigest()


def generate_split_manifest(
    *,
    fold_id: int,
    split_config: dict[str, Any],
    train_dates: tuple[pd.Timestamp, pd.Timestamp],
    val_dates: tuple[pd.Timestamp, pd.Timestamp],
    test_dates: tuple[pd.Timestamp, pd.Timestamp] | None,
    train_indices_raw_count: int,
    train_indices_final: np.ndarray,
    val_indices_raw_count: int,
    val_indices_final: np.ndarray,
    test_indices: np.ndarray | None,
) -> SplitManifest:
    """
    Generate an immutable, verifiable SplitManifest for a single fold.
    """
    train_idx_hash = _compute_index_hash(train_indices_final)
    val_idx_hash = _compute_index_hash(val_indices_final)

    train_bounds = (
        train_dates[0].strftime("%Y-%m-%d"),
        train_dates[1].strftime("%Y-%m-%d"),
    )
    val_bounds = (
        val_dates[0].strftime("%Y-%m-%d"),
        val_dates[1].strftime("%Y-%m-%d"),
    )
    test_bounds = (
        (test_dates[0].strftime("%Y-%m-%d"), test_dates[1].strftime("%Y-%m-%d")) if test_dates is not None else None
    )

    n_train_purged = train_indices_raw_count - len(train_indices_final)
    n_val_purged = val_indices_raw_count - len(val_indices_final)
    n_test = len(test_indices) if test_indices is not None else 0

    manifest_payload = {
        "fold_id": fold_id,
        "split_config": split_config,
        "train_date_bounds": train_bounds,
        "val_date_bounds": val_bounds,
        "test_date_bounds": test_bounds,
        "n_train_final": len(train_indices_final),
        "n_train_purged": n_train_purged,
        "n_val_final": len(val_indices_final),
        "n_val_purged": n_val_purged,
        "n_test": n_test,
        "train_index_hash": train_idx_hash,
        "val_index_hash": val_idx_hash,
    }

    serialized_payload = json.dumps(manifest_payload, sort_keys=True)
    manifest_hash = hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()

    return SplitManifest(
        fold_id=fold_id,
        split_config=split_config,
        train_date_bounds=train_bounds,
        val_date_bounds=val_bounds,
        test_date_bounds=test_bounds,
        n_train_raw=train_indices_raw_count,
        n_train_purged=n_train_purged,
        n_train_final=len(train_indices_final),
        n_val_raw=val_indices_raw_count,
        n_val_purged=n_val_purged,
        n_val_final=len(val_indices_final),
        n_test=n_test,
        train_index_hash=train_idx_hash,
        val_index_hash=val_idx_hash,
        manifest_hash=manifest_hash,
    )
