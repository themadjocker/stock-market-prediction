from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


METADATA_COLUMNS: tuple[str, ...] = (
    "ticker",
    "session_date",
)
TARGET_PREFIX = "target_"


@dataclass(frozen=True)
class ModelDataset:
    """Validated model-ready matrices plus point-in-time metadata."""

    X: pd.DataFrame
    y: pd.DataFrame
    metadata: pd.DataFrame

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return tuple(self.X.columns)

    @property
    def target_columns(self) -> tuple[str, ...]:
        return tuple(self.y.columns)

    @property
    def n_rows(self) -> int:
        return len(self.X)


def _as_tuple(columns: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(columns)
    if not result:
        raise ValueError(f"{name} must contain at least one column.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique column names.")
    return result


def _validate_structure(
    data: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
) -> None:
    required = set(METADATA_COLUMNS) | set(feature_columns) | set(target_columns)
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    overlap = set(feature_columns) & set(target_columns)
    if overlap:
        raise ValueError(
            "Feature and target columns overlap: " + ", ".join(sorted(overlap))
        )

    leaked_feature_names = [
        column for column in feature_columns if column.startswith(TARGET_PREFIX)
    ]
    if leaked_feature_names:
        raise ValueError(
            "Target-like columns cannot be used as features: "
            + ", ".join(sorted(leaked_feature_names))
        )

    if data[["ticker", "session_date"]].duplicated().any():
        raise ValueError("Duplicate ticker/session_date rows detected.")

    if data["session_date"].isna().any():
        raise ValueError("session_date contains missing values.")

    parsed_dates = pd.to_datetime(data["session_date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("session_date contains invalid datetime values.")

    chronological = data.assign(__parsed_session_date=parsed_dates)
    for ticker, ticker_frame in chronological.groupby("ticker", sort=False):
        if not ticker_frame["__parsed_session_date"].is_monotonic_increasing:
            raise ValueError(
                f"session_date must be chronological for ticker {ticker}."
            )

    for column in feature_columns + target_columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"{column} must be numeric.")


def validate_information_cutoff(
    data: pd.DataFrame,
    *,
    as_of_column: str = "as_of_timestamp",
    available_at_column: str = "information_timestamp",
) -> None:
    """Reject information whose availability is after the prediction cutoff."""
    missing = {as_of_column, available_at_column}.difference(data.columns)
    if missing:
        raise ValueError(
            "Missing timestamp columns: " + ", ".join(sorted(missing))
        )

    as_of = pd.to_datetime(data[as_of_column], errors="coerce", utc=True)
    available_at = pd.to_datetime(
        data[available_at_column], errors="coerce", utc=True
    )

    if (as_of.isna() | available_at.isna()).any():
        raise ValueError(
            "Prediction and information timestamps must be valid datetimes."
        )

    future_mask = available_at > as_of
    if future_mask.any():
        first_bad = int(np.flatnonzero(future_mask.to_numpy())[0])
        raise ValueError(
            "Future information detected: "
            f"{available_at.iloc[first_bad]} occurs after "
            f"prediction cutoff {as_of.iloc[first_bad]}."
        )


def build_model_dataset(
    data: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    drop_incomplete: bool = True,
) -> ModelDataset:
    """
    Build a validated model-ready dataset without fitting transforms.

    Scaling, time-series splitting, and sequence/window creation are left to
    later pipeline stages.
    """
    features = _as_tuple(feature_columns, "feature_columns")
    targets = _as_tuple(target_columns, "target_columns")

    _validate_structure(
        data,
        feature_columns=features,
        target_columns=targets,
    )

    if data.empty:
        return ModelDataset(
            X=pd.DataFrame(columns=features, index=data.index),
            y=pd.DataFrame(columns=targets, index=data.index),
            metadata=data.loc[:, list(METADATA_COLUMNS)].copy(),
        )

    frame = data.copy()
    numeric_columns = list(features + targets)
    frame.loc[:, numeric_columns] = frame.loc[:, numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    if np.isinf(frame.loc[:, list(features)].to_numpy(dtype=float)).any():
        raise ValueError("Feature columns contain non-finite values.")

    if np.isinf(frame.loc[:, list(targets)].to_numpy(dtype=float)).any():
        raise ValueError("Target columns contain non-finite values.")

    selected_columns = list(METADATA_COLUMNS + features + targets)
    selected = frame.loc[:, selected_columns].copy()

    if drop_incomplete:
        selected = selected.dropna(subset=list(features + targets), how="any")
        if selected.empty:
            raise ValueError(
                "No complete model rows remain after missing-value filtering."
            )

    metadata = selected.loc[:, list(METADATA_COLUMNS)].reset_index(drop=True)
    X = selected.loc[:, list(features)].reset_index(drop=True)
    y = selected.loc[:, list(targets)].reset_index(drop=True)

    return ModelDataset(X=X, y=y, metadata=metadata)
