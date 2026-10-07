from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml.validation.model_selection import InnerSelectionSplit
from ml.validation.preprocessor import TrainOnlyPreprocessor


@dataclass(frozen=True)
class InnerFoldData:
    """Precomputed and preprocessed inner fold data for P6 optimization."""

    fold_id: int
    X_fit: pd.DataFrame
    y_fit: pd.DataFrame
    metadata_fit: pd.DataFrame

    X_selection: pd.DataFrame
    y_selection: pd.DataFrame
    metadata_selection: pd.DataFrame

    # Used by sequence models for history
    X_context: pd.DataFrame
    metadata_context: pd.DataFrame

    preprocessor: TrainOnlyPreprocessor
    selection_start: pd.Timestamp
    split: InnerSelectionSplit | None = None


def prepare_inner_folds(
    X_raw: pd.DataFrame,
    y_raw: pd.DataFrame,
    metadata: pd.DataFrame,
    splits: list[InnerSelectionSplit],
) -> list[InnerFoldData]:
    """Prepare precomputed immutable inner folds with preprocessors fitted strictly on Inner Fit."""

    bundles = []
    for i, split in enumerate(splits, start=1):
        # 1. Isolate Inner Fit
        X_fit_raw = X_raw.iloc[split.fit_indices].reset_index(drop=True)
        y_fit = y_raw.iloc[split.fit_indices].reset_index(drop=True)
        meta_fit = metadata.iloc[split.fit_indices].reset_index(drop=True)

        # 2. Fit Preprocessor on Inner Fit ONLY
        preprocessor = TrainOnlyPreprocessor(scaler_type="standard")
        preprocessor.fit(X_fit_raw)

        # 3. Transform Fit
        X_fit = preprocessor.transform(X_fit_raw)

        # 4. Transform Selection
        X_sel_raw = X_raw.iloc[split.selection_indices].reset_index(drop=True)
        y_sel = y_raw.iloc[split.selection_indices].reset_index(drop=True)
        meta_sel = metadata.iloc[split.selection_indices].reset_index(drop=True)
        X_sel = preprocessor.transform(X_sel_raw)

        # 5. Transform Context (up to end of selection)
        # Context needs to include fit and gap (purged) rows up to the start of selection.
        # predict() will concatenate this context with the selection rows automatically.
        min_idx = split.selection_indices.min()
        X_context_raw = X_raw.iloc[:min_idx].reset_index(drop=True)
        meta_context = metadata.iloc[:min_idx].reset_index(drop=True)
        X_context = preprocessor.transform(X_context_raw)

        bundles.append(
            InnerFoldData(
                fold_id=i,
                X_fit=X_fit,
                y_fit=y_fit,
                metadata_fit=meta_fit,
                X_selection=X_sel,
                y_selection=y_sel,
                metadata_selection=meta_sel,
                X_context=X_context,
                metadata_context=meta_context,
                preprocessor=preprocessor,
                selection_start=split.selection_start,
                split=split,
            )
        )

    return bundles
