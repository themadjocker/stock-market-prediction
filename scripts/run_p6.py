import argparse
import logging
import os
import sys
from typing import Any

import numpy as np
import optuna
import pandas as pd

# The script bootstraps the repository root before importing project modules.
# Ruff I001 cannot safely reorder these imports across that runtime bootstrap.

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure root package is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.features.dataset import ModelDataset, build_model_dataset
from ml.models.lstm_model import LSTMModel
from ml.models.metrics import calculate_classification_metrics, calculate_metrics
from ml.models.target import target_spec_for
from ml.models.transformer_model import TransformerModel
from ml.models.xgboost_model import XGBoostModel
from ml.optimization.folds import prepare_inner_folds
from ml.optimization.manifest import (
    build_frozen_manifest,
    generate_frozen_manifest,
    validate_p6_frozen_manifest,
)
from ml.optimization.outer_gate import OuterValidationGate
from ml.optimization.search import (
    create_lstm_objective,
    create_transformer_objective,
    create_xgboost_objective,
    get_search_space_definition,
    get_search_space_hash,
    get_search_space_identity,
    summarize_study,
    validate_study_execution,
)
from ml.validation.model_selection import make_p6_inner_folds
from ml.validation.split import chronological_split


def make_pipeline_dataset(n_days: int = 150) -> ModelDataset:
    """Generate a synthetic dataset for P6 pipeline testing."""
    dates = pd.date_range("2026-01-01", periods=n_days, freq="B")
    rows = []

    # We need realistic column names
    feature_columns = [
        "sma_10", "sma_20", "sma_50", "sma_200", "ema_10", "ema_20", "ema_50",
        "close_to_sma_10", "close_to_sma_20", "close_to_sma_50", "close_to_sma_200",
        "rsi_14", "macd_12_26", "macd_signal_9", "macd_hist_12_26_9",
        "stoch_k_14", "stoch_d_14_3", "adx_14", "plus_di_14", "minus_di_14",
        "di_spread_14", "bb_percent_b_20_2", "bb_bandwidth_20_2",
        "trend_slope_20", "trend_slope_50",
        "volatility_5d", "volatility_20d", "volatility_60d",
        "volume_change_1d", "volume_sma_5d", "volume_zscore_5d",
        "volume_sma_20d", "volume_zscore_20d"
    ]
    
    for ticker in ["AAA", "BBB"]:
        for i, d in enumerate(dates):
            row = {
                "ticker": ticker,
                "session_date": d,
                "target_return_1d": 0.01 * (1 if i % 2 == 0 else -1),
                "target_return_5d": 0.02 * (1 if i % 2 == 0 else -1),
                "target_up_1d": 1 if i % 2 == 0 else 0,
                "target_up_5d": 1 if i % 2 == 0 else 0,
            }
            # Fill dummy features
            for f in feature_columns:
                row[f] = np.random.randn()
            rows.append(row)
            
    df = pd.DataFrame(rows)
    df = df.sort_values(["session_date", "ticker"]).reset_index(drop=True)
    return build_model_dataset(
        df,
        feature_columns=feature_columns,
        target_columns=["target_return_1d", "target_return_5d", "target_up_1d", "target_up_5d"],
    )


DEFAULT_N_TRIALS = 30
DEFAULT_N_STARTUP_TRIALS = 10
FIXED_SEED = 42


def run_p6(
    target_name: str,
    n_trials: int = DEFAULT_N_TRIALS,
    n_startup_trials: int = DEFAULT_N_STARTUP_TRIALS,
    seed: int = FIXED_SEED,
) -> dict[str, Any]:
    logger.info("Resolving active target specification...")
    target_spec = target_spec_for(target_name)
    task_type = target_spec.task_type
    target_horizon = target_spec.horizon
    logger.info(f"Target: {target_name} ({task_type}), horizon: {target_horizon} sessions")

    logger.info("Building model dataset...")
    dataset = make_pipeline_dataset(n_days=150)
    
    logger.info("Performing outer chronological split using target-specific horizon...")
    # Outer split uses target_spec.horizon dynamically (never hardcoded 5)
    outer_bundle, _test_bundle, _temporal_split = chronological_split(
        dataset,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        target_horizon=target_horizon,
        embargo_sessions=0,
    )
    
    # Generate Inner Folds using target_spec.horizon
    logger.info("Generating inner folds...")
    raw_inner_splits = make_p6_inner_folds(
        outer_bundle.metadata_train,
        target_horizon=target_horizon,
    )
    
    logger.info("Preparing precomputed inner folds with TrainOnlyPreprocessor...")
    X_train_raw = outer_bundle.X_train_raw
    if X_train_raw is None:
        raise RuntimeError(
            "P6 requires raw outer-training features for inner-fold preparation."
        )
    inner_folds = prepare_inner_folds(
        X_raw=X_train_raw,
        y_raw=outer_bundle.y_train[[target_name]],
        metadata=outer_bundle.metadata_train,
        splits=raw_inner_splits,
    )
    
    # P6 v1 is locked to exactly 30 trials and 10 random-startup trials.
    if n_trials != DEFAULT_N_TRIALS:
        raise ValueError(f"P6 v1 requires exactly {DEFAULT_N_TRIALS} Optuna trials.")
    if n_startup_trials != DEFAULT_N_STARTUP_TRIALS:
        raise ValueError(
            f"P6 v1 requires exactly {DEFAULT_N_STARTUP_TRIALS} startup trials."
        )

    logger.info(f"Starting Hyperparameter Optimization (budget: {n_trials} trials, startup: {n_startup_trials}, seed: {seed})...")
    studies: dict[str, optuna.Study] = {}
    
    objectives = [
        ("XGBoost", create_xgboost_objective(inner_folds, task_type, target_name=target_name)),
        ("LSTM", create_lstm_objective(inner_folds, task_type, target_name=target_name)),
        ("Transformer", create_transformer_objective(inner_folds, task_type, target_name=target_name)),
    ]
    
    for family, objective_fn in objectives:
        logger.info(f"Optimizing {family} with locked TPESampler...")
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)
        study = optuna.create_study(
            study_name=f"{family}_p6_study",
            direction="minimize",
            sampler=sampler,
        )
        study.optimize(objective_fn, n_trials=n_trials, n_jobs=1)
        studies[family] = study
        
    for family, study in studies.items():
        validate_study_execution(study, expected_n_trials=DEFAULT_N_TRIALS)

    best_family = None
    best_value = float("inf")
    best_params: dict[str, Any] = {}
    
    for family, study in studies.items():
        if len(study.trials) > 0 and study.best_value < best_value:
            best_value = study.best_value
            best_family = family
            best_params = study.best_params
            
    logger.info(f"Winning model family: {best_family} with inner objective score: {best_value}")
    
    # Extract inner fold geometry for audit manifest
    fold_geometry = [
        {
            "fold_id": fold.fold_id,
            "raw_fit_count": fold.split.raw_fit_count if fold.split else len(fold.X_fit),
            "purged_fit_count": fold.split.purged_fit_count if fold.split else 0,
            "selection_count": len(fold.X_selection),
            "selection_start": str(fold.selection_start),
        }
        for fold in inner_folds
    ]

    # The gate owns Outer Validation access. Inner optimization above never receives
    # the gate, so Outer Validation cannot participate in tuning or selection.
    outer_gate = OuterValidationGate(outer_bundle)

    logger.info("Writing complete frozen configuration manifest (BEFORE Outer Validation gate)...")
    winning_study = studies[str(best_family)]
    winning_search_space = get_search_space_definition(str(best_family))
    frozen_config = build_frozen_manifest(
        experiment_id=f"exp_p6_{target_name}_{best_family}",
        target=target_name,
        task_type=task_type,
        winning_model_family=str(best_family),
        best_hyperparameters=best_params,
        feature_schema=list(dataset.feature_columns),
        inner_fold_geometry=fold_geometry,
        purge_length=target_horizon,
        dataset_manifest=outer_bundle.manifest_hash,
        lookback=best_params.get("lookback"),
        model_architecture={
            "family": str(best_family),
            "locked_parameters": dict(best_params),
        },
        random_seed=seed,
        optuna_trial_budget=n_trials,
        optuna_sampler_configuration={
            "sampler": "TPESampler",
            "seed": seed,
            "n_startup_trials": n_startup_trials,
            "n_jobs": 1,
            "pruner": "NopPruner",
        },
        search_space_identity=get_search_space_identity(str(best_family)),
        search_space_definition=winning_search_space,
        optuna_trial_metadata={
            "winning_model_family": str(best_family),
            "winning_search_space_hash": get_search_space_hash(str(best_family)),
            "studies": {
                family: summarize_study(study)
                for family, study in studies.items()
            },
            "winning_study": summarize_study(winning_study),
        },
    )
    validate_p6_frozen_manifest(frozen_config)
    canonical_json, manifest_hash = generate_frozen_manifest(frozen_config)
    
    manifest_file = f"p6_frozen_{target_name}.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write(canonical_json)
        
    logger.info(f"P6 Frozen Manifest Hash: {manifest_hash} saved to {manifest_file}")

    # --- OUTER VALIDATION GATE ---
    # Only after the frozen manifest is written may the immutable gate be opened.
    outer_gate = outer_gate.freeze().open()
    outer_bundle = outer_gate.get_bundle()
    logger.info("GATE OPENED: Accessing Outer Validation for Protocol A Final Evaluation...")
    fold_2 = inner_folds[1]  # Protocol A uses Fold 2
    
    # 1. Instantiate fresh model with locked settings
    if best_family == "XGBoost":
        xgb_params = {
            "n_estimators": 1000,
            "early_stopping_rounds": 30,
            **best_params,
        }
        final_model = XGBoostModel(**xgb_params)
    elif best_family == "LSTM":
        final_model = LSTMModel(**best_params, max_lookback=10)
    elif best_family == "Transformer":
        final_model = TransformerModel(**best_params, max_lookback=10)
    else:
        raise ValueError(f"Unknown winning family: {best_family}")
        
    # 2. Train on Fold 2 Inner Fit, early stopping on Fold 2 Inner Selection
    # Pass historical feature context to sequence models so purge gap continuity is preserved.
    logger.info("Training final model on Fold 2 Inner Fit with early stopping on Fold 2 Inner Selection...")
    fit_kwargs: dict[str, Any] = {
        "X_train": fold_2.X_fit,
        "y_train": fold_2.y_fit,
        "X_val": fold_2.X_selection,
        "y_val": fold_2.y_selection,
        "metadata_train": fold_2.metadata_fit,
        "metadata_val": fold_2.metadata_selection,
    }
    if hasattr(final_model, "lookback"):
        fit_kwargs["val_history_X"] = fold_2.X_context
        fit_kwargs["val_history_metadata"] = fold_2.metadata_context
        
    final_model.fit(**fit_kwargs)
    
    best_iteration = getattr(final_model, "_selection_best_iteration", None)
    best_epoch = getattr(final_model, "_selection_best_epoch", None)
    logger.info(f"Protocol A fit complete. Best step: iteration={best_iteration}, epoch={best_epoch}")
    
    # 3. Outer Validation Prediction using Fold 2 preprocessor
    logger.info("Evaluating final model on Outer Validation...")
    X_outer_val_raw = outer_bundle.X_val_raw
    if X_outer_val_raw is None:
        raise RuntimeError(
            "P6 requires raw outer-validation features for final evaluation."
        )
    y_outer_val = outer_bundle.y_val[[target_name]]
    meta_outer_val = outer_bundle.metadata_val
    
    X_outer_val = fold_2.preprocessor.transform(X_outer_val_raw)
    
    if hasattr(final_model, "lookback"):
        # Context before validation includes the unpurged outer train features
        # (including the outer purge gap sessions), preserving continuity.
        if outer_bundle.X_train_context_raw is not None:
            X_outer_context_raw = outer_bundle.X_train_context_raw
            meta_outer_context = outer_bundle.metadata_train_context
            if meta_outer_context is None:
                raise RuntimeError(
                    "P6 sequence evaluation requires matching training-context metadata."
                )
        else:
            X_outer_context_raw = X_train_raw
            meta_outer_context = outer_bundle.metadata_train
            
        X_outer_context = fold_2.preprocessor.transform(X_outer_context_raw)
        
        if task_type == "classification":
            y_pred = final_model.predict_proba(
                X_outer_val,
                metadata=meta_outer_val,
                history_X=X_outer_context,
                history_metadata=meta_outer_context,
            )
        else:
            y_pred = final_model.predict(
                X_outer_val,
                metadata=meta_outer_val,
                history_X=X_outer_context,
                history_metadata=meta_outer_context,
            )
    else:
        if task_type == "classification":
            y_pred = final_model.predict_proba(X_outer_val)
        else:
            y_pred = final_model.predict(X_outer_val)
            
    y_true = y_outer_val.iloc[:, 0].to_numpy(dtype=float)
    
    if task_type == "classification":
        final_metrics = calculate_classification_metrics(y_true, y_pred)
    else:
        final_metrics = calculate_metrics(y_true, y_pred)
        
    logger.info(f"Outer Validation Metrics: {final_metrics}")
    logger.info("Protocol A Complete.")
    
    return {
        "target": target_name,
        "task_type": task_type,
        "target_horizon": target_horizon,
        "winning_family": best_family,
        "best_hyperparameters": best_params,
        "best_iteration": best_iteration,
        "best_epoch": best_epoch,
        "manifest_hash": manifest_hash,
        "outer_validation_metrics": final_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="target_return_5d")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--n-startup-trials", type=int, default=DEFAULT_N_STARTUP_TRIALS)
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    args = parser.parse_args()
    run_p6(
        args.target,
        n_trials=args.n_trials,
        n_startup_trials=args.n_startup_trials,
        seed=args.seed,
    )
