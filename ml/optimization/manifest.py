from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "version",
    "target",
    "task_type",
    "winning_model_family",
    "lookback",
    "best_hyperparameters",
    "model_architecture",
    "feature_schema",
    "preprocessing_configuration",
    "inner_fold_geometry",
    "purge_length",
    "scoring_row_rule",
    "sequence_scoring_alignment_rule",
    "early_stopping_configuration",
    "protocol_a_configuration",
    "random_seed",
    "device",
    "determinism_settings",
    "objective",
    "secondary_metrics",
    "aggregation_rules",
    "baseline_definition",
    "optuna_trial_budget",
    "optuna_sampler_configuration",
    "search_space_definition",
    "dataset_manifest",
    "code_provenance",
    "library_versions",
)

P6_REQUIRED_ADDITIONAL_FIELDS: tuple[str, ...] = (
    "search_space_identity",
    "optuna_trial_metadata",
)

P6_MODEL_FAMILIES = {"XGBoost", "LSTM", "Transformer"}
P6_OBJECTIVES = {
    "regression": "relative_rmse",
    "classification": "logloss",
}
P6_AGGREGATION = "mean_across_tickers_then_mean_across_folds"
P6_SCORING_RULE = "shared_max_lookback_10"
P6_SEQUENCE_ALIGNMENT_RULE = "exact_matching_ticker_and_session_date"


def get_default_library_versions() -> dict[str, str]:
    """Capture runtime versions of critical P6 libraries."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    package_map = {
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        "torch": "torch",
        "xgboost": "xgboost",
        "optuna": "optuna",
        "ta-lib": "ta-lib",
    }

    for key, package_name in package_map.items():
        try:
            versions[key] = str(package_version(package_name))
        except PackageNotFoundError:
            continue

    return versions


def get_default_code_provenance() -> dict[str, Any]:
    """Capture reproducibility-relevant source-control provenance."""
    provenance: dict[str, Any] = {
        "framework": "stock-market-prediction-p6",
        "provenance_schema": "git.v1",
    }

    try:
        provenance["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except OSError, subprocess.CalledProcessError:
        provenance["git_commit"] = None

    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        provenance["git_branch"] = branch or None
    except OSError, subprocess.CalledProcessError:
        provenance["git_branch"] = None

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        provenance["git_dirty"] = bool(status.strip())
    except OSError, subprocess.CalledProcessError:
        provenance["git_dirty"] = None

    return provenance


def build_frozen_manifest(
    *,
    experiment_id: str,
    target: str,
    task_type: str,
    winning_model_family: str,
    best_hyperparameters: dict[str, Any],
    feature_schema: list[str] | tuple[str, ...],
    inner_fold_geometry: list[dict[str, Any]],
    purge_length: int,
    dataset_manifest: dict[str, Any] | str,
    lookback: int | None = None,
    version: str = "1.0.0",
    model_architecture: dict[str, Any] | None = None,
    preprocessing_configuration: dict[str, Any] | None = None,
    scoring_row_rule: str = P6_SCORING_RULE,
    sequence_scoring_alignment_rule: str = P6_SEQUENCE_ALIGNMENT_RULE,
    early_stopping_configuration: dict[str, Any] | None = None,
    protocol_a_configuration: dict[str, Any] | None = None,
    random_seed: int = 42,
    device: str = "cpu",
    determinism_settings: dict[str, Any] | None = None,
    objective: str | None = None,
    secondary_metrics: list[str] | None = None,
    aggregation_rules: str = P6_AGGREGATION,
    baseline_definition: str | None = None,
    optuna_trial_budget: int = 30,
    optuna_sampler_configuration: dict[str, Any] | None = None,
    search_space_definition: dict[str, Any] | None = None,
    search_space_identity: str | None = None,
    optuna_trial_metadata: dict[str, Any] | None = None,
    code_provenance: dict[str, Any] | None = None,
    library_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a frozen configuration dictionary for P6."""
    if task_type == "classification":
        default_objective = "logloss"
        default_secondary = ["accuracy", "precision", "recall", "f1"]
        default_baseline = "majority_class_baseline"
    else:
        default_objective = "relative_rmse"
        default_secondary = ["rmse", "mae", "r2", "directional_accuracy", "pearson_ic"]
        default_baseline = "zero_return_baseline"

    if winning_model_family == "XGBoost":
        default_early_stopping = {
            "early_stopping_rounds": 30,
            "n_estimators": 1000,
            "selection_rule": "best_validation_iteration",
        }
    else:
        default_early_stopping = {
            "epochs": 15,
            "selection_rule": "best_validation_epoch",
        }

    config: dict[str, Any] = {
        "experiment_id": experiment_id,
        "version": version,
        "target": target,
        "task_type": task_type,
        "winning_model_family": winning_model_family,
        "lookback": lookback if lookback is not None else best_hyperparameters.get("lookback"),
        "best_hyperparameters": dict(best_hyperparameters),
        "model_architecture": model_architecture or {"family": winning_model_family},
        "feature_schema": list(feature_schema),
        "preprocessing_configuration": preprocessing_configuration
        or {
            "scaler_type": "standard",
            "fit_scope": "inner_fit_only",
        },
        "inner_fold_geometry": inner_fold_geometry,
        "purge_length": purge_length,
        "scoring_row_rule": scoring_row_rule,
        "sequence_scoring_alignment_rule": sequence_scoring_alignment_rule,
        "early_stopping_configuration": early_stopping_configuration or default_early_stopping,
        "protocol_a_configuration": protocol_a_configuration
        or {
            "fit_fold": 2,
            "selection_fold": 2,
            "refit_fresh_model": True,
            "outer_retraining": False,
            "early_stopping_source": "fold_2_inner_selection",
        },
        "random_seed": random_seed,
        "device": device,
        "determinism_settings": determinism_settings
        or {
            "torch_deterministic": True,
            "numpy_seed": random_seed,
        },
        "objective": objective or default_objective,
        "secondary_metrics": secondary_metrics or default_secondary,
        "aggregation_rules": aggregation_rules,
        "baseline_definition": baseline_definition or default_baseline,
        "optuna_trial_budget": optuna_trial_budget,
        "optuna_sampler_configuration": optuna_sampler_configuration
        or {
            "sampler": "TPESampler",
            "seed": random_seed,
            "n_startup_trials": 10,
            "n_jobs": 1,
            "pruner": "NopPruner",
        },
        "search_space_identity": search_space_identity,
        "search_space_definition": search_space_definition or {},
        "optuna_trial_metadata": optuna_trial_metadata or {},
        "dataset_manifest": dataset_manifest,
        "code_provenance": code_provenance or get_default_code_provenance(),
        "library_versions": library_versions or get_default_library_versions(),
    }
    return config


def validate_frozen_manifest(config: dict[str, Any]) -> None:
    """Validate the generic frozen-manifest schema."""
    missing = [k for k in REQUIRED_MANIFEST_FIELDS if k not in config]
    if missing:
        raise ValueError(f"Frozen manifest is missing required fields: {missing}")

    nullable_fields = {"lookback"}

    for field in REQUIRED_MANIFEST_FIELDS:
        if field in nullable_fields:
            continue

        if config[field] is None:
            raise ValueError(f"Frozen manifest field '{field}' cannot be null.")


def validate_p6_frozen_manifest(config: dict[str, Any]) -> None:
    """Validate the locked P6 v1 experiment contract."""
    validate_frozen_manifest(config)

    missing = [field for field in P6_REQUIRED_ADDITIONAL_FIELDS if field not in config]
    if missing:
        raise ValueError(f"P6 frozen manifest is missing required fields: {missing}")

    family = config["winning_model_family"]
    if family not in P6_MODEL_FAMILIES:
        raise ValueError(f"Unsupported P6 model family: {family!r}")

    task_type = config["task_type"]
    if task_type not in P6_OBJECTIVES:
        raise ValueError(f"Unsupported P6 task_type: {task_type!r}")

    expected_objective = P6_OBJECTIVES[task_type]
    if config["objective"] != expected_objective:
        raise ValueError(f"P6 objective mismatch: expected {expected_objective!r}, got {config['objective']!r}")

    if config["aggregation_rules"] != P6_AGGREGATION:
        raise ValueError("P6 aggregation rule is not locked.")

    if config["scoring_row_rule"] != P6_SCORING_RULE:
        raise ValueError("P6 scoring-row rule is not locked.")

    if config["sequence_scoring_alignment_rule"] != P6_SEQUENCE_ALIGNMENT_RULE:
        raise ValueError("P6 sequence scoring alignment rule is not locked.")

    if config["optuna_trial_budget"] != 30:
        raise ValueError("P6 v1 requires exactly 30 Optuna trials.")

    sampler = config["optuna_sampler_configuration"]
    if sampler.get("sampler") != "TPESampler":
        raise ValueError("P6 v1 requires TPESampler.")
    if sampler.get("seed") != config["random_seed"]:
        raise ValueError("P6 sampler seed must equal random_seed.")
    if sampler.get("n_startup_trials") != 10:
        raise ValueError("P6 v1 requires 10 startup trials.")
    if sampler.get("n_jobs") != 1:
        raise ValueError("P6 v1 requires n_jobs=1.")
    if sampler.get("pruner") not in {None, "NopPruner"}:
        raise ValueError("P6 v1 forbids Optuna pruning.")

    if not config["search_space_identity"]:
        raise ValueError("P6 requires search_space_identity.")

    if not isinstance(config["search_space_definition"], dict) or not config["search_space_definition"]:
        raise ValueError("P6 requires a non-empty search_space_definition.")

    if not isinstance(config["optuna_trial_metadata"], dict) or not config["optuna_trial_metadata"]:
        raise ValueError("P6 requires retained Optuna trial metadata.")

    if not isinstance(config["code_provenance"], dict) or not config["code_provenance"]:
        raise ValueError("P6 requires non-empty code provenance.")

    protocol = config["protocol_a_configuration"]
    expected_protocol = {
        "fit_fold": 2,
        "selection_fold": 2,
        "refit_fresh_model": True,
        "outer_retraining": False,
        "early_stopping_source": "fold_2_inner_selection",
    }
    for key, expected_value in expected_protocol.items():
        if protocol.get(key) != expected_value:
            raise ValueError(
                f"P6 Protocol A mismatch for {key!r}: expected {expected_value!r}, got {protocol.get(key)!r}"
            )

    early = config["early_stopping_configuration"]
    if family == "XGBoost":
        expected_early = {
            "early_stopping_rounds": 30,
            "n_estimators": 1000,
            "selection_rule": "best_validation_iteration",
        }
        for key, expected_value in expected_early.items():
            if early.get(key) != expected_value:
                raise ValueError(
                    f"P6 XGBoost early-stopping mismatch for {key!r}: "
                    f"expected {expected_value!r}, got {early.get(key)!r}"
                )
        if config["lookback"] is not None:
            raise ValueError("XGBoost P6 manifests must use lookback=None.")
    else:
        if early.get("epochs") != 15:
            raise ValueError("P6 neural search requires epochs=15.")
        if early.get("selection_rule") != "best_validation_epoch":
            raise ValueError("P6 neural search must select the best validation epoch.")
        if config["lookback"] not in {5, 10}:
            raise ValueError("Sequence-model lookback must be 5 or 10.")


def generate_frozen_manifest(config: dict[str, Any]) -> tuple[str, str]:
    """
    Validate and serialize the frozen configuration to canonical JSON and generate SHA-256.

    Returns:
        (canonical_json_string, sha256_hash_string)
    """
    validate_frozen_manifest(config)
    canonical_json = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    manifest_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return canonical_json, manifest_hash


def verify_manifest_hash(
    config: dict[str, Any],
    expected_hash: str,
) -> None:
    """Verify that a configuration reproduces an expected canonical SHA-256 hash."""
    _, actual_hash = generate_frozen_manifest(config)
    if actual_hash != expected_hash:
        raise ValueError(f"Frozen manifest hash mismatch: expected {expected_hash}, calculated {actual_hash}")
