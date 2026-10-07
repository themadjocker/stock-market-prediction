from __future__ import annotations

import copy

import numpy as np
import optuna
import pytest

from ml.optimization.manifest import (
    build_frozen_manifest,
    generate_frozen_manifest,
)
from ml.optimization.search import (
    get_search_space_definition,
    get_search_space_identity,
    summarize_study,
    validate_study_execution,
)


def valid_manifest():
    return build_frozen_manifest(
        experiment_id="test_exp",
        target="target_return_5d",
        task_type="regression",
        winning_model_family="XGBoost",
        best_hyperparameters={
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        feature_schema=["f1", "f2"],
        inner_fold_geometry=[
            {
                "fold_id": 1,
                "raw_fit_count": 100,
                "purged_fit_count": 95,
                "selection_count": 20,
                "selection_start": "2026-03-01",
                "context_count": 10,
            },
            {
                "fold_id": 2,
                "raw_fit_count": 120,
                "purged_fit_count": 115,
                "selection_count": 20,
                "selection_start": "2026-04-01",
                "context_count": 10,
            },
        ],
        purge_length=5,
        dataset_manifest="dataset-hash",
        lookback=None,
        model_architecture={
            "family": "XGBoost",
            "model_type": "gradient_boosted_tree",
            "n_estimators": 1000,
            "early_stopping_rounds": 30,
        },
        early_stopping_configuration={
            "early_stopping_rounds": 30,
            "n_estimators": 1000,
            "selection_rule": "best_validation_iteration",
        },
        protocol_a_configuration={
            "fit_fold": 2,
            "selection_fold": 2,
            "refit_fresh_model": True,
            "outer_retraining": False,
            "early_stopping_source": ("fold_2_inner_selection"),
        },
        optuna_trial_budget=30,
        optuna_sampler_configuration={
            "sampler": "TPESampler",
            "seed": 42,
            "n_startup_trials": 10,
            "n_jobs": 1,
            "pruner": "NopPruner",
        },
        search_space_identity="xgboost.v1",
        search_space_definition=(get_search_space_definition("XGBoost")),
        optuna_trial_metadata={
            "winning_trial": {"number": 1},
            "studies": {},
        },
    )


def test_search_spaces_are_explicit():
    for family in (
        "XGBoost",
        "LSTM",
        "Transformer",
    ):
        definition = get_search_space_definition(family)

        assert definition["identity"]
        assert definition["parameters"]
        assert "fixed_parameters" in definition
        assert get_search_space_identity(family) == definition["identity"]


def test_manifest_hash_is_deterministic():
    a = valid_manifest()
    b = copy.deepcopy(a)

    b["best_hyperparameters"] = {
        key: a["best_hyperparameters"][key] for key in reversed(list(a["best_hyperparameters"]))
    }

    _, hash_a = generate_frozen_manifest(a)
    _, hash_b = generate_frozen_manifest(b)

    assert hash_a == hash_b


def test_manifest_hash_changes_after_configuration_change():
    config = valid_manifest()

    _, original = generate_frozen_manifest(config)

    changed = copy.deepcopy(config)
    changed["random_seed"] = 43

    _, changed_hash = generate_frozen_manifest(changed)

    assert original != changed_hash


def test_manifest_rejects_nonfinite_values():
    config = valid_manifest()
    config["best_hyperparameters"]["learning_rate"] = np.nan

    with pytest.raises(ValueError):
        generate_frozen_manifest(config)


def test_study_metadata_is_retained():
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=1,
        ),
        pruner=optuna.pruners.NopPruner(),
    )

    study.optimize(
        lambda trial: trial.suggest_float("x", 0.0, 1.0),
        n_trials=3,
    )

    metadata = summarize_study(study)

    assert metadata["n_trials"] == 3
    assert len(metadata["trials"]) == 3
    assert metadata["best_trial_number"] is not None


def test_wrong_trial_budget_is_rejected():
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=1,
        ),
        pruner=optuna.pruners.NopPruner(),
    )

    study.optimize(
        lambda trial: trial.suggest_float("x", 0.0, 1.0),
        n_trials=2,
    )

    with pytest.raises(
        RuntimeError,
        match="trial-count mismatch",
    ):
        validate_study_execution(
            study,
            expected_n_trials=3,
        )
