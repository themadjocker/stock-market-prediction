from __future__ import annotations

import pytest

from ml.optimization.manifest import (
    REQUIRED_MANIFEST_FIELDS,
    build_frozen_manifest,
    generate_frozen_manifest,
    validate_frozen_manifest,
)


def make_sample_manifest_config() -> dict:
    return build_frozen_manifest(
        experiment_id="exp_test_001",
        target="target_return_5d",
        task_type="regression",
        winning_model_family="XGBoost",
        best_hyperparameters={"max_depth": 4, "learning_rate": 0.05},
        feature_schema=["f1", "f2"],
        inner_fold_geometry=[
            {
                "fold_id": 1,
                "raw_fit_count": 60,
                "purged_fit_count": 5,
                "selection_count": 20,
            },
            {
                "fold_id": 2,
                "raw_fit_count": 80,
                "purged_fit_count": 5,
                "selection_count": 20,
            },
        ],
        purge_length=5,
        dataset_manifest="abc123hash",
        random_seed=42,
        optuna_trial_budget=30,
    )


def test_manifest_contains_all_required_fields() -> None:
    config = make_sample_manifest_config()
    validate_frozen_manifest(config)
    for field in REQUIRED_MANIFEST_FIELDS:
        assert field in config, f"Missing required manifest field: {field}"


def test_manifest_missing_field_raises() -> None:
    config = make_sample_manifest_config()
    del config["winning_model_family"]
    with pytest.raises(ValueError, match="missing required fields"):
        generate_frozen_manifest(config)


def test_manifest_deterministic_json_and_sha256() -> None:
    config1 = make_sample_manifest_config()
    config2 = make_sample_manifest_config()

    json1, hash1 = generate_frozen_manifest(config1)
    json2, hash2 = generate_frozen_manifest(config2)

    assert json1 == json2
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex string length


def test_sha256_changes_when_configuration_changes() -> None:
    base_config = make_sample_manifest_config()
    _, base_hash = generate_frozen_manifest(base_config)

    # 1. Change hyperparameter
    c1 = make_sample_manifest_config()
    c1["best_hyperparameters"]["max_depth"] = 5
    _, h1 = generate_frozen_manifest(c1)
    assert h1 != base_hash

    # 2. Change seed
    c2 = make_sample_manifest_config()
    c2["random_seed"] = 99
    _, h2 = generate_frozen_manifest(c2)
    assert h2 != base_hash

    # 3. Change winning model family
    c3 = make_sample_manifest_config()
    c3["winning_model_family"] = "LSTM"
    _, h3 = generate_frozen_manifest(c3)
    assert h3 != base_hash

    # 4. Change purge length
    c4 = make_sample_manifest_config()
    c4["purge_length"] = 1
    _, h4 = generate_frozen_manifest(c4)
    assert h4 != base_hash
