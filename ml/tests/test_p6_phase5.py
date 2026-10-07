from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scripts.run_p6 as runner

from ml.models.target import target_spec_for
from ml.optimization.search import get_search_space_identity

ACTIVE_TARGETS = (
    "target_return_1d",
    "target_return_5d",
    "target_up_1d",
    "target_up_5d",
)


def test_phase5_active_target_matrix_is_exact() -> None:
    assert tuple(runner.target_spec_for(name).name for name in ACTIVE_TARGETS) == ACTIVE_TARGETS
    assert [target_spec_for(name).horizon for name in ACTIVE_TARGETS] == [1, 5, 1, 5]
    assert [target_spec_for(name).task_type for name in ACTIVE_TARGETS] == [
        "regression",
        "regression",
        "classification",
        "classification",
    ]

    with pytest.raises(ValueError):
        target_spec_for("target_return_20d")


def test_phase5_runner_rejects_unlocked_optuna_budget(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="exactly 30 Optuna trials"):
        runner.run_p6("target_return_1d", n_trials=1, n_startup_trials=1)

    with pytest.raises(ValueError, match="exactly 10 startup trials"):
        runner.run_p6("target_return_1d", n_trials=30, n_startup_trials=1)


def test_phase5_runner_smoke_all_targets_and_freezes_complete_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F15/F16 contract smoke: exercise all four target routes without model cost."""
    monkeypatch.chdir(tmp_path)

    class FakeXGBoost:
        def __init__(self, **kwargs):
            self._selection_best_iteration = 0

        def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
            self._is_classification = str(y_train.columns[0]).startswith("target_up_")
            return self

        def predict(self, X, **kwargs):
            return np.zeros(len(X), dtype=float)

        def predict_proba(self, X, **kwargs):
            return np.tile(np.array([[0.5, 0.5]], dtype=float), (len(X), 1))

    def fake_objective_factory(*args, **kwargs):
        def objective(trial):
            return 0.0

        return objective

    monkeypatch.setattr(runner, "XGBoostModel", FakeXGBoost)
    monkeypatch.setattr(runner, "create_xgboost_objective", fake_objective_factory)
    monkeypatch.setattr(runner, "create_lstm_objective", fake_objective_factory)
    monkeypatch.setattr(runner, "create_transformer_objective", fake_objective_factory)

    for target in ACTIVE_TARGETS:
        result = runner.run_p6(target, n_trials=30, n_startup_trials=10, seed=42)

        assert result["target"] == target
        assert result["winning_family"] == "XGBoost"
        assert len(result["manifest_hash"]) == 64

        manifest_path = tmp_path / f"p6_frozen_{target}.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["target"] == target
        assert manifest["optuna_trial_budget"] == 30
        assert manifest["optuna_sampler_configuration"]["n_startup_trials"] == 10
        assert manifest["search_space_identity"] == get_search_space_identity("XGBoost")
        assert manifest["optuna_trial_metadata"]["winning_model_family"] == "XGBoost"
        assert set(manifest["optuna_trial_metadata"]["studies"]) == {
            "XGBoost",
            "LSTM",
            "Transformer",
        }
        for study in manifest["optuna_trial_metadata"]["studies"].values():
            assert study["n_trials"] == 30
            assert study["best_trial_number"] is not None


def test_phase5_runner_uses_target_specific_horizon_in_final_result(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    class FakeXGBoost:
        def __init__(self, **kwargs):
            self._selection_best_iteration = 0

        def fit(self, X_train, y_train, **kwargs):
            return self

        def predict(self, X, **kwargs):
            return np.zeros(len(X), dtype=float)

        def predict_proba(self, X, **kwargs):
            return np.tile(np.array([[0.5, 0.5]], dtype=float), (len(X), 1))

    monkeypatch.setattr(runner, "XGBoostModel", FakeXGBoost)
    monkeypatch.setattr(runner, "create_xgboost_objective", lambda *a, **k: lambda trial: 0.0)
    monkeypatch.setattr(runner, "create_lstm_objective", lambda *a, **k: lambda trial: 0.0)
    monkeypatch.setattr(runner, "create_transformer_objective", lambda *a, **k: lambda trial: 0.0)

    one_day = runner.run_p6("target_return_1d")
    five_day = runner.run_p6("target_return_5d")
    assert one_day["target_horizon"] == 1
    assert five_day["target_horizon"] == 5
