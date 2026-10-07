from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np
import optuna

from ml.models.base import BaseModel
from ml.models.lstm_model import LSTMModel
from ml.models.metrics import calculate_classification_metrics, calculate_metrics
from ml.models.transformer_model import TransformerModel
from ml.models.xgboost_model import XGBoostModel
from ml.optimization.folds import InnerFoldData


class InvalidOptimizationTrial(RuntimeError):
    """Expected trial-level invalidity such as non-finite numerical results."""


def _finite_mean(values: list[float], *, label: str) -> float:
    """Return the arithmetic mean, rejecting empty or non-finite inputs."""
    if not values:
        raise InvalidOptimizationTrial(f"No values available for {label}.")
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise InvalidOptimizationTrial(f"Non-finite value encountered in {label}.")
    result = float(np.mean(array))
    if not np.isfinite(result):
        raise InvalidOptimizationTrial(f"Non-finite aggregate produced for {label}.")
    return result


def finite_mean(values: list[float], *, label: str) -> float:
    """Public wrapper for the P6 finite-mean contract used by tests."""
    return _finite_mean(values, label=label)


SEARCH_SPACE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "XGBoost": {
        "identity": "xgboost.v1",
        "parameters": {
            "max_depth": {
                "type": "int",
                "low": 2,
                "high": 8,
            },
            "learning_rate": {
                "type": "float",
                "low": 1e-3,
                "high": 0.3,
                "log": True,
            },
            "subsample": {
                "type": "float",
                "low": 0.5,
                "high": 1.0,
            },
            "colsample_bytree": {
                "type": "float",
                "low": 0.5,
                "high": 1.0,
            },
        },
        "fixed_parameters": {
            "n_estimators": 1000,
            "early_stopping_rounds": 30,
        },
    },
    "LSTM": {
        "identity": "lstm.v1",
        "parameters": {
            "lookback": {
                "type": "categorical",
                "choices": [5, 10],
            },
            "hidden_dim": {
                "type": "categorical",
                "choices": [16, 32, 64],
            },
            "num_layers": {
                "type": "int",
                "low": 1,
                "high": 2,
            },
            "dropout": {
                "type": "float",
                "low": 0.0,
                "high": 0.3,
            },
            "lr": {
                "type": "float",
                "low": 1e-4,
                "high": 1e-2,
                "log": True,
            },
        },
        "fixed_parameters": {
            "epochs": 15,
            "max_lookback": 10,
        },
    },
    "Transformer": {
        "identity": "transformer.v1",
        "parameters": {
            "lookback": {
                "type": "categorical",
                "choices": [5, 10],
            },
            "d_model": {
                "type": "categorical",
                "choices": [16, 32, 64],
            },
            "nhead": {
                "type": "categorical",
                "choices": [2, 4],
            },
            "num_layers": {
                "type": "int",
                "low": 1,
                "high": 2,
            },
            "dropout": {
                "type": "float",
                "low": 0.0,
                "high": 0.3,
            },
            "lr": {
                "type": "float",
                "low": 1e-4,
                "high": 1e-2,
                "log": True,
            },
        },
        "fixed_parameters": {
            "epochs": 15,
            "max_lookback": 10,
        },
    },
}


def get_search_space_definition(model_family: str) -> dict[str, Any]:
    """Return a defensive copy of the locked search-space definition."""
    if model_family not in SEARCH_SPACE_DEFINITIONS:
        raise ValueError(f"Unknown model family: {model_family!r}")
    return copy.deepcopy(SEARCH_SPACE_DEFINITIONS[model_family])


def get_search_space_identity(model_family: str) -> str:
    """Return the versioned identity of a model family's search space."""
    return str(get_search_space_definition(model_family)["identity"])


def get_search_space_hash(model_family: str) -> str:
    """Return a deterministic SHA-256 hash of a model family's search space."""
    definition = get_search_space_definition(model_family)
    canonical = json.dumps(
        definition,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evaluate_model_on_folds(
    model_cls: type[BaseModel],
    model_params: dict[str, Any],
    folds: list[InnerFoldData],
    task_type: str,
    max_lookback: int | None = None,
    trial: optuna.Trial | None = None,
    model_family: str | None = None,
    target_name: str | None = None,
) -> float:
    """Evaluate a hyperparameter configuration across precomputed inner folds."""
    fold_metrics: list[float] = []
    fold_ticker_metrics: list[dict[str, Any]] = []
    fold_best_steps: list[int | None] = []

    for fold in folds:
        params = dict(model_params)
        if max_lookback is not None and model_cls in (LSTMModel, TransformerModel):
            params["max_lookback"] = max_lookback

        model = model_cls(**params)

        fit_kwargs: dict[str, Any] = {
            "X_train": fold.X_fit,
            "y_train": fold.y_fit,
            "X_val": fold.X_selection,
            "y_val": fold.y_selection,
            "metadata_train": fold.metadata_fit,
            "metadata_val": fold.metadata_selection,
        }
        if hasattr(model, "lookback"):
            fit_kwargs["val_history_X"] = fold.X_context
            fit_kwargs["val_history_metadata"] = fold.metadata_context

        try:
            model.fit(**fit_kwargs)
        except ValueError as exc:
            if "Non-finite" in str(exc) or "non-finite" in str(exc):
                raise InvalidOptimizationTrial(str(exc)) from exc
            raise

        if hasattr(model, "lookback"):
            if task_type == "classification":
                y_pred = model.predict_proba(
                    fold.X_selection,
                    metadata=fold.metadata_selection,
                    history_X=fold.X_context,
                    history_metadata=fold.metadata_context,
                )
            else:
                y_pred = model.predict(
                    fold.X_selection,
                    metadata=fold.metadata_selection,
                    history_X=fold.X_context,
                    history_metadata=fold.metadata_context,
                )
        else:
            if task_type == "classification":
                y_pred = model.predict_proba(fold.X_selection)
            else:
                y_pred = model.predict(fold.X_selection)

        if not np.isfinite(y_pred).all():
            raise InvalidOptimizationTrial("Model generated non-finite predictions.")

        y_true = fold.y_selection.iloc[:, 0].to_numpy(dtype=float)
        if not np.isfinite(y_true).all():
            raise InvalidOptimizationTrial("Selection targets contain non-finite values.")

        tickers = fold.metadata_selection["ticker"].to_numpy()
        unique_tickers = np.unique(tickers)

        ticker_scores: list[float] = []
        ticker_metrics: dict[str, Any] = {}

        for ticker in unique_tickers:
            mask = tickers == ticker
            yt = y_true[mask]
            yp = y_pred[mask]

            if len(yt) == 0:
                continue

            if task_type == "classification":
                res = calculate_classification_metrics(yt, yp)
                score = res["logloss"]
            else:
                res = calculate_metrics(yt, yp)
                score = res["relative_rmse"]

            if not np.isfinite(score):
                raise InvalidOptimizationTrial(f"Non-finite metric calculated for ticker '{ticker}': {score}")

            ticker_scores.append(score)
            ticker_metrics[str(ticker)] = res

        if not ticker_scores:
            raise InvalidOptimizationTrial("No valid predictions for any ticker.")

        fold_score = _finite_mean(ticker_scores, label="ticker scores")

        fold_metrics.append(fold_score)
        fold_ticker_metrics.append(ticker_metrics)

        best_step = getattr(
            model,
            "_selection_best_iteration",
            getattr(model, "_selection_best_epoch", None),
        )
        fold_best_steps.append(best_step)

    if not fold_metrics:
        raise InvalidOptimizationTrial("No valid inner folds were evaluated.")

    overall_score = _finite_mean(fold_metrics, label="fold scores")

    if trial is not None:
        trial.set_user_attr("model_family", model_family)
        trial.set_user_attr("target", target_name)
        trial.set_user_attr("task_type", task_type)
        trial.set_user_attr("lookback", model_params.get("lookback"))
        trial.set_user_attr("max_lookback", max_lookback)
        trial.set_user_attr("fold_scores", fold_metrics)
        trial.set_user_attr("fold_ticker_metrics", fold_ticker_metrics)
        trial.set_user_attr("fold_best_iteration_or_epoch", fold_best_steps)
        trial.set_user_attr("objective_score", overall_score)

    return overall_score


def create_xgboost_objective(
    folds: list[InnerFoldData],
    task_type: str,
    target_name: str | None = None,
) -> Any:
    def objective(trial: optuna.Trial) -> float:
        spec = get_search_space_definition("XGBoost")
        p = spec["parameters"]
        fixed = spec["fixed_parameters"]

        params = {
            "n_estimators": fixed["n_estimators"],
            "early_stopping_rounds": fixed["early_stopping_rounds"],
            "max_depth": trial.suggest_int(
                "max_depth",
                p["max_depth"]["low"],
                p["max_depth"]["high"],
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                p["learning_rate"]["low"],
                p["learning_rate"]["high"],
                log=True,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                p["subsample"]["low"],
                p["subsample"]["high"],
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                p["colsample_bytree"]["low"],
                p["colsample_bytree"]["high"],
            ),
        }
        return evaluate_model_on_folds(
            XGBoostModel,
            params,
            folds,
            task_type,
            trial=trial,
            model_family="XGBoost",
            target_name=target_name,
        )

    return objective


def create_lstm_objective(
    folds: list[InnerFoldData],
    task_type: str,
    target_name: str | None = None,
) -> Any:
    def objective(trial: optuna.Trial) -> float:
        spec = get_search_space_definition("LSTM")
        p = spec["parameters"]
        fixed = spec["fixed_parameters"]

        params = {
            "lookback": trial.suggest_categorical(
                "lookback",
                p["lookback"]["choices"],
            ),
            "hidden_dim": trial.suggest_categorical(
                "hidden_dim",
                p["hidden_dim"]["choices"],
            ),
            "num_layers": trial.suggest_int(
                "num_layers",
                p["num_layers"]["low"],
                p["num_layers"]["high"],
            ),
            "dropout": trial.suggest_float(
                "dropout",
                p["dropout"]["low"],
                p["dropout"]["high"],
            ),
            "lr": trial.suggest_float(
                "lr",
                p["lr"]["low"],
                p["lr"]["high"],
                log=True,
            ),
            "epochs": fixed["epochs"],
        }
        return evaluate_model_on_folds(
            LSTMModel,
            params,
            folds,
            task_type,
            max_lookback=fixed["max_lookback"],
            trial=trial,
            model_family="LSTM",
            target_name=target_name,
        )

    return objective


def create_transformer_objective(
    folds: list[InnerFoldData],
    task_type: str,
    target_name: str | None = None,
) -> Any:
    def objective(trial: optuna.Trial) -> float:
        spec = get_search_space_definition("Transformer")
        p = spec["parameters"]
        fixed = spec["fixed_parameters"]

        params = {
            "lookback": trial.suggest_categorical(
                "lookback",
                p["lookback"]["choices"],
            ),
            "d_model": trial.suggest_categorical(
                "d_model",
                p["d_model"]["choices"],
            ),
            "nhead": trial.suggest_categorical(
                "nhead",
                p["nhead"]["choices"],
            ),
            "num_layers": trial.suggest_int(
                "num_layers",
                p["num_layers"]["low"],
                p["num_layers"]["high"],
            ),
            "dropout": trial.suggest_float(
                "dropout",
                p["dropout"]["low"],
                p["dropout"]["high"],
            ),
            "lr": trial.suggest_float(
                "lr",
                p["lr"]["low"],
                p["lr"]["high"],
                log=True,
            ),
            "epochs": fixed["epochs"],
        }
        return evaluate_model_on_folds(
            TransformerModel,
            params,
            folds,
            task_type,
            max_lookback=fixed["max_lookback"],
            trial=trial,
            model_family="Transformer",
            target_name=target_name,
        )

    return objective


def summarize_study(study: optuna.Study) -> dict[str, Any]:
    """Create an audit-friendly summary of every Optuna trial."""
    trials: list[dict[str, Any]] = []

    for trial in study.trials:
        trials.append(
            {
                "number": int(trial.number),
                "state": trial.state.name,
                "value": None if trial.value is None else float(trial.value),
                "params": dict(trial.params),
                "user_attrs": copy.deepcopy(dict(trial.user_attrs)),
            }
        )

    complete_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]

    best_trial_number = None
    best_value = None
    if complete_trials:
        best_trial = study.best_trial
        best_trial_number = int(best_trial.number)
        if best_trial.value is None:
            raise RuntimeError("Optuna reported a best trial without an objective value.")
        best_value = float(best_trial.value)

    state_counts: dict[str, int] = {}
    for trial in study.trials:
        state = trial.state.name
        state_counts[state] = state_counts.get(state, 0) + 1

    return {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "n_trials": len(study.trials),
        "best_trial_number": best_trial_number,
        "best_value": best_value,
        "state_counts": state_counts,
        "trials": trials,
    }


def validate_study_execution(
    study: optuna.Study,
    expected_n_trials: int = 30,
) -> None:
    """Validate the locked execution invariants for a completed P6 study."""
    if len(study.trials) != expected_n_trials:
        raise RuntimeError(f"Optuna trial-count mismatch: expected {expected_n_trials}, got {len(study.trials)}")

    pruned = [trial.number for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED]
    if pruned:
        raise RuntimeError(f"P6 forbids Optuna pruning, but trials were pruned: {pruned}")

    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("P6 study produced no completed trials.")
